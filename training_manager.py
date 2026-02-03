"""
NEAT Training Manager for Slither.io Bot
Supports parallel evaluation with multiple browser instances.
"""

import time
import os
import sys
import neat
import math
import csv
import pickle
import multiprocessing
from datetime import datetime
from queue import Empty

# Force unbuffered output
sys.stdout = sys.stderr

def log(msg):
    """Print with flush for immediate output."""
    print(msg, flush=True)


# ============================================
# WORKER PROCESS - runs in separate process
# ============================================

class BrowserManager:
    """Manages browser instance for a worker."""
    def __init__(self, worker_id, headless=False):
        self.worker_id = worker_id
        self.headless = headless
        self.browser = None

    def ensure_browser(self):
        """Ensure browser is running, create if needed."""
        if self.browser is None:
            from browser_engine import SlitherBrowser
            mode = "headless" if self.headless else "visible"
            log(f"[WORKER {self.worker_id}] Starting browser ({mode})...")
            self.browser = SlitherBrowser(headless=self.headless, nickname=f"NEAT-{self.worker_id}")
            log(f"[WORKER {self.worker_id}] Browser ready.")
        return self.browser

    def restart_browser(self):
        """Close and recreate browser."""
        if self.browser:
            try:
                self.browser.close()
            except:
                pass
        self.browser = None
        return self.ensure_browser()

    def close(self):
        """Close browser."""
        if self.browser:
            try:
                self.browser.close()
            except:
                pass
            self.browser = None


def worker_process(worker_id, task_queue, result_queue, config_path, headless=False):
    """
    Worker process that evaluates genomes.
    Each worker has its own browser instance.
    """
    from spatial_awareness import SpatialAwareness
    from ai_brain import BotAgent

    log(f"[WORKER {worker_id}] Starting...")

    # Load NEAT config
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    # Browser manager
    browser_mgr = BrowserManager(worker_id, headless=headless)
    spatial = SpatialAwareness()

    try:
        while True:
            try:
                # Get task from queue (with timeout to allow graceful shutdown)
                task = task_queue.get(timeout=5)

                if task is None:  # Poison pill - shutdown signal
                    log(f"[WORKER {worker_id}] Received shutdown signal.")
                    break

                genome_id, genome_pickle = task

                # Deserialize genome
                genome = pickle.loads(genome_pickle)

                # Evaluate
                fitness, stats = eval_genome_worker(
                    worker_id, genome, config, browser_mgr, spatial
                )

                # Send result back
                result_queue.put((genome_id, fitness, stats))

            except Empty:
                # No task available, check if we should continue
                continue
            except Exception as e:
                log(f"[WORKER {worker_id}] Error: {e}")
                import traceback
                traceback.print_exc()
                # Try to recover browser
                browser_mgr.restart_browser()

    finally:
        browser_mgr.close()
        log(f"[WORKER {worker_id}] Shutdown complete.")


def eval_genome_worker(worker_id, genome, config, browser_mgr, spatial):
    """
    Evaluates a single genome in worker process.
    Returns (fitness, stats_dict).
    """
    from ai_brain import BotAgent

    brain = BotAgent(genome, config)

    # Ensure browser is running
    browser = browser_mgr.ensure_browser()

    # Start game with recovery
    max_retries = 3
    for attempt in range(max_retries):
        try:
            browser.force_restart()
            break
        except Exception as e:
            err_msg = str(e).lower()
            if "no such window" in err_msg or "target window" in err_msg or "invalid session" in err_msg:
                log(f"[WORKER {worker_id}] Browser lost, restarting...")
                browser = browser_mgr.restart_browser()
            else:
                log(f"[WORKER {worker_id}] Attempt {attempt+1} failed: {e}")
                time.sleep(1)
                if attempt == max_retries - 1:
                    browser = browser_mgr.restart_browser()

    # Wait for game to stabilize
    time.sleep(1.5)

    start_time = time.time()
    last_eat_time = start_time
    max_len = 0
    food_eaten_count = 0
    fitness_score = 0.0
    cause_of_death = "Unknown"

    eval_timeout = 120  # 2 minutes max per genome

    # Main Game Loop
    while time.time() - start_time < eval_timeout:
        # Get game data
        try:
            data = browser.get_game_data()
        except Exception as e:
            log(f"[WORKER {worker_id}] get_game_data error: {e}")
            browser = browser_mgr.restart_browser()
            break

        if data is None:
            break

        if data.get('dead', False):
            if data.get('in_menu', False):
                try:
                    browser.force_restart()
                except:
                    browser = browser_mgr.restart_browser()
                time.sleep(3)
                continue

            if time.time() - start_time < 3:
                try:
                    browser.force_restart()
                except:
                    browser = browser_mgr.restart_browser()
                start_time = time.time()
                last_eat_time = start_time
                continue

            cause_of_death = "Collision"
            break

        my_snake = data.get('self')
        if not my_snake:
            if time.time() - start_time > 10:
                break
            time.sleep(0.1)
            continue

        # Process fitness
        current_len = my_snake.get('len', 0)

        if current_len > max_len:
            diff = current_len - max_len
            fitness_score += (diff * 25.0)  # Bonus for eating
            food_eaten_count += diff
            max_len = current_len
            last_eat_time = time.time()

        # Starvation check
        if time.time() - last_eat_time > 20:
            cause_of_death = "Starvation"
            break

        # Get neural network decision
        inputs = spatial.calculate_sectors(
            my_snake,
            data.get('enemies', []),
            data.get('foods', [])
        )

        current_heading = my_snake.get('ang', 0)
        angle, boost = brain.decide(inputs, current_heading)

        # Execute action
        try:
            browser.send_action(angle, boost)
        except:
            pass

        # Small delay to avoid overwhelming the browser
        time.sleep(0.05)  # ~20 FPS

    # Calculate final fitness
    survival_time = time.time() - start_time
    fitness_score += (survival_time * 5.0)
    fitness_score += (max_len * 5)

    stats = {
        'survival': survival_time,
        'food': food_eaten_count,
        'cause': cause_of_death,
        'len': max_len
    }

    return fitness_score, stats


# ============================================
# MAIN PROCESS - manages workers
# ============================================

class ParallelEvaluator:
    """
    Manages parallel genome evaluation using multiple browser instances.
    """
    def __init__(self, num_workers, config_path, headless=False):
        self.num_workers = num_workers
        self.config_path = config_path
        self.headless = headless
        self.workers = []
        self.task_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()

    def start_workers(self):
        """Start worker processes."""
        log(f"[MAIN] Starting {self.num_workers} parallel workers...")

        for i in range(self.num_workers):
            p = multiprocessing.Process(
                target=worker_process,
                args=(i, self.task_queue, self.result_queue, self.config_path, self.headless)
            )
            p.start()
            self.workers.append(p)
            time.sleep(2)  # Stagger startup to avoid resource contention

        log(f"[MAIN] All {self.num_workers} workers started.")

    def stop_workers(self):
        """Stop all worker processes."""
        log("[MAIN] Stopping workers...")

        # Send poison pills
        for _ in self.workers:
            self.task_queue.put(None)

        # Wait for workers to finish
        for p in self.workers:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()

        self.workers = []
        log("[MAIN] All workers stopped.")

    def evaluate_genomes(self, genomes, config):
        """
        Evaluate all genomes in parallel.
        This is called by NEAT for each generation.
        """
        # Submit all tasks - serialize genomes with pickle
        pending = {}
        for genome_id, genome in genomes:
            genome_pickle = pickle.dumps(genome)
            self.task_queue.put((genome_id, genome_pickle))
            pending[genome_id] = genome

        log(f"[MAIN] Submitted {len(pending)} genomes for evaluation...")

        # Collect results
        results_count = 0
        completed_ids = set()
        
        while results_count < len(pending):
            try:
                genome_id, fitness, stats = self.result_queue.get(timeout=180)  # 3 min timeout

                genome = pending[genome_id]
                genome.fitness = fitness
                completed_ids.add(genome_id)

                # Log result
                log(f"[RESULT] Genome {genome_id} | Fit: {fitness:.2f} | "
                    f"Time: {stats['survival']:.1f}s | Food: {stats['food']} | "
                    f"Len: {stats['len']} | Cause: {stats['cause']}")

                # CSV logging
                self._log_to_csv(genome_id, fitness, stats)

                results_count += 1

            except Empty:
                log("[MAIN] Timeout waiting for results!")
                break

        # Assign penalty fitness to genomes that didn't return results
        # This prevents NEAT from crashing due to missing fitness values
        missing_count = 0
        for genome_id, genome in pending.items():
            if genome_id not in completed_ids:
                genome.fitness = 1.0  # Minimal fitness (penalty for timeout/crash)
                missing_count += 1
                log(f"[WARN] Genome {genome_id} timed out - assigned penalty fitness 1.0")
        
        log(f"[MAIN] Generation complete. {results_count}/{len(pending)} evaluated, {missing_count} timed out.")

    def _log_to_csv(self, genome_id, fitness, stats):
        """Log result to CSV file."""
        try:
            file_exists = os.path.isfile('training_stats.csv')
            with open('training_stats.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['Timestamp', 'Generation', 'GenomeID', 'Fitness',
                                   'SurvivalTime', 'FoodEaten', 'MaxLen', 'CauseOfDeath'])
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "N/A",
                    genome_id,
                    f"{fitness:.2f}",
                    f"{stats['survival']:.2f}",
                    stats['food'],
                    stats['len'],
                    stats['cause']
                ])
        except Exception as e:
            log(f"[ERROR] CSV write failed: {e}")


def run_neat(config_path, num_workers=3, headless=False):
    """
    Run NEAT evolution with parallel evaluation.
    """
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    p = None

    # Check for checkpoint
    checkpoint_files = [f for f in os.listdir('.') if f.startswith('neat-checkpoint-')]
    if checkpoint_files:
        try:
            latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('-')[-1]))
            log(f"[RESUME] Found checkpoint: {latest_checkpoint}. Restoring...")
            p = neat.Checkpointer.restore_checkpoint(latest_checkpoint)
        except Exception as e:
            log(f"[ERROR] Failed to restore checkpoint: {e}. Starting fresh.")

    if p is None:
        log("[START] No valid checkpoint found. Starting fresh population.")
        p = neat.Population(config)

    # Add reporters
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Add checkpointer - save every generation for safety
    checkpointer = neat.Checkpointer(generation_interval=1, filename_prefix='neat-checkpoint-')
    p.add_reporter(checkpointer)

    # Create parallel evaluator
    evaluator = ParallelEvaluator(num_workers, config_path, headless=headless)

    try:
        evaluator.start_workers()

        log(f"[TRAINING] Starting evolution with {num_workers} parallel workers...")
        log("[TRAINING] Checkpoints saved every 5 generations")

        # Run evolution
        winner = p.run(evaluator.evaluate_genomes, 100)  # 100 generations

        # Save winner
        with open('best_genome.pkl', 'wb') as f:
            pickle.dump(winner, f)
        log("[TRAINING] Best genome saved to best_genome.pkl")

        print(f'\nBest genome:\n{winner}')

    finally:
        evaluator.stop_workers()


if __name__ == "__main__":
    # Configuration
    NUM_WORKERS = 5  # Number of parallel browser instances (10 may overload system)
    HEADLESS = True  # Set to True for headless mode (faster, no windows)

    try:
        log("=" * 50)
        log("  Slither.io NEAT Bot - Parallel Training")
        log(f"  Workers: {NUM_WORKERS}")
        log("=" * 50)

        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config_neat.txt')
        log(f"[CONFIG] Loading config from: {config_path}")

        run_neat(config_path, num_workers=NUM_WORKERS, headless=HEADLESS)

    except KeyboardInterrupt:
        log("\n[MAIN] Training interrupted by user.")
    except Exception as e:
        import traceback
        with open("fatal_error.log", "w") as f:
            f.write(f"Fatal crash: {e}\n")
            f.write(traceback.format_exc())
        log(f"FATAL ERROR: {e}")
        traceback.print_exc()
