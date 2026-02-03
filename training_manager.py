
import time
import os
import sys
import neat
import math
from browser_engine import SlitherBrowser
from spatial_awareness import SpatialAwareness
from ai_brain import BotAgent

# Force unbuffered output
sys.stdout = sys.stderr

# Global browser instance to reuse chrome window
browser = None

import atexit
def cleanup():
    global browser
    if browser:
        try:
            browser.close()
        except:
            pass
atexit.register(cleanup)
def log(msg):
    """Print with flush for immediate output."""
    print(msg, flush=True)

def eval_genome(genome, config):
    """
    Evaluates a single genome.
    Returns fitness score.
    """
    global browser
    
    # Ensure browser is running
    def ensure_browser():
        global browser
        if browser is None:
            log("[INIT] Starting browser...")
            browser = SlitherBrowser(headless=False)
            log("[INIT] Browser started successfully.")
    
    ensure_browser()
    
    brain = BotAgent(genome, config)
    spatial = SpatialAwareness()
    
    # Start game with recovery
    try:
         # force_restart now waits for game start internally
        browser.force_restart()
    except Exception as e:
        # Check if error is fatal (window closed)
        err_msg = str(e).lower()
        if "no such window" in err_msg or "target window already closed" in err_msg or "invalid session id" in err_msg:
            log(f"[FATAL ERROR] Browser window lost, restarting: {e}")
            try:
                browser.close()
            except:
                pass
            browser = None
            ensure_browser()
            browser.force_restart()
        else:
            log(f"[RECOVERABLE ERROR] {e}. Continuing...")
    
    # Wait a bit longer to ensure game state is fully synchronized
    time.sleep(2)
    
    start_time = time.time()
    last_eat_time = start_time
    max_len = 0
    fitness_score = 0.0
    
    eval_timeout = 180 # 3 minutes max per genome
    
    # 1. Main Game Loop
    while time.time() - start_time < eval_timeout:
        # 1. Get Data
        data = browser.get_game_data()
        
        if data is None:
            log("[ERROR] get_game_data returned None")
            break
            
        if data.get('dead', False):
            # If in menu, try to play
            if data.get('in_menu', False):
                log("[STATUS] In menu. Attempting to start game...")
                browser.force_restart()
                time.sleep(5) # Wait longer for game to actually start
                continue
                
            # If dead but just started, it's a connection/spawn issue
            if time.time() - start_time < 3:
                 log("[WARNING] Dead/Disconnected immediately. Retrying...")
                 browser.force_restart()
                 start_time = time.time()
                 last_eat_time = start_time
                 continue
            break

            
        my_snake = data.get('self')
        if not my_snake:
            # Maybe still connecting?
            if time.time() - start_time > 10:
                # Timeout connecting
                break
            continue
            
        # 2. Process fitness / state (with accumulated rewards for eating)
        current_len = my_snake.get('len', 0)
        
        # Reward for growing (eating food)
        if current_len > max_len:
            diff = current_len - max_len
            # Large bonus for every piece of food eaten
            fitness_score += (diff * 20.0) 
            max_len = current_len
            last_eat_time = time.time() # Reset starvation timer
            
        # Anti-Loop / Camping Penalty
        # If length hasn't increased in 25s -> kill
        if time.time() - last_eat_time > 25: 
           log(f"[TIMEOUT] Starved. Len: {max_len}")
           break

        # 3. Decision
        inputs = spatial.calculate_sectors(
            my_snake, 
            data.get('enemies', []), 
            data.get('foods', [])
        )
        
        angle, boost = brain.decide(inputs)
        
        # 4. Act
        browser.send_action(angle, boost)
        
    # Calculate Fitness
    # Base: Survival time (increased reward to encourage longevity)
    survival_time = time.time() - start_time
    fitness_score += (survival_time * 5.0)
    
    # Final length bonus (redundant but good for baseline)
    fitness_score += (max_len * 5)
    
    return fitness_score

def run_neat(config_path):
    config = neat.Config(
        neat.DefaultGenome, 
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet, 
        neat.DefaultStagnation,
        config_path
    )
    
    p = None
    
    # Check for latest checkpoint
    checkpoint_files = [f for f in os.listdir('.') if f.startswith('neat-checkpoint-')]
    if checkpoint_files:
        # Sort by generation number (neat-checkpoint-X)
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
    
    # Add checkpointer to save progress every 5 generations
    checkpointer = neat.Checkpointer(generation_interval=5, filename_prefix='neat-checkpoint-')
    p.add_reporter(checkpointer)
    
    log("[TRAINING] Starting evolution for 50 generations...")
    log("[TRAINING] Checkpoints will be saved every 5 generations")
    
    # Train
    winner = p.run(eval_genomes_wrapper, 50) # 50 generations
    
    # Save winner
    import pickle
    with open('best_genome.pkl', 'wb') as f:
        pickle.dump(winner, f)
    log("[TRAINING] Best genome saved to best_genome.pkl")
    
    print('\\nBest genome:\\n{!s}'.format(winner))

def eval_genomes_wrapper(genomes, config):
    """
    Wrapper for batch evaluation.
    """
    for genome_id, genome in genomes:
        log(f"[EVAL] Evaluating Genome {genome_id}...")
        genome.fitness = eval_genome(genome, config)
        log(f"[RESULT] Genome {genome_id} Fitness: {genome.fitness:.2f}")

if __name__ == "__main__":
    try:
        log("========================================")
        log("  Slither.io NEAT Bot - Training Start")
        log("========================================")
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config_neat.txt')
        log(f"[CONFIG] Loading config from: {config_path}")
        run_neat(config_path)
    except Exception as e:
        import traceback
        with open("fatal_error.log", "w") as f:
            f.write(f"Fatal crash: {e}\n")
            f.write(traceback.format_exc())
        log(f"FATAL ERROR: {e}")
        traceback.print_exc()
