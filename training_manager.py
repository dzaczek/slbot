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

def log(msg):
    """Print with flush for immediate output."""
    print(msg, flush=True)

def eval_genome(genome, config):
    """
    Evaluates a single genome.
    Returns fitness score.
    """
    global browser
    if browser is None:
        log("[INIT] Starting browser...")
        browser = SlitherBrowser(headless=False)
        log("[INIT] Browser started successfully.")
        
    brain = BotAgent(genome, config)
    spatial = SpatialAwareness()
    
    # Start game
    browser.force_restart()
    # Wait a moment for connection
    time.sleep(1)
    
    start_time = time.time()
    last_eat_time = start_time
    max_len = 0
    
    # Anti-loop / activity vars
    last_pos = (0, 0)
    stuck_time = 0
    
    while True:
        # 1. Get Data
        data = browser.get_game_data()
        
        if data.get('dead', False):
            break
            
        my_snake = data.get('self')
        if not my_snake:
            # Maybe still connecting?
            if time.time() - start_time > 5:
                # Timeout connecting
                break
            continue
            
        # 2. Process fitness / state
        current_len = my_snake.get('len', 0)
        if current_len > max_len:
            max_len = current_len
            last_eat_time = time.time() # Reset starvation timer
            
        # Anti-Loop / Camping Penalty
        # If length hasn't increased in 10s -> kill (to prevent safe circling forever)
        if time.time() - last_eat_time > 20: # 20s strictly for starvation
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
        
        # Throttle loop slightly ideally, but Selenium overhead is already high
        # time.sleep(0.01)

    # Calculate Fitness
    # Combines survival time and length
    survival_time = time.time() - start_time
    fitness = survival_time + (max_len * 10) 
    # High reward for length, low for just camping
    
    return fitness

def run_neat(config_path):
    config = neat.Config(
        neat.DefaultGenome, 
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet, 
        neat.DefaultStagnation,
        config_path
    )
    
    p = neat.Population(config)
    
    # Add reporters
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    # Train
    # We need a wrapper to match neat's requirement of (genomes, config) if we use run()
    # Or we can use ParallelEvaluator if needed, but for browser it's single threaded usually.
    # Standard p.run takes a function `eval_genomes(genomes, config)`
    winner = p.run(eval_genomes_wrapper, 50) # 50 generations
    
    print('\nBest genome:\n{!s}'.format(winner))

def eval_genomes_wrapper(genomes, config):
    """
    Wrapper for batch evaluation.
    """
    for genome_id, genome in genomes:
        log(f"[EVAL] Evaluating Genome {genome_id}...")
        genome.fitness = eval_genome(genome, config)
        log(f"[RESULT] Genome {genome_id} Fitness: {genome.fitness:.2f}")

if __name__ == "__main__":
    log("========================================")
    log("  Slither.io NEAT Bot - Training Start")
    log("========================================")
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config_neat.txt')
    log(f"[CONFIG] Loading config from: {config_path}")
    run_neat(config_path)
