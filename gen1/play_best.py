"""
Play the best trained genome in Slither.io
Loads best_genome.pkl and runs it in a visible browser window.
"""

import time
import os
import pickle
import neat

from browser_engine import SlitherBrowser
from spatial_awareness import SpatialAwareness
from ai_brain import BotAgent


def play_best_genome(genome_path='best_genome.pkl', config_path='config_neat.txt'):
    """
    Load and play the best trained genome.
    """
    # Load NEAT config
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    # Load the best genome
    print(f"Loading genome from: {genome_path}")
    with open(genome_path, 'rb') as f:
        genome = pickle.load(f)
    
    print(f"Genome loaded! Fitness: {genome.fitness if hasattr(genome, 'fitness') else 'N/A'}")

    # Create brain from genome
    brain = BotAgent(genome, config)
    spatial = SpatialAwareness()

    # Start browser (visible mode for watching)
    print("Starting browser...")
    browser = SlitherBrowser(headless=False, nickname="NEAT-Champion")
    
    print("\n" + "=" * 50)
    print("  Bot is running! Watch the game in the browser.")
    print("  Press Ctrl+C to stop.")
    print("=" * 50 + "\n")

    games_played = 0
    
    try:
        while True:
            games_played += 1
            print(f"\n--- Game #{games_played} ---")
            
            # Start game
            browser.force_restart()
            time.sleep(2)  # Wait for game to start
            
            start_time = time.time()
            max_len = 0
            food_eaten = 0
            
            # Main game loop
            while True:
                try:
                    data = browser.get_game_data()
                except Exception as e:
                    print(f"Error getting game data: {e}")
                    break

                if data is None:
                    time.sleep(0.1)
                    continue

                # Check if dead
                if data.get('dead', False):
                    survival_time = time.time() - start_time
                    print(f"Game Over! Survived: {survival_time:.1f}s | Food: {food_eaten} | Max Length: {max_len}")
                    time.sleep(3)  # Pause before restart
                    break

                my_snake = data.get('self')
                if not my_snake:
                    time.sleep(0.1)
                    continue

                # Track stats
                current_len = my_snake.get('len', 0)
                if current_len > max_len:
                    food_eaten += (current_len - max_len)
                    max_len = current_len
                    print(f"  Length: {max_len} | Food eaten: {food_eaten}")

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

                time.sleep(0.05)  # ~20 FPS

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    finally:
        browser.close()
        print("Browser closed.")


def play_from_checkpoint(checkpoint_path, config_path='config_neat.txt'):
    """
    Load the best genome from a checkpoint and play it.
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    pop = neat.Checkpointer.restore_checkpoint(checkpoint_path)
    
    # Find the best genome in the population
    best_genome = None
    best_fitness = float('-inf')
    
    for species in pop.species.species.values():
        for genome_id, genome in species.members.items():
            if genome.fitness is not None and genome.fitness > best_fitness:
                best_fitness = genome.fitness
                best_genome = genome
    
    if best_genome is None:
        print("No genome with fitness found in checkpoint!")
        return
    
    print(f"Best genome fitness: {best_fitness}")
    
    # Save it temporarily and play
    with open('_temp_best.pkl', 'wb') as f:
        pickle.dump(best_genome, f)
    
    play_best_genome('_temp_best.pkl', config_path)
    
    # Cleanup
    if os.path.exists('_temp_best.pkl'):
        os.remove('_temp_best.pkl')


if __name__ == "__main__":
    import sys
    
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config_neat.txt')
    
    if len(sys.argv) > 1:
        # If argument provided, treat as checkpoint path
        checkpoint = sys.argv[1]
        if checkpoint.startswith('neat-checkpoint'):
            play_from_checkpoint(checkpoint, config_path)
        else:
            play_best_genome(checkpoint, config_path)
    else:
        # Default: load best_genome.pkl
        genome_path = os.path.join(local_dir, 'best_genome.pkl')
        
        if os.path.exists(genome_path):
            play_best_genome(genome_path, config_path)
        else:
            print("ERROR: best_genome.pkl not found!")
            print("Either train the bot first or provide a checkpoint path:")
            print("  python play_best.py neat-checkpoint-100")
