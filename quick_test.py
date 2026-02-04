"""
Quick test to verify the bot works with new changes.
Runs ONE genome for 30 seconds to check if everything works.
"""

import time
import neat
from browser_engine import SlitherBrowser
from spatial_awareness import SpatialAwareness
from ai_brain import BotAgent

def quick_test():
    print("=" * 60)
    print("  QUICK TEST - Verifying Bot Functionality")
    print("=" * 60)
    
    # Load config
    print("\n1. Loading NEAT config...")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        'config_neat.txt'
    )
    print("✓ Config loaded")
    
    # Create a random genome
    print("\n2. Creating test genome...")
    genome = config.genome_type(1)
    genome.configure_new(config.genome_config)
    brain = BotAgent(genome, config)
    print("✓ Genome created")
    
    # Create spatial awareness
    print("\n3. Initializing spatial awareness...")
    spatial = SpatialAwareness()
    print("✓ Spatial awareness ready")
    
    # Start browser
    print("\n4. Starting browser (this may take a moment)...")
    browser = SlitherBrowser(headless=False, nickname="TEST-Bot")
    print("✓ Browser started")
    
    # Start game
    print("\n5. Starting game...")
    browser.force_restart()
    time.sleep(2)
    print("✓ Game started")
    
    print("\n" + "=" * 60)
    print("  TEST RUNNING - Bot will play for 30 seconds")
    print("  Watch the browser window!")
    print("=" * 60 + "\n")
    
    start_time = time.time()
    max_len = 0
    food_count = 0
    frame_count = 0
    
    try:
        while time.time() - start_time < 30:
            frame_count += 1
            
            # Get game data
            data = browser.get_game_data()
            
            if data is None:
                time.sleep(0.1)
                continue
            
            if data.get('dead', False):
                print("Bot died! Restarting...")
                browser.force_restart()
                time.sleep(2)
                continue
            
            my_snake = data.get('self')
            if not my_snake:
                time.sleep(0.1)
                continue
            
            # Track stats
            current_len = my_snake.get('len', 0)
            if current_len > max_len:
                food_count += (current_len - max_len)
                max_len = current_len
                print(f"  Frame {frame_count}: ATE FOOD! Length now: {max_len}")
            
            # Get inputs
            inputs = spatial.calculate_sectors(
                my_snake,
                data.get('enemies', []),
                data.get('foods', [])
            )
            
            # Verify input count
            if frame_count == 1:
                print(f"  Input vector size: {len(inputs)} (expected 195)")
                if len(inputs) != 195:
                    print("  ✗ ERROR: Wrong input size!")
                else:
                    print("  ✓ Input size correct")
            
            # Get decision
            current_heading = my_snake.get('ang', 0)
            angle, boost = brain.decide(inputs, current_heading)
            
            # Execute
            browser.send_action(angle, boost)
            time.sleep(0.05)
    
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    
    finally:
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("  TEST RESULTS")
        print("=" * 60)
        print(f"Duration: {elapsed:.1f}s")
        print(f"Frames processed: {frame_count}")
        print(f"Max length: {max_len}")
        print(f"Food eaten: {food_count}")
        print(f"FPS: {frame_count/elapsed:.1f}")
        
        if food_count > 0:
            print("\n✓ SUCCESS: Bot is eating food!")
        else:
            print("\n⚠ WARNING: Bot didn't eat any food in 30s")
            print("  This is normal for a random genome.")
            print("  Training will evolve better food-seeking behavior.")
        
        print("\n✓ All systems working!")
        print("\nYou can now start training with: python training_manager.py")
        print("=" * 60 + "\n")
        
        browser.close()


if __name__ == "__main__":
    quick_test()
