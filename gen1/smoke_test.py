import sys
import os
import neat
import numpy as np

# Add current dir to path
sys.path.append(os.getcwd())

def test_imports():
    print("Testing imports...")
    try:
        from browser_engine import SlitherBrowser
        from spatial_awareness import SpatialAwareness
        from ai_brain import BotAgent
        import training_manager
        print("Imports successful.")
    except ImportError as e:
        print(f"Import failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error during imports: {e}")
        sys.exit(1)

def test_logic_instantiation():
    print("Testing logic instantiation...")
    try:
        from spatial_awareness import SpatialAwareness
        spatial = SpatialAwareness()
        
        # Test sector calculation with dummy data
        my_snake = {'x': 1000, 'y': 1000, 'ang': 0, 'len': 10}
        enemies = [{'x': 1100, 'y': 1100, 'ang': 0, 'sp': 6, 'pts': [[1100, 1100]]}]
        foods = [[1050, 1050, 1]]
        
        inputs = spatial.calculate_sectors(my_snake, enemies, foods)
        if len(inputs) == 145: # 24*6 + 1
            print(f"Spatial Awareness inputs generated correctly: {len(inputs)}")
        else:
            print(f"Spatial Awareness logic error. Expected 145 inputs, got {len(inputs)}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Logic instantiation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_imports()
    test_logic_instantiation()
    print("Smoke test passed!")
