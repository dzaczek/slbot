import sys
import os
import math
import numpy as np

# Add gen2 to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../gen2')))

from slither_env import SlitherEnv

# Mock browser
class MockBrowser:
    def __init__(self, headless=True, nickname="test"):
        pass
    def scan_game_variables(self):
        return {}
    def get_game_data(self):
        return {}
    def force_restart(self):
        pass
    def inject_view_plus_overlay(self):
        pass
    def update_view_plus_overlay(self, *args, **kwargs):
        pass
    def close(self):
        pass

class TestEnv(SlitherEnv):
    def __init__(self):
        # Skip super init to avoid selenium
        self.browser = MockBrowser()
        self.map_radius = 21600
        self.map_center_x = 21600
        self.map_center_y = 21600
        self.matrix_size = 84
        self.view_plus = False
        self.view_size = 500
        self.scale = self.matrix_size / (self.view_size * 2)
        self._map_vars_printed = False
        self.last_dist_to_wall = 99999
        self.near_wall_frames = 0
        self.prev_length = 0
        self.prev_food_dist = None
        self.frame_skip = 4
        # Curriculum params
        self.food_reward = 10
        self.food_shaping = 0.01
        self.survival_reward = 0.05
        self.death_wall_penalty = -15
        self.death_snake_penalty = -15
        self.straight_penalty = 0.0
        self.length_bonus = 0.0

def save_matrix_image(matrix, filename="test_render.ppm"):
    # Matrix is 3x84x84 float
    # Convert to RGB int
    # CH0=Food(G), CH1=Enemy(R), CH2=Self(B)

    h, w = matrix.shape[1], matrix.shape[2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # Scale 0-1 to 0-255
    # Red = Enemy (Ch1)
    rgb[:,:,0] = (matrix[1] * 255).astype(np.uint8)
    # Green = Food (Ch0)
    rgb[:,:,1] = (matrix[0] * 255).astype(np.uint8)
    # Blue = Self (Ch2)
    rgb[:,:,2] = (matrix[2] * 255).astype(np.uint8)

    # PPM format P6
    with open(filename, 'wb') as f:
        f.write(f"P6\n{w} {h}\n255\n".encode())
        f.write(rgb.tobytes())
    print(f"Saved {filename}")

def run_test():
    env = TestEnv()

    # 1. Test Wall Detection Logic
    print("Testing Wall Detection Logic...")

    # Case A: Far inside map
    data_inside = {
        'self': {'x': 21600, 'y': 21600},
        'dist_to_wall': 21600,
        'enemies': []
    }
    pen, cause = env._get_death_reward_and_cause(data_inside)
    print(f"Inside Map (21600,21600): Cause={cause}")
    assert cause == "SnakeCollision" # Default fallback when no enemies/walls

    # Case B: Close to wall (JS says so)
    data_near_wall = {
        'self': {'x': 40000, 'y': 21600},
        'dist_to_wall': 1000,
        'enemies': []
    }
    pen, cause = env._get_death_reward_and_cause(data_near_wall)
    print(f"Near Wall JS (dist=1000): Cause={cause}")
    assert cause == "Wall"

    # Case C: Outside Map (Python Calc Override)
    # Map Center 21600, Radius 21600.
    # Pos 50000, 21600 -> Dist 28400 -> Outside by 6800
    data_outside = {
        'self': {'x': 50000, 'y': 21600},
        'dist_to_wall': 99999, # JS might be confused
        'enemies': []
    }
    pen, cause = env._get_death_reward_and_cause(data_outside)
    print(f"Strict Outside Map (50000,21600): Cause={cause}")
    assert cause == "Wall"

    # 2. Test Rendering (Thickness)
    print("\nTesting Rendering...")

    # Self at center
    # Enemy 1: Normal (sc=1.0)
    # Enemy 2: Huge (sc=3.0)

    data_render = {
        'self': {
            'x': 21600, 'y': 21600, 'sc': 1.0,
            'pts': [[21600, 21600], [21600, 21550]] # Tail pointing up
        },
        'enemies': [
            {
                'x': 21600 + 200, 'y': 21600, 'sc': 1.0, # Right
                'pts': [[21600+200, 21600], [21600+200, 21500]]
            },
            {
                'x': 21600 - 200, 'y': 21600, 'sc': 3.0, # Left (Huge)
                'pts': [[21600-200, 21600], [21600-200, 21500]]
            }
        ],
        'view_radius': 500,
        'map_radius': 21600,
        'map_center_x': 21600,
        'map_center_y': 21600,
        'dist_to_wall': 21600
    }

    matrix = env._process_data_to_matrix(data_render)

    # Check pixel counts
    # Channel 1 is enemies
    enemy_pixels = matrix[1]

    # Split left and right halves to count pixels for each enemy
    mid_x = 42
    left_half = enemy_pixels[:, :mid_x]
    right_half = enemy_pixels[:, mid_x:]

    left_count = np.sum(left_half > 0)
    right_count = np.sum(right_half > 0)

    print(f"Normal Snake Pixels (Right): {right_count}")
    print(f"Huge Snake Pixels (Left): {left_count}")

    assert left_count > right_count * 2, "Huge snake should have significantly more pixels"

    save_matrix_image(matrix, "tests/rendering_test.ppm")

if __name__ == "__main__":
    try:
        run_test()
        print("TEST PASSED")
    except AssertionError as e:
        print(f"TEST FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        exit(1)
