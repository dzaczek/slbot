
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add parent dir to path to import slither_env
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the module to be tested
import slither_env

class TestDeathClassification(unittest.TestCase):
    def setUp(self):
        # We need to patch the SlitherBrowser class used inside slither_env
        self.patcher = patch('slither_env.SlitherBrowser')
        self.MockBrowser = self.patcher.start()

        # Configure the mock instance
        self.mock_browser_instance = self.MockBrowser.return_value

        # Instantiate SlitherEnv (it will use the mock)
        self.env = slither_env.SlitherEnv(headless=True)

        # Set default penalties for testing
        self.env.death_wall_penalty = -100
        self.env.death_snake_penalty = -10

        # Default map settings
        self.env.map_radius = 21600
        self.env.map_center_x = 21600
        self.env.map_center_y = 21600
        self.env.boundary_type = 'circle'
        self.env.last_dist_to_wall = 99999

    def tearDown(self):
        self.patcher.stop()

    def test_snake_collision_near_wall(self):
        """
        Scenario 1: Near wall (1000 < 2000) AND hit snake (200 < 500).
        This currently fails (returns Wall) but should return SnakeCollision.
        """
        data = {
            'dist_to_wall': 1000,  # Near wall (< 2000)
            'self': {'x': 20000, 'y': 20000}, # Somewhere valid
            'enemies': [
                {'x': 20100, 'y': 20100, 'pts': []} # Distance ~141 (< 500)
            ],
            'boundary_type': 'circle',
            'boundary_vertices': []
        }
        # We need to manually update env state as if update_from_game_data ran
        self.env._update_from_game_data(data)

        # Call the method under test
        penalty, cause = self.env._get_death_reward_and_cause(data)

        print(f"\nScenario 1 (Near Wall + Hit Snake): Cause={cause}, Penalty={penalty}")

        # Expect SnakeCollision (Priority 2) over Wall (Priority 3)
        self.assertEqual(cause, "SnakeCollision")

    def test_wall_proximity_no_snake(self):
        """
        Scenario 2: Near wall (1000) AND NO snake collision (enemy far away).
        Should return Wall.
        """
        data = {
            'dist_to_wall': 1000,
            'self': {'x': 20000, 'y': 20000},
            'enemies': [
                {'x': 25000, 'y': 25000, 'pts': []} # Far away
            ],
            'boundary_type': 'circle'
        }
        self.env._update_from_game_data(data)
        penalty, cause = self.env._get_death_reward_and_cause(data)
        print(f"Scenario 2 (Near Wall + No Snake): Cause={cause}, Penalty={penalty}")
        self.assertEqual(cause, "Wall")

    def test_outside_wall_hit_snake(self):
        """
        Scenario 3: Outside wall (-100) AND hit snake.
        Should return Wall because outside map is definitive.
        """
        data = {
            'dist_to_wall': -100, # Outside
            'self': {'x': 40000, 'y': 40000},
            'enemies': [
                {'x': 40100, 'y': 40100, 'pts': []} # Close
            ],
            'boundary_type': 'circle'
        }
        self.env._update_from_game_data(data)
        penalty, cause = self.env._get_death_reward_and_cause(data)
        print(f"Scenario 3 (Outside Wall + Hit Snake): Cause={cause}, Penalty={penalty}")
        self.assertEqual(cause, "Wall")

    def test_far_from_wall_hit_snake(self):
        """
        Scenario 4: Far from wall (5000) AND hit snake.
        Should return SnakeCollision.
        """
        data = {
            'dist_to_wall': 5000,
            'self': {'x': 21600, 'y': 21600},
            'enemies': [
                {'x': 21700, 'y': 21700, 'pts': []} # Close (~141)
            ],
            'boundary_type': 'circle'
        }
        self.env._update_from_game_data(data)
        penalty, cause = self.env._get_death_reward_and_cause(data)
        print(f"Scenario 4 (Far Wall + Hit Snake): Cause={cause}, Penalty={penalty}")
        self.assertEqual(cause, "SnakeCollision")

if __name__ == '__main__':
    unittest.main()
