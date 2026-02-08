import sys
import os
import unittest
import numpy as np
from unittest.mock import MagicMock

# Add gen2 to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock Browser Engine BEFORE importing SlitherEnv
sys.modules['browser_engine'] = MagicMock()

from slither_env import SlitherEnv
from coord_transform import world_to_grid

class TestSlitherEnv(unittest.TestCase):
    def setUp(self):
        self.env = SlitherEnv(headless=True, nickname="TestBot", matrix_size=84, view_plus=False)
        self.env.browser = MagicMock()
        # Mock get_game_data to avoid browser calls
        self.env.browser.get_game_data = MagicMock(return_value={})

    def test_radial_rendering(self):
        # Create a snake near the map boundary (21600)
        # Radius 21600. Center (21600, 21600).
        # Safe position: (21600, 21600) -> dist 0.
        # Near wall: x = 21600 + 21500 = 43100. y = 21600. Dist = 21500. Wall dist = 100.

        data = {
            'boundary_type': 'circle', # Irrelevant, forced anyway
            'map_radius': 21600,
            'map_center_x': 21600,
            'map_center_y': 21600,
            'dist_to_wall': 100, # Fake JS value, should be ignored/recalc
            'view_radius': 500,
            'self': {'x': 43100, 'y': 21600, 'len': 10},
            'gsc': 1.0
        }

        matrix = self.env._process_data_to_matrix(data)

        # Center (42, 42) is snake head.
        # At (42, 42), distance is 21500.
        # Wall threshold = 21600 - 500 = 21100.
        # 21500 > 21100, so it IS Wall (Warning Zone).
        self.assertEqual(matrix[1, 42, 42], 1.0)

        # Check Wall rendering
        # View radius 500. Scale = 84 / 1000 = 0.084.
        # Wall is at x=43200 (dist 21600).
        # Snake at x=43100.
        # dx_wall = 100.
        # gx = 42 + 100 * 0.084 = 50.4 -> 50.
        # So at gx=52, we should be strictly outside.
        self.assertEqual(matrix[1, 42, 55], 1.0)

    def test_update_from_game_data_ignores_nan(self):
        self.env.last_dist_to_wall = 1234
        self.env.map_radius = 21600
        self.env.map_center_x = 21600
        self.env.map_center_y = 21600

        data = {
            'dist_to_wall': float('nan'),
            'map_radius': float('nan'),
            'map_center_x': float('nan'),
            'map_center_y': float('nan')
        }

        self.env._update_from_game_data(data)

        self.assertEqual(self.env.last_dist_to_wall, 1234)
        self.assertEqual(self.env.map_radius, 21600)
        self.assertEqual(self.env.map_center_x, 21600)
        self.assertEqual(self.env.map_center_y, 21600)

    def test_invalid_frame_returns_last_matrix(self):
        self.env.browser.send_action = MagicMock()
        self.env.last_matrix = np.ones((3, self.env.matrix_size, self.env.matrix_size), dtype=np.float32)
        self.env.browser.get_game_data = MagicMock(return_value={
            'dead': False,
            'valid': False,
            'self': {'x': 0, 'y': 0, 'len': 0}
        })

        state, reward, done, info = self.env.step(0)

        self.assertFalse(done)
        self.assertEqual(reward, 0.0)
        self.assertEqual(info.get('cause'), 'InvalidFrame')
        self.assertTrue(np.array_equal(state, self.env.last_matrix))

if __name__ == '__main__':
    unittest.main()
