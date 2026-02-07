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

    def test_polygon_rendering(self):
        # Create a polygon boundary
        data = {
            'boundary_type': 'polygon',
            'boundary_vertices': [[100, 100], [200, 100], [200, 200], [100, 200]],
            'map_radius': 0,
            'map_center_x': 0,
            'map_center_y': 0,
            'dist_to_wall': 500,
            'view_radius': 200, # Large view to see the polygon
            'self': {'x': 150, 'y': 150, 'len': 10}, # Inside polygon
            'gsc': 1.0
        }

        # Manually trigger process
        # matrix_size=84. view_size=200. scale = 84 / 400 = 0.21
        # Polygon coords: (100, 100) -> (200, 200)
        # Center is (150, 150).
        # (100, 100) relative to center (-50, -50).
        # Matrix coords: (-50 * 0.21) + 42 = -10.5 + 42 = 31.5 -> 31

        matrix = self.env._process_data_to_matrix(data)

        # Check if wall channel (1) has values.
        # Inside polygon (center) should be 0. Outside should be 1.
        # But polygon is [100, 200] range. View is radius 200 -> [ -50, 350 ].
        # So polygon is entirely within view.
        # Inside polygon should be 0.

        # Check center (150, 150) -> matrix center (42, 42)
        # matrix[1] is wall channel
        self.assertEqual(matrix[1, 42, 42], 0.0)

        # Check outside polygon (e.g. 50, 50)
        # Relative (-100, -100).
        # Matrix: (-100 * 0.21) + 42 = -21 + 42 = 21.
        # Note: scale is exactly 84 / 400 = 0.21.
        # x = 50. mx = 150. dx = -100. gx = -100 * 0.21 + 42 = 21.
        self.assertEqual(matrix[1, 21, 21], 1.0)

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
