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

if __name__ == '__main__':
    unittest.main()
