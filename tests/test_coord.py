import sys
import os
import unittest

# Add gen2 to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coord_transform import world_to_grid

class TestCoordTransform(unittest.TestCase):
    def test_center(self):
        # Center of map should be center of grid
        gx, gy = world_to_grid(100, 100, 100, 100, 1.0, 84)
        self.assertEqual(gx, 42)
        self.assertEqual(gy, 42)

    def test_scale(self):
        # Scale 0.1. dx=10 -> 1 pixel
        gx, gy = world_to_grid(110, 100, 100, 100, 0.1, 84)
        self.assertEqual(gx, 43)
        self.assertEqual(gy, 42)

if __name__ == '__main__':
    unittest.main()
