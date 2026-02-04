import unittest
import os
# Mock SDL for headless
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

from game import SnakeGameAI, Point, BLOCK_SIZE

class TestCollision(unittest.TestCase):
    def test_collision_boundary(self):
        # Default size w=640, h=480
        game = SnakeGameAI(w=640, h=480)

        # Test safe points
        safe_point = Point(20, 20)
        self.assertFalse(game.is_collision(safe_point))

        # Test boundary points (should be collision if using >= width)

        # Right boundary
        # x = 640 should be collision (indices 0..639)
        collision_right = Point(640, 20)
        self.assertTrue(game.is_collision(collision_right), "x=640 should be collision")

        # x = 620 should be safe
        collision_right_in = Point(620, 20)
        self.assertFalse(game.is_collision(collision_right_in), "x=620 should be safe")

        # Left boundary
        collision_left = Point(-20, 20)
        self.assertTrue(game.is_collision(collision_left), "x=-20 should be collision")

        collision_left_in = Point(0, 20)
        self.assertFalse(game.is_collision(collision_left_in), "x=0 should be safe")

        # Bottom boundary
        # y = 480 should be collision
        collision_bottom = Point(20, 480)
        self.assertTrue(game.is_collision(collision_bottom), "y=480 should be collision")

        collision_bottom_in = Point(20, 460)
        self.assertFalse(game.is_collision(collision_bottom_in), "y=460 should be safe")

        # Top boundary
        collision_top = Point(20, -20)
        self.assertTrue(game.is_collision(collision_top), "y=-20 should be collision")

        collision_top_in = Point(20, 0)
        self.assertFalse(game.is_collision(collision_top_in), "y=0 should be safe")

if __name__ == '__main__':
    unittest.main()
