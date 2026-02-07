import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import math

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# Constants
BLOCK_SIZE = 20
SPEED = 40
GRID_SIZE = 20

# Colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

class SnakeGameAI:
    def __init__(self, w=GRID_SIZE*BLOCK_SIZE, h=GRID_SIZE*BLOCK_SIZE, render=True):
        self.w = w
        self.h = h
        self.render_mode = render

        # Lazy Init pygame
        if not pygame.get_init():
            pygame.init()

        # Init display
        if self.render_mode:
            self.font = pygame.font.Font(pygame.font.get_default_font(), 25)
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake')
        else:
            self.display = None
            self.font = None

        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # Init game state
        self.direction = Direction.RIGHT

        # Start in the middle
        self.head = Point(GRID_SIZE // 2, GRID_SIZE // 2)
        self.snake = [self.head,
                      Point(self.head.x-1, self.head.y),
                      Point(self.head.x-2, self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.steps_without_food = 0

        # Calculate initial distance to food for reward shaping
        self.prev_distance = self._get_distance_to_food()
        return self.get_state_data()

    def _place_food(self):
        x = random.randint(0, GRID_SIZE-1)
        y = random.randint(0, GRID_SIZE-1)
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def _get_distance_to_food(self):
        if self.food is None:
            return 0
        return math.sqrt((self.head.x - self.food.x)**2 + (self.head.y - self.food.y)**2)

    def play_step(self, action):
        self.frame_iteration += 1
        self.steps_without_food += 1

        # 1. Collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. Move
        self._move(action) # Update the head
        self.snake.insert(0, self.head)

        # 3. Check if game over
        reward = -0.005 # Step penalty
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -20
            return reward, game_over, self.score

        # 4. Place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self.steps_without_food = 0
            self._place_food()
            self.prev_distance = self._get_distance_to_food() # Reset distance ref
        else:
            self.snake.pop()

            # Distance reward shaping
            current_distance = self._get_distance_to_food()
            if current_distance < self.prev_distance:
                reward += 0.05
            else:
                reward -= 0.05
            self.prev_distance = current_distance

        # 5. Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)

        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Hits boundary
        if pt.x > GRID_SIZE - 1 or pt.x < 0 or pt.y > GRID_SIZE - 1 or pt.y < 0:
            return True
        # Hits itself
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        if not self.render_mode:
            return

        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x*BLOCK_SIZE, pt.y*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x*BLOCK_SIZE+4, pt.y*BLOCK_SIZE+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x*BLOCK_SIZE, self.food.y*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

        if self.font:
            text = self.font.render("Score: " + str(self.score), True, WHITE)
            self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn
        elif np.array_equal(action, [0, 0, 1]):
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn
        else:
            raise ValueError(f"Invalid action: {action}. Expected one-hot [straight, right, left].")

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += 1
        elif self.direction == Direction.LEFT:
            x -= 1
        elif self.direction == Direction.DOWN:
            y += 1
        elif self.direction == Direction.UP:
            y -= 1

        self.head = Point(x, y)

    def get_state_data(self):
        """
        Returns the raw state data for the agent to process:
        - Grid (20x20) with values: 0 (empty), 0.5 (body), 1.0 (head), -1.0 (food)
        - Head Direction (One-hot 4 vector)
        """
        # Grid
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

        # Body
        for pt in self.snake:
            if 0 <= pt.x < GRID_SIZE and 0 <= pt.y < GRID_SIZE:
                grid[pt.y, pt.x] = 0.5

        # Head (overwrite body if needed, though head is in snake list)
        if 0 <= self.head.x < GRID_SIZE and 0 <= self.head.y < GRID_SIZE:
            grid[self.head.y, self.head.x] = 1.0

        # Food
        if 0 <= self.food.x < GRID_SIZE and 0 <= self.food.y < GRID_SIZE:
            grid[self.food.y, self.food.x] = -1.0

        # Orientation
        # [Right, Left, Up, Down] match Direction enum
        # But enum is 1-based. Let's return explicit vector.
        # Order: [RIGHT, LEFT, UP, DOWN]
        dir_vec = [0, 0, 0, 0]
        if self.direction == Direction.RIGHT:
            dir_vec[0] = 1
        elif self.direction == Direction.LEFT:
            dir_vec[1] = 1
        elif self.direction == Direction.UP:
            dir_vec[2] = 1
        elif self.direction == Direction.DOWN:
            dir_vec[3] = 1

        return grid, np.array(dir_vec, dtype=np.float32)
