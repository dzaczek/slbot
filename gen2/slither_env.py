import numpy as np
import math
import sys
import os

# Add parent directory to path to import browser_engine
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from browser_engine import SlitherBrowser

class SlitherEnv:
    def __init__(self, headless=True, nickname="MatrixBot", matrix_size=84):
        self.browser = SlitherBrowser(headless=headless, nickname=nickname)
        self.map_radius = 21600
        self.view_size = 1200 # Radius of view
        self.matrix_size = matrix_size # Output grid size (default 84x84)
        self.scale = self.matrix_size / (self.view_size * 2) # Scale factor

        # Wall tracking (only wall matters for death classification)
        self.near_wall_frames = 0
        
        # Track previous length to detect food eating
        self.prev_length = 0

    def reset(self):
        """Resets the game and returns initial matrix state."""
        self.browser.force_restart()
        self.near_wall_frames = 0
        
        # Get initial state and set prev_length to actual starting length
        data = self.browser.get_game_data()
        if data and data.get('self'):
            self.prev_length = data['self'].get('len', 0)
        else:
            self.prev_length = 0
            
        return self._process_data_to_matrix(data)

    def _get_death_reward_and_cause(self):
        """
        Slither.io death causes (only 2 possibilities):
        1. Wall collision - hit the map boundary
        2. Snake collision - our head hit another snake's body
        
        Note: You CAN hide your head in your OWN body (coiling) - no self-collision!
        """
        if self.near_wall_frames > 3:
            return -50, "Wall"  # Preventable - should learn to avoid edges
        else:
            return -10, "SnakeCollision"  # Normal gameplay death

    def step(self, action):
        """
        Executes action and returns (next_state, reward, done, info).
        Actions:
        0: Keep current direction
        1: Turn Left small (~17 deg)
        2: Turn Right small (~17 deg)
        3: Turn Left big (~45 deg)
        4: Turn Right big (~45 deg)
        5: Boost (speed up, loses mass)
        """
        angle_change = 0
        boost = 0

        if action == 0: pass
        elif action == 1: angle_change = -0.3
        elif action == 2: angle_change = 0.3
        elif action == 3: angle_change = -0.8
        elif action == 4: angle_change = 0.8
        elif action == 5: boost = 1

        # Get current state
        data = self.browser.get_game_data()
        if not data:
            return self._matrix_zeros(), -5, True, {"cause": "Unknown"}

        if data.get('dead'):
            reward, cause = self._get_death_reward_and_cause()
            return self._matrix_zeros(), reward, True, {"cause": cause}

        my_snake = data.get('self', {})
        current_ang = my_snake.get('ang', 0)
        mx, my = my_snake.get('x', 0), my_snake.get('y', 0)
        current_len = my_snake.get('len', 0)

        # Track wall proximity
        dist_from_center = math.hypot(mx - self.map_radius, my - self.map_radius)
        dist_to_wall = self.map_radius - dist_from_center
        if dist_to_wall < 500:
            self.near_wall_frames += 1
        else:
            self.near_wall_frames = max(0, self.near_wall_frames - 1)

        # Execute action
        target_ang = current_ang + angle_change
        self.browser.send_action(target_ang, boost)

        # Get new state after action
        data = self.browser.get_game_data()
        state = self._process_data_to_matrix(data)

        # Check if died after action
        if not data or data.get('dead'):
            reward, cause = self._get_death_reward_and_cause()
            return state, reward, True, {"cause": cause}

        # === REWARD CALCULATION ===
        reward = 0
        new_snake = data.get('self', {})
        new_len = new_snake.get('len', 0)
        
        # 1. Survival reward (small positive for staying alive)
        reward += 0.1
        
        # 2. Food reward (big positive for eating)
        if new_len > self.prev_length:
            food_gained = new_len - self.prev_length
            reward += food_gained * 5.0  # Strong reward for eating!
        
        # 3. Wall proximity penalty (encourage staying away from edges)
        if dist_to_wall < 1000:
            reward -= 0.5  # Small penalty for being near wall
        
        # Update tracked length
        self.prev_length = new_len

        return state, reward, False, {"length": new_len}

    def _get_state(self):
        data = self.browser.get_game_data()
        return self._process_data_to_matrix(data)

    def _matrix_zeros(self):
        return np.zeros((3, self.matrix_size, self.matrix_size), dtype=np.float32)

    def _process_data_to_matrix(self, data):
        """
        Converts raw JSON data to a 3-channel Matrix (64x64).
        Channel 0: Food
        Channel 1: Enemies (Head = 1.0, Body = 0.5) + Walls (1.0) -> DANGER
        Channel 2: Self (Head = 1.0, Body = 0.5) -> SAFE/SELF
        """
        matrix = np.zeros((3, self.matrix_size, self.matrix_size), dtype=np.float32)

        if not data or data.get('dead'):
            return matrix

        my_snake = data.get('self')
        if not my_snake:
            return matrix

        mx, my = my_snake['x'], my_snake['y']

        # Helper to map world coord to matrix coord
        def world_to_matrix(wx, wy):
            # 1. Translate to ego-centric (relative to head)
            dx = wx - mx
            dy = wy - my

            # 2. Add offset (center of matrix)
            # Matrix 0,0 is top-left. Center is 32,32.
            # World dx=0 should be Matrix x=32
            # dx is -view_size to +view_size

            # Scale
            # range -1200..1200 -> 0..64
            # (dx + 1200) / 2400 * 64

            # Using self.scale
            # (dx * scale) + (size / 2)

            mat_x = int((dx * self.scale) + (self.matrix_size / 2))
            mat_y = int((dy * self.scale) + (self.matrix_size / 2))

            return mat_x, mat_y

        # 1. Food (Channel 0)
        foods = data.get('foods', [])
        for f in foods:
            if len(f) < 2: continue
            fx, fy = f[0], f[1]
            mx_x, mx_y = world_to_matrix(fx, fy)
            if 0 <= mx_x < self.matrix_size and 0 <= mx_y < self.matrix_size:
                # Size of food dot
                matrix[0, mx_y, mx_x] = 1.0 # Simple dot

        # 2. Enemies (Channel 1)
        enemies = data.get('enemies', [])
        for e in enemies:
            # Head
            ex, ey = e['x'], e['y']
            hx, hy = world_to_matrix(ex, ey)
            if 0 <= hx < self.matrix_size and 0 <= hy < self.matrix_size:
                matrix[1, hy, hx] = 1.0 # Head

            # Body
            for pt in e.get('pts', []):
                 px, py = pt[0], pt[1]
                 bx, by = world_to_matrix(px, py)
                 if 0 <= bx < self.matrix_size and 0 <= by < self.matrix_size:
                     matrix[1, by, bx] = 0.5 # Body

        # 3. Self & Walls (Channel 2)
        # Self Head
        cx, cy = self.matrix_size // 2, self.matrix_size // 2
        matrix[2, cy, cx] = 1.0

        # Self Body (now available via 'pts' in 'self')
        for pt in my_snake.get('pts', []):
            px, py = pt[0], pt[1]
            bx, by = world_to_matrix(px, py)
            if 0 <= bx < self.matrix_size and 0 <= by < self.matrix_size:
                matrix[2, by, bx] = 0.5 # Body

        # Walls (Channel 1 - DANGER)
        # Check boundary intersection efficiently using numpy
        map_center = 21600
        dist_from_map_center = math.hypot(mx - map_center, my - map_center)
        
        # Only calculate if wall is potentially visible (within view range)
        if dist_from_map_center + self.view_size > map_center:
             y_grid, x_grid = np.ogrid[:self.matrix_size, :self.matrix_size]
             
             # Convert matrix coords to world coords
             # mat_x = (dx * scale) + size/2  => dx = (mat_x - size/2) / scale
             dx_world = (x_grid - (self.matrix_size / 2)) / self.scale
             dy_world = (y_grid - (self.matrix_size / 2)) / self.scale
             
             wx = mx + dx_world
             wy = my + dy_world
             
             # Squared distance from map center
             dist_sq = (wx - map_center)**2 + (wy - map_center)**2
             
             # Radius squared
             radius_sq = self.map_radius**2
             
             # Mask where dist > radius (outside map)
             wall_mask = dist_sq > radius_sq
             
             # Draw walls as solid obstacles in ENEMY channel (Danger)
             matrix[1][wall_mask] = 1.0

        return matrix

    def close(self):
        self.browser.close()
