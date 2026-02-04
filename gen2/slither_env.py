import numpy as np
import math
import sys
import os

# Add parent directory to path to import browser_engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from browser_engine import SlitherBrowser

class SlitherEnv:
    def __init__(self, headless=True, nickname="MatrixBot"):
        self.browser = SlitherBrowser(headless=headless, nickname=nickname)
        self.map_radius = 21600
        self.view_size = 1200 # Radius of view
        self.matrix_size = 64 # Output grid size (64x64)
        self.scale = self.matrix_size / (self.view_size * 2) # Scale factor

        # Collision tracking
        self.near_wall_frames = 0
        self.near_enemy_frames = 0

    def reset(self):
        """Resets the game and returns initial matrix state."""
        self.browser.force_restart()
        self.near_wall_frames = 0
        self.near_enemy_frames = 0
        return self._get_state()

    def step(self, action):
        """
        Executes action and returns (next_state, reward, done, info).
        Action: 0-7 (Direction sectors) + 8 (Boost) maybe?
        Simpler: Action is an angle index (0-15) and boost is implied or separate?
        Let's stick to discrete actions for DQN.
        Actions:
        0: Keep current
        1: Turn Left small
        2: Turn Right small
        3: Turn Left fast
        4: Turn Right fast
        5: Boost
        """
        # DQN usually outputs discrete index.
        # Let's map 0-11 to 12 clock directions.
        # Or relative changes.
        # Given the game physics, relative steering is smoother.
        # 0: Straight
        # 1: Left 15 deg
        # 2: Right 15 deg
        # 3: Left 45 deg
        # 4: Right 45 deg

        angle_change = 0
        boost = 0

        if action == 0: pass
        elif action == 1: angle_change = -0.3 # ~17 deg
        elif action == 2: angle_change = 0.3
        elif action == 3: angle_change = -0.8 # ~45 deg
        elif action == 4: angle_change = 0.8
        elif action == 5: boost = 1

        # We need current angle to apply relative change
        data = self.browser.get_game_data()
        if not data:
             return self._matrix_zeros(), -10, True, {"cause": "Unknown"}

        if data.get('dead'):
            # Determine cause of death using tracked stats
            cause = "BodyCollision"
            if self.near_wall_frames > 3:
                cause = "Wall"
                reward = -500 # Heavy penalty for wall
            elif self.near_enemy_frames > 3:
                cause = "HeadCollision"
                reward = -50
            else:
                reward = -20 # Standard body collision
            return self._matrix_zeros(), reward, True, {"cause": cause}

        my_snake = data.get('self', {})
        current_ang = my_snake.get('ang', 0)
        mx, my = my_snake.get('x', 0), my_snake.get('y', 0)

        # UPDATE COLLISION TRACKING
        # Check wall distance
        dist_from_center = math.hypot(mx - self.map_radius, my - self.map_radius)
        dist_to_wall = self.map_radius - dist_from_center
        if dist_to_wall < 500:
            self.near_wall_frames += 1
        else:
            self.near_wall_frames = max(0, self.near_wall_frames - 1)

        # Check enemy head distance
        enemies = data.get('enemies', [])
        found_near_enemy = False
        for e in enemies:
            edist = math.hypot(e['x'] - mx, e['y'] - my)
            if edist < 200:
                found_near_enemy = True
                break

        if found_near_enemy:
            self.near_enemy_frames += 1
        else:
            self.near_enemy_frames = max(0, self.near_enemy_frames - 1)

        target_ang = current_ang + angle_change

        self.browser.send_action(target_ang, boost)

        # Get new state
        data = self.browser.get_game_data()
        state = self._process_data_to_matrix(data)

        # Reward calculation
        reward = 0
        done = False
        if not data or data.get('dead'):
            done = True
            # Determine cause again if dead now
            cause = "BodyCollision"
            if self.near_wall_frames > 3:
                cause = "Wall"
                reward = -500
            elif self.near_enemy_frames > 3:
                cause = "HeadCollision"
                reward = -50
            else:
                reward = -20
            return state, reward, done, {"cause": cause}
        else:
            # Survival reward
            reward += 0.1
            # Length reward
            current_len = my_snake.get('len', 0)
            reward += current_len * 0.001

        return state, reward, done, {}

    def _get_state(self):
        data = self.browser.get_game_data()
        return self._process_data_to_matrix(data)

    def _matrix_zeros(self):
        return np.zeros((3, self.matrix_size, self.matrix_size), dtype=np.float32)

    def _process_data_to_matrix(self, data):
        """
        Converts raw JSON data to a 3-channel Matrix (64x64).
        Channel 0: Food
        Channel 1: Enemies (Head = 1.0, Body = 0.5)
        Channel 2: Self (Head = 1.0, Body = 0.5) + Walls (1.0)
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
        # Always center? Yes, but drawing it helps convolution see it
        cx, cy = self.matrix_size // 2, self.matrix_size // 2
        matrix[2, cy, cx] = 1.0

        # Self Body?
        # browser_engine.get_game_data doesn't return my body points in 'self'
        # It only returns 'pts' length.
        # So we can't draw our own body unless we track it or modify browser_engine.
        # For now, ignore self body (assuming we know where we are).

        # Walls
        # Check corners of view
        # Map radius is 21600. Center is 21600, 21600.
        # If mx + view_size > 2 * radius ... etc
        # Simple rasterization: check every pixel? Too slow.
        # Check boundary intersection.
        # Or just checking if a point in matrix corresponds to out of bounds.

        # Optimization: Just check distance from center
        map_center = 21600
        dist_from_map_center = math.hypot(mx - map_center, my - map_center)
        if dist_from_map_center + self.view_size > map_center:
             # Wall is visible
             pass
             # For every pixel in matrix, calc world coord, check dist > radius
             # This is slow in python.
             # Approximate: Draw a big circle?
             # Let's skip drawing walls for now, assume inputs handle it?
             # Or simpler:
             # Calculate nearest point on wall circle.
             pass

        return matrix

    def close(self):
        self.browser.close()
