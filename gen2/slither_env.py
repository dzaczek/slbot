import numpy as np
import math
import sys
import os
import time

# Add parent directory to path to import browser_engine
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from browser_engine import SlitherBrowser

class SlitherEnv:
    def __init__(self, headless=True, nickname="MatrixBot", matrix_size=84, view_plus=False):
        self.browser = SlitherBrowser(headless=headless, nickname=nickname)
        self.map_radius = 21600  # Will be updated dynamically from game
        self.map_center_x = 21600
        self.map_center_y = 21600
        self.matrix_size = matrix_size # Output grid size (default 84x84)
        self.view_plus = view_plus  # Enable visual overlay
        
        # Dynamic view size - will be updated from game data
        self.view_size = 500  # Initial estimate, updated dynamically
        self.scale = self.matrix_size / (self.view_size * 2)
        
        # Flag to print map vars only once
        self._map_vars_printed = False

        # Wall tracking (only wall matters for death classification)
        self.near_wall_frames = 0
        
        # Track previous length to detect food eating
        self.prev_length = 0
        
        # Track distance to nearest food for shaping reward
        self.prev_food_dist = None
        
        # Frame skip - repeat action for N frames
        self.frame_skip = 4

    def reset(self):
        """Resets the game and returns initial matrix state."""
        self.browser.force_restart()
        self.near_wall_frames = 0
        
        # Inject view-plus overlay if enabled
        if self.view_plus:
            self.browser.inject_view_plus_overlay()
        
        # Get initial state and set prev_length to actual starting length
        data = self.browser.get_game_data()
        if data and data.get('self'):
            self.prev_length = data['self'].get('len', 0)
        else:
            self.prev_length = 0
            
        matrix = self._process_data_to_matrix(data)
        
        # Update overlay with initial state (include gsc and view_radius for debugging)
        if self.view_plus and data:
            gsc = data.get('gsc', 0)
            view_r = data.get('view_radius', 0)
            debug_info = data.get('debug', {})
            self.browser.update_view_plus_overlay(matrix, gsc=gsc, view_radius=view_r, debug_info=debug_info)
            
        return matrix

    def _get_death_reward_and_cause(self, last_data=None):
        """
        Slither.io death causes (only 2 possibilities):
        1. Wall collision - hit the map boundary
        2. Snake collision - our head hit another snake's body
        """
        # Check actual position at death time (most accurate)
        if last_data and last_data.get('self'):
            mx = last_data['self'].get('x', 0)
            my = last_data['self'].get('y', 0)
            dist_from_center = math.hypot(mx - self.map_center_x, my - self.map_center_y)
            dist_to_wall = self.map_radius - dist_from_center
            
            # Debug log for death analysis
            print(f"DEBUG DEATH: Pos=({mx:.1f}, {my:.1f}) DistToWall={dist_to_wall:.1f} MapR={self.map_radius}")
            
            # If very close to wall at death, it's wall collision
            # Increased threshold from 300 to 1200 because wall visual boundary is fuzzy
            if dist_to_wall < 1200:  
                return -50, "Wall"
        
        # Fallback: use frame tracking
        if self.near_wall_frames > 0:
            return -50, "Wall"  # Preventable - should learn to avoid edges
        else:
            return -10, "SnakeCollision"  # Normal gameplay death

    def step(self, action):
        """
        Executes action and returns (next_state, reward, done, info).
        Actions:
        0: Keep current direction
        1: Turn Left small (~22 deg)
        2: Turn Right small (~22 deg)
        3: Turn Left big (~60 deg)
        4: Turn Right big (~60 deg)
        5: Boost (speed up, loses mass)
        """
        angle_change = 0
        boost = 0

        # Increased turn angles for more responsive steering
        if action == 0: pass
        elif action == 1: angle_change = -0.7  # ~40 deg (was 0.4)
        elif action == 2: angle_change = 0.7   # ~40 deg
        elif action == 3: angle_change = -1.8  # ~103 deg (was 1.0)
        elif action == 4: angle_change = 1.8   # ~103 deg
        elif action == 5: boost = 1

        # Get current state before action
        data = self.browser.get_game_data()
        if not data:
            return self._matrix_zeros(), -5, True, {"cause": "BrowserError"}

        if data.get('dead'):
            reward, cause = self._get_death_reward_and_cause(data)
            mx = data['self'].get('x', 0) if data.get('self') else 0
            my = data['self'].get('y', 0) if data.get('self') else 0
            return self._matrix_zeros(), reward, True, {"cause": cause, "pos": (mx, my)}

        my_snake = data.get('self', {})
        current_ang = my_snake.get('ang', 0)
        mx, my = my_snake.get('x', 0), my_snake.get('y', 0)
        current_len = my_snake.get('len', 0)

        # Track wall proximity
        dist_from_center = math.hypot(mx - self.map_center_x, my - self.map_center_y)
        dist_to_wall = self.map_radius - dist_from_center
        # Increased detection range for "near wall" state
        if dist_to_wall < 1500: # Was 500
            self.near_wall_frames += 1
        else:
            self.near_wall_frames = max(0, self.near_wall_frames - 1)

        # Save pre-action data for death detection
        pre_action_data = data

        # Calculate distance to nearest food BEFORE action
        foods = data.get('foods', [])
        if foods:
            food_dists = [math.hypot(f[0] - mx, f[1] - my) for f in foods if len(f) >= 2]
            current_food_dist = min(food_dists) if food_dists else None
        else:
            current_food_dist = None

        # Execute action with FRAME SKIP (repeat action for multiple frames)
        target_ang = current_ang + angle_change
        for _ in range(self.frame_skip):
            self.browser.send_action(target_ang, boost)
            # Small delay to let action take effect
            time.sleep(0.02)

        # Get new state after action
        data = self.browser.get_game_data()
        state = self._process_data_to_matrix(data)
        
        # Update view-plus overlay with gsc and view_radius for debugging
        if self.view_plus and data:
            gsc = data.get('gsc', 0)
            view_r = data.get('view_radius', 0)
            debug_info = data.get('debug', {})
            self.browser.update_view_plus_overlay(state, gsc=gsc, view_radius=view_r, debug_info=debug_info)

        # Check if died after action
        if not data or data.get('dead'):
            reward, cause = self._get_death_reward_and_cause(pre_action_data)
            mx = pre_action_data['self'].get('x', 0) if pre_action_data and pre_action_data.get('self') else 0
            my = pre_action_data['self'].get('y', 0) if pre_action_data and pre_action_data.get('self') else 0
            return state, reward, True, {"cause": cause, "pos": (mx, my)}

        # === REWARD CALCULATION ===
        reward = 0
        new_snake = data.get('self', {})
        new_len = new_snake.get('len', 0)
        new_x, new_y = new_snake.get('x', 0), new_snake.get('y', 0)
        
        # 1. Survival reward (small positive for staying alive)
        reward += 0.1
        
        # 2. Food reward (big positive for eating)
        if new_len > self.prev_length:
            food_gained = new_len - self.prev_length
            reward += food_gained * 5.0  # Strong reward for eating!
        
        # 3. SHAPING REWARD: Reward for moving TOWARDS food
        new_foods = data.get('foods', [])
        if new_foods:
            new_food_dists = [math.hypot(f[0] - new_x, f[1] - new_y) for f in new_foods if len(f) >= 2]
            new_food_dist = min(new_food_dists) if new_food_dists else None
        else:
            new_food_dist = None
            
        if current_food_dist is not None and new_food_dist is not None:
            # Reward for getting closer to food
            dist_delta = current_food_dist - new_food_dist
            # Scale: ~0.5 reward for moving 100 units closer
            shaping_reward = dist_delta * 0.005
            # Clamp to avoid extreme values
            shaping_reward = max(-0.5, min(0.5, shaping_reward))
            reward += shaping_reward
        
        # 4. Wall proximity penalty (encourage staying away from edges)
        new_dist_from_center = math.hypot(new_x - self.map_center_x, new_y - self.map_center_y)
        new_dist_to_wall = self.map_radius - new_dist_from_center
        
        if new_dist_to_wall < 1000:
            # Reduced penalty to avoid massive negative scores for survival strategies
            # Was 0.5, now 0.05 (half of survival reward)
            reward -= 0.05  

        
        # 5. Small penalty for going straight too much (encourage exploration)
        if action == 0:
            reward -= 0.05  # Tiny penalty for not turning
        
        # Update tracked values
        self.prev_length = new_len
        self.prev_food_dist = new_food_dist

        # Include cause=None for alive steps (trainer will detect MaxSteps)
        return state, reward, False, {
            "length": new_len, 
            "cause": None, 
            "pos": (new_x, new_y), 
            "wall_dist": new_dist_to_wall
        }

    def _get_state(self):
        data = self.browser.get_game_data()
        return self._process_data_to_matrix(data)

    def _matrix_zeros(self):
        return np.zeros((3, self.matrix_size, self.matrix_size), dtype=np.float32)

    def _process_data_to_matrix(self, data):
        """
        Converts raw JSON data to a 3-channel Matrix (84x84).
        Channel 0: Food
        Channel 1: Enemies (Head = 1.0, Body = 0.5) + Walls (1.0) -> DANGER
        Channel 2: Self (Head = 1.0, Body = 0.5) -> SAFE/SELF
        
        Uses DYNAMIC view_radius from game for correct scaling!
        """
        matrix = np.zeros((3, self.matrix_size, self.matrix_size), dtype=np.float32)

        if not data or data.get('dead'):
            return matrix

        my_snake = data.get('self')
        if not my_snake:
            return matrix

        mx, my = my_snake['x'], my_snake['y']
        
        # UPDATE view_size from actual game data!
        game_view_radius = data.get('view_radius')
        if game_view_radius and game_view_radius > 0:
            self.view_size = game_view_radius
            self.scale = self.matrix_size / (self.view_size * 2)
        
        # UPDATE map_radius and center from actual game data!
        game_map_radius = data.get('map_radius')
        if game_map_radius and game_map_radius > 0:
            self.map_radius = game_map_radius
        
        # Get map center (calculated from food positions)
        self.map_center_x = data.get('map_center_x', self.map_radius)
        self.map_center_y = data.get('map_center_y', self.map_radius)
        
        # Print map variables ONCE for debugging
        debug_info = data.get('debug', {})
        if not self._map_vars_printed and debug_info.get('map_vars'):
            print("\n" + "="*50)
            print("SLITHER.IO MAP VARIABLES FOUND:")
            print("="*50)
            map_vars = debug_info.get('map_vars', {})
            for key, val in map_vars.items():
                print(f"  {key}: {val}")
            print(f"\nUsing: source={debug_info.get('boundary_source')}")
            print(f"       map_radius={debug_info.get('map_radius')}")
            print(f"       center=({debug_info.get('map_cx')}, {debug_info.get('map_cy')})")
            print("="*50 + "\n")
            self._map_vars_printed = True

        # Helper to map world coord to matrix coord
        def world_to_matrix(wx, wy):
            # 1. Translate to ego-centric (relative to head)
            dx = wx - mx
            dy = wy - my

            # 2. Scale and center
            # World range: -view_size to +view_size
            # Matrix range: 0 to matrix_size
            # Formula: (dx / view_size) * (matrix_size/2) + (matrix_size/2)
            #        = dx * scale + matrix_size/2
            
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
        # Use dist_to_wall from game data (distance to nearest boundary point)
        dist_to_wall = data.get('dist_to_wall', float('inf'))
        
        # Only calculate walls if they could be visible (snake close to boundary)
        if dist_to_wall < self.view_size * 2.5:
            map_center_x = getattr(self, 'map_center_x', self.map_radius)
            map_center_y = getattr(self, 'map_center_y', self.map_radius)
            
            y_grid, x_grid = np.ogrid[:self.matrix_size, :self.matrix_size]
            
            # Convert matrix coords to world coords
            dx_world = (x_grid - (self.matrix_size / 2)) / self.scale
            dy_world = (y_grid - (self.matrix_size / 2)) / self.scale
            
            wx = mx + dx_world
            wy = my + dy_world
            
            # For polygon boundary, we approximate with circle using detected radius
            # But the dist_to_wall from JS is more accurate for the actual boundary
            dist_sq = (wx - map_center_x)**2 + (wy - map_center_y)**2
            radius_sq = self.map_radius**2
            
            # Create wall mask - use circular approximation but with correct center/radius
            wall_mask = dist_sq > radius_sq
            
            # Draw walls as solid obstacles in ENEMY channel (Danger)
            matrix[1][wall_mask] = 1.0

        return matrix

    def close(self):
        self.browser.close()
