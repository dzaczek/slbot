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
        self.map_radius = 0  # Legacy, not used for wall detection anymore
        self.map_center_x = 0
        self.map_center_y = 0
        self.matrix_size = matrix_size # Output grid size (default 84x84)
        self.view_plus = view_plus  # Enable visual overlay
        
        # Dynamic view size - will be updated from game data
        self.view_size = 500  # Initial estimate, updated dynamically
        self.scale = self.matrix_size / (self.view_size * 2)
        
        # Flag to print map vars only once
        self._map_vars_printed = False

        # Wall tracking - stores actual dist_to_wall from JS (pbx vertices)
        self.last_dist_to_wall = 99999
        self.near_wall_frames = 0
        
        # Track previous length to detect food eating
        self.prev_length = 0
        
        # Track distance to nearest food for shaping reward
        self.prev_food_dist = None
        
        # Frame skip - repeat action for N frames
        self.frame_skip = 4

        # === CURRICULUM REWARD PARAMETERS (set by trainer via set_curriculum_stage) ===
        self.food_reward = 10.0       # Stage 1 default: high food reward
        self.food_shaping = 0.01      # Reward for moving towards food
        self.survival_reward = 0.05   # Per-step survival bonus
        self.death_wall_penalty = -15 # Increased default to prevent suicide eating
        self.death_snake_penalty = -15
        self.straight_penalty = 0.0   # Stage 1: no straight penalty
        self.length_bonus = 0.0       # Stage 3: per-step bonus for snake length

    def set_curriculum_stage(self, stage_config):
        """Set reward parameters from curriculum stage config dict."""
        self.food_reward = stage_config.get('food_reward', 5.0)
        self.food_shaping = stage_config.get('food_shaping', 0.005)
        self.survival_reward = stage_config.get('survival', 0.1)
        self.death_wall_penalty = stage_config.get('death_wall', -100)
        self.death_snake_penalty = stage_config.get('death_snake', -10)
        self.straight_penalty = stage_config.get('straight_penalty', 0.05)
        self.length_bonus = stage_config.get('length_bonus', 0.0)
        print(f"  ENV: food={self.food_reward} surv={self.survival_reward} "
              f"wall={self.death_wall_penalty} snake={self.death_snake_penalty}")

    def _update_from_game_data(self, data):
        """Update wall distance and map info from game data."""
        if not data:
            return
        
        # dist_to_wall comes from JS (empirical: distance from grd-based center minus estimated radius)
        dtw = data.get('dist_to_wall')
        if dtw is not None:
            self.last_dist_to_wall = dtw
        
        # Update map params from JS
        mr = data.get('map_radius')
        if mr and mr > 0:
            self.map_radius = mr
        cx = data.get('map_center_x')
        cy = data.get('map_center_y')
        if cx and cy:
            self.map_center_x = cx
            self.map_center_y = cy
        
        # Track wall proximity
        if self.last_dist_to_wall < 2000:
            self.near_wall_frames += 1
        else:
            self.near_wall_frames = max(0, self.near_wall_frames - 1)
    
    def _calc_dist_to_wall(self, x, y):
        """
        Returns the last known distance to wall from JS.
        We no longer calculate this ourselves - JS has the real pbx vertex data.
        """
        return self.last_dist_to_wall
    
    def _is_outside_map(self, x, y):
        """Returns True if position is outside the map boundary."""
        return self.last_dist_to_wall <= 0
    
    def _is_near_wall(self, x, y, threshold=2000):
        """Returns True if position is within threshold of wall."""
        return self.last_dist_to_wall < threshold

    # =====================================================
    # DEATH CLASSIFICATION (Forensic approach)
    # =====================================================

    def _get_death_reward_and_cause(self, last_data=None):
        """
        Death classification: Forensic + Geometric hybrid.
        
        1. Enemy nearby (< 500 units) -> SnakeCollision (certain)
        2. No enemy nearby BUT close to wall (dist_to_wall < 2000) -> Wall (certain)
        3. No enemy nearby AND far from wall -> SnakeCollision (default)
           (enemy likely appeared during frame_skip and wasn't in pre_action_data)
        """
        mx, my = 0, 0
        if last_data and last_data.get('self'):
            mx = last_data['self'].get('x', 0)
            my = last_data['self'].get('y', 0)
        
        # Get geometric wall distance
        dist_to_wall = last_data.get('dist_to_wall', 99999) if last_data else 99999
        
        # Check if any enemy was close to our head at the moment before death
        enemies = last_data.get('enemies', []) if last_data else []
        min_enemy_dist = float('inf')
        
        for e in enemies:
            # Check enemy head
            ex, ey = e.get('x', 0), e.get('y', 0)
            dist = math.hypot(ex - mx, ey - my)
            min_enemy_dist = min(min_enemy_dist, dist)
            
            # Check enemy body points
            for pt in e.get('pts', []):
                if len(pt) >= 2:
                    px, py = pt[0], pt[1]
                    dist = math.hypot(px - mx, py - my)
                    min_enemy_dist = min(min_enemy_dist, dist)
        
        # Classification logic
        COLLISION_RADIUS = 500  # Increased to account for frame_skip movement
        WALL_PROXIMITY = 2000   # Must be geometrically near wall to classify as Wall
        
        if min_enemy_dist < COLLISION_RADIUS:
            # Enemy was close -> definitely SnakeCollision
            cause = "SnakeCollision"
            penalty = self.death_snake_penalty
        elif dist_to_wall < WALL_PROXIMITY:
            # No enemy close BUT we're near the wall -> Wall death
            cause = "Wall"
            penalty = self.death_wall_penalty
        else:
            # No enemy in data AND far from wall -> likely snake that appeared during frame_skip
            cause = "SnakeCollision"
            penalty = self.death_snake_penalty
        
        print(f"DEBUG DEATH: Pos=({mx:.0f},{my:.0f}) NearestEnemy={min_enemy_dist:.0f} WallDist={dist_to_wall:.0f} -> {cause} ({penalty})")
        
        return penalty, cause

    def reset(self):
        """Resets the game and returns initial matrix state."""
        self.browser.force_restart()
        self.near_wall_frames = 0
        
        # One-time game variable scan
        if not self._map_vars_printed:
            scan = self.browser.scan_game_variables()
            if scan:
                print("\n" + "="*60)
                print("GAME VARIABLE SCAN")
                print("="*60)
                if scan.get('snake_pos'):
                    print(f"Snake pos: {scan['snake_pos']}")
                print(f"\n--- Specific vars ---")
                for k, v in sorted(scan.get('specific', {}).items()):
                    print(f"  {k}: {v}")
                print(f"\n--- Numeric vars (map-range) ---")
                for k, v in sorted(scan.get('numeric', {}).items()):
                    if isinstance(v, (int, float)) and v > 1000:
                        print(f"  {k}: {v}")
                print(f"\n--- Arrays ---")
                for k, v in sorted(scan.get('arrays', {}).items()):
                    print(f"  {k}: len={v.get('length')} first={v.get('first3')}")
                print(f"\n--- Boundary-related functions ---")
                for f in scan.get('boundary_funcs', []):
                    print(f"  {f}()")
                print("="*60 + "\n")
        
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

    def step(self, action):
        """
        Executes action and returns (next_state, reward, done, info).
        Actions:
        0: Keep current direction
        1: Turn Left small (~40 deg)
        2: Turn Right small (~40 deg)
        3: Turn Left big (~103 deg)
        4: Turn Right big (~103 deg)
        5: Boost (speed up, loses mass)
        """
        angle_change = 0
        boost = 0

        if action == 0: pass
        elif action == 1: angle_change = -0.7  # ~40 deg
        elif action == 2: angle_change = 0.7   # ~40 deg
        elif action == 3: angle_change = -1.8  # ~103 deg
        elif action == 4: angle_change = 1.8   # ~103 deg
        elif action == 5: boost = 1

        # Get current state before action
        data = self.browser.get_game_data()
        if not data:
            return self._matrix_zeros(), -5, True, {"cause": "BrowserError"}

        # Update map params IMMEDIATELY from fresh game data
        self._update_from_game_data(data)

        if data.get('dead'):
            reward, cause = self._get_death_reward_and_cause(data)
            mx = data['self'].get('x', 0) if data.get('self') else 0
            my = data['self'].get('y', 0) if data.get('self') else 0
            dtw = self._calc_dist_to_wall(mx, my)
            return self._matrix_zeros(), reward, True, {"cause": cause, "pos": (mx, my), "wall_dist": dtw}

        my_snake = data.get('self', {})
        current_ang = my_snake.get('ang', 0)
        mx, my = my_snake.get('x', 0), my_snake.get('y', 0)
        current_len = my_snake.get('len', 0)

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
            time.sleep(0.02)

        # Get new state after action
        data = self.browser.get_game_data()
        state = self._process_data_to_matrix(data)
        
        # Update view-plus overlay
        if self.view_plus and data:
            gsc = data.get('gsc', 0)
            view_r = data.get('view_radius', 0)
            debug_info = data.get('debug', {})
            self.browser.update_view_plus_overlay(state, gsc=gsc, view_radius=view_r, debug_info=debug_info)

        # Check if died after action
        if not data or data.get('dead'):
            reward, cause = self._get_death_reward_and_cause(pre_action_data)
            pmx = pre_action_data['self'].get('x', 0) if pre_action_data and pre_action_data.get('self') else 0
            pmy = pre_action_data['self'].get('y', 0) if pre_action_data and pre_action_data.get('self') else 0
            dtw = self._calc_dist_to_wall(pmx, pmy)
            return state, reward, True, {"cause": cause, "pos": (pmx, pmy), "wall_dist": dtw}

        # === REWARD CALCULATION ===
        reward = 0
        new_snake = data.get('self', {})
        new_len = new_snake.get('len', 0)
        new_x, new_y = new_snake.get('x', 0), new_snake.get('y', 0)
        
        # Update wall tracking from post-action data
        self._update_from_game_data(data)
        new_dist_to_wall = data.get('dist_to_wall', 99999)
        
        # 1. Survival reward
        reward += self.survival_reward
        
        # 2. Food reward (parametrized by curriculum stage)
        food_eaten = 0
        if new_len > self.prev_length:
            food_eaten = new_len - self.prev_length
            reward += food_eaten * self.food_reward
        
        # 3. Length bonus (Stage 3: reward for being a big snake)
        if self.length_bonus > 0 and new_len > 0:
            reward += self.length_bonus * new_len
        
        # 4. SHAPING REWARD: Reward for moving TOWARDS food
        new_foods = data.get('foods', [])
        if new_foods:
            new_food_dists = [math.hypot(f[0] - new_x, f[1] - new_y) for f in new_foods if len(f) >= 2]
            new_food_dist = min(new_food_dists) if new_food_dists else None
        else:
            new_food_dist = None
            
        if current_food_dist is not None and new_food_dist is not None:
            dist_delta = current_food_dist - new_food_dist
            shaping_reward = dist_delta * self.food_shaping
            shaping_reward = max(-0.5, min(0.5, shaping_reward))
            reward += shaping_reward
        
        # 5. Straight penalty (encourage turning/exploration)
        if self.straight_penalty > 0 and action == 0:
            reward -= self.straight_penalty
        
        # Update tracked values
        self.prev_length = new_len
        self.prev_food_dist = new_food_dist

        return state, reward, False, {
            "length": new_len,
            "food_eaten": food_eaten,
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
        
        # UPDATE map params from game data
        self._update_from_game_data(data)
        
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
            print(f"       dist_to_wall={debug_info.get('dist_to_wall')}")
            print("="*50 + "\n")
            self._map_vars_printed = True

        # Helper to map world coord to matrix coord
        def world_to_matrix(wx, wy):
            dx = wx - mx
            dy = wy - my
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
                matrix[0, mx_y, mx_x] = 1.0

        # 2. Enemies (Channel 1)
        enemies = data.get('enemies', [])
        for e in enemies:
            ex, ey = e['x'], e['y']
            hx, hy = world_to_matrix(ex, ey)
            if 0 <= hx < self.matrix_size and 0 <= hy < self.matrix_size:
                matrix[1, hy, hx] = 1.0

            for pt in e.get('pts', []):
                 px, py = pt[0], pt[1]
                 bx, by = world_to_matrix(px, py)
                 if 0 <= bx < self.matrix_size and 0 <= by < self.matrix_size:
                     matrix[1, by, bx] = 0.5

        # 3. Self (Channel 2)
        cx, cy = self.matrix_size // 2, self.matrix_size // 2
        matrix[2, cy, cx] = 1.0

        for pt in my_snake.get('pts', []):
            px, py = pt[0], pt[1]
            bx, by = world_to_matrix(px, py)
            if 0 <= bx < self.matrix_size and 0 <= by < self.matrix_size:
                matrix[2, by, bx] = 0.5

        # 4. Walls (Channel 1 - DANGER) 
        # We draw wall based on grd circle, but this is approximate.
        # The real wall learning comes from death penalties (forensic detection).
        # Visual wall in matrix helps the bot see the boundary coming.
        dist_to_wall = data.get('dist_to_wall', 99999)
        
        if dist_to_wall < 5000 and self.map_radius > 0 and self.map_center_x > 0:
            y_grid, x_grid = np.ogrid[:self.matrix_size, :self.matrix_size]
            
            dx_world = (x_grid - (self.matrix_size / 2)) / self.scale
            dy_world = (y_grid - (self.matrix_size / 2)) / self.scale
            
            wx = mx + dx_world
            wy = my + dy_world
            
            dist_sq = (wx - self.map_center_x)**2 + (wy - self.map_center_y)**2
            radius_sq = self.map_radius**2
            
            wall_mask = dist_sq > radius_sq
            matrix[1][wall_mask] = 1.0

        return matrix

    def close(self):
        self.browser.close()
