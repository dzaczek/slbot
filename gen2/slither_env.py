import numpy as np
import math
import sys
import os
import time
import json
import matplotlib.pyplot as plt
# Ensure matplotlib uses a non-interactive backend for headless environments
plt.switch_backend('Agg')

# Add parent directory to path to import browser_engine
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from browser_engine import SlitherBrowser
from coord_transform import world_to_grid
from matplotlib.path import Path as MplPath

class SlitherEnv:
    def __init__(self, headless=True, nickname="MatrixBot", matrix_size=84, view_plus=False, base_url="http://slither.io", frame_skip=4):
        self.browser = SlitherBrowser(headless=headless, nickname=nickname, base_url=base_url)
        # Fixed Map Constants (Standard Slither.io)
        self.MAP_RADIUS = 21600
        self.MAP_CENTER_X = 21600
        self.MAP_CENTER_Y = 21600

        self.map_radius = self.MAP_RADIUS
        self.map_center_x = self.MAP_CENTER_X
        self.map_center_y = self.MAP_CENTER_Y
        self.boundary_type = 'circle' # Force circle
        self.boundary_vertices = []
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
        self.frame_skip = frame_skip

        # === CURRICULUM REWARD PARAMETERS (set by trainer via set_curriculum_stage) ===
        self.food_reward = 10.0       # Stage 1 default: high food reward
        self.food_shaping = 0.01      # Reward for moving towards food
        self.survival_reward = 0.05   # Per-step survival bonus
        self.death_wall_penalty = -15 # Increased default to prevent suicide eating
        self.death_snake_penalty = -15
        self.straight_penalty = 0.0   # Stage 1: no straight penalty
        self.length_bonus = 0.0       # Stage 3: per-step bonus for snake length

        # Threat awareness
        self.wall_alert_dist = 2000
        self.enemy_alert_dist = 800
        self.wall_proximity_penalty = 0.0
        self.enemy_proximity_penalty = 0.0

        # Frame validation (from tsrgy0)
        self.last_matrix = self._matrix_zeros()
        self.last_valid_data = None
        self.invalid_frame_count = 0
        self.max_invalid_frames = 15

    def _has_valid_coordinates(self, data):
        """Checks if the frame contains plausible coordinates (ignores dead status)."""
        if not data:
            return False
        snake = data.get('self') or {}
        mx = snake.get('x')
        my = snake.get('y')
        if mx is None or my is None:
            return False
        if not math.isfinite(mx) or not math.isfinite(my):
            return False
        # Basic sanity check for coordinates (reject 0,0 initialization glitch)
        # Map center is (21600, 21600), so (0,0) is far outside.
        if abs(mx) <= 1000 and abs(my) <= 1000:
            return False
        return True

    def _is_valid_frame(self, data):
        """Checks if the frame data is valid for training (alive and good coords)."""
        if not data or data.get('dead'):
            return False
        if data.get('valid') is False:
            return False
        return self._has_valid_coordinates(data)

    def set_curriculum_stage(self, stage_config):
        """Set reward parameters from curriculum stage config dict."""
        self.food_reward = stage_config.get('food_reward', 5.0)
        self.food_shaping = stage_config.get('food_shaping', 0.005)
        self.survival_reward = stage_config.get('survival', 0.1)
        self.death_wall_penalty = stage_config.get('death_wall', -100)
        self.death_snake_penalty = stage_config.get('death_snake', -10)
        self.straight_penalty = stage_config.get('straight_penalty', 0.05)
        self.length_bonus = stage_config.get('length_bonus', 0.0)

        # Update alert distances if provided, otherwise keep defaults
        self.wall_alert_dist = stage_config.get('wall_alert_dist', 2000)
        self.enemy_alert_dist = stage_config.get('enemy_alert_dist', 800)
        self.wall_proximity_penalty = stage_config.get('wall_proximity_penalty', 0.0)
        self.enemy_proximity_penalty = stage_config.get('enemy_proximity_penalty', 0.0)

        print(f"  ENV: food={self.food_reward} surv={self.survival_reward} "
              f"wall={self.death_wall_penalty} snake={self.death_snake_penalty}")

    def _update_from_game_data(self, data):
        """Update wall distance and map info from game data."""
        if not data:
            return
        
        # We ignore JS dist_to_wall to fix false positives with polygon logic
        # and enforce Python radial logic.
        
        # Update map params if they look reasonable, else stick to defaults
        mr = data.get('map_radius')
        if mr and math.isfinite(mr) and mr > 10000:
             self.map_radius = mr

        cx = data.get('map_center_x')
        cy = data.get('map_center_y')
        if cx and cy and math.isfinite(cx) and math.isfinite(cy) and cx > 10000:
             self.map_center_x = cx
             self.map_center_y = cy

        # Force Circle logic regardless of what JS says
        self.boundary_type = 'circle'
        self.boundary_vertices = data.get('boundary_vertices', [])
        
        # Recalculate dist_to_wall using Python Radial Logic
        mx = data.get('self', {}).get('x', 0)
        my = data.get('self', {}).get('y', 0)

        # Only update if we have valid coordinates
        if abs(mx) > 100 and abs(my) > 100:
            self.last_dist_to_wall = self._calc_dist_to_wall(mx, my)

        # Track wall proximity
        if self.last_dist_to_wall < 2000:
            self.near_wall_frames += 1
        else:
            self.near_wall_frames = max(0, self.near_wall_frames - 1)
    
    def _calc_dist_to_wall(self, x, y):
        """
        Calculates distance to wall using pure radial logic.
        Standard Slither.io map is a circle.
        """
        dist_from_center = math.hypot(x - self.map_center_x, y - self.map_center_y)
        return self.map_radius - dist_from_center

    def _snake_width_world(self, snake):
        sc = snake.get('sc', 1.0) if snake else 1.0
        return sc * 29.0

    def _head_radius_world(self, snake):
        width = self._snake_width_world(snake)
        return (width / 2.0) * 1.2

    def _min_enemy_distance(self, enemies, mx, my):
        min_enemy_dist = float('inf')
        for e in enemies or []:
            ex, ey = e.get('x', 0), e.get('y', 0)
            min_enemy_dist = min(min_enemy_dist, math.hypot(ex - mx, ey - my))
            for pt in e.get('pts', []):
                if len(pt) >= 2:
                    px, py = pt[0], pt[1]
                    min_enemy_dist = min(min_enemy_dist, math.hypot(px - mx, py - my))
        return min_enemy_dist
    
    def _is_outside_map(self, x, y):
        """Returns True if position is outside the map boundary."""
        return self.last_dist_to_wall <= 0
    
    def _is_near_wall(self, x, y, threshold=2000):
        """Returns True if position is within threshold of wall."""
        return self.last_dist_to_wall < threshold

    def _draw_circle(self, matrix, channel, cx, cy, r, value):
        """Draws a filled circle on the matrix."""
        r_int = int(math.ceil(r))
        x_min = max(0, int(cx - r_int))
        x_max = min(self.matrix_size, int(cx + r_int + 1))
        y_min = max(0, int(cy - r_int))
        y_max = min(self.matrix_size, int(cy + r_int + 1))

        r_sq = r * r

        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                dx = x - cx
                dy = y - cy
                if dx*dx + dy*dy <= r_sq:
                    matrix[channel, y, x] = value

    def _draw_thick_line(self, matrix, channel, x0, y0, x1, y1, width, value):
        """Draws a thick line by interpolating circles along the segment."""
        radius = width / 2.0
        dist = math.hypot(x1 - x0, y1 - y0)

        if dist == 0:
            self._draw_circle(matrix, channel, x0, y0, radius, value)
            return

        # Interpolate circles
        # Step size should be radius to ensure coverage
        steps = int(dist / max(0.5, radius * 0.5)) + 1

        for i in range(steps + 1):
            t = i / max(1, steps)
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            self._draw_circle(matrix, channel, x, y, radius, value)

    # =====================================================
    # DEATH CLASSIFICATION (Forensic approach)
    # =====================================================

    def save_death_packet(self, matrix, reward, cause, final_data):
        """
        Saves a comprehensive death event packet (JSON + Image).
        Replaces legacy save_debug_image.
        """
        try:
            timestamp = int(time.time())
            date_str = time.strftime("%Y%m%d_%H%M%S")
            debug_dir = os.path.join(os.path.dirname(__file__), 'events')
            os.makedirs(debug_dir, exist_ok=True)

            # 1. Save Image
            img_filename = f"event_{date_str}_{cause}.png"
            img_path = os.path.join(debug_dir, img_filename)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            titles = ["Food", "Enemies/Wall", "Self"]
            for i in range(3):
                axes[i].imshow(matrix[i], cmap='gray', origin='upper')
                axes[i].set_title(titles[i])
                axes[i].axis('off')

            snake = final_data.get('self', {})
            pos_str = f"({snake.get('x',0):.0f}, {snake.get('y',0):.0f})"
            wall_d = final_data.get('dist_to_wall', -1)

            plt.suptitle(f"Cause: {cause} | Reward: {reward:.2f}\nPos: {pos_str} | Wall Dist: {wall_d:.0f}")
            plt.tight_layout()
            plt.savefig(img_path)
            plt.close(fig)

            # 2. Save JSON
            json_filename = f"event_{date_str}_{cause}.json"
            json_path = os.path.join(debug_dir, json_filename)

            packet = {
                "timestamp": timestamp,
                "date": date_str,
                "cause": cause,
                "reward": reward,
                "snake": {
                    "x": snake.get('x'),
                    "y": snake.get('y'),
                    "len": snake.get('len'),
                    "ang": snake.get('ang')
                },
                "env": {
                    "dist_to_wall": wall_d,
                    "map_radius": final_data.get('map_radius'),
                    "view_radius": final_data.get('view_radius'),
                    "boundary_type": self.boundary_type
                },
                "debug": final_data.get('debug', {})
            }

            with open(json_path, 'w') as f:
                json.dump(packet, f, indent=2)

            print(f"Saved Death Packet: {json_filename}")

        except Exception as e:
            print(f"Failed to save death packet: {e}")

    def _get_death_reward_and_cause(self, last_data=None):
        """
        Death classification: Forensic + Geometric hybrid.
        
        Priority Logic:
        1. Strictly outside map (dist < 0) -> Wall
        2. Enemy collision (dist < 500) -> SnakeCollision
        3. Near wall (dist < 2000) -> Wall
        4. Default -> SnakeCollision
        """
        mx, my = 0, 0
        if last_data and last_data.get('self'):
            mx = last_data['self'].get('x', 0)
            my = last_data['self'].get('y', 0)
        
        # Python-side Strict Check (PRIMARY)
        # We calculate wall distance ourselves to be sure
        dist_to_wall_py = self._calc_dist_to_wall(mx, my)

        # Check if any enemy was close to our head at the moment before death
        enemies = last_data.get('enemies', []) if last_data else []
        min_enemy_dist = self._min_enemy_distance(enemies, mx, my)
        head_radius = self._head_radius_world(last_data.get('self') if last_data else None)
        
        # Classification logic
        # Tighter thresholds for more accurate classification
        COLLISION_BUFFER = 40
        WALL_BUFFER = 40
        
        cause = "Unknown"
        penalty = self.death_snake_penalty

        # Priority 1: Strictly outside map (Absolute Wall Death)
        if dist_to_wall_py < -50:
             cause = "Wall"
             penalty = self.death_wall_penalty

        # Priority 2: High confidence Enemy Collision
        elif min_enemy_dist != float('inf') and min_enemy_dist <= (head_radius + COLLISION_BUFFER):
            cause = "SnakeCollision"
            penalty = self.death_snake_penalty

        # Priority 3: Proximity to wall (if no enemy hit detected)
        elif dist_to_wall_py <= (head_radius + WALL_BUFFER):
             cause = "Wall"
             penalty = self.death_wall_penalty

        # Priority 4: Default (Assumed Snake Collision if inside map)
        else:
            cause = "Unknown" if min_enemy_dist == float('inf') else "SnakeCollision"
            penalty = self.death_snake_penalty
        
        min_enemy_display = min_enemy_dist if min_enemy_dist != float('inf') else -1
        print(f"DEBUG DEATH: Pos=({mx:.0f},{my:.0f}) NearestEnemy={min_enemy_display:.0f} WallDistPY={dist_to_wall_py:.0f} -> {cause} ({penalty})")
        
        return penalty, cause

    def reset(self):
        """Resets the game and returns initial matrix state."""
        self.browser.force_restart()
        self.near_wall_frames = 0
        self.invalid_frame_count = 0
        
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
        # We enforce strict coordinate validation to avoid (0,0) starts
        data = None
        start_time = time.time()
        while time.time() - start_time < 10.0: # 10 second timeout
            data = self.browser.get_game_data()
            if self._is_valid_frame(data):
                mx = data.get('self', {}).get('x', 0)
                my = data.get('self', {}).get('y', 0)
                # Double check to be absolutely sure
                if abs(mx) > 100 and abs(my) > 100:
                    break
            time.sleep(0.2)

        # If still invalid, we might want to try force_restart again or just warn
        if not data or not self._is_valid_frame(data):
            print("WARNING: reset() failed to get valid coordinates after 10s. Trying again...")
            self.browser.force_restart()
            time.sleep(2.0)
            # Try one more time quickly
            data = self.browser.get_game_data()

        if data and data.get('self'):
            self.prev_length = data['self'].get('len', 0)
        else:
            self.prev_length = 0
            
        matrix = self._process_data_to_matrix(data)
        self.last_matrix = matrix
        self.last_valid_data = data if self._is_valid_frame(data) else None
        
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

        # Robust check (Validation Logic from tsrgy0)
        if not data:
            return self._matrix_zeros(), -5, True, {"cause": "BrowserError"}

        # Check for valid coordinates regardless of dead status
        has_valid_coords = self._has_valid_coordinates(data)

        if not data.get('dead') and not has_valid_coords:
            self.invalid_frame_count += 1
            if self.invalid_frame_count >= self.max_invalid_frames:
                return self.last_matrix, -5, True, {"cause": "InvalidFrame"}
            # Return last valid state
            return self.last_matrix, 0.0, False, {"cause": "InvalidFrame"}

        self.invalid_frame_count = 0

        # Only update last_valid_data if coordinates are plausible
        if has_valid_coords:
            self.last_valid_data = data

        # Update map params IMMEDIATELY from fresh game data
        self._update_from_game_data(data)

        if data.get('dead'):
            # If death frame has invalid coords (0,0 bug), use last valid data
            final_data = data
            if not has_valid_coords and self.last_valid_data:
                final_data = self.last_valid_data
                # print(f"DEBUG: Using last_valid_data for death classification (Current: {data.get('self', {}).get('x')})")

            reward, cause = self._get_death_reward_and_cause(final_data)

            snake = final_data.get('self', {})
            mx = snake.get('x', 0)
            my = snake.get('y', 0)
            dtw = self._calc_dist_to_wall(mx, my)
            min_enemy_dist = self._min_enemy_distance(final_data.get('enemies', []), mx, my)

            # Use last valid data for death packet if current is empty?
            # data is 'dead', so it might be empty.
            # But we have last_valid_data

            return self._matrix_zeros(), reward, True, {
                "cause": cause,
                "pos": (mx, my),
                "wall_dist": dtw,
                "enemy_dist": min_enemy_dist if min_enemy_dist != float('inf') else -1,
            }

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

        # Validate again
        if data and not data.get('dead') and not self._is_valid_frame(data):
             self.invalid_frame_count += 1
             if self.invalid_frame_count >= self.max_invalid_frames:
                 return self.last_matrix, -5, True, {"cause": "InvalidFrame"}
             return self.last_matrix, 0.0, False, {"cause": "InvalidFrame"}

        state = self._process_data_to_matrix(data)
        self.last_matrix = state
        if self._is_valid_frame(data):
            self.last_valid_data = data
            self.invalid_frame_count = 0
        
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
            min_enemy_dist = self._min_enemy_distance(pre_action_data.get('enemies', []), pmx, pmy)

            # Save debug image using LAST VALID STATE
            debug_matrix = self._process_data_to_matrix(pre_action_data)
            self.save_death_packet(debug_matrix, reward, cause, pre_action_data)

            return state, reward, True, {
                "cause": cause,
                "pos": (pmx, pmy),
                "wall_dist": dtw,
                "enemy_dist": min_enemy_dist if min_enemy_dist != float('inf') else -1,
            }

        # === REWARD CALCULATION ===
        reward = 0
        new_snake = data.get('self', {})
        new_len = new_snake.get('len', 0)
        new_x, new_y = new_snake.get('x', 0), new_snake.get('y', 0)
        
        # Update wall tracking from post-action data
        self._update_from_game_data(data)
        new_dist_to_wall = data.get('dist_to_wall', 99999)
        if not math.isfinite(new_dist_to_wall):
            new_dist_to_wall = 99999
        min_enemy_dist = self._min_enemy_distance(data.get('enemies', []), new_x, new_y)
        
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

        # 6. Threat proximity shaping (keep distance from wall/enemies)
        if self.wall_proximity_penalty > 0 and new_dist_to_wall < self.wall_alert_dist:
            wall_ratio = max(0.0, 1.0 - (new_dist_to_wall / max(self.wall_alert_dist, 1)))
            reward -= self.wall_proximity_penalty * wall_ratio

        if self.enemy_proximity_penalty > 0 and min_enemy_dist != float('inf') and min_enemy_dist < self.enemy_alert_dist:
            enemy_ratio = max(0.0, 1.0 - (min_enemy_dist / max(self.enemy_alert_dist, 1)))
            reward -= self.enemy_proximity_penalty * enemy_ratio
        
        # Update tracked values
        self.prev_length = new_len
        self.prev_food_dist = new_food_dist

        return state, reward, False, {
            "length": new_len,
            "food_eaten": food_eaten,
            "cause": None,
            "pos": (new_x, new_y),
            "wall_dist": new_dist_to_wall,
            "enemy_dist": min_enemy_dist if min_enemy_dist != float('inf') else -1
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
            # Enforce minimum view radius to ensure bot sees enough context
            self.view_size = max(float(game_view_radius), 500.0)
            self.scale = self.matrix_size / (self.view_size * 2)
        else:
            self.view_size = 500.0
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

        # 1. Food (Channel 0)
        foods = data.get('foods', [])
        nearest_food = None
        min_dist_sq = float('inf')

        for f in foods:
            if len(f) < 2: continue
            fx, fy = f[0], f[1]

            # Track nearest food
            dist_sq = (fx - mx)**2 + (fy - my)**2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                nearest_food = (fx, fy)

            mx_x, mx_y = world_to_grid(fx, fy, mx, my, self.scale, self.matrix_size)
            if 0 <= mx_x < self.matrix_size and 0 <= mx_y < self.matrix_size:
                matrix[0, mx_y, mx_x] = 1.0

        # Highlighting Nearest Food (Compass/Focus)
        if nearest_food:
            nfx, nfy = nearest_food
            # Calculate grid coordinates even if off-screen
            dx = (nfx - mx) * self.scale
            dy = (nfy - my) * self.scale
            cx = self.matrix_size / 2
            cy = self.matrix_size / 2

            nmx_x = int(cx + dx)
            nmx_y = int(cy + dy)

            # Check if on screen
            if 0 <= nmx_x < self.matrix_size and 0 <= nmx_y < self.matrix_size:
                # On screen: Draw bigger (radius 2.0) to highlight
                self._draw_circle(matrix, 0, nmx_x, nmx_y, 2.0, 1.0)
            else:
                # Off screen: Draw compass marker on edge
                # Normalize vector
                mag = math.hypot(dx, dy)
                if mag > 0:
                    ndx = dx / mag
                    ndy = dy / mag

                    # Project to edge (box projection)
                    # We want to find t such that (cx + t*ndx, cy + t*ndy) is on edge
                    # Edge is x=0, x=W, y=0, y=H

                    # Max dist to edge from center is size/2
                    half_size = self.matrix_size / 2 - 2 # -2 padding to keep marker fully inside

                    # Calculate t for X and Y boundaries
                    tx = float('inf')
                    ty = float('inf')

                    if abs(ndx) > 1e-6:
                        tx = half_size / abs(ndx)
                    if abs(ndy) > 1e-6:
                        ty = half_size / abs(ndy)

                    t = min(tx, ty)

                    ex = cx + t * ndx
                    ey = cy + t * ndy

                    # Draw marker (radius 1.5)
                    self._draw_circle(matrix, 0, ex, ey, 1.5, 0.8)

        # 2. Enemies (Channel 1)
        enemies = data.get('enemies', [])

        # Explicit scaling logic as requested
        # Center of grid
        cx_grid = self.matrix_size / 2
        cy_grid = self.matrix_size / 2

        for e in enemies:
            ex, ey = e['x'], e['y']

            # Relative position
            dx = ex - mx
            dy = ey - my

            # Grid coordinates (no rotation, north-up)
            hx = cx_grid + dx * self.scale
            hy = cx_grid + dy * self.scale

            # Calculate snake width in matrix pixels
            sc = e.get('sc', 1.0)
            width_world = sc * 29.0
            width_matrix = max(1.0, width_world * self.scale)

            # Draw Head
            head_radius = (width_matrix / 2.0) * 1.2

            # Only draw if roughly on screen (optimization)
            if -50 < hx < self.matrix_size + 50 and -50 < hy < self.matrix_size + 50:
                 self._draw_circle(matrix, 1, hx, hy, head_radius, 1.0)

            pts = e.get('pts', [])

            # Draw Body
            prev_x, prev_y = hx, hy

            for pt in pts:
                px_world, py_world = pt[0], pt[1]
                dx_p = px_world - mx
                dy_p = py_world - my

                px_grid = cx_grid + dx_p * self.scale
                py_grid = cx_grid + dy_p * self.scale

                self._draw_thick_line(matrix, 1, prev_x, prev_y, px_grid, py_grid, width_matrix, 0.5)
                prev_x, prev_y = px_grid, py_grid

        # 3. Self (Channel 2)
        cx, cy = self.matrix_size // 2, self.matrix_size // 2

        my_pts = my_snake.get('pts', [])
        my_sc = my_snake.get('sc', 1.0)
        my_width_matrix = max(1.0, (my_sc * 29.0) * self.scale)

        # Head to first point
        if my_pts:
             px, py = my_pts[0][0], my_pts[0][1]
             bx, by = world_to_grid(px, py, mx, my, self.scale, self.matrix_size)
             self._draw_thick_line(matrix, 2, cx, cy, bx, by, my_width_matrix, 0.5)

        for i in range(len(my_pts) - 1):
            p1 = my_pts[i]
            p2 = my_pts[i+1]
            x1, y1 = world_to_grid(p1[0], p1[1], mx, my, self.scale, self.matrix_size)
            x2, y2 = world_to_grid(p2[0], p2[1], mx, my, self.scale, self.matrix_size)
            self._draw_thick_line(matrix, 2, x1, y1, x2, y2, my_width_matrix, 0.5)

        # Head
        my_head_radius = (my_width_matrix / 2.0) * 1.2
        self._draw_circle(matrix, 2, cx, cy, my_head_radius, 1.0)

        # 4. Walls (Channel 1 - DANGER)
        # Radial / Circular Wall Rendering (Force Python Logic)
        
        # We always check wall distance in Python now
        dist_to_wall_py = self._calc_dist_to_wall(mx, my)

        # Only draw if wall is potentially visible on the grid
        # Max view distance ~ view_size.
        # If dist > view_size, we probably don't see it.
        if dist_to_wall_py < self.view_size * 1.5:
             # Standard Circular Wall
             y_grid, x_grid = np.ogrid[:self.matrix_size, :self.matrix_size]

             # Convert grid coords to relative world coords
             dx_world = (x_grid - (self.matrix_size / 2)) / self.scale
             dy_world = (y_grid - (self.matrix_size / 2)) / self.scale

             # Absolute world coords
             wx = mx + dx_world
             wy = my + dy_world

             # Check distance from map center
             dist_sq = (wx - self.map_center_x)**2 + (wy - self.map_center_y)**2

             # Safety margin: behave as if wall is slightly closer (radius - 500)
             # This ensures we don't accidentally go out.
             # Note: map_radius is 21600.
             radius_sq = (self.map_radius - 500)**2

             wall_mask = dist_sq > radius_sq
             matrix[1][wall_mask] = 1.0

        return matrix

    def close(self):
        self.browser.close()
