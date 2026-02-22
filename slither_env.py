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
from coord_transform import world_to_grid


def _create_browser(backend, headless, nickname, base_url, ws_server_url=""):
    """Factory: create SlitherBrowser using selected backend."""
    if backend == "websocket":
        # CDP hybrid: Chrome handles anti-bot, CDP intercepts WS frames for speed
        from browser_engine import SlitherBrowser
        return SlitherBrowser(headless=headless, nickname=nickname,
                              base_url=base_url, use_cdp=True)
    else:
        from browser_engine import SlitherBrowser
        return SlitherBrowser(headless=headless, nickname=nickname,
                              base_url=base_url)
from matplotlib.path import Path as MplPath

class SlitherEnv:
    def __init__(self, headless=True, nickname="MatrixBot", matrix_size=84, view_plus=False, base_url="http://slither.io", frame_skip=4, backend="selenium", ws_server_url=""):
        self.backend = backend
        self.browser = _create_browser(backend, headless, nickname, base_url, ws_server_url)
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

        # Auto-calibration: collect death positions to detect real map radius
        self._wall_death_samples = []  # list of dist_from_center at suspected wall deaths
        self._radius_calibrated = False
        
        # Track previous length to detect food eating
        self.prev_length = 0
        
        # Track distance to nearest food for shaping reward
        self.prev_food_dist = None
        
        # Frame skip - repeat action for N frames
        self.frame_skip = frame_skip

        # Cached post-action data from previous step (avoids extra get_game_data call)
        self._cached_data = None

        # === CURRICULUM REWARD PARAMETERS (set by trainer via set_curriculum_stage) ===
        self.food_reward = 10.0       # Stage 1 default: high food reward
        self.food_shaping = 0.01      # Reward for moving towards food
        self.survival_reward = 0.05   # Per-step survival bonus
        self.death_wall_penalty = -15 # Increased default to prevent suicide eating
        self.death_snake_penalty = -15
        self.straight_penalty = 0.0   # Stage 1: no straight penalty
        self.length_bonus = 0.0       # Stage 3: per-step bonus for snake length

        # Escalating survival
        self.survival_escalation = 0.0
        self.steps_in_episode = 0

        # Threat awareness
        self.wall_alert_dist = 2000
        self.enemy_alert_dist = 800
        self.wall_proximity_penalty = 0.0
        self.enemy_proximity_penalty = 0.0

        # New reward components
        self.enemy_approach_penalty = 0.0
        self.boost_penalty = 0.0
        self.mass_loss_penalty = 0.0
        self.prev_enemy_dist = None

        # Starvation penalty: escalating penalty when bot hasn't eaten for too long
        self.starvation_penalty = 0.0        # per-step penalty rate (set by curriculum)
        self.starvation_grace_steps = 50     # steps before penalty kicks in
        self.starvation_max_penalty = 2.0    # cap per step
        self.steps_since_food = 0

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
        self.survival_escalation = stage_config.get('survival_escalation', 0.0)

        # Update alert distances if provided, otherwise keep defaults
        self.wall_alert_dist = stage_config.get('wall_alert_dist', 2000)
        self.enemy_alert_dist = stage_config.get('enemy_alert_dist', 800)
        self.wall_proximity_penalty = stage_config.get('wall_proximity_penalty', 0.0)
        self.enemy_proximity_penalty = stage_config.get('enemy_proximity_penalty', 0.0)

        # New reward components
        self.enemy_approach_penalty = stage_config.get('enemy_approach_penalty', 0.0)
        self.boost_penalty = stage_config.get('boost_penalty', 0.0)
        self.mass_loss_penalty = stage_config.get('mass_loss_penalty', 0.0)

        # Starvation penalty
        self.starvation_penalty = stage_config.get('starvation_penalty', 0.0)
        self.starvation_grace_steps = stage_config.get('starvation_grace_steps', 50)
        self.starvation_max_penalty = stage_config.get('starvation_max_penalty', 0.5)

        print(f"  ENV: food={self.food_reward} shaping={self.food_shaping} surv={self.survival_reward} "
              f"wall={self.death_wall_penalty} snake={self.death_snake_penalty} "
              f"enemy_approach={self.enemy_approach_penalty} boost_pen={self.boost_penalty} "
              f"starv_pen={self.starvation_penalty}")

    def _update_from_game_data(self, data):
        """Update wall distance and map info from game data."""
        if not data:
            return
        
        # We ignore JS dist_to_wall to fix false positives with polygon logic
        # and enforce Python radial logic.
        
        # Update map params if they look reasonable (supports eslither ~32k, standard ~21600)
        mr = data.get('map_radius')
        if mr and math.isfinite(mr) and 5000 < mr < 100000:
             self.map_radius = mr

        cx = data.get('map_center_x')
        cy = data.get('map_center_y')
        if cx and cy and math.isfinite(cx) and math.isfinite(cy) and cx > 1000:
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
        # Wider buffers to account for frame_skip=4 movement distance
        COLLISION_BUFFER = 120   # doubled for frame_skip=8
        WALL_BUFFER = 240       # doubled for frame_skip=8

        cause = "SnakeCollision"
        penalty = self.death_snake_penalty

        # Priority 1: Strictly outside map (Absolute Wall Death)
        if dist_to_wall_py < -50:
             cause = "Wall"
             penalty = self.death_wall_penalty

        # Priority 2: Near wall AND no close enemy → Wall death
        elif dist_to_wall_py <= (head_radius + WALL_BUFFER) and (min_enemy_dist == float('inf') or min_enemy_dist > 500):
             cause = "Wall"
             penalty = self.death_wall_penalty

        # Priority 3: High confidence Enemy Collision (close enemy)
        elif min_enemy_dist != float('inf') and min_enemy_dist <= (head_radius + COLLISION_BUFFER):
            cause = "SnakeCollision"
            penalty = self.death_snake_penalty

        # Priority 4: Near wall with nearby enemy — heuristic: if wall < 200 and enemy > 500, prefer Wall
        elif dist_to_wall_py < 200 and (min_enemy_dist == float('inf') or min_enemy_dist > 500):
             cause = "Wall"
             penalty = self.death_wall_penalty

        # Priority 5: Default → SnakeCollision
        else:
            cause = "SnakeCollision"
            penalty = self.death_snake_penalty

        min_enemy_display = min_enemy_dist if min_enemy_dist != float('inf') else -1
        enemy_detected = min_enemy_dist != float('inf')
        dist_from_center = math.hypot(mx - self.map_center_x, my - self.map_center_y)
        print(f"DEBUG DEATH: Pos=({mx:.0f},{my:.0f}) DistFromCenter={dist_from_center:.0f} MapRadius={self.map_radius:.0f} NearestEnemy={min_enemy_display:.0f} EnemyDetected={enemy_detected} WallDistPY={dist_to_wall_py:.0f} -> {cause} ({penalty})")

        # Auto-calibrate map radius from wall death positions
        # If no enemies nearby, death was likely a wall hit — use position as upper bound
        if not self._radius_calibrated and not enemy_detected and dist_from_center > 1000:
            self._wall_death_samples.append(dist_from_center)
            print(f"  [RADIUS CALIBRATION] Sample #{len(self._wall_death_samples)}: dist_from_center={dist_from_center:.0f}")

            if len(self._wall_death_samples) >= 3:
                # Use the minimum death distance as the real radius (with small buffer)
                estimated_radius = min(self._wall_death_samples) - 100
                if estimated_radius < self.map_radius * 0.8:  # Only apply if significantly smaller
                    old_radius = self.map_radius
                    self.map_radius = estimated_radius
                    self._radius_calibrated = True
                    print(f"  [RADIUS CALIBRATION] Map radius adjusted: {old_radius:.0f} -> {estimated_radius:.0f} "
                          f"(from {len(self._wall_death_samples)} samples: {[round(s) for s in self._wall_death_samples]})")

        return penalty, cause

    def reset(self):
        """Resets the game and returns initial matrix state."""
        self.browser.force_restart()
        self.near_wall_frames = 0
        self.invalid_frame_count = 0
        self.steps_in_episode = 0
        self.steps_since_food = 0
        # NAV debug separator
        with open("logs/nav_debug.log", "a") as _f:
            _f.write(f"\n{'='*140}\n  NEW EPISODE\n{'='*140}\n")
        self.prev_enemy_dist = None
        self._cached_data = None  # Clear cache on reset
        
        # One-time game variable scan
        if not self._map_vars_printed:
            self._map_vars_printed = True  # Set BEFORE scan to prevent re-runs
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
            self.prev_length = data['self'].get('len', 10)
        else:
            self.prev_length = 10
            
        matrix = self._process_data_to_matrix(data)
        sectors = self._compute_sectors(data)
        obs = {'matrix': matrix, 'sectors': sectors}
        self.last_matrix = obs
        self.last_valid_data = data if self._is_valid_frame(data) else None

        # Update overlay with initial state (include gsc and view_radius for debugging)
        if self.view_plus and data:
            gsc = data.get('gsc', 0)
            view_r = data.get('view_radius', 0)
            debug_info = data.get('debug', {})
            self.browser.update_view_plus_overlay(matrix, gsc=gsc, view_radius=view_r, debug_info=debug_info)

        return obs

    def step(self, action):
        """
        Executes action and returns (next_state, reward, done, info).
        Actions (10):
        0: Keep current direction
        1: Turn Left gentle  (~20 deg)  - fine correction
        2: Turn Right gentle (~20 deg)  - fine correction
        3: Turn Left medium  (~40 deg)  - standard evasion
        4: Turn Right medium (~40 deg)  - standard evasion
        5: Turn Left sharp   (~69 deg)  - aggressive evasion
        6: Turn Right sharp  (~69 deg)  - aggressive evasion
        7: Turn Left u-turn  (~103 deg) - emergency escape
        8: Turn Right u-turn (~103 deg) - emergency escape
        9: Boost (speed up, loses mass)
        """
        angle_change = 0
        boost = 0

        if action == 0:   pass
        elif action == 1:  angle_change = -0.35  # ~20 deg
        elif action == 2:  angle_change =  0.35  # ~20 deg
        elif action == 3:  angle_change = -0.7   # ~40 deg
        elif action == 4:  angle_change =  0.7   # ~40 deg
        elif action == 5:  angle_change = -1.2   # ~69 deg
        elif action == 6:  angle_change =  1.2   # ~69 deg
        elif action == 7:  angle_change = -1.8   # ~103 deg
        elif action == 8:  angle_change =  1.8   # ~103 deg
        elif action == 9:  boost = 1

        # Get current state before action (use cache from previous step if available)
        if self._cached_data is not None:
            data = self._cached_data
            self._cached_data = None
        else:
            data = self.browser.get_game_data()

        # Robust check (Validation Logic from tsrgy0)
        if not data:
            zeros = self._matrix_zeros()
            return zeros, -5, True, {"cause": "BrowserError"}

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

            zeros = self._matrix_zeros()
            return zeros, reward, True, {
                "cause": cause,
                "pos": (mx, my),
                "wall_dist": dtw,
                "enemy_dist": min_enemy_dist if min_enemy_dist != float('inf') else -1,
            }

        my_snake = data.get('self', {})
        current_ang = my_snake.get('ang', 0)
        mx, my = my_snake.get('x', 0), my_snake.get('y', 0)
        current_len = my_snake.get('len', 0)

        # Store pre-action position for movement vector analysis
        pre_x, pre_y = mx, my
        pre_ang = current_ang
        pre_wang = my_snake.get('wang', None)
        pre_eang = my_snake.get('eang', None)

        # Save pre-action data for death detection
        pre_action_data = data

        # Calculate distance to nearest food BEFORE action
        foods = data.get('foods', [])
        if foods:
            food_dists = [math.hypot(f[0] - mx, f[1] - my) for f in foods if len(f) >= 2]
            current_food_dist = min(food_dists) if food_dists else None
        else:
            current_food_dist = None

        # Execute action — relative to current heading (ang)
        target_ang = current_ang + angle_change
        # Normalize to [-pi, pi] to avoid accumulation
        while target_ang > math.pi: target_ang -= 2 * math.pi
        while target_ang < -math.pi: target_ang += 2 * math.pi
        if hasattr(self.browser, 'send_action_get_data'):
            # Combined: send action + read pre-wait state in ONE call
            self.browser.send_action_get_data(target_ang, boost)
        else:
            self.browser.send_action(target_ang, boost)
        if self.backend == "websocket":
            # CDP: send is fire-and-forget (~1ms), state reads from memory (instant)
            # Just wait for server to process action + push updates
            time.sleep(self.frame_skip * 0.008)  # ~32ms for frame_skip=4
        else:
            time.sleep(self.frame_skip * 0.010)  # ~40ms for frame_skip=4

        # Get new state after action
        data = self.browser.get_game_data()

        # Validate again
        if data and not data.get('dead') and not self._is_valid_frame(data):
             self.invalid_frame_count += 1
             if self.invalid_frame_count >= self.max_invalid_frames:
                 return self.last_matrix, -5, True, {"cause": "InvalidFrame"}
             return self.last_matrix, 0.0, False, {"cause": "InvalidFrame"}

        matrix = self._process_data_to_matrix(data)
        sectors = self._compute_sectors(data)
        state = {'matrix': matrix, 'sectors': sectors}
        self.last_matrix = state
        if self._is_valid_frame(data):
            self.last_valid_data = data
            self.invalid_frame_count = 0

        # Update view-plus overlay
        if self.view_plus and data:
            gsc = data.get('gsc', 0)
            view_r = data.get('view_radius', 0)
            debug_info = data.get('debug', {})
            self.browser.update_view_plus_overlay(matrix, gsc=gsc, view_radius=view_r, debug_info=debug_info)

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
        self.steps_in_episode += 1
        new_snake = data.get('self', {})
        new_len = new_snake.get('len', 0)
        new_x, new_y = new_snake.get('x', 0), new_snake.get('y', 0)

        # === MOVEMENT VECTOR ANALYSIS ===
        # Log every step for first 20 steps, then every 10
        if self.steps_in_episode <= 20 or self.steps_in_episode % 10 == 0:
            dx = new_x - pre_x
            dy = new_y - pre_y
            move_dist = math.hypot(dx, dy)
            # Actual movement direction (atan2 in slither convention: 0=East, Y-down)
            move_ang = math.atan2(dy, dx) if move_dist > 0.5 else float('nan')
            move_deg = math.degrees(move_ang) if not math.isnan(move_ang) else 0

            post_ang = new_snake.get('ang', 0)
            post_wang = new_snake.get('wang', None)
            post_eang = new_snake.get('eang', None)

            def _fmt_ang(v):
                return f"{math.degrees(v):+7.1f}°" if isinstance(v, (int, float)) else "     N/A"

            def _ang_diff_deg(a, b):
                if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                    return "    N/A"
                if math.isnan(a): return "    N/A"
                d = a - b
                while d > math.pi: d -= 2*math.pi
                while d < -math.pi: d += 2*math.pi
                return f"{math.degrees(d):+7.1f}°"

            act_names = ['FWD','L1','R1','L2','R2','L3','R3','LU','RU','BST']
            act_name = act_names[action] if action < len(act_names) else f'?{action}'
            target_ang_val = pre_ang + angle_change

            line = (
                f"step={self.steps_in_episode:3d} {act_name:3s} | "
                f"PRE({pre_x:6.0f},{pre_y:6.0f}) POST({new_x:6.0f},{new_y:6.0f}) | "
                f"dx={dx:+6.0f} dy={dy:+6.0f} dist={move_dist:5.0f} | "
                f"MOVE={move_deg:+7.1f}° | "
                f"pre_ang={_fmt_ang(pre_ang)} wang={_fmt_ang(pre_wang)} eang={_fmt_ang(pre_eang)} | "
                f"target={_fmt_ang(target_ang_val)} post_ang={_fmt_ang(post_ang)} post_wang={_fmt_ang(post_wang)} post_eang={_fmt_ang(post_eang)} | "
                f"ERR(move-ang)={_ang_diff_deg(move_ang, pre_ang)} ERR(move-eang)={_ang_diff_deg(move_ang, pre_eang)}"
            )
            with open("logs/nav_debug.log", "a") as _f:
                _f.write(line + "\n")

        # Update wall tracking from post-action data (Python radial logic)
        self._update_from_game_data(data)
        new_dist_to_wall = self.last_dist_to_wall
        min_enemy_dist = self._min_enemy_distance(data.get('enemies', []), new_x, new_y)

        # 1. Escalating survival reward (grows with episode length)
        escalation = self.survival_escalation * self.steps_in_episode
        survival_reward = self.survival_reward * (1.0 + escalation)
        reward += survival_reward
        
        # 2. Food reward (parametrized by curriculum stage)
        food_eaten = 0
        if new_len > self.prev_length:
            food_eaten = new_len - self.prev_length
            reward += food_eaten * self.food_reward
            self.steps_since_food = 0  # reset starvation counter
        else:
            self.steps_since_food += 1
            # Mass loss penalty: feel the pain of losing length (e.g. from boosting)
            mass_lost = self.prev_length - new_len
            if mass_lost > 0 and self.mass_loss_penalty > 0:
                reward -= mass_lost * self.mass_loss_penalty
        
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
            shaping_reward = max(-2.0, min(2.0, shaping_reward))
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

        # 7. Enemy approach penalty (penalize getting closer to enemies)
        if self.enemy_approach_penalty > 0 and min_enemy_dist != float('inf'):
            if self.prev_enemy_dist is not None and self.prev_enemy_dist != float('inf'):
                approach_delta = self.prev_enemy_dist - min_enemy_dist
                if approach_delta > 0 and min_enemy_dist < self.enemy_alert_dist:
                    reward -= self.enemy_approach_penalty * (approach_delta / max(self.enemy_alert_dist, 1))

        # 8. Boost penalty — scales with snake size (small snake = boost free, big = costly)
        if self.boost_penalty > 0 and action == 9:
            size_factor = max(0.0, (new_len - 30) / 70.0)  # 0 at len≤30, 1.0 at len=100
            reward -= self.boost_penalty * min(size_factor, 1.0)

        # 9. Starvation penalty: escalating penalty for not eating
        # Kicks in after grace period, grows linearly with steps without food
        if self.starvation_penalty > 0 and self.steps_since_food > self.starvation_grace_steps:
            hunger = self.steps_since_food - self.starvation_grace_steps
            penalty = min(self.starvation_penalty * hunger, self.starvation_max_penalty)
            reward -= penalty

        # Update tracked values
        self.prev_length = new_len
        self.prev_food_dist = new_food_dist
        self.prev_enemy_dist = min_enemy_dist

        # Cache post-action data for next step's pre-action read
        self._cached_data = data

        return state, reward, False, {
            "length": new_len,
            "food_eaten": food_eaten,
            "cause": None,
            "pos": (new_x, new_y),
            "wall_dist": new_dist_to_wall,
            "enemy_dist": min_enemy_dist if min_enemy_dist != float('inf') else -1,
            "server_id": data.get('server_id', ''),
        }

    def _get_state(self):
        data = self.browser.get_game_data()
        matrix = self._process_data_to_matrix(data)
        sectors = self._compute_sectors(data)
        return {'matrix': matrix, 'sectors': sectors}

    def _matrix_zeros(self):
        return {
            'matrix': np.zeros((3, self.matrix_size, self.matrix_size), dtype=np.float32),
            'sectors': np.zeros(99, dtype=np.float32),
        }

    def _compute_sectors(self, data):
        """
        Compute 75-float sector vector (egocentric).
        24 sectors × 15° covering 360°. Sector 0 = straight ahead.

        Layout (99 floats):
          [0..23]  food_score[i]      — closest food per sector, 1 - dist/scope
          [24..47] obstacle_score[i]  — closest enemy/wall per sector
          [48..71] obstacle_type[i]   — -1=none, 0=body/wall, 1=head
          [72..95] enemy_approach[i]  — +1=heading toward me, -1=away, 0=none
          [96]     wall_dist_norm     — dist_to_wall / scope
          [97]     snake_length_norm  — length / 500
          [98]     speed_norm         — speed / 20
        """
        NUM_SECTORS = 24
        SCOPE = 2000.0
        SECTOR_ANGLE = 2 * math.pi / NUM_SECTORS  # 15° in radians

        sectors = np.zeros(99, dtype=np.float32)
        # Initialize obstacle types to -1 (no obstacle)
        sectors[48:72] = -1.0

        if not data or data.get('dead'):
            return sectors

        my_snake = data.get('self')
        if not my_snake:
            return sectors

        mx = my_snake.get('x', 0)
        my_ = my_snake.get('y', 0)
        ang = my_snake.get('ang', 0)
        snake_len = my_snake.get('len', 0)
        spd = my_snake.get('sp', 0)

        sin_a = math.sin(ang)
        cos_a = math.cos(ang)

        def to_ego_angle_dist(dx, dy):
            """World-relative (dx,dy) -> egocentric (angle, dist).
            Returns angle in [0, 2*pi) where 0=ahead, clockwise."""
            # Egocentric rotation (same as _ego_raw)
            rx = -sin_a * dx + cos_a * dy
            ry = -cos_a * dx - sin_a * dy
            # In ego frame: ahead = -ry direction (up in matrix)
            # Convert to angle: 0=ahead(up), clockwise
            # atan2 with ahead=-y: angle = atan2(rx, -ry)
            angle = math.atan2(rx, -ry)
            if angle < 0:
                angle += 2 * math.pi
            dist = math.hypot(dx, dy)
            return angle, dist

        def sector_index(angle):
            idx = int(angle / SECTOR_ANGLE) % NUM_SECTORS
            return idx

        def score_distance(d):
            return max(0.0, 1.0 - d / SCOPE)

        # --- Food scores (weighted by size — death remains are worth more) ---
        foods = data.get('foods', [])
        for f in foods:
            if len(f) < 2:
                continue
            fx, fy = f[0], f[1]
            f_sz = f[2] if len(f) > 2 else 1.0  # food size (death remains: 10-20, normal: 1)
            dx, dy = fx - mx, fy - my_
            angle, dist = to_ego_angle_dist(dx, dy)
            if dist > SCOPE:
                continue
            si = sector_index(angle)
            # Weight by food size: big food (death remains) scores higher
            size_weight = min(f_sz / 2.0, 3.0)  # normal=0.5, remains≈3.0 → capped at 3.0
            sc = score_distance(dist) * size_weight
            if sc > sectors[si]:
                sectors[si] = sc

        # --- Obstacle scores (enemies) ---
        enemies = data.get('enemies', [])
        for e in enemies:
            ex, ey = e.get('x', 0), e.get('y', 0)
            e_sc = e.get('sc', 1.0)
            e_ang = e.get('ang', 0)
            half_width = e_sc * 29.0 * 0.5  # body half-width

            # Head
            dx, dy = ex - mx, ey - my_
            angle, dist = to_ego_angle_dist(dx, dy)
            effective_dist = max(0.0, dist - half_width)
            if effective_dist < SCOPE:
                si = sector_index(angle)
                sc = score_distance(effective_dist)
                if sc > sectors[24 + si]:
                    sectors[24 + si] = sc
                    sectors[48 + si] = 1.0  # head type

                    # Enemy approach: dot product of enemy velocity toward us
                    # Enemy heading vector (slither: ang=0 → East, Y-down)
                    e_vx = math.cos(e_ang)
                    e_vy = math.sin(e_ang)
                    # Vector from enemy to us
                    to_us_x, to_us_y = mx - ex, my_ - ey
                    to_us_len = max(dist, 1.0)
                    # Dot product: +1 = heading straight at us, -1 = away
                    approach = (e_vx * to_us_x + e_vy * to_us_y) / to_us_len
                    sectors[72 + si] = max(-1.0, min(1.0, approach))

            # Body points
            for pt in e.get('pts', []):
                if len(pt) < 2:
                    continue
                px, py = pt[0], pt[1]
                dx, dy = px - mx, py - my_
                angle, dist = to_ego_angle_dist(dx, dy)
                effective_dist = max(0.0, dist - half_width)
                if effective_dist < SCOPE:
                    si = sector_index(angle)
                    sc = score_distance(effective_dist)
                    if sc > sectors[24 + si]:
                        sectors[24 + si] = sc
                        sectors[48 + si] = 0.0  # body type

        # --- Wall per sector (ray-circle intersection) ---
        # Map is circle centered at (map_center_x, map_center_y) with radius map_radius
        # Snake at (mx, my_). For each sector, cast ray and find intersection distance.
        for si in range(NUM_SECTORS):
            # Ray direction in ego frame: sector center angle
            ego_angle = si * SECTOR_ANGLE + SECTOR_ANGLE / 2
            # ego direction: ahead=up=-y, clockwise
            # rx = sin(ego_angle), ry = -cos(ego_angle)
            ray_rx = math.sin(ego_angle)
            ray_ry = -math.cos(ego_angle)

            # Convert ray direction from ego to world
            # Inverse rotation: dx = -sin(ang)*rx - cos(ang)*ry
            #                   dy = cos(ang)*rx - sin(ang)*ry
            ray_dx = -sin_a * ray_rx - cos_a * ray_ry
            ray_dy = cos_a * ray_rx - sin_a * ray_ry

            # Ray-circle intersection
            # Ray: P = (mx, my_) + t * (ray_dx, ray_dy)
            # Circle: |P - C|^2 = R^2
            # (mx + t*rdx - cx)^2 + (my_ + t*rdy - cy)^2 = R^2
            ocx = mx - self.map_center_x
            ocy = my_ - self.map_center_y
            a = ray_dx * ray_dx + ray_dy * ray_dy  # always 1 for unit vector but keep general
            b = 2.0 * (ocx * ray_dx + ocy * ray_dy)
            c = ocx * ocx + ocy * ocy - self.map_radius * self.map_radius

            discriminant = b * b - 4.0 * a * c
            if discriminant >= 0:
                sqrt_disc = math.sqrt(discriminant)
                t1 = (-b - sqrt_disc) / (2.0 * a)
                t2 = (-b + sqrt_disc) / (2.0 * a)
                # We want the positive t (forward along ray)
                # t2 is always the exit point; t1 may be behind us if we're inside circle
                wall_t = t2 if t2 > 0 else t1
                if wall_t > 0 and wall_t < SCOPE:
                    wall_sc = score_distance(wall_t)
                    if wall_sc > sectors[24 + si]:
                        sectors[24 + si] = wall_sc
                        sectors[48 + si] = 0.0  # wall = body type (solid obstacle)

        # --- Global features ---
        sectors[96] = min(1.0, self.last_dist_to_wall / SCOPE)
        sectors[97] = min(1.0, snake_len / 500.0)
        sectors[98] = min(1.0, spd / 20.0) if spd else 0.0

        return sectors

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
        ang = my_snake.get('ang', 0)

        # Egocentric rotation: rotate world so snake heading = matrix "up"
        sin_a = math.sin(ang)
        cos_a = math.cos(ang)

        def _ego(dx, dy):
            """World-relative (dx,dy) → egocentric grid coords."""
            rx = -sin_a * dx + cos_a * dy
            ry = -cos_a * dx - sin_a * dy
            return int(rx * self.scale + self.matrix_size / 2), int(ry * self.scale + self.matrix_size / 2)

        def _ego_raw(dx, dy):
            """World-relative (dx,dy) → egocentric (rx,ry) unscaled."""
            return -sin_a * dx + cos_a * dy, -cos_a * dx - sin_a * dy

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

            gx, gy = _ego(fx - mx, fy - my)
            if 0 <= gx < self.matrix_size and 0 <= gy < self.matrix_size:
                matrix[0, gy, gx] = 1.0

        # Highlighting Nearest Food (Compass/Focus)
        if nearest_food:
            nfx, nfy = nearest_food
            # Calculate egocentric grid coordinates even if off-screen
            rx, ry = _ego_raw(nfx - mx, nfy - my)
            dx = rx * self.scale
            dy = ry * self.scale
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

            # Relative position → egocentric rotation
            rx, ry = _ego_raw(ex - mx, ey - my)

            # Grid coordinates (egocentric: heading = up)
            hx = cx_grid + rx * self.scale
            hy = cy_grid + ry * self.scale

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
                rx_p, ry_p = _ego_raw(px_world - mx, py_world - my)

                px_grid = cx_grid + rx_p * self.scale
                py_grid = cy_grid + ry_p * self.scale

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
             bx, by = _ego(px - mx, py - my)
             self._draw_thick_line(matrix, 2, cx, cy, bx, by, my_width_matrix, 0.5)

        for i in range(len(my_pts) - 1):
            p1 = my_pts[i]
            p2 = my_pts[i+1]
            x1, y1 = _ego(p1[0] - mx, p1[1] - my)
            x2, y2 = _ego(p2[0] - mx, p2[1] - my)
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

             # Grid coords → egocentric relative coords
             rx_grid = (x_grid - (self.matrix_size / 2)) / self.scale
             ry_grid = (y_grid - (self.matrix_size / 2)) / self.scale

             # Inverse rotation: egocentric → world-relative
             dx_world = -sin_a * rx_grid - cos_a * ry_grid
             dy_world =  cos_a * rx_grid - sin_a * ry_grid

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

        # 5. Wall Compass (Channel 1 - DANGER)
        # Show direction to nearest wall even when wall is off-screen
        # Brightness proportional to proximity (brighter = closer)
        if dist_to_wall_py < self.wall_alert_dist:
            # Direction from snake to nearest wall point (outward from map center)
            wall_dx = mx - self.map_center_x
            wall_dy = my - self.map_center_y
            wall_mag = math.hypot(wall_dx, wall_dy)

            if wall_mag > 100:  # avoid div-by-zero near center
                wall_dx /= wall_mag
                wall_dy /= wall_mag

                # Convert world direction to egocentric
                rx, ry = _ego_raw(wall_dx * self.view_size, wall_dy * self.view_size)
                dx_s = rx * self.scale
                dy_s = ry * self.scale

                # Normalize to unit vector
                mag_s = math.hypot(dx_s, dy_s)
                if mag_s > 0:
                    ndx = dx_s / mag_s
                    ndy = dy_s / mag_s

                    # Project to matrix edge
                    half_size = self.matrix_size / 2 - 2
                    tx = half_size / abs(ndx) if abs(ndx) > 1e-6 else float('inf')
                    ty = half_size / abs(ndy) if abs(ndy) > 1e-6 else float('inf')
                    t = min(tx, ty)

                    ex = self.matrix_size / 2 + t * ndx
                    ey = self.matrix_size / 2 + t * ndy

                    # Brightness: 0.3 at alert_dist, 1.0 at wall
                    proximity = 1.0 - (dist_to_wall_py / max(self.wall_alert_dist, 1))
                    brightness = 0.3 + 0.7 * proximity
                    radius = 1.5 + 1.5 * proximity  # bigger when closer

                    self._draw_circle(matrix, 1, ex, ey, radius, brightness)

        return matrix

    def close(self):
        self.browser.close()
