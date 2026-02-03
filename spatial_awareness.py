import math
import numpy as np

class SpatialAwareness:
    """
    Processes raw game data into network inputs.
    """
    def __init__(self):
        self.num_sectors = 24
        self.fov_range = 800  # Max distance to "see"
        self.sector_angle = (2 * math.pi) / self.num_sectors
        self.map_radius = 21600 # Approx slither.io radius

    def calculate_sectors(self, my_snake, other_snakes, foods):
        """
        Generates 99 inputs based on elliott-wen logic.
        Inputs: 24 sectors * 4 features + 3 global
        Features per sector:
        1. Food Value (1 - dist/1000)
        2. Food Size
        3. Perimeter/Body Value (1 - dist/1000)
        4. Enemy Angle
        Global:
        5. Self X, Self Y, Self Angle (normalized)
        """
        # Constants from elliott-wen
        MAX_SCOPE = 1000.0
        
        # Initialize sector arrays
        # 24 sectors
        food_val = [0.0] * self.num_sectors
        food_sz = [0.0] * self.num_sectors
        body_val = [0.0] * self.num_sectors
        enemy_ang = [-1.0] * self.num_sectors # Default -1
        
        mx, my = my_snake['x'], my_snake['y']
        
        # 1. Process Foods
        for f in foods:
            if len(f) < 2: continue
            fx, fy = f[0], f[1]
            dist = math.hypot(fx - mx, fy - my)
            
            if dist < MAX_SCOPE:
                # Absolute angle to food
                angle = math.atan2(fy - my, fx - mx)
                # Angle relative to map (0..2PI) for sector indexing?
                # Elliott-wen uses simple sector index based on angle relative to coordinate system
                # But typically bots use ego-centric.
                # Looking at app.py: getAngleIndex uses absolute difference but logic implies absolute sectors?
                # Actually app.py: np.arctan((pos[1] - snakepos[1])/(pos[0] - snakepos[0]))
                # It calculates angle from snake to object.
                # Let's use standard absolute angle sectors for consistency with that logic,
                # unless we want ego-centric.
                # Given the 'snakeangle' output is absolute 0-2PI, inputs should likely be absolute sectors.
                
                # Map angle -PI..PI to 0..24
                angle_idx = int((angle + math.pi) / (2 * math.pi) * self.num_sectors) % self.num_sectors
                
                val = 1.0 - (dist / MAX_SCOPE)
                sz_val = f[2] / 20.0 # Normalize size
                
                if val > food_val[angle_idx]:
                    food_val[angle_idx] = val
                    food_sz[angle_idx] = sz_val

        # 2. Process Enemies (Bodies for collision/perimeter)
        for s in other_snakes:
            # Check segments
             # We iterate existing points. 'pts' in our get_game_data is list of [x, y]
            points = s.get('pts', [])
            # Also include head
            points.append([s['x'], s['y']])
            
            for p in points:
                px, py = p[0], p[1]
                dist = math.hypot(px - mx, py - my)
                
                if dist < MAX_SCOPE:
                    angle = math.atan2(py - my, px - mx)
                    angle_idx = int((angle + math.pi) / (2 * math.pi) * self.num_sectors) % self.num_sectors
                    
                    val = 1.0 - (dist / MAX_SCOPE)
                    
                    if val > body_val[angle_idx]:
                        body_val[angle_idx] = val
                        # Store angle of this enemy's head if this is the head?
                        # Elliott-wen stores 'sn['ang']' if it's a head, else -1.
                        # We need to identifying if 'p' is head.
                        # Simplify: If it's a body part, we just care about collision.
                        # If we match the head, we store its angle.
                        if p[0] == s['x'] and p[1] == s['y']:
                             enemy_ang[angle_idx] = s['ang'] / (2 * math.pi) # Normalize 0-1
                        else:
                             enemy_ang[angle_idx] = -1.0

        # 3. Process Walls (Treat as body/obstacle)
        grd = self.map_radius
        dist_from_center = math.hypot(mx, my)
        dist_to_wall = grd - dist_from_center
        
        # If we are close to wall
        if dist_to_wall < MAX_SCOPE:
            # Wall direction is same as my position vector from center
            wall_angle = math.atan2(my, mx)
            angle_idx = int((wall_angle + math.pi) / (2 * math.pi) * self.num_sectors) % self.num_sectors
             
            # Wall creates a "wall" of danger. 
            # Simple approach: Block the forward sector towards wall.
            # Elliott-wen scans a semicircle for wall distance.
            # We will block the sector pointing to wall.
            val = 1.0 - (dist_to_wall / MAX_SCOPE)
            if val > body_val[angle_idx]:
                body_val[angle_idx] = val
                enemy_ang[angle_idx] = -1.0 # Wall has no angle
            
            # Spread to adjacent
            prev_sec = (angle_idx - 1) % self.num_sectors
            next_sec = (angle_idx + 1) % self.num_sectors
            body_val[prev_sec] = max(body_val[prev_sec], val * 0.8)
            body_val[next_sec] = max(body_val[next_sec], val * 0.8)

        # Global Inputs
        norm_x = mx / 45000.0
        norm_y = my / 45000.0
        norm_ang = my_snake['ang'] / (2 * math.pi)

        # Flatten inputs
        # Order: food_val, food_sz, body_val, enemy_ang, globals
        final_inputs = food_val + food_sz + body_val + enemy_ang + [norm_x, norm_y, norm_ang]
        
        return final_inputs

    def detect_encirclement(self, my_pos, enemy_pts):
        """
        Detects if existing snake points encircle the player.
        Logic: Calculate angles of all enemy points relative to player.
        Sort angles. Check largest gap.
        If largest gap < 90 deg (meaning coverage > 270), it is a trap.
        """
        if not enemy_pts or len(enemy_pts) < 10:
            return 0.0
            
        mx, my = my_pos
        angles = []
        for p in enemy_pts:
            # p provides [x, y]
            px, py = p[0], p[1]
            dist = math.hypot(px - mx, py - my)
            # Only consider points that are reasonably close to constitute a trap
            if dist < 600: 
                ang = math.atan2(py - my, px - mx)
                angles.append(ang)
        
        if len(angles) < 5:
            return 0.0
            
        # Sort angles -PI to PI
        angles.sort()
        
        # Check gaps
        max_gap = 0
        for i in range(len(angles)):
            # Current angle to next angle
            curr = angles[i]
            next_a = angles[(i + 1) % len(angles)]
            
            # Normalize diff
            diff = next_a - curr
            if diff < 0: # Wrap around case (PI to -PI)
                diff += 2 * math.pi
            
            if diff > max_gap:
                max_gap = diff
                
        # If the largest open gap is small (e.g. < 90 deg = PI/2), we are trapped > 270 deg
        if max_gap < (math.pi / 2): # 90 degrees
            return 1.0 # TRAP!
            
        return 0.0

    def _normalize_angle(self, angle):
        """ Normalize angle to -PI..PI """
        while angle > math.pi: angle -= 2 * math.pi
        while angle < -math.pi: angle += 2 * math.pi
        return angle
