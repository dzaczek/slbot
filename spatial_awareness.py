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
        Main function to generate 145 inputs.
        Returns flattened list of inputs.
        """
        # Initialize sectors [food_dist, enemy_dist, is_big, is_boost, is_trap, wall_dist]
        # Distances are inverted: 1.0 = close, 0.0 = far/none
        sectors = np.zeros((self.num_sectors, 6))
        
        mx, my = my_snake['x'], my_snake['y']
        mang = my_snake['ang']
        
        # 1. Process Foods
        for f in foods:
            if len(f) < 2: continue
            fx, fy = f[0], f[1]
            dist = math.hypot(fx - mx, fy - my)
            
            if dist < self.fov_range:
                angle = math.atan2(fy - my, fx - mx)
                # Adjust angle relative to snake's heading? 
                # Usually absolute sectors are easier for general navigation, 
                # but relative to head is better for "ego-centric" control.
                # Let's use RELATIVE to head direction for standard bot behavior.
                rel_angle = self._normalize_angle(angle - mang)
                
                sector_idx = int((rel_angle + math.pi) / (2 * math.pi) * self.num_sectors) % self.num_sectors
                
                # Input value: 1 - (dist / max_view), so closer = higher
                # Clamped to 0-1
                val = 1.0 - (dist / self.fov_range)
                
                # Keep max food value seen in this sector
                if val > sectors[sector_idx][0]:
                    sectors[sector_idx][0] = val

        # 2. Process Enemies
        for s in other_snakes:
            # Enemy head
            ex, ey = s['x'], s['y']
            dist = math.hypot(ex - mx, ey - my)
            
            # Check boost
            is_boosting = 1.0 if s.get('sp', 5) > 8 else 0.0 # Normal speed ~5.7
            
            # Check if big (heuristic: if simple length available, or based on 'pts')
            # Assuming 'pts' field exists and has length
            is_big = 1.0 if len(s.get('pts', [])) > 50 else 0.0 
            
            # Trap Detection
            # We pass the WHOLE body of this enemy to check if it surrounds us
            is_trap = self.detect_encirclement((mx, my), s.get('pts', []))

            # Populate sectors for the Head
            if dist < self.fov_range:
                angle = math.atan2(ey - my, ex - mx)
                rel_angle = self._normalize_angle(angle - mang)
                sector_idx = int((rel_angle + math.pi) / (2 * math.pi) * self.num_sectors) % self.num_sectors
                
                val = 1.0 - (dist / self.fov_range)
                if val > sectors[sector_idx][1]:
                    sectors[sector_idx][1] = val
                    sectors[sector_idx][2] = is_big
                    sectors[sector_idx][3] = is_boosting
                    
            # If trap detected, mark ALL sectors that contain this enemy's body as "TRAP"
            # Or just mark the general direction. 
            # Requirement says: "Assign TRAP_WARNING flag for that sector". 
            # Since a trap surrounds us, it likely affects multiple sectors.
            # We can iterate body points to fill sectors.
            if is_trap == 1.0 and dist < self.fov_range:
                 for p in s.get('pts', []):
                     px, py = p[0], p[1]
                     p_dist = math.hypot(px - mx, py - my)
                     if p_dist < self.fov_range:
                         p_angle = math.atan2(py - my, px - mx)
                         p_rel = self._normalize_angle(p_angle - mang)
                         p_sec = int((p_rel + math.pi) / (2 * math.pi) * self.num_sectors) % self.num_sectors
                         sectors[p_sec][4] = 1.0

        # 3. Process Walls (Border)
        # Distance from center (0,0)
        # Map is circular. Radius ~21600.
        dist_from_center = math.hypot(mx, my)
        dist_to_wall = self.map_radius - dist_from_center
        # We need direction to wall. The closest wall point is along the vector from center to us.
        # Vector Center->Me is (mx, my). Angle = atan2(my, mx).
        wall_angle = math.atan2(my, mx) # Angle pointing OUT towards nearest wall
        
        # We want "Distance to wall" in relevant sectors.
        # If we are looking AT the wall, the distance is short.
        # We can map the wall distance to the sectors that align with `wall_angle`.
        # However, purely "distance to wall" usually implies "how close am I to death by wall".
        # A simple approach: Calculating distance to wall for EACH sector ray.
        # Ray casting approach for walls:
        for i in range(self.num_sectors):
            # Angle of this sector (global)
            # sector_angle_rel = -PI + i * (2PI/24) + (PI/24) center
            # global = sector_angle_rel + snake_ang
            
            # Simple approximation: Just put the global wall distance into the sector pointing at the wall?
            # Better: Only warn if close.
             pass
        
        # Simplified Wall Logic for inputs:
        # Just calculate closest wall point and put it in the matching sector?
        # Or better: Provide distance to wall in the sector that points towards the wall.
        if dist_to_wall < self.fov_range:
             wall_rel_angle = self._normalize_angle(wall_angle - mang)
             wall_sec = int((wall_rel_angle + math.pi) / (2 * math.pi) * self.num_sectors) % self.num_sectors
             
             # Fill a few sectors around the direction
             # 1.0 = touching wall, 0.0 = safe
             w_val = 1.0 - (dist_to_wall / self.fov_range)
             sectors[wall_sec][5] = w_val
             # Helper: bleed into adjacent sectors slightly
             prev_sec = (wall_sec - 1) % self.num_sectors
             next_sec = (wall_sec + 1) % self.num_sectors
             sectors[prev_sec][5] = max(sectors[prev_sec][5], w_val * 0.8)
             sectors[next_sec][5] = max(sectors[next_sec][5], w_val * 0.8)

        # Flatten
        return sectors.flatten().tolist() + [1.0] # + Bias

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
