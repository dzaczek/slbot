import math

class SpatialAwareness:
    """
    Processes raw game data into neural network inputs.
    Uses EGO-CENTRIC coordinate system (relative to snake's heading).
    
    Sector 0 = directly AHEAD
    Sector 6 = RIGHT
    Sector 12 = BEHIND  
    Sector 18 = LEFT
    """
    def __init__(self):
        self.num_sectors = 24  # 15 degrees per sector
        self.sector_angle = (2 * math.pi) / self.num_sectors
        self.map_radius = 21600  # Slither.io map radius
        self.view_distance = 1500  # How far the snake can "see"

    def calculate_sectors(self, my_snake, other_snakes, foods):
        """
        Generates 195 inputs for the neural network.
        
        All angles are EGO-CENTRIC (sector 0 = directly ahead, sector 12 = behind)
        
        FOOD (24 * 3 = 72 inputs):
        1. food_proximity: Closest food in sector (1.0 = very close)
        2. food_size: Size of closest food
        3. food_density: Total food "score" in sector
        
        ENEMY BODIES (24 * 2 = 48 inputs):
        4. body_proximity: Closest enemy body segment (DANGER - collision = death)
        5. body_density: How many body segments in sector (trap detection)
        
        ENEMY HEADS (24 * 3 = 72 inputs):
        6. head_proximity: Closest enemy head (HIGH DANGER)
        7. head_direction: Where the head is pointing (0=away, 0.5=sideways, 1=towards us)
        8. head_speed: How fast enemy is moving (normalized)
        
        GLOBAL (3 inputs):
        - Distance from map edge (0 = center, 1 = at edge)
        - Angle to center (to help avoid edges)
        - Wall danger (1 if close to wall in front)
        
        Total: 72 + 48 + 72 + 3 = 195 inputs
        """
        MAX_DIST = self.view_distance
        
        # Initialize all sector arrays
        # FOOD
        food_proximity = [0.0] * self.num_sectors
        food_size = [0.0] * self.num_sectors
        food_density = [0.0] * self.num_sectors
        
        # ENEMY BODIES (segments)
        body_proximity = [0.0] * self.num_sectors
        body_density = [0.0] * self.num_sectors
        
        # ENEMY HEADS
        head_proximity = [0.0] * self.num_sectors
        head_direction = [0.5] * self.num_sectors  # 0.5 = neutral/no head
        head_speed = [0.0] * self.num_sectors
        
        # RELATIVE SIZE (0.0 = I'm bigger or no enemy, 1.0 = enemy is bigger)
        enemy_relative_size = [0.0] * self.num_sectors
        
        mx, my = my_snake['x'], my_snake['y']
        my_angle = my_snake.get('ang', 0)
        my_speed = my_snake.get('sp', 0)
        my_len = my_snake.get('len', 0)  # Added missing definition
        
        # ============================================
        # 1. PROCESS FOOD
        # ============================================
        for f in foods:
            if len(f) < 2:
                continue
            
            fx, fy = f[0], f[1]
            dist = math.hypot(fx - mx, fy - my)
            
            if dist < MAX_DIST and dist > 1:
                # Ego-centric angle
                abs_angle = math.atan2(fy - my, fx - mx)
                rel_angle = self._normalize_angle(abs_angle - my_angle)
                sector = self._angle_to_sector(rel_angle)
                
                # Proximity (closer = higher value)
                proximity = 1.0 - (dist / MAX_DIST)
                
                # Size (normalized)
                size = (f[2] if len(f) >= 3 else 1) / 15.0
                size = min(size, 1.0)
                
                # Update closest food
                if proximity > food_proximity[sector]:
                    food_proximity[sector] = proximity
                    food_size[sector] = size
                
                # Accumulate density
                food_density[sector] += proximity * size
        
        # Normalize density
        max_density = max(food_density) if max(food_density) > 0 else 1.0
        food_density = [min(v / max(max_density, 1.0), 1.0) for v in food_density]
        
        # ============================================
        # 2. PROCESS ENEMY SNAKES
        # ============================================
        for snake in other_snakes:
            enemy_x = snake.get('x', 0)
            enemy_y = snake.get('y', 0)
            enemy_angle = snake.get('ang', 0)
            enemy_speed = snake.get('sp', 0)
            body_points = snake.get('pts', [])
            
            # -----------------------------------------
            # 2A. PROCESS ENEMY HEAD (most dangerous!)
            # -----------------------------------------
            head_dist = math.hypot(enemy_x - mx, enemy_y - my)
            
            if head_dist < MAX_DIST and head_dist > 1:
                abs_angle = math.atan2(enemy_y - my, enemy_x - mx)
                rel_angle = self._normalize_angle(abs_angle - my_angle)
                sector = self._angle_to_sector(rel_angle)
                
                proximity = 1.0 - (head_dist / MAX_DIST)
                
                # HEAD is very dangerous - boost proximity
                boosted_proximity = min(proximity * 1.3, 1.0)
                
                if boosted_proximity > head_proximity[sector]:
                    head_proximity[sector] = boosted_proximity
                    
                    # Calculate if enemy head is pointing towards us
                    # angle FROM enemy TO us
                    angle_enemy_to_us = math.atan2(my - enemy_y, mx - enemy_x)
                    # Difference between enemy's heading and direction to us
                    heading_diff = abs(self._normalize_angle(enemy_angle - angle_enemy_to_us))
                    # 0 = pointing directly at us, PI = pointing away
                    # Convert to 0-1 where 1 = coming at us
                    direction_threat = 1.0 - (heading_diff / math.pi)
                    head_direction[sector] = direction_threat
                    
                    # Normalize speed (typical max ~20)
                    head_speed[sector] = min(enemy_speed / 20.0, 1.0)
                    
                    # RELATIVE SIZE CALCULATION
                    enemy_len = len(snake.get('pts', []))
                    if enemy_len > my_len:
                        ratio = min(enemy_len / max(my_len, 1), 2.0)
                        size_danger = (ratio - 1.0) 
                        enemy_relative_size[sector] = min(max(size_danger, 0.0), 1.0)
                    else:
                        enemy_relative_size[sector] = 0.0 

                # ==========================================================
                # PREDICTIVE CUTOFF DETECTION (Anti-Kamikaze Logic)
                # ==========================================================
                # Even if enemy is small, if they boost in front of us -> DEATH.
                # We project their position into the future based on speed.
                
                # Predict where enemy head will be in 10-15 frames (~0.5s)
                # Speed factor: Boosting snakes move faster
                pred_frames = 12
                pred_x = enemy_x + math.cos(enemy_angle) * (enemy_speed * pred_frames)
                pred_y = enemy_y + math.sin(enemy_angle) * (enemy_speed * pred_frames)
                
                # Check distance of this FUTURE position to our CURRENT head
                pred_dist = math.hypot(pred_x - mx, pred_y - my)
                
                # If their future position is very close to our head (cutoff attempt)
                if pred_dist < 300: # Danger zone
                    # Calculate angle to this danger zone
                    pred_abs_angle = math.atan2(pred_y - my, pred_x - mx)
                    pred_rel_angle = self._normalize_angle(pred_abs_angle - my_angle)
                    pred_sector = self._angle_to_sector(pred_rel_angle)
                    
                    # Treat this predicted spot as a SOLID BODY/WALL
                    # This overrides "size" advantage. Collision is collision.
                    cutoff_danger = 1.0 - (pred_dist / 600)
                    cutoff_danger = min(cutoff_danger * 2.0, 1.0) # Panic mode!
                    
                    if cutoff_danger > body_proximity[pred_sector]:
                        body_proximity[pred_sector] = cutoff_danger
                        # Also flag as "head danger" to discourage biting
                        head_proximity[pred_sector] = max(head_proximity[pred_sector], cutoff_danger)

            # -----------------------------------------
            # 2B. PROCESS ENEMY BODY SEGMENTS
            
            # -----------------------------------------
            # 2B. PROCESS ENEMY BODY SEGMENTS
            # -----------------------------------------
            for p in body_points:
                if len(p) < 2:
                    continue
                
                px, py = p[0], p[1]
                dist = math.hypot(px - mx, py - my)
                
                if dist < MAX_DIST and dist > 1:
                    abs_angle = math.atan2(py - my, px - mx)
                    rel_angle = self._normalize_angle(abs_angle - my_angle)
                    sector = self._angle_to_sector(rel_angle)
                    
                    proximity = 1.0 - (dist / MAX_DIST)
                    
                    # Update closest body
                    if proximity > body_proximity[sector]:
                        body_proximity[sector] = proximity
                    
                    # Count body density (for trap detection)
                    body_density[sector] += 0.1  # Each segment adds to density
        
        # Normalize body density
        body_density = [min(v, 1.0) for v in body_density]
        
        # ============================================
        # 3. PROCESS WALLS (CRITICAL - IMPROVED!)
        # ============================================
        center_x, center_y = self.map_radius, self.map_radius
        dist_from_center = math.hypot(mx - center_x, my - center_y)
        dist_to_wall = self.map_radius - dist_from_center
        
        # Check wall danger in EACH sector independently
        for i in range(self.num_sectors):
            # Calculate the direction this sector points to
            sector_angle = i * self.sector_angle
            sector_abs_angle = my_angle + sector_angle # Sector 0 is ahead (relative angle 0)
            
            # Check distance to wall in this direction
            test_x = mx + MAX_DIST * math.cos(sector_abs_angle)
            test_y = my + MAX_DIST * math.sin(sector_abs_angle)
            test_dist_from_center = math.hypot(test_x - center_x, test_y - center_y)
            
            # If test point would be beyond wall
            if test_dist_from_center > self.map_radius:
                # Calculate actual distance to wall in this direction
                # Use strict map radius (21600) instead of 0.95 buffer to fix sensor blindness
                actual_dist_to_wall = self.map_radius - dist_from_center
                
                if actual_dist_to_wall < MAX_DIST:
                    wall_danger = 1.0 - (actual_dist_to_wall / MAX_DIST)
                    wall_danger = max(wall_danger, 0.0)
                    wall_danger = min(wall_danger * 2.0, 1.0)  # Higher boost for wall danger
                    
                    # Set as body danger (collision = death)
                    if wall_danger > body_proximity[i]:
                        body_proximity[i] = wall_danger
        
        # ============================================
        # 4. GLOBAL INPUTS
        # ============================================
        # Distance from edge (0 = center, 1 = at edge)
        edge_proximity = dist_from_center / self.map_radius
        
        # Angle to center (ego-centric) - helps navigate away from edges
        angle_to_center = math.atan2(center_y - my, center_x - mx)
        rel_angle_to_center = self._normalize_angle(angle_to_center - my_angle)
        norm_center_angle = (rel_angle_to_center + math.pi) / (2 * math.pi)
        
        # Is wall directly ahead? (sector 0 danger)
        wall_ahead = body_proximity[0] if dist_to_wall < MAX_DIST else 0.0
        
        # ============================================
        # 5. BUILD FINAL INPUT VECTOR
        # ============================================
        final_inputs = (
            # Food (72)
            food_proximity +      # 24 - where is closest food
            food_size +           # 24 - how big is it  
            food_density +        # 24 - how much food total
            
            # Enemy bodies (48)
            body_proximity +      # 24 - where are body segments (DANGER)
            body_density +        # 24 - how dense (trap detection)
            
            # Enemy heads (72)
            head_proximity +      # 24 - where are enemy heads (HIGH DANGER)
            head_direction +      # 24 - are they coming towards us?
            head_speed +          # 24 - how fast are they moving
            
            # Relative Size (24) - NEW
            enemy_relative_size + # 24 - 0.0 if I'm bigger, 1.0 if they are bigger
            
            # Globals (3)
            [edge_proximity, norm_center_angle, wall_ahead]
        )
        
        return final_inputs
    
    def _normalize_angle(self, angle):
        """Normalize angle to -PI..PI"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def _angle_to_sector(self, rel_angle):
        """
        Convert relative angle to sector index.
        Sector 0 = directly ahead (angle ~0)
        Sector 12 = directly behind (angle ~PI or ~-PI)
        """
        # rel_angle is -PI to PI
        # We want: 0 rad (ahead) -> sector 0
        # PI/2 (right) -> sector 6
        # PI (behind) -> sector 12
        # -PI/2 (left) -> sector 18
        
        # Normalize to 0..2PI
        if rel_angle < 0:
            rel_angle += 2 * math.pi
        
        # Map to sector
        sector = int(rel_angle / self.sector_angle) % self.num_sectors
        return sector

    def get_best_food_direction(self, my_snake, foods):
        """
        Helper: Returns angle to best nearby food.
        """
        mx, my = my_snake['x'], my_snake['y']
        my_angle = my_snake.get('ang', 0)
        
        best_score = 0
        best_angle = my_angle
        
        for f in foods:
            if len(f) < 2:
                continue
            
            fx, fy = f[0], f[1]
            dist = math.hypot(fx - mx, fy - my)
            
            if dist < self.view_distance and dist > 1:
                size = f[2] if len(f) >= 3 else 1
                score = size / (dist + 1)
                
                if score > best_score:
                    best_score = score
                    best_angle = math.atan2(fy - my, fx - mx)
        
        return best_angle, best_score
    
    def get_nearest_threat(self, my_snake, other_snakes):
        """
        Helper: Returns info about nearest enemy head.
        """
        mx, my = my_snake['x'], my_snake['y']
        
        nearest_dist = float('inf')
        nearest_info = None
        
        for snake in other_snakes:
            ex, ey = snake.get('x', 0), snake.get('y', 0)
            dist = math.hypot(ex - mx, ey - my)
            
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_info = {
                    'dist': dist,
                    'x': ex,
                    'y': ey,
                    'angle': snake.get('ang', 0),
                    'speed': snake.get('sp', 0)
                }
        
        return nearest_info

    def detect_encirclement(self, my_pos, other_snakes):
        """
        Detects if enemy snake bodies form a trap around player.
        Returns value 0.0 to 1.0 indicating trap severity.
        """
        mx, my = my_pos
        angles = []
        
        for snake in other_snakes:
            for p in snake.get('pts', []):
                if len(p) < 2:
                    continue
                px, py = p[0], p[1]
                dist = math.hypot(px - mx, py - my)
                
                if dist < 600:  # Only nearby points
                    ang = math.atan2(py - my, px - mx)
                    angles.append(ang)
        
        if len(angles) < 8:
            return 0.0
            
        angles.sort()
        
        # Find largest gap
        max_gap = 0
        for i in range(len(angles)):
            curr = angles[i]
            next_a = angles[(i + 1) % len(angles)]
            
            diff = next_a - curr
            if diff < 0:
                diff += 2 * math.pi
            
            if diff > max_gap:
                max_gap = diff
        
        # Smaller gap = more surrounded
        # gap < PI/2 (90°) = very trapped
        # gap > PI (180°) = safe
        if max_gap < math.pi / 2:
            return 1.0  # Fully trapped
        elif max_gap < math.pi:
            return 1.0 - ((max_gap - math.pi/2) / (math.pi/2))
        else:
            return 0.0
