#!/usr/bin/env python3
"""
Diagnostic tool to check what the bot actually "sees" in the game.
Run this to verify that game data extraction works correctly.
"""

import time
import math
from browser_engine import SlitherBrowser
from spatial_awareness import SpatialAwareness

def log(msg):
    print(msg, flush=True)

def _sector_to_direction(sector):
    """Convert sector index to human-readable direction."""
    directions = {
        0: "AHEAD",
        1: "AHEAD-R",
        2: "AHEAD-R",
        3: "F-RIGHT",
        4: "F-RIGHT",
        5: "F-RIGHT",
        6: "RIGHT",
        7: "B-RIGHT",
        8: "B-RIGHT",
        9: "B-RIGHT",
        10: "BEHIND",
        11: "BEHIND",
        12: "BEHIND",
        13: "BEHIND",
        14: "B-LEFT",
        15: "B-LEFT",
        16: "B-LEFT",
        17: "B-LEFT",
        18: "LEFT",
        19: "F-LEFT",
        20: "F-LEFT",
        21: "F-LEFT",
        22: "AHEAD-L",
        23: "AHEAD-L",
    }
    return directions.get(sector, f"S{sector}")

def run_diagnostic():
    log("=" * 60)
    log("  SLITHER.IO BOT - VISION DIAGNOSTIC TEST")
    log("=" * 60)
    
    log("\n[1/4] Starting browser...")
    browser = SlitherBrowser(headless=False)
    
    log("[2/4] Waiting for game to start...")
    browser.force_restart()
    time.sleep(3)
    
    spatial = SpatialAwareness()
    
    log("[3/4] Running diagnostic loop (30 seconds)...")
    log("-" * 60)
    
    start_time = time.time()
    sample_count = 0
    
    while time.time() - start_time < 30:
        data = browser.get_game_data()
        sample_count += 1
        
        if data is None:
            log(f"\n[SAMPLE {sample_count}] ERROR: get_game_data returned None!")
            time.sleep(1)
            continue
            
        if data.get('dead', False):
            log(f"\n[SAMPLE {sample_count}] STATUS: Dead or in menu")
            if data.get('in_menu', False):
                log("  -> In menu, attempting restart...")
                browser.force_restart()
                time.sleep(3)
            continue
        
        my_snake = data.get('self')
        foods = data.get('foods', [])
        enemies = data.get('enemies', [])
        
        log(f"\n{'='*60}")
        log(f"[SAMPLE {sample_count}] Time: {time.time() - start_time:.1f}s")
        log(f"{'='*60}")
        
        # 1. My Snake Info
        if my_snake:
            log(f"\n[MY SNAKE]")
            log(f"  Position: ({my_snake.get('x', 'N/A'):.1f}, {my_snake.get('y', 'N/A'):.1f})")
            log(f"  Angle: {my_snake.get('ang', 'N/A'):.3f} rad ({math.degrees(my_snake.get('ang', 0)):.1f}°)")
            log(f"  Speed: {my_snake.get('sp', 'N/A')}")
            log(f"  Length: {my_snake.get('len', 'N/A')} segments")
            
            # Check map boundaries
            map_radius = 21600
            dist_from_center = math.hypot(my_snake['x'] - map_radius, my_snake['y'] - map_radius)
            dist_to_wall = map_radius - dist_from_center
            log(f"  Distance to wall: {dist_to_wall:.0f} units")
        else:
            log(f"\n[MY SNAKE] ERROR: No snake data!")
        
        # 2. Food Info
        log(f"\n[FOOD] Total visible: {len(foods)}")
        if foods:
            # Find closest food
            if my_snake:
                mx, my_y = my_snake['x'], my_snake['y']
                closest_dist = float('inf')
                closest_food = None
                
                for f in foods[:100]:  # Check first 100
                    if len(f) >= 2:
                        dist = math.hypot(f[0] - mx, f[1] - my_y)
                        if dist < closest_dist:
                            closest_dist = dist
                            closest_food = f
                
                if closest_food:
                    angle_to_food = math.atan2(closest_food[1] - my_y, closest_food[0] - mx)
                    log(f"  Closest food: dist={closest_dist:.0f}, angle={math.degrees(angle_to_food):.0f}°")
                    log(f"  Food position: ({closest_food[0]:.0f}, {closest_food[1]:.0f})")
                    if len(closest_food) >= 3:
                        log(f"  Food size: {closest_food[2]}")
            
            # Show sample of food data
            log(f"  Sample (first 3): {foods[:3]}")
        else:
            log("  WARNING: No food detected!")
        
        # 3. Enemy Info
        log(f"\n[ENEMIES] Total visible: {len(enemies)}")
        if enemies:
            for i, enemy in enumerate(enemies[:3]):  # Show first 3
                log(f"  Enemy {i+1}:")
                log(f"    Position: ({enemy.get('x', 'N/A'):.0f}, {enemy.get('y', 'N/A'):.0f})")
                log(f"    Angle: {enemy.get('ang', 'N/A'):.2f} rad")
                pts_count = len(enemy.get('pts', []))
                log(f"    Body segments: {pts_count}")
                
                if my_snake:
                    dist = math.hypot(enemy['x'] - mx, enemy['y'] - my_y)
                    log(f"    Distance to me: {dist:.0f}")
        else:
            log("  No enemies in view (could be normal if alone)")
        
        # 4. Test Spatial Awareness processing
        if my_snake:
            log(f"\n[SPATIAL AWARENESS TEST - EGO-CENTRIC]")
            log(f"  (Sector 0 = AHEAD, Sector 6 = RIGHT, Sector 12 = BEHIND, Sector 18 = LEFT)")
            try:
                inputs = spatial.calculate_sectors(my_snake, enemies, foods)
                log(f"  Input vector length: {len(inputs)} (expected: 195)")
                
                # Parse new format (195 inputs)
                idx = 0
                food_prox = inputs[idx:idx+24]; idx += 24
                food_size = inputs[idx:idx+24]; idx += 24
                food_dens = inputs[idx:idx+24]; idx += 24
                body_prox = inputs[idx:idx+24]; idx += 24
                body_dens = inputs[idx:idx+24]; idx += 24
                head_prox = inputs[idx:idx+24]; idx += 24
                head_dir = inputs[idx:idx+24]; idx += 24
                head_spd = inputs[idx:idx+24]; idx += 24
                globals_in = inputs[idx:idx+3]
                
                # === FOOD ===
                food_sectors = [(i, v) for i, v in enumerate(food_prox) if v > 0.01]
                log(f"\n  FOOD - Sectors with food: {len(food_sectors)}/24")
                if food_sectors:
                    top_food = sorted(food_sectors, key=lambda x: -x[1])[:5]
                    for sec, val in top_food:
                        log(f"    Sector {sec:2d} ({_sector_to_direction(sec):8s}): prox={val:.3f}, size={food_size[sec]:.2f}, density={food_dens[sec]:.3f}")
                else:
                    log(f"    No food detected in range!")
                
                # === ENEMY BODIES ===
                body_sectors = [(i, v) for i, v in enumerate(body_prox) if v > 0.01]
                log(f"\n  ENEMY BODIES - Sectors with bodies: {len(body_sectors)}/24")
                if body_sectors:
                    top_body = sorted(body_sectors, key=lambda x: -x[1])[:5]
                    for sec, val in top_body:
                        log(f"    Sector {sec:2d} ({_sector_to_direction(sec):8s}): prox={val:.3f}, density={body_dens[sec]:.2f}")
                
                # === ENEMY HEADS (most important!) ===
                head_sectors = [(i, v) for i, v in enumerate(head_prox) if v > 0.01]
                log(f"\n  ENEMY HEADS - Sectors with heads: {len(head_sectors)}/24")
                if head_sectors:
                    top_heads = sorted(head_sectors, key=lambda x: -x[1])[:5]
                    for sec, val in top_heads:
                        direction_str = "COMING AT US!" if head_dir[sec] > 0.7 else "sideways" if head_dir[sec] > 0.3 else "moving away"
                        log(f"    Sector {sec:2d} ({_sector_to_direction(sec):8s}): prox={val:.3f}, dir={head_dir[sec]:.2f} ({direction_str}), speed={head_spd[sec]:.2f}")
                else:
                    log(f"    No enemy heads in range")
                
                # === GLOBALS ===
                log(f"\n  GLOBALS:")
                log(f"    Edge proximity: {globals_in[0]:.3f} (0=center, 1=edge)")
                log(f"    Angle to center: {globals_in[1]:.3f}")
                log(f"    Wall ahead: {globals_in[2]:.3f}")
                
                # === THREAT ASSESSMENT ===
                nearest = spatial.get_nearest_threat(my_snake, enemies)
                if nearest:
                    log(f"\n  NEAREST THREAT:")
                    log(f"    Distance: {nearest['dist']:.0f}")
                    log(f"    Speed: {nearest['speed']:.1f}")
                
                # === ENCIRCLEMENT CHECK ===
                trap_level = spatial.detect_encirclement((mx, my_snake['y']), enemies)
                if trap_level > 0.3:
                    log(f"\n  ⚠️  ENCIRCLEMENT WARNING: {trap_level:.1%} surrounded!")
                
                # === RECOMMENDATION ===
                best_angle, best_score = spatial.get_best_food_direction(my_snake, foods)
                if best_score > 0:
                    rel_angle = best_angle - my_snake.get('ang', 0)
                    while rel_angle > math.pi: rel_angle -= 2*math.pi
                    while rel_angle < -math.pi: rel_angle += 2*math.pi
                    turn = "RIGHT" if rel_angle > 0 else "LEFT"
                    log(f"\n  RECOMMENDATION: Turn {abs(math.degrees(rel_angle)):.0f}° {turn} for best food")
                
            except Exception as e:
                log(f"  ERROR in spatial processing: {e}")
                import traceback
                traceback.print_exc()
        
        time.sleep(2)  # Sample every 2 seconds
    
    log("\n" + "=" * 60)
    log("[4/4] Diagnostic complete!")
    log("=" * 60)
    
    browser.close()

if __name__ == "__main__":
    try:
        run_diagnostic()
    except KeyboardInterrupt:
        log("\nDiagnostic interrupted by user.")
    except Exception as e:
        log(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
