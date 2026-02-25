
import os
import re
import time
from datetime import datetime, timedelta

# Directories
BACKUP_DIR = "backup_models"
EVENTS_DIR = "events"

def cleanup_checkpoints(keep_top=20):
    """Keep only models with high steps/food and the most recent ones."""
    if not os.path.exists(BACKUP_DIR):
        return
    
    files = [f for f in os.listdir(BACKUP_DIR) if f.endswith('.pth')]
    if not files:
        return

    # Regex to extract: ep (episode), s (steps), f (food)
    # Example: best_model_20260217-f6da4990_ep7321_s586_f179.pth
    pattern = re.compile(r'_ep(\d+)_s(\d+)_f(\d+)\.pth')
    
    model_data = []
    for f in files:
        full_path = os.path.join(BACKUP_DIR, f)
        m = pattern.search(f)
        if m:
            ep, steps, food = map(int, m.groups())
            # Scoring: steps are primary (survival), food is secondary
            score = steps * 10 + food 
            model_data.append({
                'name': f,
                'path': full_path,
                'ep': ep,
                'steps': steps,
                'food': food,
                'score': score,
                'mtime': os.path.getmtime(full_path)
            })
        else:
            # Files without proper pattern (keep them just in case if they are new)
            if time.time() - os.path.getmtime(full_path) > 86400: # older than 1 day
                # print(f"Removing unknown file: {f}")
                # os.remove(full_path)
                pass

    # 1. Keep TOP X by Score (Best performers)
    top_by_score = sorted(model_data, key=lambda x: x['score'], reverse=True)[:keep_top]
    
    # 2. Keep TOP X by Recency (Latest checkpoints)
    top_by_time = sorted(model_data, key=lambda x: x['mtime'], reverse=True)[:keep_top]
    
    # Combine sets of names to keep
    to_keep = {m['name'] for m in top_by_score} | {m['name'] for m in top_by_time}
    
    removed_count = 0
    for m in model_data:
        if m['name'] not in to_keep:
            try:
                os.remove(m['path'])
                removed_count += 1
            except Exception as e:
                print(f"Error removing {m['name']}: {e}")
                
    if removed_count > 0:
        print(f"[Cleanup] Removed {removed_count} old/low-performance checkpoints. Kept {len(to_keep)} best/recent models.")

def cleanup_events(days_to_keep=3):
    """Remove old event logs and screenshots."""
    if not os.path.exists(EVENTS_DIR):
        return
    
    files = os.listdir(EVENTS_DIR)
    cutoff = time.time() - (days_to_keep * 86400)
    
    removed_count = 0
    for f in files:
        full_path = os.path.join(EVENTS_DIR, f)
        if os.path.getmtime(full_path) < cutoff:
            try:
                os.remove(full_path)
                removed_count += 1
            except Exception as e:
                print(f"Error removing event {f}: {e}")
                
    if removed_count > 0:
        print(f"[Cleanup] Removed {removed_count} old event files (older than {days_to_keep} days).")

if __name__ == "__main__":
    print(f"--- Data Cleanup Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
    cleanup_checkpoints(keep_top=15)
    cleanup_events(days_to_keep=2)
    print("--- Cleanup Finished ---")
