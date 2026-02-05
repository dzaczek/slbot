"""
Script to restart training from scratch with backups.
Usage: python restart_training.py
"""

import os
import sys
import glob
import time
import shutil
import subprocess
from datetime import datetime

def restart_training():
    print("=" * 42)
    print("  Slither.io Bot - Training Restart")
    print("=" * 42)
    print("")

    # Check for existing checkpoints
    checkpoints = glob.glob("neat-checkpoint-*")
    
    if checkpoints:
        print("Found existing checkpoints.")
        print("")
        print("Options:")
        print("  1) Continue from last checkpoint (keeps old learning)")
        print("  2) Start FRESH - backup old and restart (RECOMMENDED)")
        print("  3) Cancel")
        print("")
        
        choice = input("Choose option [1/2/3]: ").strip()
        
        if choice == '1':
            print("Continuing from existing checkpoint...")
            run_training()
            
        elif choice == '2':
            print("\nCreating backup...")
            
            # Create backup directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = f"backup_{timestamp}"
            os.makedirs(backup_dir, exist_ok=True)
            
            # Move checkpoints
            for cp in checkpoints:
                shutil.move(cp, os.path.join(backup_dir, cp))
            print(f"✓ Moved checkpoints to {backup_dir}/")
            
            # Move best genome
            if os.path.exists("best_genome.pkl"):
                shutil.move("best_genome.pkl", os.path.join(backup_dir, "best_genome.pkl"))
                print(f"✓ Moved best_genome.pkl to {backup_dir}/")
            
            # Copy stats
            if os.path.exists("training_stats.csv"):
                shutil.copy("training_stats.csv", os.path.join(backup_dir, "training_stats_old.csv"))
                print(f"✓ Copied training_stats.csv to {backup_dir}/")
                
            print("\nStarting FRESH training with improved parameters...")
            run_training()
            
        elif choice == '3':
            print("Cancelled.")
            sys.exit(0)
        else:
            print("Invalid option.")
            sys.exit(1)
            
    else:
        print("No existing checkpoints found. Starting fresh training...")
        run_training()

def run_training():
    try:
        # Run training_manager.py using the same python interpreter
        subprocess.run([sys.executable, "training_manager.py"], check=True)
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    except Exception as e:
        print(f"Error running training: {e}")

if __name__ == "__main__":
    restart_training()
