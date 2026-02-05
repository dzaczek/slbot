#!/bin/bash
# Script to restart training from scratch with backups

echo "=========================================="
echo "  Slither.io Bot - Training Restart"
echo "=========================================="
echo ""

# Check if there are checkpoints
if ls neat-checkpoint-* 1> /dev/null 2>&1; then
    echo "Found existing checkpoints."
    echo ""
    echo "Options:"
    echo "  1) Continue from last checkpoint (keeps old learning)"
    echo "  2) Start FRESH - backup old and restart (RECOMMENDED)"
    echo "  3) Cancel"
    echo ""
    read -p "Choose option [1/2/3]: " choice
    
    case $choice in
        1)
            echo "Continuing from existing checkpoint..."
            python training_manager.py
            ;;
        2)
            echo ""
            echo "Creating backup..."
            
            # Create backup directory with timestamp
            backup_dir="backup_$(date +%Y%m%d_%H%M%S)"
            mkdir -p "$backup_dir"
            
            # Move checkpoints
            if ls neat-checkpoint-* 1> /dev/null 2>&1; then
                mv neat-checkpoint-* "$backup_dir/"
                echo "✓ Moved checkpoints to $backup_dir/"
            fi
            
            # Move best genome
            if [ -f "best_genome.pkl" ]; then
                mv best_genome.pkl "$backup_dir/"
                echo "✓ Moved best_genome.pkl to $backup_dir/"
            fi
            
            # Copy (not move) stats for reference
            if [ -f "training_stats.csv" ]; then
                cp training_stats.csv "$backup_dir/training_stats_old.csv"
                echo "✓ Copied training_stats.csv to $backup_dir/"
            fi
            
            echo ""
            echo "Starting FRESH training with improved parameters..."
            python training_manager.py
            ;;
        3)
            echo "Cancelled."
            exit 0
            ;;
        *)
            echo "Invalid option."
            exit 1
            ;;
    esac
else
    echo "No existing checkpoints found. Starting fresh training..."
    python training_manager.py
fi
