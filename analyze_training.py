"""
Advanced Training Analysis Tool for Slither.io NEAT Bot
Shows statistics and ASCII charts in the terminal.

Usage:
    python analyze_training.py                  # Analyze training_stats.csv
    python analyze_training.py --live           # Live mode (updates every 10s)
    python analyze_training.py myfile.csv       # Analyze specific file
"""

import pandas as pd
import sys
import os
import time
import argparse

try:
    import plotext as plt
    HAS_PLOTEXT = True
except ImportError:
    HAS_PLOTEXT = False
    print("Warning: plotext not installed. Install with: pip install plotext")


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def analyze_training(csv_path='training_stats.csv', show_charts=True):
    """Analyze training data and display statistics with charts."""
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File '{csv_path}' not found!")
        return
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    if len(df) == 0:
        print("No data in CSV file yet.")
        return
    
    total_evals = len(df)
    
    print("=" * 70)
    print("  SLITHER.IO NEAT BOT - TRAINING ANALYSIS")
    print("=" * 70)
    print(f"\nFile: {csv_path}")
    print(f"Total evaluations: {total_evals}")
    
    # ============================================
    # OVERALL STATISTICS
    # ============================================
    print("\n" + "─" * 70)
    print("  OVERALL STATISTICS")
    print("─" * 70)
    
    # Convert columns to numeric, handling errors
    df['Fitness'] = pd.to_numeric(df['Fitness'], errors='coerce')
    df['SurvivalTime'] = pd.to_numeric(df['SurvivalTime'], errors='coerce')
    df['FoodEaten'] = pd.to_numeric(df['FoodEaten'], errors='coerce')
    df['MaxLen'] = pd.to_numeric(df['MaxLen'], errors='coerce')
    
    print(f"\n{'Metric':<25} {'Average':>12} {'Max':>12} {'Min':>12}")
    print("-" * 61)
    print(f"{'Fitness':<25} {df['Fitness'].mean():>12.1f} {df['Fitness'].max():>12.1f} {df['Fitness'].min():>12.1f}")
    print(f"{'Survival Time (s)':<25} {df['SurvivalTime'].mean():>12.1f} {df['SurvivalTime'].max():>12.1f} {df['SurvivalTime'].min():>12.1f}")
    print(f"{'Food Eaten':<25} {df['FoodEaten'].mean():>12.1f} {df['FoodEaten'].max():>12.0f} {df['FoodEaten'].min():>12.0f}")
    print(f"{'Max Length':<25} {df['MaxLen'].mean():>12.1f} {df['MaxLen'].max():>12.0f} {df['MaxLen'].min():>12.0f}")
    
    # ============================================
    # CAUSE OF DEATH
    # ============================================
    print("\n" + "─" * 70)
    print("  CAUSE OF DEATH")
    print("─" * 70 + "\n")
    
    death_counts = df['CauseOfDeath'].value_counts()
    for cause, count in death_counts.items():
        pct = (count / total_evals) * 100
        bar_len = int(pct / 2)
        bar = "█" * bar_len
        print(f"  {cause:<15} {count:>6} ({pct:>5.1f}%) {bar}")
    
    # ============================================
    # RECENT PERFORMANCE (Last 100)
    # ============================================
    if len(df) >= 50:
        print("\n" + "─" * 70)
        print("  RECENT PERFORMANCE (Last 100 vs First 100)")
        print("─" * 70)
        
        first_100 = df.head(100)
        last_100 = df.tail(100)
        
        print(f"\n{'Metric':<25} {'First 100':>12} {'Last 100':>12} {'Change':>12}")
        print("-" * 61)
        
        metrics = [
            ('Avg Fitness', 'Fitness'),
            ('Avg Food', 'FoodEaten'),
            ('Avg Length', 'MaxLen'),
            ('Avg Survival', 'SurvivalTime')
        ]
        
        for name, col in metrics:
            first_val = first_100[col].mean()
            last_val = last_100[col].mean()
            change = last_val - first_val
            change_pct = (change / max(first_val, 0.1)) * 100
            
            arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
            color_code = "" # Could add ANSI colors here
            
            print(f"{name:<25} {first_val:>12.1f} {last_val:>12.1f} {arrow} {change:>+10.1f}")
        
        # Food eating improvement
        first_ate = (first_100['FoodEaten'] > 0).sum()
        last_ate = (last_100['FoodEaten'] > 0).sum()
        print(f"\n  Bots that ate food: {first_ate}% → {last_ate}%")
    
    # ============================================
    # TOP PERFORMERS
    # ============================================
    print("\n" + "─" * 70)
    print("  TOP 10 PERFORMERS")
    print("─" * 70 + "\n")
    
    top10 = df.nlargest(10, 'Fitness')
    print(f"{'#':<3} {'GenomeID':<10} {'Fitness':>10} {'Time':>8} {'Food':>6} {'Len':>6} {'Death':<15}")
    print("-" * 65)
    for i, (idx, row) in enumerate(top10.iterrows(), 1):
        print(f"{i:<3} {row['GenomeID']:<10} {row['Fitness']:>10.1f} {row['SurvivalTime']:>7.1f}s {row['FoodEaten']:>6.0f} {row['MaxLen']:>6.0f} {row['CauseOfDeath']:<15}")
    
    # ============================================
    # CHARTS (if plotext available)
    # ============================================
    if show_charts and HAS_PLOTEXT and len(df) >= 20:
        
        # Group data into chunks for smoothing
        chunk_size = max(1, len(df) // 50)
        
        # Prepare data for charts
        fitness_chunks = []
        food_chunks = []
        length_chunks = []
        
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            fitness_chunks.append(chunk['Fitness'].mean())
            food_chunks.append(chunk['FoodEaten'].mean())
            length_chunks.append(chunk['MaxLen'].mean())
        
        x_values = list(range(len(fitness_chunks)))
        
        # ─────────────────────────────────────────
        # Chart 1: Fitness Progress
        # ─────────────────────────────────────────
        print("\n" + "─" * 70)
        print("  FITNESS PROGRESS (Rolling Average)")
        print("─" * 70)
        
        plt.clear_figure()
        plt.plot(x_values, fitness_chunks, marker="braille")
        plt.title("Fitness Over Time")
        plt.xlabel("Evaluation Batch")
        plt.ylabel("Avg Fitness")
        plt.theme("pro")
        plt.plot_size(70, 15)
        plt.show()
        
        # ─────────────────────────────────────────
        # Chart 2: Food Eating Progress
        # ─────────────────────────────────────────
        print("\n" + "─" * 70)
        print("  FOOD EATING PROGRESS")
        print("─" * 70)
        
        plt.clear_figure()
        plt.plot(x_values, food_chunks, marker="braille", color="green")
        plt.title("Average Food Eaten Over Time")
        plt.xlabel("Evaluation Batch")
        plt.ylabel("Avg Food")
        plt.theme("pro")
        plt.plot_size(70, 15)
        plt.show()
        
        # ─────────────────────────────────────────
        # Chart 3: Snake Length Progress
        # ─────────────────────────────────────────
        print("\n" + "─" * 70)
        print("  SNAKE LENGTH PROGRESS")
        print("─" * 70)
        
        plt.clear_figure()
        plt.plot(x_values, length_chunks, marker="braille", color="cyan")
        plt.title("Average Snake Length Over Time")
        plt.xlabel("Evaluation Batch")
        plt.ylabel("Avg Length")
        plt.theme("pro")
        plt.plot_size(70, 15)
        plt.show()
        
        # ─────────────────────────────────────────
        # Chart 4: Death Causes Pie Chart (Bar version)
        # ─────────────────────────────────────────
        print("\n" + "─" * 70)
        print("  DEATH CAUSES DISTRIBUTION")
        print("─" * 70)
        
        plt.clear_figure()
        causes = list(death_counts.index)
        counts = list(death_counts.values)
        plt.bar(causes, counts, orientation="horizontal")
        plt.title("Deaths by Cause")
        plt.theme("pro")
        plt.plot_size(70, 10)
        plt.show()
    
    elif not HAS_PLOTEXT:
        print("\n[Charts disabled - install plotext: pip install plotext]")
    
    # ============================================
    # LEARNING ASSESSMENT
    # ============================================
    print("\n" + "─" * 70)
    print("  LEARNING ASSESSMENT")
    print("─" * 70)
    
    if len(df) >= 200:
        first_half = df.head(len(df) // 2)
        second_half = df.tail(len(df) // 2)
        
        fitness_imp = second_half['Fitness'].mean() - first_half['Fitness'].mean()
        food_imp = second_half['FoodEaten'].mean() - first_half['FoodEaten'].mean()
        
        print("\n  Improvement (2nd half vs 1st half):")
        
        if fitness_imp > 10:
            print(f"  ✓ Fitness:  +{fitness_imp:.1f} (GOOD - Bot is learning!)")
        elif fitness_imp > 0:
            print(f"  ~ Fitness:  +{fitness_imp:.1f} (Slow progress)")
        else:
            print(f"  ✗ Fitness:  {fitness_imp:.1f} (NO IMPROVEMENT - Check parameters!)")
        
        if food_imp > 1:
            print(f"  ✓ Food:     +{food_imp:.1f} (GOOD - Learning to eat!)")
        elif food_imp > 0:
            print(f"  ~ Food:     +{food_imp:.1f} (Slow progress)")
        else:
            print(f"  ✗ Food:     {food_imp:.1f} (NOT LEARNING TO EAT!)")
        
        # Recommendations
        print("\n  Recommendations:")
        
        starvation_rate = (df['CauseOfDeath'] == 'Starvation').sum() / len(df) * 100
        wall_rate = (df['CauseOfDeath'] == 'Wall').sum() / len(df) * 100 if 'Wall' in df['CauseOfDeath'].values else 0
        
        if starvation_rate > 80:
            print("  ⚠ 80%+ starvation - Increase food reward or decrease starvation timeout")
        if wall_rate > 20:
            print("  ⚠ 20%+ wall deaths - Increase wall danger detection range")
        if df['FoodEaten'].mean() < 2:
            print("  ⚠ Very low food intake - Bot may need simpler initial task")
        if fitness_imp <= 0:
            print("  ⚠ No fitness improvement - Consider restarting with fresh population")
        
        if fitness_imp > 10 and food_imp > 1:
            print("  ✓ Bot is learning well! Continue training.")
    else:
        print("\n  Need at least 200 evaluations for learning assessment.")
    
    print("\n" + "=" * 70)
    print(f"  Analysis completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")


def live_mode(csv_path='training_stats.csv', interval=10):
    """Continuously update analysis every interval seconds."""
    print(f"Live mode enabled. Updating every {interval} seconds. Press Ctrl+C to stop.\n")
    
    last_count = 0
    
    try:
        while True:
            clear_screen()
            
            try:
                df = pd.read_csv(csv_path)
                current_count = len(df)
                
                if current_count > last_count:
                    new_evals = current_count - last_count
                    print(f"[LIVE] +{new_evals} new evaluations since last update\n")
                    last_count = current_count
                
            except:
                pass
            
            analyze_training(csv_path, show_charts=True)
            
            print(f"\n[Live mode - refreshing in {interval}s. Ctrl+C to stop]")
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nLive mode stopped.")


def main():
    parser = argparse.ArgumentParser(description='Analyze Slither.io NEAT Bot training')
    parser.add_argument('csv_file', nargs='?', default='training_stats.csv', 
                        help='CSV file to analyze (default: training_stats.csv)')
    parser.add_argument('--live', '-l', action='store_true',
                        help='Live mode - continuously update')
    parser.add_argument('--interval', '-i', type=int, default=10,
                        help='Live mode update interval in seconds (default: 10)')
    parser.add_argument('--no-charts', action='store_true',
                        help='Disable charts (text only)')
    
    args = parser.parse_args()
    
    if args.live:
        live_mode(args.csv_file, args.interval)
    else:
        analyze_training(args.csv_file, show_charts=not args.no_charts)


if __name__ == "__main__":
    main()
