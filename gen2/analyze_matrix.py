#!/usr/bin/env python3
"""
Slither.io MatrixBot Training Analyzer
Generates beautiful console reports and charts from training data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import sys
from datetime import datetime

# Colors for console output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

def colored(text, color):
    return f"{color}{text}{Colors.ENDC}"

def print_header(text):
    width = 60
    print()
    print(colored("╔" + "═" * (width-2) + "╗", Colors.CYAN))
    print(colored("║" + text.center(width-2) + "║", Colors.CYAN + Colors.BOLD))
    print(colored("╚" + "═" * (width-2) + "╝", Colors.CYAN))

def print_section(text):
    print()
    print(colored(f"┌─ {text} " + "─" * (55 - len(text)), Colors.BLUE))

def print_stat(label, value, color=Colors.ENDC):
    print(f"│ {label:<25} {colored(str(value), color)}")

def print_table_row(cols, widths, color=Colors.ENDC):
    row = "│"
    for col, width in zip(cols, widths):
        row += f" {str(col):<{width}} │"
    print(colored(row, color))

def analyze():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'matrix_stats.csv')
    
    if not os.path.exists(csv_path):
        print(colored("❌ No stats file found!", Colors.RED))
        print(f"   Expected: {csv_path}")
        print("   Run 'python gen2/trainer.py' first.")
        return
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(colored(f"❌ Error reading CSV: {e}", Colors.RED))
        return
    
    if len(df) < 2:
        print(colored("⚠ Not enough data yet (need at least 2 episodes)", Colors.YELLOW))
        return
    
    # === Calculate metrics ===
    df['SMA10'] = df['Reward'].rolling(window=10, min_periods=1).mean()
    df['SMA50'] = df['Reward'].rolling(window=50, min_periods=1).mean()
    df['SMA100'] = df['Reward'].rolling(window=100, min_periods=1).mean()
    
    # === CONSOLE REPORT ===
    print_header("🐍 SLITHER.IO MATRIXBOT TRAINING REPORT")
    
    # General Stats
    print_section("📊 GENERAL STATISTICS")
    print_stat("Total Episodes:", f"{len(df):,}")
    print_stat("Total Steps:", f"{df['Steps'].sum():,}")
    print_stat("Current Epsilon:", f"{df['Epsilon'].iloc[-1]:.4f}")
    
    if 'MemorySize' in df.columns:
        print_stat("Memory Size:", f"{df['MemorySize'].iloc[-1]:,}")
    if 'LearningRate' in df.columns:
        print_stat("Learning Rate:", f"{df['LearningRate'].iloc[-1]:.6f}")
    
    # Reward Stats
    print_section("🏆 REWARD STATISTICS")
    print_stat("Best Reward:", f"{df['Reward'].max():.2f}", Colors.GREEN)
    print_stat("Worst Reward:", f"{df['Reward'].min():.2f}", Colors.RED)
    print_stat("Average Reward:", f"{df['Reward'].mean():.2f}")
    print_stat("Std Deviation:", f"{df['Reward'].std():.2f}")
    print_stat("Last 10 Avg:", f"{df['Reward'].tail(10).mean():.2f}", Colors.CYAN)
    print_stat("Last 50 Avg:", f"{df['Reward'].tail(50).mean():.2f}", Colors.CYAN)
    print_stat("Last 100 Avg:", f"{df['Reward'].tail(100).mean():.2f}", Colors.CYAN)
    
    # Steps Stats
    print_section("👣 STEPS PER EPISODE")
    print_stat("Average Steps:", f"{df['Steps'].mean():.1f}")
    print_stat("Max Steps:", f"{df['Steps'].max()}")
    print_stat("Min Steps:", f"{df['Steps'].min()}")
    print_stat("Last 10 Avg Steps:", f"{df['Steps'].tail(10).mean():.1f}")
    
    # Death Causes
    if 'Cause' in df.columns:
        print_section("💀 DEATH CAUSES")
        cause_counts = df['Cause'].value_counts()
        total = len(df)
        for cause, count in cause_counts.items():
            pct = count / total * 100
            bar_len = int(pct / 5)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            color = Colors.RED if cause == "Wall" else Colors.YELLOW if cause == "SnakeCollision" else Colors.DIM
            print_stat(f"{cause}:", f"{count:>5} ({pct:>5.1f}%) {bar}", color)
    
    # Progress Analysis
    print_section("📈 LEARNING PROGRESS")
    
    if len(df) >= 100:
        first_100_avg = df['Reward'].head(100).mean()
        last_100_avg = df['Reward'].tail(100).mean()
        improvement = last_100_avg - first_100_avg
        
        print_stat("First 100 Avg:", f"{first_100_avg:.2f}")
        print_stat("Last 100 Avg:", f"{last_100_avg:.2f}")
        
        if improvement > 0:
            print_stat("Improvement:", f"+{improvement:.2f}", Colors.GREEN)
        else:
            print_stat("Change:", f"{improvement:.2f}", Colors.RED)
    
    # Trend (last 50 episodes)
    if len(df) >= 50:
        recent = df.tail(50)
        slope = np.polyfit(range(50), recent['Reward'].values, 1)[0]
        trend = "📈 Improving" if slope > 0.1 else "📉 Declining" if slope < -0.1 else "➡️ Stable"
        trend_color = Colors.GREEN if slope > 0.1 else Colors.RED if slope < -0.1 else Colors.YELLOW
        print_stat("Recent Trend:", trend, trend_color)
    
    # Epsilon Progress
    print_section("🎲 EXPLORATION (EPSILON)")
    eps_start = df['Epsilon'].iloc[0]
    eps_current = df['Epsilon'].iloc[-1]
    eps_progress = (1 - eps_current / eps_start) * 100
    print_stat("Starting Epsilon:", f"{eps_start:.4f}")
    print_stat("Current Epsilon:", f"{eps_current:.4f}")
    print_stat("Decay Progress:", f"{eps_progress:.1f}%")
    
    # Top 10 Episodes
    print_section("🥇 TOP 10 BEST EPISODES")
    top10 = df.nlargest(10, 'Reward')[['Episode', 'Steps', 'Reward']].reset_index(drop=True)
    
    widths = [10, 10, 12]
    print("├" + "─" * 12 + "┬" + "─" * 12 + "┬" + "─" * 14 + "┤")
    print_table_row(["Episode", "Steps", "Reward"], widths, Colors.BOLD)
    print("├" + "─" * 12 + "┼" + "─" * 12 + "┼" + "─" * 14 + "┤")
    for _, row in top10.iterrows():
        print_table_row([int(row['Episode']), int(row['Steps']), f"{row['Reward']:.2f}"], widths, Colors.GREEN)
    print("└" + "─" * 12 + "┴" + "─" * 12 + "┴" + "─" * 14 + "┘")
    
    # === MOVING AVERAGE WITH STANDARD DEVIATION (Console) ===
    print_section("📉 MOVING AVERAGE ± STD DEV (last 200 episodes)")
    
    if len(df) >= 50:
        # Divide into chunks for text display
        window = 50
        n_points = min(10, len(df) // window)
        
        print("│")
        print("│  Episode Range      │   Avg Reward   │   Std Dev   │  Range (±1σ)")
        print("├" + "─" * 22 + "┼" + "─" * 16 + "┼" + "─" * 13 + "┼" + "─" * 20)
        
        for i in range(n_points):
            start_idx = len(df) - (n_points - i) * window
            end_idx = start_idx + window
            if start_idx < 0:
                continue
            chunk = df.iloc[start_idx:end_idx]
            avg = chunk['Reward'].mean()
            std = chunk['Reward'].std()
            ep_start = int(chunk['Episode'].iloc[0])
            ep_end = int(chunk['Episode'].iloc[-1])
            
            # Color based on average
            if avg > 0:
                color = Colors.GREEN
            elif avg > -20:
                color = Colors.YELLOW
            else:
                color = Colors.RED
            
            range_str = f"[{avg-std:>7.1f} to {avg+std:>7.1f}]"
            print(colored(f"│  {ep_start:>5} - {ep_end:<5}     │  {avg:>10.2f}    │  {std:>8.2f}   │  {range_str}", color))
        
        print("└" + "─" * 22 + "┴" + "─" * 16 + "┴" + "─" * 13 + "┴" + "─" * 20)
    else:
        print("│  (Need at least 50 episodes for this analysis)")
    
    # === REWARD HISTOGRAM (Console ASCII) ===
    print_section("📊 REWARD DISTRIBUTION (ASCII Histogram)")
    
    rewards = df['Reward'].values
    
    # Create bins
    n_bins = 15
    hist_min = df['Reward'].min()
    hist_max = df['Reward'].max()
    bin_width = (hist_max - hist_min) / n_bins
    
    # Compute histogram
    hist_counts = []
    bin_edges = []
    for i in range(n_bins):
        bin_start = hist_min + i * bin_width
        bin_end = bin_start + bin_width
        bin_edges.append((bin_start, bin_end))
        count = ((rewards >= bin_start) & (rewards < bin_end)).sum()
        hist_counts.append(count)
    
    # Handle last bin edge
    hist_counts[-1] += (rewards == hist_max).sum()
    
    max_count = max(hist_counts) if hist_counts else 1
    bar_max_width = 35
    
    print("│")
    print("│     Reward Range     │ Count │ Distribution")
    print("├" + "─" * 22 + "┼" + "─" * 7 + "┼" + "─" * (bar_max_width + 2))
    
    for i, ((bin_start, bin_end), count) in enumerate(zip(bin_edges, hist_counts)):
        bar_len = int((count / max_count) * bar_max_width) if max_count > 0 else 0
        bar = "█" * bar_len
        
        # Color based on reward value
        mid = (bin_start + bin_end) / 2
        if mid > 0:
            color = Colors.GREEN
        elif mid > -50:
            color = Colors.YELLOW
        else:
            color = Colors.RED
        
        range_str = f"{bin_start:>7.1f} to {bin_end:>7.1f}"
        print(colored(f"│ {range_str} │ {count:>5} │ {bar}", color))
    
    print("└" + "─" * 22 + "┴" + "─" * 7 + "┴" + "─" * (bar_max_width + 2))
    
    # Percentiles
    print("│")
    print("│  " + colored("Percentiles:", Colors.BOLD))
    p10 = np.percentile(rewards, 10)
    p25 = np.percentile(rewards, 25)
    p50 = np.percentile(rewards, 50)
    p75 = np.percentile(rewards, 75)
    p90 = np.percentile(rewards, 90)
    print(f"│    10%: {p10:>8.2f}   25%: {p25:>8.2f}   50% (median): {p50:>8.2f}")
    print(f"│    75%: {p75:>8.2f}   90%: {p90:>8.2f}")
    print("│")
    
    # === SPARKLINE TREND (ASCII mini-chart) ===
    print_section("📈 REWARD TREND (Sparkline)")
    
    # Sample data for sparkline (take ~60 points across all episodes)
    n_points = min(60, len(df))
    step = max(1, len(df) // n_points)
    sampled = df['SMA50'].iloc[::step].values[-60:]  # Last 60 sampled points
    
    if len(sampled) > 5:
        # Normalize to 0-7 range for sparkline characters
        spark_chars = "▁▂▃▄▅▆▇█"
        min_val = sampled.min()
        max_val = sampled.max()
        
        if max_val > min_val:
            normalized = (sampled - min_val) / (max_val - min_val)
            sparkline = ""
            for val in normalized:
                idx = min(7, int(val * 8))
                sparkline += spark_chars[idx]
        else:
            sparkline = "▄" * len(sampled)
        
        print("│")
        print(f"│  Min: {min_val:>8.2f}  Max: {max_val:>8.2f}")
        print("│")
        print("│  " + colored(sparkline, Colors.CYAN))
        print("│  " + Colors.DIM + "└" + "─" * (len(sparkline) - 2) + "┘" + Colors.ENDC)
        print("│  " + Colors.DIM + f"oldest{' ' * (len(sparkline) - 12)}newest" + Colors.ENDC)
        print("│")
        
        # Trend arrow
        if len(sampled) >= 10:
            recent_trend = sampled[-10:].mean() - sampled[:10].mean()
            if recent_trend > 5:
                trend_icon = colored("⬆️  IMPROVING", Colors.GREEN)
            elif recent_trend < -5:
                trend_icon = colored("⬇️  DECLINING", Colors.RED)
            else:
                trend_icon = colored("➡️  STABLE", Colors.YELLOW)
            print(f"│  Overall Trend: {trend_icon}")
    
    print()
    print(colored("=" * 60, Colors.DIM))
    
    # === GENERATE CHARTS ===
    print()
    print(colored("📊 Generating charts...", Colors.CYAN))
    
    # Set style
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Slither.io MatrixBot Training Analysis', fontsize=16, fontweight='bold', color='white')
    
    # 1. Main Reward Plot (large)
    ax1 = plt.subplot(2, 2, 1)
    ax1.fill_between(df['Episode'], df['Reward'], alpha=0.3, color='gray', label='Reward')
    ax1.plot(df['Episode'], df['SMA10'], color='#00ff88', linewidth=1, label='SMA 10', alpha=0.7)
    ax1.plot(df['Episode'], df['SMA50'], color='#ff6600', linewidth=2, label='SMA 50')
    if len(df) >= 100:
        ax1.plot(df['Episode'], df['SMA100'], color='#ff0066', linewidth=2, label='SMA 100')
    ax1.axhline(y=0, color='white', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Episode', color='white')
    ax1.set_ylabel('Reward', color='white')
    ax1.set_title('Reward Over Time', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.2)
    
    # 2. Steps per Episode
    ax2 = plt.subplot(2, 2, 2)
    steps_sma = df['Steps'].rolling(window=20, min_periods=1).mean()
    ax2.fill_between(df['Episode'], df['Steps'], alpha=0.3, color='#4488ff')
    ax2.plot(df['Episode'], steps_sma, color='#00ccff', linewidth=2, label='SMA 20')
    ax2.set_xlabel('Episode', color='white')
    ax2.set_ylabel('Steps', color='white')
    ax2.set_title('Steps per Episode (Survival Time)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.2)
    
    # 3. Death Causes Pie Chart
    ax3 = plt.subplot(2, 2, 3)
    if 'Cause' in df.columns:
        cause_counts = df['Cause'].value_counts()
        colors_pie = ['#ff4444', '#ffaa00', '#4488ff', '#44ff44', '#ff44ff']
        explode = [0.05] * len(cause_counts)
        wedges, texts, autotexts = ax3.pie(
            cause_counts.values, 
            labels=cause_counts.index,
            autopct='%1.1f%%',
            colors=colors_pie[:len(cause_counts)],
            explode=explode,
            shadow=True,
            startangle=90
        )
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax3.set_title('Death Causes', fontsize=12, fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No cause data available', ha='center', va='center', color='gray')
        ax3.set_title('Death Causes (No Data)', fontsize=12)
    
    # 4. Epsilon Decay + Learning Progress
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(df['Episode'], df['Epsilon'], color='#ff6600', linewidth=2, label='Epsilon')
    ax4.fill_between(df['Episode'], df['Epsilon'], alpha=0.3, color='#ff6600')
    ax4.set_xlabel('Episode', color='white')
    ax4.set_ylabel('Epsilon', color='#ff6600')
    ax4.tick_params(axis='y', labelcolor='#ff6600')
    ax4.set_title('Exploration (Epsilon) Decay', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.2)
    
    # If we have AvgReward50, add it as secondary axis
    if 'AvgReward50' in df.columns:
        ax4b = ax4.twinx()
        ax4b.plot(df['Episode'], df['AvgReward50'], color='#00ff88', linewidth=2, alpha=0.7, label='Avg Reward (50)')
        ax4b.set_ylabel('Avg Reward (50 ep)', color='#00ff88')
        ax4b.tick_params(axis='y', labelcolor='#00ff88')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save
    output_file = os.path.join(script_dir, 'training_plot.png')
    plt.savefig(output_file, dpi=150, facecolor='#1a1a2e', edgecolor='none')
    print(colored(f"✅ Chart saved: {output_file}", Colors.GREEN))
    
    # === Additional detailed chart ===
    if len(df) >= 20:  # Generate after 20 episodes
        fig2 = plt.figure(figsize=(14, 6))
        fig2.suptitle('Detailed Learning Progress', fontsize=14, fontweight='bold', color='white')
        
        # Adaptive window size based on data available
        window_size = min(50, max(5, len(df) // 3))
        sma_adaptive = df['Reward'].rolling(window=window_size, min_periods=1).mean()
        rolling_std = df['Reward'].rolling(window=window_size, min_periods=1).std()
        
        # Rolling statistics
        ax5 = plt.subplot(1, 2, 1)
        ax5.plot(df['Episode'], sma_adaptive, color='#00ff88', linewidth=2, label=f'Avg Reward ({window_size} ep)')
        
        # Add confidence band
        ax5.fill_between(df['Episode'], 
                         sma_adaptive - rolling_std, 
                         sma_adaptive + rolling_std, 
                         alpha=0.2, color='#00ff88', label='±1 Std Dev')
        ax5.axhline(y=0, color='white', linestyle='--', alpha=0.3)
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('Reward')
        ax5.set_title('Moving Average with Variance')
        ax5.legend(loc='upper left')
        ax5.grid(True, alpha=0.2)
        
        # Histogram of rewards - adaptive bins
        n_bins = min(50, max(10, len(df) // 2))
        ax6 = plt.subplot(1, 2, 2)
        ax6.hist(df['Reward'], bins=n_bins, color='#4488ff', alpha=0.7, edgecolor='white')
        ax6.axvline(df['Reward'].mean(), color='#ff6600', linestyle='--', linewidth=2, label=f'Mean: {df["Reward"].mean():.1f}')
        ax6.axvline(df['Reward'].median(), color='#00ff88', linestyle='--', linewidth=2, label=f'Median: {df["Reward"].median():.1f}')
        ax6.set_xlabel('Reward')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Reward Distribution')
        ax6.legend()
        ax6.grid(True, alpha=0.2)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        
        output_file2 = os.path.join(script_dir, 'training_detailed.png')
        plt.savefig(output_file2, dpi=150, facecolor='#1a1a2e', edgecolor='none')
        print(colored(f"✅ Detailed chart saved: {output_file2}", Colors.GREEN))
    
    # === CONSOLE PLOT (if plotext available) ===
    try:
        import plotext as pltx
        print()
        print(colored("📺 Console Preview:", Colors.CYAN))
        pltx.clear_figure()
        pltx.theme('dark')
        pltx.plot(df['Episode'].tolist(), df['SMA50'].tolist(), label='Avg Reward (50)')
        pltx.scatter(df['Episode'].tolist()[::10], df['Reward'].tolist()[::10], label='Reward (sampled)', marker='dot')
        pltx.title("Training Progress")
        pltx.xlabel("Episodes")
        pltx.ylabel("Reward")
        pltx.canvas_color("black")
        pltx.axes_color("black")
        pltx.ticks_color("white")
        pltx.plotsize(80, 20)
        pltx.show()
    except ImportError:
        print(colored("💡 Tip: Install 'plotext' for console charts: pip install plotext", Colors.DIM))
    
    print()
    print(colored("✨ Analysis complete!", Colors.GREEN + Colors.BOLD))

if __name__ == "__main__":
    analyze()
