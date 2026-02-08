#!/usr/bin/env python3
"""
Slither.io MatrixBot Training Analyzer
Generates beautiful console reports, Markdown analysis, and charts from training data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import sys
from datetime import datetime

# ==========================================
#  🎨 UI / UX CONSTANTS & HELPERS
# ==========================================

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

def get_trend_icon(value, baseline, rising_is_good=True):
    diff = value - baseline
    if diff == 0: return "➡️", Colors.DIM

    if rising_is_good:
        if diff > 0: return "↗️", Colors.GREEN
        else: return "↘️", Colors.RED
    else:
        if diff < 0: return "↘️", Colors.GREEN
        else: return "↗️", Colors.RED

def format_number(n):
    if abs(n) >= 1000: return f"{n:,.0f}"
    if abs(n) >= 10: return f"{n:.1f}"
    return f"{n:.4f}"

# ==========================================
#  📊 STATISTICS CALCULATION
# ==========================================

def calculate_statistics(df):
    """Calculates comprehensive statistics for a given DataFrame."""
    stats = {}
    
    # Numeric columns to analyze (Dynamic)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude non-metric columns if necessary (like Episode, though it's useful for range)
    # kept Episode in for now as it doesn't hurt, but let's prioritize metrics
    priority_cols = ['Reward', 'Steps', 'Epsilon', 'Loss', 'Beta', 'LR', 'Food', 'Stage']
    
    # Sort columns: priority first, then others alphabetically
    numeric_cols.sort(key=lambda x: (0 if x in priority_cols else 1, x))
    
    for col in numeric_cols:
        if col == 'Episode': continue # Skip Episode stats
        series = df[col]
        stats[col] = {
            'mean': series.mean(),
            'median': series.median(),
            'std': series.std(),
            'min': series.min(),
            'max': series.max(),
            'last': series.iloc[-1]
        }

    # Categorical: Cause
    if 'Cause' in df.columns:
        stats['Cause'] = df['Cause'].value_counts().to_dict()
        stats['Cause_Pct'] = (df['Cause'].value_counts(normalize=True) * 100).to_dict()

    stats['count'] = len(df)
    return stats

# ==========================================
#  📝 REPORT GENERATION
# ==========================================

def generate_markdown_report(df, full_stats, recent_stats, output_path):
    """Generates a detailed Markdown report."""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # Header
        f.write(f"# 🐍 Slither.io MatrixBot Training Report\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Total Episodes:** {full_stats['count']}  \n")
        f.write(f"**Recent Analysis Window:** Last {recent_stats['count']} episodes\n\n")
        
        # 1. Executive Summary
        f.write("## 📊 Executive Summary (Recent Performance)\n")
        f.write("| Metric | Average | Trend (vs All-Time) | Best | Worst |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        
        metrics_to_show = [
            ('Reward', 'Reward', True),
            ('Survival (Steps)', 'Steps', True),
            ('Food Eaten', 'Food', True),
            ('Loss', 'Loss', False)
        ]
        
        for label, key, rising_good in metrics_to_show:
            if key in recent_stats:
                curr = recent_stats[key]['mean']
                base = full_stats[key]['mean']
                diff = curr - base
                icon = "↗️" if (diff > 0 and rising_good) or (diff < 0 and not rising_good) else "↘️"
                if abs(diff) < 0.001: icon = "➡️"
                
                trend_str = f"{icon} {diff:+.2f}"
                best = recent_stats[key]['max']
                worst = recent_stats[key]['min']
                
                f.write(f"| **{label}** | {curr:.2f} | {trend_str} | {best:.2f} | {worst:.2f} |\n")
        f.write("\n")

        # 2. Detailed Statistics
        f.write("## 📈 Detailed Statistics\n")
        f.write("### Recent vs Full History\n")
        f.write("| Metric | Recent Mean | Recent Std Dev | All-Time Mean | All-Time Std Dev |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")

        for key in full_stats:
            if key in ['Cause', 'Cause_Pct', 'count']: continue
            rec = recent_stats.get(key, {})
            full = full_stats.get(key, {})
            f.write(f"| {key} | {rec.get('mean', 0):.4f} | {rec.get('std', 0):.4f} | {full.get('mean', 0):.4f} | {full.get('std', 0):.4f} |\n")
        f.write("\n")

        # 3. Death Analysis
        f.write("## 💀 Death Analysis\n")
        if 'Cause' in full_stats:
            f.write("| Cause | Recent Count | Recent % | All-Time % |\n")
            f.write("| :--- | :---: | :---: | :---: |\n")
            
            all_causes = set(full_stats['Cause'].keys()) | set(recent_stats.get('Cause', {}).keys())

            for cause in sorted(all_causes):
                rec_count = recent_stats.get('Cause', {}).get(cause, 0)
                rec_pct = recent_stats.get('Cause_Pct', {}).get(cause, 0)
                full_pct = full_stats.get('Cause_Pct', {}).get(cause, 0)
                f.write(f"| {cause} | {rec_count} | {rec_pct:.1f}% | {full_pct:.1f}% |\n")
        f.write("\n")

        # 4. Training Parameters
        f.write("## ⚙️ Training Health\n")
        f.write(f"- **Current Epsilon:** {full_stats['Epsilon']['last']:.4f}\n")
        f.write(f"- **Current Learning Rate:** {full_stats['LR']['last']:.6f}\n")
        if 'Beta' in full_stats:
            f.write(f"- **Current Beta (PER):** {full_stats['Beta']['last']:.4f}\n")
        f.write("\n")

        # 5. Visuals
        f.write("## 🖼️ Charts\n")
        f.write("![Training Plot](training_plot.png)\n")
        f.write("\n")
        f.write("![Detailed Analysis](training_detailed.png)\n")

    print(colored(f"✅ Markdown report generated: {output_path}", Colors.GREEN))

# ==========================================
#  📺 CONSOLE DASHBOARD
# ==========================================

def print_dashboard(df, full_stats, recent_stats):
    """Prints a 'Gustowny' (Tasteful) UI Dashboard to the console."""
    
    width = 70
    line = "─" * width
    thick_line = "═" * width
    
    print("\n" + colored(thick_line, Colors.CYAN))
    print(colored(f"🐍  SLITHER.IO MATRIXBOT ANALYTICS DASHBOARD  🐍".center(width), Colors.CYAN + Colors.BOLD))
    print(colored(thick_line, Colors.CYAN))
    
    # --- RECENT PERFORMANCE ---
    print(colored(f"\n[ RECENT PERFORMANCE (Last {recent_stats['count']} Episodes) ]", Colors.YELLOW + Colors.BOLD))
    print(colored(line, Colors.DIM))
    
    # Reward & Food Row
    r_mean = recent_stats['Reward']['mean']
    r_trend, r_color = get_trend_icon(r_mean, full_stats['Reward']['mean'], True)
    
    f_mean = recent_stats.get('Food', {}).get('mean', 0)
    f_trend, f_color = get_trend_icon(f_mean, full_stats.get('Food', {}).get('mean', 0), True)
    
    print(f" {colored('Reward:', Colors.BOLD):<15} {r_mean:>8.2f} {r_trend}  (All-Time: {full_stats['Reward']['mean']:.2f})")
    print(f" {colored('Food:', Colors.BOLD):<15} {f_mean:>8.2f} {f_trend}  (All-Time: {full_stats.get('Food', {}).get('mean', 0):.2f})")
    
    # Steps Row
    s_mean = recent_stats['Steps']['mean']
    s_trend, s_color = get_trend_icon(s_mean, full_stats['Steps']['mean'], True)
    print(f" {colored('Survival:', Colors.BOLD):<15} {s_mean:>8.1f}s {s_trend} (All-Time: {full_stats['Steps']['mean']:.1f}s)")

    # --- TRAINING HEALTH ---
    print(colored(f"\n[ TRAINING HEALTH ]", Colors.BLUE + Colors.BOLD))
    print(colored(line, Colors.DIM))
    
    # Loss & LR
    if 'Loss' in recent_stats:
        l_mean = recent_stats['Loss']['mean']
        l_trend, _ = get_trend_icon(l_mean, full_stats['Loss']['mean'], False) # Lower is good
        print(f" {colored('Avg Loss:', Colors.BOLD):<15} {l_mean:>8.4f} {l_trend}")
        
    print(f" {colored('Epsilon:', Colors.BOLD):<15} {full_stats['Epsilon']['last']:>8.4f} (Exploration)")
    print(f" {colored('Learn Rate:', Colors.BOLD):<15} {full_stats['LR']['last']:>8.6f}")
    
    if 'Beta' in full_stats:
        print(f" {colored('Beta (PER):', Colors.BOLD):<15} {full_stats['Beta']['last']:>8.4f}")

    # --- DEATH CAUSES ---
    if 'Cause' in recent_stats:
        print(colored(f"\n[ RECENT DEATH CAUSES ]", Colors.RED + Colors.BOLD))
        print(colored(line, Colors.DIM))
        
        causes = recent_stats['Cause']
        total = recent_stats['count']
        
        # Sort by count desc
        sorted_causes = sorted(causes.items(), key=lambda x: x[1], reverse=True)
        
        for cause, count in sorted_causes:
            pct = (count / total) * 100
            bar_len = int(pct / 4)
            bar = "█" * bar_len + "░" * (25 - bar_len)

            c_color = Colors.RED if cause == "Wall" else Colors.YELLOW if cause == "SnakeCollision" else Colors.DIM
            print(f" {cause:<15} {count:>3} ({pct:>5.1f}%) {colored(bar, c_color)}")

    print("\n" + colored(line, Colors.DIM))


# ==========================================
#  📉 PLOTTING
# ==========================================

def plot_charts(df, script_dir):
    """Generates comprehensive charts."""
    print(colored("📊 Generating charts...", Colors.CYAN))
    
    plt.style.use('dark_background')
    
    # --- CHART 1: OVERVIEW (Reward, Steps, Epsilon, Loss) ---
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Slither.io MatrixBot Training Overview', fontsize=16, fontweight='bold', color='white')

    # 1. Reward
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(df['Episode'], df['Reward'], alpha=0.3, color='gray', label='Raw')
    ax1.plot(df['Episode'], df['Reward'].rolling(50, min_periods=1).mean(), color='#00ff88', linewidth=2, label='SMA 50')
    ax1.set_title('Reward History')
    ax1.grid(True, alpha=0.2)
    ax1.legend()
    
    # 2. Steps (Survival)
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(df['Episode'], df['Steps'].rolling(20, min_periods=1).mean(), color='#00ccff', label='Steps (SMA 20)')
    if 'Food' in df.columns:
         ax2_twin = ax2.twinx()
         ax2_twin.plot(df['Episode'], df['Food'].rolling(50, min_periods=1).mean(), color='#ffaa00', alpha=0.7, label='Food (SMA 50)')
         ax2_twin.set_ylabel('Food', color='#ffaa00')
    ax2.set_title('Survival & Food')
    ax2.grid(True, alpha=0.2)
    ax2.legend(loc='upper left')
    
    # 3. Loss
    ax3 = plt.subplot(2, 2, 3)
    if 'Loss' in df.columns:
        ax3.plot(df['Episode'], df['Loss'].rolling(50, min_periods=1).mean(), color='#ff4444', label='Loss (SMA 50)')
        ax3.set_title('Training Loss')
    else:
        ax3.text(0.5, 0.5, 'No Loss Data', ha='center', va='center')
    ax3.grid(True, alpha=0.2)
    
    # 4. Epsilon & Beta
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(df['Episode'], df['Epsilon'], color='#ff6600', label='Epsilon')
    if 'Beta' in df.columns:
        ax4_twin = ax4.twinx()
        ax4_twin.plot(df['Episode'], df['Beta'], color='#cc00ff', label='Beta')
        ax4_twin.set_ylabel('Beta', color='#cc00ff')
    ax4.set_title('Hyperparameters')
    ax4.grid(True, alpha=0.2)
    ax4.legend(loc='upper left')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(script_dir, 'training_plot.png'), dpi=100, facecolor='#1a1a2e')
    
    # --- CHART 2: DETAILED ANALYSIS ---
    fig2 = plt.figure(figsize=(16, 8))
    
    # 1. Reward Distribution
    ax5 = plt.subplot(1, 2, 1)
    ax5.hist(df['Reward'], bins=50, color='#4488ff', alpha=0.7)
    ax5.axvline(df['Reward'].mean(), color='white', linestyle='--', label=f'Mean: {df["Reward"].mean():.1f}')
    ax5.set_title('Reward Distribution (All-Time)')
    ax5.legend()

    # 2. Death Causes
    ax6 = plt.subplot(1, 2, 2)
    if 'Cause' in df.columns:
        causes = df['Cause'].value_counts()
        ax6.pie(causes, labels=causes.index, autopct='%1.1f%%', startangle=90)
        ax6.set_title('Death Causes')
    else:
        ax6.text(0.5, 0.5, 'No Cause Data', ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'training_detailed.png'), dpi=100, facecolor='#1a1a2e')

    print(colored("✅ Charts saved.", Colors.GREEN))


# ==========================================
#  🚀 MAIN ENTRY POINT
# ==========================================

def analyze():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Locate CSV
    possible_files = ['training_stats.csv', 'matrix_stats.csv']
    csv_path = None

    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        csv_path = sys.argv[1]
    else:
        for f in possible_files:
            p = os.path.join(script_dir, f)
            if os.path.exists(p):
                csv_path = p
                break

    if not csv_path:
        print(colored("❌ No stats file found!", Colors.RED))
        return

    # Load Data
    try:
        df = pd.read_csv(csv_path)
        # Normalize columns
        if 'LR' in df.columns and 'LearningRate' not in df.columns:
            df['LearningRate'] = df['LR']
    except Exception as e:
        print(colored(f"❌ Error reading CSV: {e}", Colors.RED))
        return
        
    if len(df) < 2:
        print(colored("⚠ Not enough data.", Colors.YELLOW))
        return
        
    # Calculate Stats
    full_stats = calculate_statistics(df)

    recent_window = min(50, len(df))
    recent_df = df.tail(recent_window)
    recent_stats = calculate_statistics(recent_df)

    # 1. Console Dashboard
    print_dashboard(df, full_stats, recent_stats)
    
    # 2. Markdown Report
    md_path = os.path.join(script_dir, 'analysis_report.md')
    generate_markdown_report(df, full_stats, recent_stats, md_path)

    # 3. Charts
    plot_charts(df, script_dir)

    # 4. Console Sparklines (Optional Plotext)
    try:
        import plotext as pltx
        print(colored("\n[ RECENT REWARD TREND ]", Colors.CYAN + Colors.BOLD))
        pltx.clear_figure()
        pltx.theme('dark')
        # Use simple list for x-axis to avoid plotext date/time issues if index is huge
        y_data = df['Reward'].tail(100).tolist()
        pltx.plot(y_data, label='Reward')
        pltx.plotsize(80, 15)
        pltx.show()
    except ImportError:
        pass

if __name__ == "__main__":
    analyze()
