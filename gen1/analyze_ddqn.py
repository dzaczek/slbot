import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def analyze():
    if not os.path.exists('ddqn_stats.csv'):
        print("No stats file found (ddqn_stats.csv). Run 'python agent.py' first.")
        return

    try:
        df = pd.read_csv('ddqn_stats.csv')
    except Exception as e:
        print(f"Error reading csv: {e}")
        return

    if len(df) < 2:
        print("Not enough data to plot yet.")
        return

    # Calculate Rolling Average (e.g., last 50 games)
    df['MA50'] = df['Score'].rolling(window=50, min_periods=1).mean()

    # Console Stats
    print("\n=== Training Statistics ===")
    print(f"Total Games: {len(df)}")
    print(f"Current Record: {df['Record'].iloc[-1]}")
    print(f"Last 100 Avg Score: {df['Score'].tail(100).mean():.2f}")
    print(f"Current Epsilon: {df['Epsilon'].iloc[-1]}")
    print("===========================\n")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['Game'], df['Score'], label='Score', alpha=0.3, color='gray')
    plt.plot(df['Game'], df['MA50'], label='Avg Score (50)', color='blue', linewidth=2)
    plt.xlabel('Games')
    plt.ylabel('Score')
    plt.title('Snake DDQN Training Progress')
    plt.legend()
    plt.grid(True)

    output_file = 'ddqn_progress.png'
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

    # Try using plotext for console plot if available
    try:
        import plotext as plt_console
        plt_console.clear_figure()
        plt_console.plot(df['Game'], df['Score'], label='Score', marker="dot")
        plt_console.plot(df['Game'], df['MA50'], label='Avg (50)')
        plt_console.title("Training Progress")
        plt_console.xlabel("Games")
        plt_console.ylabel("Score")
        plt_console.show()
    except ImportError:
        pass

if __name__ == "__main__":
    analyze()
