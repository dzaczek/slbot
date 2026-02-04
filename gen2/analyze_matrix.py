import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def analyze():
    # Path relative to script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'matrix_stats.csv')

    if not os.path.exists(csv_path):
        print(f"No stats file found at {csv_path}. Run 'python gen2/trainer.py' first.")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading csv: {e}")
        return

    if len(df) < 2:
        print("Not enough data to plot yet.")
        return

    # Calculate Rolling Average
    df['SMA10'] = df['Reward'].rolling(window=10, min_periods=1).mean()

    # Console Stats
    print("\n=== MatrixBot Training Statistics ===")
    print(f"Total Episodes: {len(df)}")
    print(f"Best Reward: {df['Reward'].max():.2f}")
    print(f"Last 10 Avg Reward: {df['Reward'].tail(10).mean():.2f}")
    print(f"Current Epsilon: {df['Epsilon'].iloc[-1]:.4f}")
    print("=====================================\n")

    # Save PNG Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['Episode'], df['Reward'], label='Reward', alpha=0.3, color='gray')
    plt.plot(df['Episode'], df['SMA10'], label='Avg Reward (10)', color='red', linewidth=2)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Slither MatrixBot Learning Progress')
    plt.legend()
    plt.grid(True)

    output_file = os.path.join(script_dir, 'training_plot.png')
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

    # Console Plot
    try:
        import plotext as plt_console
        plt_console.clear_figure()
        plt_console.plot(df['Episode'], df['Reward'], label='Reward', marker="dot")
        plt_console.plot(df['Episode'], df['SMA10'], label='Avg (10)')
        plt_console.title("Slither MatrixBot Progress")
        plt_console.xlabel("Episodes")
        plt_console.ylabel("Reward")
        plt_console.show()
    except ImportError:
        print("Install 'plotext' for console plots: pip install plotext")

if __name__ == "__main__":
    analyze()
