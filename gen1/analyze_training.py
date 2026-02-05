"""
Advanced Training Analysis Tool for Slither.io NEAT Bot
Shows statistics and ASCII charts in the terminal.
Supports export to TXT and HTML files.

Usage:
    python analyze_training.py                  # Analyze training_stats.csv
    python analyze_training.py --live           # Live mode (updates every 10s)
    python analyze_training.py --output report  # Export to report.txt and report.html
    python analyze_training.py myfile.csv       # Analyze specific file
"""

import pandas as pd
import sys
import os
import time
import argparse
from datetime import datetime

try:
    import plotext as plt
    HAS_PLOTEXT = True
except ImportError:
    HAS_PLOTEXT = False
    print("Warning: plotext not installed. Install with: pip install plotext")


class ReportBuilder:
    """Builds report in multiple formats (console, TXT, HTML)."""
    
    def __init__(self):
        self.lines = []
        self.html_parts = []
        self.charts_data = []
    
    def add_header(self, text, level=1):
        """Add a header."""
        self.lines.append("=" * 70)
        self.lines.append(f"  {text}")
        self.lines.append("=" * 70)
        
        self.html_parts.append(f"<h{level}>{text}</h{level}>")
    
    def add_section(self, text):
        """Add a section header."""
        self.lines.append("")
        self.lines.append("─" * 70)
        self.lines.append(f"  {text}")
        self.lines.append("─" * 70)
        
        self.html_parts.append(f"<h3>{text}</h3>")
    
    def add_line(self, text=""):
        """Add a text line."""
        self.lines.append(text)
        self.html_parts.append(f"<p>{text}</p>" if text else "<br>")
    
    def add_table(self, headers, rows):
        """Add a table."""
        # Console/TXT format
        col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
        
        header_line = "  ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))
        self.lines.append(header_line)
        self.lines.append("-" * len(header_line))
        
        for row in rows:
            row_line = "  ".join(f"{str(row[i]):<{col_widths[i]}}" for i in range(len(row)))
            self.lines.append(row_line)
        
        # HTML format
        html = "<table border='1' cellpadding='8' cellspacing='0' style='border-collapse: collapse;'>"
        html += "<tr style='background-color: #f0f0f0;'>"
        for h in headers:
            html += f"<th>{h}</th>"
        html += "</tr>"
        for row in rows:
            html += "<tr>"
            for cell in row:
                html += f"<td>{cell}</td>"
            html += "</tr>"
        html += "</table>"
        self.html_parts.append(html)
    
    def add_bar_chart(self, labels, values, title=""):
        """Add a horizontal bar chart."""
        self.lines.append("")
        if title:
            self.lines.append(f"  {title}")
        
        max_val = max(values) if values else 1
        for label, val in zip(labels, values):
            bar_len = int((val / max_val) * 40)
            bar = "█" * bar_len
            pct = (val / sum(values)) * 100 if sum(values) > 0 else 0
            self.lines.append(f"  {label:<15} {val:>6} ({pct:>5.1f}%) {bar}")
        
        # HTML bar chart
        html = f"<h4>{title}</h4>" if title else ""
        html += "<div style='font-family: monospace;'>"
        for label, val in zip(labels, values):
            pct = (val / sum(values)) * 100 if sum(values) > 0 else 0
            bar_width = int((val / max_val) * 200) if max_val > 0 else 0
            html += f"<div style='margin: 5px 0;'>"
            html += f"<span style='display: inline-block; width: 120px;'>{label}</span>"
            html += f"<span style='display: inline-block; width: 80px;'>{val} ({pct:.1f}%)</span>"
            html += f"<span style='display: inline-block; width: {bar_width}px; height: 20px; background-color: #4CAF50;'></span>"
            html += "</div>"
        html += "</div>"
        self.html_parts.append(html)
    
    def add_chart_data(self, x_values, y_values, title, color="#2196F3"):
        """Store chart data for HTML rendering."""
        self.charts_data.append({
            'x': x_values,
            'y': y_values,
            'title': title,
            'color': color
        })
    
    def get_console_output(self):
        """Get console/TXT output."""
        return "\n".join(self.lines)
    
    def get_html_output(self, include_charts=True):
        """Get HTML output with optional charts."""
        html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Slither.io NEAT Bot - Training Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { 
            font-family: 'Segoe UI', Arial, sans-serif; 
            margin: 40px; 
            background-color: #f5f5f5;
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            background: white; 
            padding: 30px; 
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        h3 { color: #7f8c8d; border-left: 4px solid #3498db; padding-left: 10px; }
        table { width: 100%; margin: 20px 0; }
        th { background-color: #3498db; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .chart-container { width: 100%; height: 300px; margin: 20px 0; }
        .good { color: #27ae60; font-weight: bold; }
        .bad { color: #e74c3c; font-weight: bold; }
        .warning { color: #f39c12; font-weight: bold; }
        .timestamp { color: #95a5a6; font-size: 0.9em; }
    </style>
</head>
<body>
<div class="container">
"""
        html += "\n".join(self.html_parts)
        
        # Add Chart.js charts
        if include_charts and self.charts_data:
            for i, chart in enumerate(self.charts_data):
                html += f"""
<div class="chart-container">
    <canvas id="chart{i}"></canvas>
</div>
<script>
new Chart(document.getElementById('chart{i}'), {{
    type: 'line',
    data: {{
        labels: {chart['x']},
        datasets: [{{
            label: '{chart['title']}',
            data: {chart['y']},
            borderColor: '{chart['color']}',
            backgroundColor: '{chart['color']}22',
            fill: true,
            tension: 0.3
        }}]
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
            title: {{
                display: true,
                text: '{chart['title']}',
                font: {{ size: 16 }}
            }}
        }},
        scales: {{
            y: {{ beginAtZero: true }}
        }}
    }}
}});
</script>
"""
        
        html += f"""
<p class="timestamp">Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</div>
</body>
</html>"""
        return html
    
    def save_txt(self, filepath):
        """Save report to TXT file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.get_console_output())
        print(f"✓ Saved TXT report to: {filepath}")
    
    def save_html(self, filepath):
        """Save report to HTML file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.get_html_output())
        print(f"✓ Saved HTML report to: {filepath}")


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def analyze_training(csv_path='training_stats.csv', show_charts=True, output_file=None):
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
    
    # Create report builder
    report = ReportBuilder()
    
    total_evals = len(df)
    
    report.add_header("SLITHER.IO NEAT BOT - TRAINING ANALYSIS")
    report.add_line(f"File: {csv_path}")
    report.add_line(f"Total evaluations: {total_evals}")
    
    # ============================================
    # OVERALL STATISTICS
    # ============================================
    report.add_section("OVERALL STATISTICS")
    
    # Convert columns to numeric
    df['Fitness'] = pd.to_numeric(df['Fitness'], errors='coerce')
    df['SurvivalTime'] = pd.to_numeric(df['SurvivalTime'], errors='coerce')
    df['FoodEaten'] = pd.to_numeric(df['FoodEaten'], errors='coerce')
    df['MaxLen'] = pd.to_numeric(df['MaxLen'], errors='coerce')
    
    stats_headers = ['Metric', 'Average', 'Max', 'Min']
    stats_rows = [
        ['Fitness', f"{df['Fitness'].mean():.1f}", f"{df['Fitness'].max():.1f}", f"{df['Fitness'].min():.1f}"],
        ['Survival Time (s)', f"{df['SurvivalTime'].mean():.1f}", f"{df['SurvivalTime'].max():.1f}", f"{df['SurvivalTime'].min():.1f}"],
        ['Food Eaten', f"{df['FoodEaten'].mean():.1f}", f"{df['FoodEaten'].max():.0f}", f"{df['FoodEaten'].min():.0f}"],
        ['Max Length', f"{df['MaxLen'].mean():.1f}", f"{df['MaxLen'].max():.0f}", f"{df['MaxLen'].min():.0f}"],
    ]
    report.add_table(stats_headers, stats_rows)
    
    # ============================================
    # CAUSE OF DEATH
    # ============================================
    report.add_section("CAUSE OF DEATH")
    
    death_counts = df['CauseOfDeath'].value_counts()
    report.add_bar_chart(
        list(death_counts.index), 
        list(death_counts.values),
        "Deaths by Cause"
    )
    
    # ============================================
    # RECENT PERFORMANCE
    # ============================================
    if len(df) >= 50:
        report.add_section("RECENT PERFORMANCE (Last 100 vs First 100)")
        
        first_100 = df.head(100)
        last_100 = df.tail(100)
        
        perf_headers = ['Metric', 'First 100', 'Last 100', 'Change']
        perf_rows = []
        
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
            arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
            perf_rows.append([name, f"{first_val:.1f}", f"{last_val:.1f}", f"{arrow} {change:+.1f}"])
        
        report.add_table(perf_headers, perf_rows)
        
        first_ate = (first_100['FoodEaten'] > 0).sum()
        last_ate = (last_100['FoodEaten'] > 0).sum()
        report.add_line(f"Bots that ate food: {first_ate}% → {last_ate}%")
    
    # ============================================
    # TOP PERFORMERS
    # ============================================
    report.add_section("TOP 10 PERFORMERS")
    
    top10 = df.nlargest(10, 'Fitness')
    top_headers = ['#', 'GenomeID', 'Fitness', 'Time', 'Food', 'Len', 'Death']
    top_rows = []
    for i, (idx, row) in enumerate(top10.iterrows(), 1):
        top_rows.append([
            i, 
            row['GenomeID'], 
            f"{row['Fitness']:.1f}", 
            f"{row['SurvivalTime']:.1f}s",
            f"{row['FoodEaten']:.0f}",
            f"{row['MaxLen']:.0f}",
            row['CauseOfDeath']
        ])
    report.add_table(top_headers, top_rows)
    
    # ============================================
    # PREPARE CHART DATA
    # ============================================
    if len(df) >= 20:
        chunk_size = max(1, len(df) // 50)
        
        fitness_chunks = []
        food_chunks = []
        length_chunks = []
        
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            fitness_chunks.append(round(chunk['Fitness'].mean(), 1))
            food_chunks.append(round(chunk['FoodEaten'].mean(), 1))
            length_chunks.append(round(chunk['MaxLen'].mean(), 1))
        
        x_values = list(range(len(fitness_chunks)))
        
        # Store chart data for HTML
        report.add_chart_data(x_values, fitness_chunks, "Fitness Progress", "#2196F3")
        report.add_chart_data(x_values, food_chunks, "Food Eaten Progress", "#4CAF50")
        report.add_chart_data(x_values, length_chunks, "Snake Length Progress", "#FF9800")
    
    # ============================================
    # LEARNING ASSESSMENT
    # ============================================
    report.add_section("LEARNING ASSESSMENT")
    
    if len(df) >= 200:
        first_half = df.head(len(df) // 2)
        second_half = df.tail(len(df) // 2)
        
        fitness_imp = second_half['Fitness'].mean() - first_half['Fitness'].mean()
        food_imp = second_half['FoodEaten'].mean() - first_half['FoodEaten'].mean()
        
        report.add_line("Improvement (2nd half vs 1st half):")
        
        if fitness_imp > 10:
            report.add_line(f"✓ Fitness:  +{fitness_imp:.1f} (GOOD - Bot is learning!)")
        elif fitness_imp > 0:
            report.add_line(f"~ Fitness:  +{fitness_imp:.1f} (Slow progress)")
        else:
            report.add_line(f"✗ Fitness:  {fitness_imp:.1f} (NO IMPROVEMENT)")
        
        if food_imp > 1:
            report.add_line(f"✓ Food:     +{food_imp:.1f} (GOOD - Learning to eat!)")
        elif food_imp > 0:
            report.add_line(f"~ Food:     +{food_imp:.1f} (Slow progress)")
        else:
            report.add_line(f"✗ Food:     {food_imp:.1f} (NOT LEARNING TO EAT!)")
        
        # Recommendations
        report.add_line("")
        report.add_line("Recommendations:")
        
        starvation_rate = (df['CauseOfDeath'] == 'Starvation').sum() / len(df) * 100
        wall_rate = (df['CauseOfDeath'] == 'Wall').sum() / len(df) * 100 if 'Wall' in df['CauseOfDeath'].values else 0
        
        if starvation_rate > 80:
            report.add_line("⚠ 80%+ starvation - Increase food reward or decrease starvation timeout")
        if wall_rate > 20:
            report.add_line("⚠ 20%+ wall deaths - Increase wall danger detection range")
        if df['FoodEaten'].mean() < 2:
            report.add_line("⚠ Very low food intake - Bot may need simpler initial task")
        if fitness_imp <= 0:
            report.add_line("⚠ No fitness improvement - Consider restarting with fresh population")
        
        if fitness_imp > 10 and food_imp > 1:
            report.add_line("✓ Bot is learning well! Continue training.")
    else:
        report.add_line("Need at least 200 evaluations for learning assessment.")
    
    report.add_line("")
    report.add_line(f"Analysis completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ============================================
    # OUTPUT
    # ============================================
    
    # Always print to console
    print(report.get_console_output())
    
    # Show plotext charts in console
    if show_charts and HAS_PLOTEXT and len(df) >= 20:
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
    
    # Save to files if requested
    if output_file:
        txt_path = f"{output_file}.txt"
        html_path = f"{output_file}.html"
        
        report.save_txt(txt_path)
        report.save_html(html_path)
        
        print(f"\n📄 Reports saved:")
        print(f"   - {txt_path}")
        print(f"   - {html_path}")
    
    return report


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
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Export report to files (creates <name>.txt and <name>.html)')
    
    args = parser.parse_args()
    
    if args.live:
        live_mode(args.csv_file, args.interval)
    else:
        analyze_training(args.csv_file, show_charts=not args.no_charts, output_file=args.output)


if __name__ == "__main__":
    main()
