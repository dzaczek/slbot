"""
Analyze training progress from CSV and show statistics.
"""

import pandas as pd
import sys

def analyze_training(csv_path='training_stats.csv'):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    print("=" * 60)
    print("TRAINING ANALYSIS")
    print("=" * 60)
    
    total_evals = len(df)
    print(f"\nTotal evaluations: {total_evals}")
    
    # Overall statistics
    print("\n--- OVERALL STATISTICS ---")
    print(f"Average Fitness: {df['Fitness'].mean():.2f}")
    print(f"Max Fitness: {df['Fitness'].max():.2f}")
    print(f"Average Survival Time: {df['SurvivalTime'].mean():.2f}s")
    print(f"Max Survival Time: {df['SurvivalTime'].max():.2f}s")
    print(f"Average Food Eaten: {df['FoodEaten'].mean():.2f}")
    print(f"Max Food Eaten: {df['FoodEaten'].max():.0f}")
    print(f"Average Length: {df['MaxLen'].mean():.2f}")
    print(f"Max Length: {df['MaxLen'].max():.0f}")
    
    # Cause of death
    print("\n--- CAUSE OF DEATH ---")
    death_counts = df['CauseOfDeath'].value_counts()
    for cause, count in death_counts.items():
        pct = (count / total_evals) * 100
        print(f"{cause}: {count} ({pct:.1f}%)")
    
    # Recent performance (last 100)
    if len(df) > 100:
        recent = df.tail(100)
        print("\n--- LAST 100 EVALUATIONS ---")
        print(f"Average Fitness: {recent['Fitness'].mean():.2f}")
        print(f"Max Fitness: {recent['Fitness'].max():.2f}")
        print(f"Average Food Eaten: {recent['FoodEaten'].mean():.2f}")
        print(f"Max Food Eaten: {recent['FoodEaten'].max():.0f}")
        print(f"Average Survival: {recent['SurvivalTime'].mean():.2f}s")
        
        recent_deaths = recent['CauseOfDeath'].value_counts()
        print("\nRecent deaths:")
        for cause, count in recent_deaths.items():
            pct = (count / len(recent)) * 100
            print(f"  {cause}: {count} ({pct:.1f}%)")
    
    # Best performers
    print("\n--- TOP 10 PERFORMERS ---")
    top10 = df.nlargest(10, 'Fitness')
    for idx, row in top10.iterrows():
        print(f"Genome {row['GenomeID']}: Fit={row['Fitness']:.2f}, "
              f"Time={row['SurvivalTime']:.1f}s, Food={row['FoodEaten']:.0f}, "
              f"Len={row['MaxLen']:.0f}, Death={row['CauseOfDeath']}")
    
    # Trend analysis
    if len(df) > 500:
        print("\n--- TREND ANALYSIS ---")
        first_500 = df.head(500)
        last_500 = df.tail(500)
        
        fit_change = last_500['Fitness'].mean() - first_500['Fitness'].mean()
        food_change = last_500['FoodEaten'].mean() - first_500['FoodEaten'].mean()
        
        print(f"Fitness improvement: {fit_change:+.2f} ({fit_change/first_500['Fitness'].mean()*100:+.1f}%)")
        print(f"Food improvement: {food_change:+.2f} ({food_change/max(first_500['FoodEaten'].mean(), 0.1)*100:+.1f}%)")
        
        if fit_change > 0:
            print("✓ Bot is IMPROVING!")
        else:
            print("✗ Bot is NOT improving - may need parameter adjustment")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'training_stats.csv'
    analyze_training(csv_path)
