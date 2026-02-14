# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-14 21:41:47  
**Total Episodes:** 1747  
**Training Sessions:** 10

## Verdict: NOT LEARNING (Confidence: 50%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

### Warnings
- Rewards flat: change = 6.9

### Positive Signals
- Epsilon low (0.113) - exploiting policy

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 773 | 89.9 | 59.9 | 24.9 | 0.5461 | 0.0% | 0.0% | 0.0% |
| S2 | WALL_AVOID | 688 | 111.7 | 77.5 | 32.1 | 0.6594 | 0.0% | 0.0% | 0.0% |
| S3 | ENEMY_AVOID | 286 | 170.1 | 80.6 | 33.8 | 0.9935 | 0.0% | 0.0% | 0.0% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 111.63 | 140.10 | -50.00 | 45.97 | 91.55 | 153.79 | 292.33 | 1417.15 |
| Steps | 70.25 | 67.02 | 1.00 | 23.00 | 56.00 | 98.50 | 190.00 | 500.00 |
| Food | 29.22 | 17.13 | 0.00 | 21.00 | 27.00 | 39.00 | 60.00 | 97.00 |
| Loss | 19.19 | 103.48 | 0.00 | 1.47 | 2.54 | 4.24 | 12.71 | 996.87 |
| Food/Step | 0.66 | 0.92 | 0.00 | 0.34 | 0.45 | 0.65 | 2.00 | 14.67 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 405.87 | 532.25 | +10.6420 | 0.0833 |
| Last 100 | 270.61 | 412.24 | +5.3802 | 0.1419 |
| Last 200 | 194.85 | 310.37 | +1.7956 | 0.1116 |
| Last 500 | 110.24 | 224.66 | +0.6953 | 0.1996 |
| Last 1000 | 115.91 | 171.75 | +0.0564 | 0.0090 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 97 | 6,000 | 1.6% |
| Survival | 500 steps | 1,800 steps | 27.8% |

## Recommendations

Fine-tune hyperparameters, increase training duration.

1. No critical issues. Continue training.

## Charts

### Main Dashboard
![Main Dashboard](chart_01_dashboard.png)

### Stage Progression
![Stage Progression](chart_02_stage_progression.png)

### Per-Stage Distributions
![Per-Stage Distributions](chart_03_stage_distributions.png)

### Hyperparameter Tracking
![Hyperparameter Tracking](chart_04_hyperparameters.png)

### Metric Correlations (Scatter)
![Metric Correlations (Scatter)](chart_05_correlations.png)

### Correlation Heatmap & Rankings
![Correlation Heatmap & Rankings](chart_05b_correlation_heatmap.png)

### Performance Percentile Bands
![Performance Percentile Bands](chart_06_performance_bands.png)

### Death Analysis
![Death Analysis](chart_07_death_analysis.png)

### Food Efficiency
![Food Efficiency](chart_08_food_efficiency.png)

### Reward Distributions
![Reward Distributions](chart_09_reward_distributions.png)

### Learning Detection
![Learning Detection](chart_10_learning_detection.png)

### Goal Progress
![Goal Progress](chart_11_goal_gauges.png)

### Hyperparameter Analysis
![Hyperparameter Analysis](chart_12_hyperparameter_analysis.png)

### Q-Value & Gradient Analysis
![Q-Value & Gradient Analysis](chart_13_qvalue_gradients.png)

### Action Distribution Analysis
![Action Distribution Analysis](chart_14_action_distribution.png)

### Active Agents Over Time
![Active Agents Over Time](chart_15_auto_scaling.png)

