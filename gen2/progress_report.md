# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-14 21:37:57  
**Total Episodes:** 1710  
**Training Sessions:** 8

## Verdict: NOT LEARNING (Confidence: 50%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

### Warnings
- Rewards flat: change = -9.4

### Positive Signals
- Epsilon low (0.117) - exploiting policy

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 773 | 89.9 | 59.9 | 24.9 | 0.5461 | 0.0% | 0.0% | 0.0% |
| S2 | WALL_AVOID | 688 | 111.7 | 77.5 | 32.1 | 0.6594 | 0.0% | 0.0% | 0.0% |
| S3 | ENEMY_AVOID | 249 | 118.6 | 62.6 | 34.6 | 0.9728 | 0.0% | 0.0% | 0.0% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 102.87 | 96.38 | -50.00 | 45.33 | 91.27 | 151.92 | 281.42 | 531.78 |
| Steps | 67.39 | 57.19 | 1.00 | 23.00 | 56.00 | 97.00 | 186.55 | 337.00 |
| Food | 29.23 | 16.99 | 0.00 | 21.00 | 27.00 | 39.00 | 60.00 | 97.00 |
| Loss | 19.47 | 104.58 | 0.00 | 1.44 | 2.50 | 4.16 | 13.01 | 996.87 |
| Food/Step | 0.65 | 0.87 | 0.00 | 0.34 | 0.45 | 0.65 | 2.00 | 14.67 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 114.02 | 117.93 | +0.0821 | 0.0001 |
| Last 100 | 117.08 | 121.97 | +0.1596 | 0.0014 |
| Last 200 | 124.19 | 120.47 | -0.0452 | 0.0005 |
| Last 500 | 70.43 | 117.91 | +0.4137 | 0.2564 |
| Last 1000 | 100.33 | 105.90 | -0.0315 | 0.0074 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 97 | 6,000 | 1.6% |
| Survival | 337 steps | 1,800 steps | 18.7% |

## Recommendations

Fine-tune hyperparameters, increase training duration.

1. Episodes too short. Reduce death penalties or add survival bonus.

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

