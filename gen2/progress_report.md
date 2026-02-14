# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-14 22:15:35  
**Total Episodes:** 2084  
**Training Sessions:** 12

## Verdict: NOT LEARNING (Confidence: 45%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

### Warnings
- Rewards flat: change = 31.0
- Loss very high (28.26) - unstable

### Positive Signals
- Epsilon low (0.096) - exploiting policy

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 773 | 89.9 | 59.9 | 24.9 | 0.5461 | 0.0% | 0.0% | 0.0% |
| S2 | WALL_AVOID | 688 | 111.7 | 77.5 | 32.1 | 0.6594 | 0.0% | 0.0% | 0.0% |
| S3 | ENEMY_AVOID | 292 | 191.1 | 89.2 | 33.7 | 0.9742 | 0.0% | 0.0% | 0.0% |
| S4 | MASS_MANAGEMENT | 331 | 201.3 | 94.5 | 37.3 | 0.9233 | 0.0% | 0.0% | 0.0% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 128.98 | 223.20 | -7111.38 | 53.02 | 103.81 | 176.12 | 355.17 | 1476.21 |
| Steps | 75.34 | 129.11 | 1.00 | 25.00 | 58.00 | 101.00 | 198.00 | 5000.00 |
| Food | 30.50 | 17.90 | 0.00 | 21.00 | 28.00 | 40.00 | 63.00 | 128.00 |
| Loss | 18.20 | 94.91 | 0.00 | 1.66 | 2.96 | 5.72 | 24.33 | 996.87 |
| Food/Step | 0.70 | 0.99 | 0.00 | 0.34 | 0.46 | 0.67 | 2.20 | 14.67 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 38.72 | 1027.45 | +14.3050 | 0.0404 |
| Last 100 | 119.99 | 734.90 | -0.5545 | 0.0005 |
| Last 200 | 190.55 | 539.59 | -1.1599 | 0.0154 |
| Last 500 | 217.59 | 412.04 | -0.1127 | 0.0016 |
| Last 1000 | 144.69 | 309.42 | +0.2124 | 0.0393 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 128 | 6,000 | 2.1% |
| Survival | 5000 steps | 1,800 steps | 277.8% |

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

