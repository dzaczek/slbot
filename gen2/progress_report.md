# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-15 18:05:30  
**Total Episodes:** 218  
**Training Sessions:** 4

## Verdict: NOT LEARNING (Confidence: 45%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

### Warnings
- Rewards flat: change = -15.5

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 200 | 99.2 | 82.8 | 32.8 | 0.7589 | 8.0% | 92.0% | 0.0% |
| S2 | WALL_AVOID | 1 | 227.2 | 179.0 | 25.0 | 0.1400 | 0.0% | 100.0% | 0.0% |
| S3 | ENEMY_AVOID | 17 | 161.0 | 76.6 | 30.2 | 0.4415 | 0.0% | 100.0% | 0.0% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 104.61 | 77.42 | -16.90 | 51.73 | 84.48 | 135.76 | 243.99 | 403.17 |
| Steps | 82.79 | 65.18 | 1.00 | 30.00 | 73.00 | 115.75 | 218.90 | 293.00 |
| Food | 32.56 | 18.36 | 0.00 | 21.00 | 28.00 | 41.00 | 69.00 | 111.00 |
| Loss | 4.99 | 3.71 | 0.60 | 2.62 | 3.96 | 6.00 | 13.58 | 21.62 |
| Food/Step | 0.73 | 0.97 | 0.00 | 0.31 | 0.42 | 0.62 | 3.00 | 6.67 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 115.62 | 84.03 | +1.9015 | 0.1066 |
| Last 100 | 100.21 | 73.19 | +0.5859 | 0.0534 |
| Last 200 | 102.56 | 78.46 | -0.0495 | 0.0013 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 16 | 7.3% | 185.4 | 209.0 |
| SnakeCollision | 202 | 92.7% | 74.7 | 96.3 |

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 111 | 6,000 | 1.8% |
| Survival | 293 steps | 1,800 steps | 16.3% |

## Recommendations

Fine-tune hyperparameters, increase training duration.

1. Epsilon 0.564 still high. Consider faster decay.

2. Episodes too short. Reduce death penalties or add survival bonus.

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

### Goal Progress Over Time
![Goal Progress Over Time](chart_11b_goal_over_time.png)

### Hyperparameter Analysis
![Hyperparameter Analysis](chart_12_hyperparameter_analysis.png)

### Q-Value & Gradient Analysis
![Q-Value & Gradient Analysis](chart_13_qvalue_gradients.png)

### Action Distribution Analysis
![Action Distribution Analysis](chart_14_action_distribution.png)

### Active Agents Over Time
![Active Agents Over Time](chart_15_auto_scaling.png)

### MaxSteps Analysis
![MaxSteps Analysis](chart_16_maxsteps_analysis.png)

### Survival Percentiles
![Survival Percentiles](chart_17_survival_percentiles.png)

