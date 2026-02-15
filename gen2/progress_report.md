# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-15 16:13:09  
**Total Episodes:** 8374  
**Training Sessions:** 20

## Verdict: NOT LEARNING (Confidence: 45%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

### Warnings
- Rewards flat: change = 13.1
- Loss very high (13.25) - unstable

### Positive Signals
- Epsilon low (0.136) - exploiting policy

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 1293 | 320.7 | 62.2 | 28.5 | 0.7114 | 14.7% | 85.1% | 0.2% |
| S2 | WALL_AVOID | 1449 | 108.7 | 74.9 | 33.0 | 0.7296 | 9.0% | 91.0% | 0.0% |
| S3 | ENEMY_AVOID | 2264 | 241.4 | 79.2 | 33.0 | 0.8756 | 9.0% | 90.1% | 1.0% |
| S4 | MASS_MANAGEMENT | 3368 | 218.0 | 82.0 | 35.6 | 0.9065 | 13.7% | 86.1% | 0.2% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 221.28 | 246.89 | -7111.38 | 81.27 | 152.40 | 288.24 | 680.16 | 3256.19 |
| Steps | 76.94 | 101.21 | 1.00 | 26.00 | 60.00 | 105.00 | 202.00 | 5000.00 |
| Food | 33.37 | 18.54 | 0.00 | 21.00 | 30.00 | 43.00 | 70.00 | 128.00 |
| Loss | 17.91 | 49.31 | 0.00 | 4.10 | 9.91 | 20.72 | 44.95 | 996.87 |
| Food/Step | 0.84 | 1.12 | 0.00 | 0.37 | 0.49 | 0.75 | 3.17 | 16.50 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 275.77 | 254.17 | -6.3475 | 0.1299 |
| Last 100 | 266.04 | 229.32 | -0.4366 | 0.0030 |
| Last 200 | 224.62 | 190.60 | +0.5957 | 0.0326 |
| Last 500 | 234.60 | 209.45 | -0.0190 | 0.0002 |
| Last 1000 | 261.42 | 253.63 | -0.0915 | 0.0108 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 985 | 11.8% | 127.4 | 290.7 |
| SnakeCollision | 7357 | 87.9% | 66.3 | 208.8 |
| MaxSteps | 32 | 0.4% | 981.2 | 947.6 |

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 128 | 6,000 | 2.1% |
| Survival | 5000 steps | 1,800 steps | 277.8% |

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

