# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-15 15:12:47  
**Total Episodes:** 7940  
**Training Sessions:** 16

## Verdict: LEARNING (Confidence: 60%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

### Warnings
- Loss very high (48.69) - unstable

### Positive Signals
- Rewards improving: +62.7

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 1293 | 320.7 | 62.2 | 28.5 | 0.7114 | 14.7% | 85.1% | 0.2% |
| S2 | WALL_AVOID | 1449 | 108.7 | 74.9 | 33.0 | 0.7296 | 9.0% | 91.0% | 0.0% |
| S3 | ENEMY_AVOID | 1830 | 244.8 | 77.9 | 32.9 | 0.9042 | 8.0% | 90.8% | 1.2% |
| S4 | MASS_MANAGEMENT | 3368 | 218.0 | 82.0 | 35.6 | 0.9065 | 13.7% | 86.1% | 0.2% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 220.96 | 249.45 | -7111.38 | 80.70 | 151.04 | 287.41 | 681.57 | 3256.19 |
| Steps | 76.51 | 102.64 | 1.00 | 25.00 | 59.00 | 104.00 | 201.00 | 5000.00 |
| Food | 33.37 | 18.53 | 0.00 | 21.00 | 30.00 | 43.00 | 70.00 | 128.00 |
| Loss | 17.56 | 50.32 | 0.00 | 3.90 | 9.55 | 20.11 | 43.65 | 996.87 |
| Food/Step | 0.84 | 1.12 | 0.00 | 0.37 | 0.49 | 0.75 | 3.17 | 16.50 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 275.49 | 308.25 | +2.8798 | 0.0182 |
| Last 100 | 263.10 | 259.86 | +0.9880 | 0.0120 |
| Last 200 | 278.91 | 256.14 | -0.0292 | 0.0000 |
| Last 500 | 283.29 | 268.14 | -0.0163 | 0.0001 |
| Last 1000 | 277.29 | 274.37 | +0.0356 | 0.0014 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 928 | 11.7% | 125.1 | 282.0 |
| SnakeCollision | 6980 | 87.9% | 65.9 | 209.5 |
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

