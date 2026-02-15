# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-15 17:13:33  
**Total Episodes:** 8818  
**Training Sessions:** 24

## Verdict: NOT LEARNING (Confidence: 45%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

### Warnings
- Rewards flat: change = -24.1
- Loss very high (11.49) - unstable

### Positive Signals
- Epsilon low (0.100) - exploiting policy

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 1293 | 320.7 | 62.2 | 28.5 | 0.7114 | 14.7% | 85.1% | 0.2% |
| S2 | WALL_AVOID | 1449 | 108.7 | 74.9 | 33.0 | 0.7296 | 9.0% | 91.0% | 0.0% |
| S3 | ENEMY_AVOID | 2708 | 245.1 | 81.3 | 33.2 | 0.8759 | 9.6% | 89.4% | 1.0% |
| S4 | MASS_MANAGEMENT | 3368 | 218.0 | 82.0 | 35.6 | 0.9065 | 13.7% | 86.1% | 0.2% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 223.42 | 258.57 | -7111.38 | 81.42 | 153.16 | 288.82 | 679.63 | 4227.53 |
| Steps | 77.69 | 102.02 | 1.00 | 26.00 | 60.00 | 106.00 | 203.00 | 5000.00 |
| Food | 33.39 | 18.60 | 0.00 | 21.00 | 30.00 | 43.00 | 70.00 | 154.00 |
| Loss | 17.77 | 48.20 | 0.00 | 4.25 | 10.04 | 20.57 | 44.77 | 996.87 |
| Food/Step | 0.84 | 1.13 | 0.00 | 0.36 | 0.49 | 0.75 | 3.23 | 16.50 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 204.17 | 130.38 | +0.7787 | 0.0074 |
| Last 100 | 221.40 | 154.10 | -0.6165 | 0.0133 |
| Last 200 | 208.39 | 162.06 | +0.0978 | 0.0012 |
| Last 500 | 264.15 | 404.10 | -0.2404 | 0.0074 |
| Last 1000 | 248.65 | 320.59 | +0.0055 | 0.0000 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 1042 | 11.8% | 129.6 | 300.5 |
| SnakeCollision | 7740 | 87.8% | 66.5 | 208.1 |
| MaxSteps | 36 | 0.4% | 983.3 | 1279.2 |

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 154 | 6,000 | 2.6% |
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

