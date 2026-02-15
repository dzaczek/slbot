# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-15 08:31:35  
**Total Episodes:** 4274  
**Training Sessions:** 13

## Verdict: LEARNING (Confidence: 65%)

**Goal Feasibility:** POSSIBLE (25-60%)

### Positive Signals
- Rewards improving: +186.0

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 1182 | 286.8 | 62.3 | 27.9 | 0.6945 | 0.0% | 0.0% | 0.0% |
| S2 | WALL_AVOID | 688 | 111.7 | 77.5 | 32.1 | 0.6594 | 0.0% | 0.0% | 0.0% |
| S3 | ENEMY_AVOID | 292 | 191.1 | 89.2 | 33.7 | 0.9742 | 0.0% | 0.0% | 0.0% |
| S4 | MASS_MANAGEMENT | 2112 | 230.9 | 90.3 | 36.8 | 0.9087 | 0.0% | 0.0% | 0.0% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 224.47 | 271.55 | -7111.38 | 80.50 | 152.61 | 294.30 | 691.59 | 3256.19 |
| Steps | 80.46 | 124.48 | 1.00 | 26.00 | 61.00 | 109.00 | 206.00 | 5000.00 |
| Food | 33.39 | 19.00 | 0.00 | 21.00 | 30.00 | 43.00 | 70.00 | 128.00 |
| Loss | 19.14 | 67.02 | 0.00 | 2.94 | 8.13 | 19.41 | 42.51 | 996.87 |
| Food/Step | 0.81 | 1.16 | 0.00 | 0.36 | 0.47 | 0.71 | 3.14 | 16.50 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 745.09 | 309.51 | -1.4033 | 0.0043 |
| Last 100 | 704.50 | 323.09 | +0.5258 | 0.0022 |
| Last 200 | 708.83 | 324.28 | -0.0460 | 0.0001 |
| Last 500 | 582.25 | 339.31 | +0.9526 | 0.1642 |
| Last 1000 | 408.06 | 327.26 | +0.6482 | 0.3270 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 128 | 6,000 | 2.1% |
| Survival | 5000 steps | 1,800 steps | 277.8% |

## Recommendations

Keep training. Monitor for sustained improvement.

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

