# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-16 17:15:29  
**Total Episodes:** 901  
**Training Sessions:** 7

## Verdict: NOT LEARNING (Confidence: 55%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

### Warnings
- Rewards flat: change = 6.7

### Positive Signals
- Food collection improving (slope=0.0258/ep)
- Epsilon low (0.082) - exploiting policy

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 214 | 141.4 | 125.9 | 39.6 | 0.5520 | 0.9% | 80.8% | 17.8% |
| S2 | WALL_AVOID | 400 | 486.0 | 269.1 | 70.2 | 0.4690 | 0.2% | 69.8% | 29.0% |
| S3 | ENEMY_AVOID | 287 | 341.3 | 299.8 | 74.3 | 0.5403 | 0.3% | 98.6% | 1.0% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 358.09 | 325.42 | -144.80 | 110.95 | 256.06 | 537.82 | 920.25 | 2629.01 |
| Steps | 244.85 | 260.87 | 1.00 | 63.00 | 163.00 | 351.00 | 563.00 | 2000.00 |
| Food | 64.25 | 46.99 | 0.00 | 30.00 | 55.00 | 88.00 | 135.00 | 415.00 |
| Loss | 4.66 | 5.24 | 0.00 | 2.01 | 3.31 | 5.12 | 14.21 | 40.93 |
| Food/Step | 0.51 | 0.71 | 0.00 | 0.23 | 0.30 | 0.49 | 1.27 | 6.67 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 425.72 | 414.72 | -6.0621 | 0.0445 |
| Last 100 | 382.65 | 355.18 | +0.5506 | 0.0020 |
| Last 200 | 316.86 | 291.73 | +0.8738 | 0.0299 |
| Last 500 | 377.49 | 337.96 | -0.1882 | 0.0065 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 4 | 0.4% | 216.0 | 248.8 |
| SnakeCollision | 735 | 81.6% | 195.1 | 278.8 |
| MaxSteps | 157 | 17.4% | 480.3 | 731.1 |
| BrowserError | 5 | 0.6% | 189.8 | 390.6 |

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 415 | 6,000 | 6.9% |
| Survival | 2000 steps | 1,800 steps | 111.1% |

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

### Steps vs Food vs Episode (3D)
![Steps vs Food vs Episode (3D)](chart_18_3d_steps_food_episode.png)

