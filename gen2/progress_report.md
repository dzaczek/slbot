# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-16 17:36:05  
**Total Episodes:** 977  
**Training Sessions:** 7

## Verdict: NOT LEARNING (Confidence: 50%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

### Warnings
- Rewards flat: change = -1.4
- Reward in plateau (< 1% change over 200 eps)

### Positive Signals
- Food collection improving (slope=0.0211/ep)
- Epsilon low (0.081) - exploiting policy

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 214 | 141.4 | 125.9 | 39.6 | 0.5520 | 0.9% | 80.8% | 17.8% |
| S2 | WALL_AVOID | 400 | 486.0 | 269.1 | 70.2 | 0.4690 | 0.2% | 69.8% | 29.0% |
| S3 | ENEMY_AVOID | 363 | 347.5 | 285.8 | 72.7 | 0.5347 | 0.3% | 98.9% | 0.8% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 359.06 | 321.82 | -144.80 | 113.85 | 262.45 | 519.84 | 920.06 | 2629.01 |
| Steps | 243.90 | 257.66 | 1.00 | 64.00 | 164.00 | 343.00 | 579.20 | 2000.00 |
| Food | 64.43 | 46.27 | 0.00 | 30.00 | 56.00 | 87.00 | 136.00 | 415.00 |
| Loss | 4.73 | 5.11 | 0.00 | 2.05 | 3.41 | 5.44 | 13.50 | 40.93 |
| Food/Step | 0.51 | 0.70 | 0.00 | 0.23 | 0.31 | 0.50 | 1.28 | 6.67 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 357.63 | 302.15 | -8.0598 | 0.1482 |
| Last 100 | 362.29 | 333.78 | -1.0841 | 0.0088 |
| Last 200 | 362.53 | 311.04 | +0.2411 | 0.0020 |
| Last 500 | 356.95 | 280.99 | +0.0671 | 0.0012 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 4 | 0.4% | 216.0 | 248.8 |
| SnakeCollision | 811 | 83.0% | 198.6 | 287.4 |
| MaxSteps | 157 | 16.1% | 480.3 | 731.1 |
| BrowserError | 5 | 0.5% | 189.8 | 390.6 |

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

