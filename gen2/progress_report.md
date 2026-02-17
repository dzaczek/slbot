# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-17 15:07:24  
**Total Episodes:** 3678  
**Training Sessions:** 15

## Verdict: NOT LEARNING (Confidence: 50%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

### Critical Issues
- Rewards DECLINING: -1611.7

### Positive Signals
- Episodes getting longer (slope=0.091/ep)
- Epsilon low (0.080) - exploiting policy

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 723 | 172.9 | 165.7 | 58.8 | 0.5998 | 0.4% | 70.4% | 29.0% |
| S2 | WALL_AVOID | 400 | 486.0 | 269.1 | 70.2 | 0.4690 | 0.2% | 69.8% | 29.0% |
| S3 | ENEMY_AVOID | 714 | 16.7 | 276.3 | 64.5 | 0.6791 | 0.1% | 96.8% | 3.1% |
| S4 | MASS_MANAGEMENT | 1841 | -958.3 | 440.3 | 59.1 | 0.6278 | 0.2% | 87.2% | 12.6% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | -389.59 | 3633.55 | -35299.32 | 109.11 | 274.69 | 471.02 | 1041.92 | 4632.57 |
| Steps | 335.86 | 499.23 | 1.00 | 69.00 | 165.00 | 320.00 | 2000.00 | 2000.00 |
| Food | 61.32 | 41.82 | 0.00 | 31.00 | 56.00 | 83.00 | 130.15 | 418.00 |
| Loss | 7.90 | 10.37 | 0.00 | 2.43 | 5.13 | 9.54 | 22.45 | 159.03 |
| Food/Step | 0.61 | 0.96 | 0.00 | 0.24 | 0.36 | 0.57 | 1.86 | 11.00 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 208.03 | 152.20 | +3.1758 | 0.0907 |
| Last 100 | 209.93 | 145.46 | +0.4593 | 0.0083 |
| Last 200 | 157.57 | 144.25 | +1.0172 | 0.1657 |
| Last 500 | 185.39 | 151.03 | -0.1415 | 0.0183 |
| Last 1000 | -1878.79 | 5636.44 | +3.7490 | 0.0369 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 8 | 0.2% | 343.6 | 515.3 |
| SnakeCollision | 3084 | 83.8% | 195.0 | 379.1 |
| MaxSteps | 580 | 15.8% | 1084.5 | -4487.3 |
| InvalidFrame | 1 | 0.0% | 1197.0 | -5534.5 |
| BrowserError | 5 | 0.1% | 189.8 | 390.6 |

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 418 | 6,000 | 7.0% |
| Survival | 2000 steps | 1,800 steps | 111.1% |

## Recommendations

Fine-tune hyperparameters, increase training duration.

1. Average reward negative. Reduce penalties or boost food_reward.

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

