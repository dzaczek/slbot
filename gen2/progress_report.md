# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-16 16:54:55  
**Total Episodes:** 833  
**Training Sessions:** 7

## Verdict: NOT LEARNING (Confidence: 55%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

### Warnings
- Rewards flat: change = 35.9

### Positive Signals
- Food collection improving (slope=0.0297/ep)
- Epsilon low (0.084) - exploiting policy

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 214 | 141.4 | 125.9 | 39.6 | 0.5520 | 0.9% | 80.8% | 17.8% |
| S2 | WALL_AVOID | 400 | 486.0 | 269.1 | 70.2 | 0.4690 | 0.2% | 69.8% | 29.0% |
| S3 | ENEMY_AVOID | 219 | 318.1 | 299.2 | 75.9 | 0.5337 | 0.0% | 98.6% | 1.4% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 353.34 | 318.79 | -144.80 | 110.84 | 253.65 | 528.69 | 913.88 | 2629.01 |
| Steps | 240.21 | 252.36 | 1.00 | 64.00 | 163.00 | 343.00 | 500.00 | 2000.00 |
| Food | 63.85 | 46.82 | 0.00 | 29.00 | 55.00 | 87.00 | 132.40 | 415.00 |
| Loss | 4.49 | 5.26 | 0.00 | 1.88 | 3.16 | 4.74 | 14.21 | 40.93 |
| Food/Step | 0.51 | 0.70 | 0.00 | 0.23 | 0.30 | 0.49 | 1.24 | 6.67 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 293.97 | 216.38 | +0.9433 | 0.0040 |
| Last 100 | 246.80 | 195.21 | +1.6223 | 0.0575 |
| Last 200 | 342.93 | 248.68 | -1.5859 | 0.1356 |
| Last 500 | 401.45 | 331.78 | -0.5654 | 0.0605 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 3 | 0.4% | 214.3 | 233.6 |
| SnakeCollision | 668 | 80.2% | 184.3 | 264.8 |
| MaxSteps | 157 | 18.8% | 480.3 | 731.1 |
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

