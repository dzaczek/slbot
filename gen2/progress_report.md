# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-16 18:06:58  
**Total Episodes:** 1089  
**Training Sessions:** 7

## Verdict: NOT LEARNING (Confidence: 55%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

### Warnings
- Rewards flat: change = 45.9

### Positive Signals
- Food collection improving (slope=0.0155/ep)
- Epsilon low (0.081) - exploiting policy

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 214 | 141.4 | 125.9 | 39.6 | 0.5520 | 0.9% | 80.8% | 17.8% |
| S2 | WALL_AVOID | 400 | 486.0 | 269.1 | 70.2 | 0.4690 | 0.2% | 69.8% | 29.0% |
| S3 | ENEMY_AVOID | 414 | 349.3 | 279.2 | 71.8 | 0.5430 | 0.2% | 99.0% | 0.7% |
| S4 | MASS_MANAGEMENT | 61 | 482.8 | 225.5 | 64.4 | 0.5820 | 0.0% | 100.0% | 0.0% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 366.14 | 329.52 | -144.80 | 118.58 | 271.83 | 530.32 | 926.77 | 3071.88 |
| Steps | 242.36 | 255.75 | 1.00 | 62.00 | 164.00 | 345.00 | 603.00 | 2000.00 |
| Food | 64.47 | 45.39 | 0.00 | 31.00 | 56.00 | 88.00 | 134.60 | 415.00 |
| Loss | 4.83 | 4.96 | 0.00 | 2.12 | 3.55 | 5.62 | 13.20 | 40.93 |
| Food/Step | 0.52 | 0.68 | 0.00 | 0.23 | 0.31 | 0.51 | 1.44 | 6.67 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 489.62 | 482.74 | -0.1004 | 0.0000 |
| Last 100 | 434.83 | 392.45 | +2.2555 | 0.0275 |
| Last 200 | 390.39 | 341.12 | +0.9434 | 0.0255 |
| Last 500 | 386.26 | 313.28 | +0.0070 | 0.0000 |
| Last 1000 | 390.48 | 332.65 | +0.0306 | 0.0007 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 4 | 0.4% | 216.0 | 248.8 |
| SnakeCollision | 923 | 84.8% | 202.3 | 304.4 |
| MaxSteps | 157 | 14.4% | 480.3 | 731.1 |
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

