# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-16 16:34:17  
**Total Episodes:** 749  
**Training Sessions:** 7

## Verdict: LEARNING (Confidence: 95%)

**Goal Feasibility:** LIKELY (>60%)

### Positive Signals
- Rewards improving: +80.3
- Positive reward trend (slope=0.3799, R²=0.063)
- Episodes getting longer (slope=0.212/ep)
- Food collection improving (slope=0.0460/ep)
- Epsilon low (0.086) - exploiting policy

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 214 | 141.4 | 125.9 | 39.6 | 0.5520 | 0.9% | 80.8% | 17.8% |
| S2 | WALL_AVOID | 400 | 486.0 | 269.1 | 70.2 | 0.4690 | 0.2% | 69.8% | 29.0% |
| S3 | ENEMY_AVOID | 135 | 360.5 | 359.2 | 86.5 | 0.5162 | 0.0% | 97.8% | 2.2% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 364.94 | 327.99 | -144.80 | 110.84 | 266.96 | 564.09 | 919.50 | 2629.01 |
| Steps | 244.40 | 257.99 | 1.00 | 64.00 | 164.00 | 354.00 | 500.00 | 2000.00 |
| Food | 64.42 | 48.24 | 0.00 | 29.00 | 56.00 | 91.00 | 133.00 | 415.00 |
| Loss | 4.44 | 5.40 | 0.00 | 1.80 | 3.02 | 4.55 | 14.33 | 40.93 |
| Food/Step | 0.50 | 0.70 | 0.00 | 0.23 | 0.30 | 0.48 | 1.15 | 6.67 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 303.24 | 224.71 | -6.1096 | 0.1539 |
| Last 100 | 392.96 | 258.93 | -3.3518 | 0.1396 |
| Last 200 | 426.26 | 259.78 | -0.7100 | 0.0249 |
| Last 500 | 443.72 | 338.59 | -0.3795 | 0.0262 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 3 | 0.4% | 214.3 | 233.6 |
| SnakeCollision | 584 | 78.0% | 181.6 | 266.9 |
| MaxSteps | 157 | 21.0% | 480.3 | 731.1 |
| BrowserError | 5 | 0.7% | 189.8 | 390.6 |

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 415 | 6,000 | 6.9% |
| Survival | 2000 steps | 1,800 steps | 111.1% |

## Recommendations

Training looks healthy. Continue and monitor.

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

