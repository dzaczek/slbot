# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-16 15:32:36  
**Total Episodes:** 494  
**Training Sessions:** 6

## Verdict: LEARNING (Confidence: 95%)

**Goal Feasibility:** LIKELY (>60%)

### Positive Signals
- Rewards improving: +316.1
- Positive reward trend (slope=1.1591, R²=0.213)
- Episodes getting longer (slope=0.884/ep)
- Food collection improving (slope=0.1729/ep)
- Epsilon low (0.105) - exploiting policy

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 214 | 141.4 | 125.9 | 39.6 | 0.5520 | 0.9% | 80.8% | 17.8% |
| S2 | WALL_AVOID | 200 | 538.5 | 299.5 | 75.3 | 0.4394 | 0.0% | 61.5% | 36.5% |
| S3 | ENEMY_AVOID | 80 | 505.4 | 495.4 | 111.4 | 0.4291 | 0.0% | 96.2% | 3.8% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 361.11 | 358.50 | -144.80 | 89.90 | 237.75 | 551.85 | 1013.25 | 2629.01 |
| Steps | 256.04 | 285.81 | 1.00 | 62.25 | 167.00 | 343.75 | 534.35 | 2000.00 |
| Food | 65.66 | 54.68 | 0.00 | 26.00 | 53.00 | 93.00 | 145.05 | 415.00 |
| Loss | 3.89 | 5.12 | 0.00 | 1.63 | 2.74 | 4.27 | 10.71 | 40.93 |
| Food/Step | 0.49 | 0.73 | 0.00 | 0.24 | 0.29 | 0.44 | 1.12 | 6.67 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 410.19 | 447.96 | -5.3325 | 0.0295 |
| Last 100 | 493.92 | 495.61 | -2.7287 | 0.0253 |
| Last 200 | 531.59 | 406.04 | -1.0131 | 0.0208 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 2 | 0.4% | 277.0 | 251.0 |
| SnakeCollision | 373 | 75.5% | 190.6 | 254.3 |
| MaxSteps | 114 | 23.1% | 472.8 | 711.4 |
| BrowserError | 5 | 1.0% | 189.8 | 390.6 |

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

