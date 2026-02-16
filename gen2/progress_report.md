# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-16 16:03:29  
**Total Episodes:** 630  
**Training Sessions:** 7

## Verdict: LEARNING (Confidence: 95%)

**Goal Feasibility:** LIKELY (>60%)

### Positive Signals
- Rewards improving: +180.8
- Positive reward trend (slope=0.5564, R²=0.090)
- Episodes getting longer (slope=0.342/ep)
- Food collection improving (slope=0.0736/ep)
- Epsilon low (0.092) - exploiting policy

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 214 | 141.4 | 125.9 | 39.6 | 0.5520 | 0.9% | 80.8% | 17.8% |
| S2 | WALL_AVOID | 302 | 494.9 | 272.2 | 70.9 | 0.4702 | 0.0% | 68.2% | 30.5% |
| S3 | ENEMY_AVOID | 114 | 390.8 | 383.2 | 91.7 | 0.5205 | 0.0% | 97.4% | 2.6% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 355.99 | 338.22 | -144.80 | 101.52 | 244.89 | 534.83 | 959.25 | 2629.01 |
| Steps | 242.57 | 265.49 | 1.00 | 62.25 | 163.50 | 330.75 | 500.00 | 2000.00 |
| Food | 64.00 | 50.38 | 0.00 | 28.00 | 54.00 | 87.00 | 134.00 | 415.00 |
| Loss | 4.72 | 5.82 | 0.00 | 1.79 | 3.06 | 4.86 | 15.83 | 40.93 |
| Food/Step | 0.51 | 0.74 | 0.00 | 0.24 | 0.30 | 0.47 | 1.16 | 6.67 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 471.57 | 262.33 | +1.9975 | 0.0121 |
| Last 100 | 414.31 | 246.37 | +2.6804 | 0.0986 |
| Last 200 | 377.37 | 350.44 | +0.0884 | 0.0002 |
| Last 500 | 422.44 | 348.52 | +0.1026 | 0.0018 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 2 | 0.3% | 277.0 | 251.0 |
| SnakeCollision | 490 | 77.8% | 179.4 | 256.0 |
| MaxSteps | 133 | 21.1% | 476.7 | 724.6 |
| BrowserError | 5 | 0.8% | 189.8 | 390.6 |

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

