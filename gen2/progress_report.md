# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-16 15:12:04  
**Total Episodes:** 444  
**Training Sessions:** 5

## Verdict: LEARNING (Confidence: 95%)

**Goal Feasibility:** LIKELY (>60%)

### Positive Signals
- Rewards improving: +402.4
- Positive reward trend (slope=1.5209, R²=0.316)
- Episodes getting longer (slope=1.022/ep)
- Food collection improving (slope=0.1955/ep)
- Epsilon low (0.123) - exploiting policy

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 214 | 141.4 | 125.9 | 39.6 | 0.5520 | 0.9% | 80.8% | 17.8% |
| S2 | WALL_AVOID | 200 | 538.5 | 299.5 | 75.3 | 0.4394 | 0.0% | 61.5% | 36.5% |
| S3 | ENEMY_AVOID | 30 | 664.0 | 685.7 | 143.1 | 0.2663 | 0.0% | 93.3% | 6.7% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 355.59 | 346.55 | -144.80 | 89.36 | 235.75 | 553.13 | 981.65 | 2629.01 |
| Steps | 241.94 | 251.57 | 1.00 | 60.00 | 164.50 | 338.00 | 500.00 | 2000.00 |
| Food | 62.65 | 49.29 | 0.00 | 26.00 | 51.00 | 90.00 | 133.00 | 415.00 |
| Loss | 3.83 | 5.04 | 0.00 | 1.61 | 2.69 | 4.18 | 11.06 | 40.93 |
| Food/Step | 0.48 | 0.71 | 0.00 | 0.24 | 0.29 | 0.44 | 1.14 | 6.67 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 577.65 | 525.89 | +3.5991 | 0.0098 |
| Last 100 | 568.95 | 427.54 | -0.3036 | 0.0004 |
| Last 200 | 543.59 | 370.38 | +0.6032 | 0.0088 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 2 | 0.5% | 277.0 | 251.0 |
| SnakeCollision | 324 | 73.0% | 166.7 | 235.0 |
| MaxSteps | 113 | 25.5% | 459.3 | 701.6 |
| BrowserError | 5 | 1.1% | 189.8 | 390.6 |

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

