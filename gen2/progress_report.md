# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-16 15:42:55  
**Total Episodes:** 542  
**Training Sessions:** 7

## Verdict: LEARNING (Confidence: 90%)

**Goal Feasibility:** LIKELY (>60%)

### Warnings
- Loss very high (13.85) - unstable

### Positive Signals
- Rewards improving: +236.2
- Positive reward trend (slope=0.6953, R²=0.097)
- Episodes getting longer (slope=0.540/ep)
- Food collection improving (slope=0.1114/ep)
- Epsilon low (0.100) - exploiting policy

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 214 | 141.4 | 125.9 | 39.6 | 0.5520 | 0.9% | 80.8% | 17.8% |
| S2 | WALL_AVOID | 214 | 518.7 | 286.4 | 73.1 | 0.4577 | 0.0% | 64.0% | 34.1% |
| S3 | ENEMY_AVOID | 114 | 390.8 | 383.2 | 91.7 | 0.5205 | 0.0% | 97.4% | 2.6% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 342.86 | 348.72 | -144.80 | 88.26 | 210.43 | 486.92 | 985.21 | 2629.01 |
| Steps | 243.39 | 277.48 | 1.00 | 59.00 | 155.50 | 318.75 | 505.70 | 2000.00 |
| Food | 63.75 | 53.01 | 0.00 | 27.00 | 51.50 | 85.75 | 140.90 | 415.00 |
| Loss | 4.71 | 6.06 | 0.00 | 1.74 | 2.89 | 4.74 | 15.96 | 40.93 |
| Food/Step | 0.51 | 0.73 | 0.00 | 0.24 | 0.30 | 0.47 | 1.19 | 6.67 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 152.66 | 106.83 | +2.6113 | 0.1244 |
| Last 100 | 288.61 | 350.06 | -4.2491 | 0.1228 |
| Last 200 | 431.99 | 416.49 | -2.7329 | 0.1435 |
| Last 500 | 364.57 | 354.08 | +0.6022 | 0.0603 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 2 | 0.4% | 277.0 | 251.0 |
| SnakeCollision | 421 | 77.7% | 181.7 | 242.9 |
| MaxSteps | 114 | 21.0% | 472.8 | 711.4 |
| BrowserError | 5 | 0.9% | 189.8 | 390.6 |

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

