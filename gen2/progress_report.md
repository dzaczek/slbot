# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-16 15:53:13  
**Total Episodes:** 585  
**Training Sessions:** 7

## Verdict: LEARNING (Confidence: 95%)

**Goal Feasibility:** LIKELY (>60%)

### Positive Signals
- Rewards improving: +206.1
- Positive reward trend (slope=0.6015, R²=0.088)
- Episodes getting longer (slope=0.413/ep)
- Food collection improving (slope=0.0886/ep)
- Epsilon low (0.096) - exploiting policy

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 214 | 141.4 | 125.9 | 39.6 | 0.5520 | 0.9% | 80.8% | 17.8% |
| S2 | WALL_AVOID | 257 | 500.3 | 274.8 | 71.4 | 0.4489 | 0.0% | 67.3% | 31.1% |
| S3 | ENEMY_AVOID | 114 | 390.8 | 383.2 | 91.7 | 0.5205 | 0.0% | 97.4% | 2.6% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 347.69 | 342.30 | -144.80 | 92.96 | 230.78 | 490.17 | 980.92 | 2629.01 |
| Steps | 241.43 | 270.87 | 1.00 | 62.00 | 155.00 | 318.00 | 500.00 | 2000.00 |
| Food | 63.72 | 51.64 | 0.00 | 28.00 | 53.00 | 86.00 | 135.80 | 415.00 |
| Loss | 4.84 | 6.01 | 0.00 | 1.78 | 3.05 | 4.97 | 16.42 | 40.93 |
| Food/Step | 0.50 | 0.71 | 0.00 | 0.24 | 0.30 | 0.47 | 1.14 | 6.67 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 381.55 | 235.07 | +7.0146 | 0.1854 |
| Last 100 | 290.69 | 248.11 | +3.0626 | 0.1270 |
| Last 200 | 389.28 | 400.15 | -1.3336 | 0.0370 |
| Last 500 | 391.04 | 351.40 | +0.3527 | 0.0210 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 2 | 0.3% | 277.0 | 251.0 |
| SnakeCollision | 457 | 78.1% | 180.2 | 249.7 |
| MaxSteps | 121 | 20.7% | 474.4 | 717.5 |
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

