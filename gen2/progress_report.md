# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-16 15:01:47  
**Total Episodes:** 423  
**Training Sessions:** 5

## Verdict: LEARNING (Confidence: 95%)

**Goal Feasibility:** LIKELY (>60%)

### Positive Signals
- Rewards improving: +395.6
- Positive reward trend (slope=1.5009, R²=0.342)
- Episodes getting longer (slope=0.841/ep)
- Food collection improving (slope=0.1624/ep)
- Epsilon low (0.140) - exploiting policy

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 214 | 141.4 | 125.9 | 39.6 | 0.5520 | 0.9% | 80.8% | 17.8% |
| S2 | WALL_AVOID | 200 | 538.5 | 299.5 | 75.3 | 0.4394 | 0.0% | 61.5% | 36.5% |
| S3 | ENEMY_AVOID | 9 | 560.4 | 637.6 | 128.7 | 0.2821 | 0.0% | 100.0% | 0.0% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 338.08 | 313.53 | -144.80 | 86.88 | 215.41 | 544.08 | 932.94 | 1383.10 |
| Steps | 218.89 | 197.72 | 1.00 | 58.00 | 158.00 | 300.50 | 500.00 | 1344.00 |
| Food | 58.35 | 40.48 | 0.00 | 25.00 | 48.00 | 84.50 | 126.70 | 259.00 |
| Loss | 3.84 | 5.15 | 0.00 | 1.55 | 2.64 | 4.19 | 11.28 | 40.93 |
| Food/Step | 0.49 | 0.73 | 0.00 | 0.24 | 0.30 | 0.44 | 1.14 | 6.67 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 436.04 | 343.42 | +1.9318 | 0.0066 |
| Last 100 | 544.06 | 326.13 | -2.2776 | 0.0406 |
| Last 200 | 538.43 | 325.40 | -0.2683 | 0.0023 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 2 | 0.5% | 277.0 | 251.0 |
| SnakeCollision | 305 | 72.1% | 141.6 | 216.5 |
| MaxSteps | 111 | 26.2% | 431.5 | 671.3 |
| BrowserError | 5 | 1.2% | 189.8 | 390.6 |

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 259 | 6,000 | 4.3% |
| Survival | 1344 steps | 1,800 steps | 74.7% |

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

