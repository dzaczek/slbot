# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-16 14:51:30  
**Total Episodes:** 397  
**Training Sessions:** 5

## Verdict: LEARNING (Confidence: 90%)

**Goal Feasibility:** LIKELY (>60%)

### Positive Signals
- Rewards improving: +406.1
- Positive reward trend (slope=1.7021, R²=0.404)
- Episodes getting longer (slope=0.865/ep)
- Food collection improving (slope=0.1713/ep)

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 214 | 141.4 | 125.9 | 39.6 | 0.5520 | 0.9% | 80.8% | 17.8% |
| S2 | WALL_AVOID | 183 | 552.7 | 307.6 | 77.0 | 0.4216 | 0.0% | 59.0% | 38.8% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 331.03 | 307.01 | -144.80 | 86.32 | 212.39 | 528.69 | 920.72 | 1203.89 |
| Steps | 209.68 | 171.80 | 1.00 | 59.00 | 154.00 | 300.00 | 500.00 | 500.00 |
| Food | 56.84 | 36.83 | 0.00 | 25.00 | 47.00 | 84.00 | 122.20 | 163.00 |
| Loss | 3.80 | 5.29 | 0.00 | 1.48 | 2.57 | 3.94 | 11.74 | 40.93 |
| Food/Step | 0.49 | 0.73 | 0.00 | 0.24 | 0.30 | 0.44 | 1.14 | 6.67 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 577.58 | 295.48 | -6.6872 | 0.1067 |
| Last 100 | 577.15 | 288.19 | -0.4840 | 0.0023 |
| Last 200 | 532.59 | 308.87 | +0.6427 | 0.0144 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 2 | 0.5% | 277.0 | 251.0 |
| SnakeCollision | 281 | 70.8% | 124.0 | 199.3 |
| MaxSteps | 109 | 27.5% | 430.3 | 669.4 |
| BrowserError | 5 | 1.3% | 189.8 | 390.6 |

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 163 | 6,000 | 2.7% |
| Survival | 500 steps | 1,800 steps | 27.8% |

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

