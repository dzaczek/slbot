# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-16 14:41:13  
**Total Episodes:** 357  
**Training Sessions:** 5

## Verdict: LEARNING (Confidence: 90%)

**Goal Feasibility:** LIKELY (>60%)

### Positive Signals
- Rewards improving: +379.4
- Positive reward trend (slope=1.8849, R²=0.428)
- Episodes getting longer (slope=0.963/ep)
- Food collection improving (slope=0.1942/ep)

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 214 | 141.4 | 125.9 | 39.6 | 0.5520 | 0.9% | 80.8% | 17.8% |
| S2 | WALL_AVOID | 143 | 552.3 | 303.8 | 77.0 | 0.3326 | 0.0% | 60.8% | 36.4% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 306.03 | 296.94 | -144.80 | 81.75 | 186.04 | 419.34 | 924.90 | 1203.89 |
| Steps | 197.15 | 164.51 | 1.00 | 56.00 | 150.00 | 300.00 | 500.00 | 500.00 |
| Food | 54.55 | 36.10 | 0.00 | 24.00 | 45.00 | 78.00 | 123.80 | 163.00 |
| Loss | 3.90 | 5.55 | 0.00 | 1.32 | 2.57 | 3.99 | 12.25 | 40.93 |
| Food/Step | 0.46 | 0.60 | 0.00 | 0.25 | 0.30 | 0.45 | 1.09 | 6.67 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 619.08 | 267.83 | +0.5829 | 0.0010 |
| Last 100 | 521.13 | 290.86 | +3.6878 | 0.1339 |
| Last 200 | 464.85 | 309.47 | +2.0734 | 0.1496 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 2 | 0.6% | 277.0 | 251.0 |
| SnakeCollision | 260 | 72.8% | 121.1 | 190.9 |
| MaxSteps | 90 | 25.2% | 415.6 | 635.3 |
| BrowserError | 5 | 1.4% | 189.8 | 390.6 |

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

