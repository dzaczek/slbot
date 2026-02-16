# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-16 14:20:38  
**Total Episodes:** 288  
**Training Sessions:** 5

## Verdict: LEARNING (Confidence: 90%)

**Goal Feasibility:** LIKELY (>60%)

### Positive Signals
- Rewards improving: +285.5
- Positive reward trend (slope=1.9420, R²=0.366)
- Episodes getting longer (slope=1.007/ep)
- Food collection improving (slope=0.2244/ep)

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 214 | 141.4 | 125.9 | 39.6 | 0.5520 | 0.9% | 80.8% | 17.8% |
| S2 | WALL_AVOID | 74 | 542.1 | 284.0 | 76.9 | 0.3305 | 0.0% | 58.1% | 36.5% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 244.40 | 266.93 | -144.80 | 70.75 | 149.80 | 316.03 | 927.16 | 1203.89 |
| Steps | 166.55 | 146.37 | 1.00 | 49.50 | 118.50 | 297.25 | 500.00 | 500.00 |
| Food | 49.16 | 34.69 | 0.00 | 22.75 | 37.00 | 72.25 | 125.60 | 163.00 |
| Loss | 4.11 | 6.12 | 0.00 | 1.22 | 2.47 | 4.22 | 14.86 | 40.93 |
| Food/Step | 0.50 | 0.64 | 0.00 | 0.27 | 0.32 | 0.48 | 1.14 | 6.67 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 509.96 | 322.95 | -1.5112 | 0.0046 |
| Last 100 | 473.97 | 328.27 | +2.2741 | 0.0400 |
| Last 200 | 311.59 | 293.51 | +2.8707 | 0.3189 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 2 | 0.7% | 277.0 | 251.0 |
| SnakeCollision | 216 | 75.0% | 99.8 | 147.1 |
| MaxSteps | 65 | 22.6% | 383.1 | 556.2 |
| BrowserError | 5 | 1.7% | 189.8 | 390.6 |

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

