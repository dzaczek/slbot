# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-16 16:24:02  
**Total Episodes:** 705  
**Training Sessions:** 7

## Verdict: LEARNING (Confidence: 95%)

**Goal Feasibility:** LIKELY (>60%)

### Positive Signals
- Rewards improving: +137.2
- Positive reward trend (slope=0.5056, R²=0.096)
- Episodes getting longer (slope=0.280/ep)
- Food collection improving (slope=0.0584/ep)
- Epsilon low (0.087) - exploiting policy

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 214 | 141.4 | 125.9 | 39.6 | 0.5520 | 0.9% | 80.8% | 17.8% |
| S2 | WALL_AVOID | 377 | 494.1 | 275.1 | 71.0 | 0.4575 | 0.0% | 68.7% | 30.2% |
| S3 | ENEMY_AVOID | 114 | 390.8 | 383.2 | 91.7 | 0.5205 | 0.0% | 97.4% | 2.6% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 370.33 | 332.78 | -144.80 | 110.95 | 273.24 | 568.47 | 926.90 | 2629.01 |
| Steps | 247.31 | 258.31 | 1.00 | 67.00 | 170.00 | 361.00 | 500.00 | 2000.00 |
| Food | 64.78 | 48.71 | 0.00 | 29.00 | 56.00 | 91.00 | 132.80 | 415.00 |
| Loss | 4.52 | 5.55 | 0.00 | 1.76 | 3.03 | 4.58 | 15.11 | 40.93 |
| Food/Step | 0.50 | 0.71 | 0.00 | 0.23 | 0.30 | 0.47 | 1.14 | 6.67 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 508.82 | 239.43 | -2.5794 | 0.0242 |
| Last 100 | 500.17 | 257.28 | -0.5473 | 0.0038 |
| Last 200 | 405.68 | 261.18 | +1.8347 | 0.1645 |
| Last 500 | 466.57 | 345.62 | -0.2715 | 0.0129 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 2 | 0.3% | 277.0 | 251.0 |
| SnakeCollision | 543 | 77.0% | 181.3 | 267.8 |
| MaxSteps | 155 | 22.0% | 480.0 | 730.3 |
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

