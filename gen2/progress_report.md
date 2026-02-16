# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-16 17:46:22  
**Total Episodes:** 1015  
**Training Sessions:** 7

## Verdict: NOT LEARNING (Confidence: 55%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

### Warnings
- Rewards flat: change = 5.8

### Positive Signals
- Food collection improving (slope=0.0182/ep)
- Epsilon low (0.081) - exploiting policy

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 214 | 141.4 | 125.9 | 39.6 | 0.5520 | 0.9% | 80.8% | 17.8% |
| S2 | WALL_AVOID | 400 | 486.0 | 269.1 | 70.2 | 0.4690 | 0.2% | 69.8% | 29.0% |
| S3 | ENEMY_AVOID | 401 | 346.1 | 278.3 | 71.6 | 0.5416 | 0.2% | 99.0% | 0.7% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 358.07 | 319.48 | -144.80 | 113.64 | 262.45 | 520.76 | 917.82 | 2629.01 |
| Steps | 242.52 | 255.66 | 1.00 | 63.00 | 164.00 | 343.50 | 581.30 | 2000.00 |
| Food | 64.33 | 45.94 | 0.00 | 30.00 | 56.00 | 89.00 | 135.00 | 415.00 |
| Loss | 4.75 | 5.05 | 0.00 | 2.08 | 3.46 | 5.51 | 13.33 | 40.93 |
| Food/Step | 0.52 | 0.69 | 0.00 | 0.23 | 0.31 | 0.50 | 1.33 | 6.67 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 312.22 | 233.18 | +1.0361 | 0.0041 |
| Last 100 | 355.69 | 273.25 | -1.7825 | 0.0355 |
| Last 200 | 373.06 | 314.92 | -0.4960 | 0.0083 |
| Last 500 | 364.90 | 278.57 | -0.0773 | 0.0016 |
| Last 1000 | 362.52 | 319.75 | +0.1089 | 0.0097 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 4 | 0.4% | 216.0 | 248.8 |
| SnakeCollision | 849 | 83.6% | 199.0 | 289.4 |
| MaxSteps | 157 | 15.5% | 480.3 | 731.1 |
| BrowserError | 5 | 0.5% | 189.8 | 390.6 |

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 415 | 6,000 | 6.9% |
| Survival | 2000 steps | 1,800 steps | 111.1% |

## Recommendations

Fine-tune hyperparameters, increase training duration.

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

