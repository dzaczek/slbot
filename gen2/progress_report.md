# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-16 17:05:13  
**Total Episodes:** 869  
**Training Sessions:** 7

## Verdict: NOT LEARNING (Confidence: 55%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

### Warnings
- Rewards flat: change = 14.3

### Positive Signals
- Food collection improving (slope=0.0279/ep)
- Epsilon low (0.083) - exploiting policy

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 214 | 141.4 | 125.9 | 39.6 | 0.5520 | 0.9% | 80.8% | 17.8% |
| S2 | WALL_AVOID | 400 | 486.0 | 269.1 | 70.2 | 0.4690 | 0.2% | 69.8% | 29.0% |
| S3 | ENEMY_AVOID | 255 | 331.6 | 297.4 | 75.1 | 0.5158 | 0.4% | 98.4% | 1.2% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 355.84 | 319.21 | -144.80 | 111.03 | 256.06 | 537.82 | 916.27 | 2629.01 |
| Steps | 242.11 | 253.16 | 1.00 | 64.00 | 164.00 | 348.00 | 534.60 | 2000.00 |
| Food | 64.11 | 46.71 | 0.00 | 30.00 | 56.00 | 88.00 | 134.00 | 415.00 |
| Loss | 4.58 | 5.21 | 0.00 | 1.94 | 3.24 | 5.01 | 14.21 | 40.93 |
| Food/Step | 0.50 | 0.69 | 0.00 | 0.23 | 0.30 | 0.49 | 1.18 | 6.67 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 389.04 | 305.99 | +0.6416 | 0.0009 |
| Last 100 | 327.01 | 263.98 | +2.0659 | 0.0510 |
| Last 200 | 331.55 | 256.76 | -0.4305 | 0.0094 |
| Last 500 | 382.91 | 327.77 | -0.3081 | 0.0184 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 4 | 0.5% | 216.0 | 248.8 |
| SnakeCollision | 703 | 80.9% | 189.4 | 272.4 |
| MaxSteps | 157 | 18.1% | 480.3 | 731.1 |
| BrowserError | 5 | 0.6% | 189.8 | 390.6 |

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

