# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-16 20:31:27  
**Total Episodes:** 1609  
**Training Sessions:** 7

## Verdict: LEARNING (Confidence: 65%)

**Goal Feasibility:** POSSIBLE (25-60%)

### Warnings
- Loss very high (12.43) - unstable

### Positive Signals
- Rewards improving: +97.1
- Epsilon low (0.080) - exploiting policy

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 214 | 141.4 | 125.9 | 39.6 | 0.5520 | 0.9% | 80.8% | 17.8% |
| S2 | WALL_AVOID | 400 | 486.0 | 269.1 | 70.2 | 0.4690 | 0.2% | 69.8% | 29.0% |
| S3 | ENEMY_AVOID | 414 | 349.3 | 279.2 | 71.8 | 0.5430 | 0.2% | 99.0% | 0.7% |
| S4 | MASS_MANAGEMENT | 581 | 482.1 | 229.9 | 66.1 | 0.6470 | 0.3% | 99.5% | 0.2% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 403.60 | 652.05 | -20947.60 | 148.98 | 316.66 | 582.07 | 1082.75 | 3071.88 |
| Steps | 238.50 | 251.32 | 1.00 | 64.00 | 162.00 | 330.00 | 655.60 | 2000.00 |
| Food | 65.06 | 43.23 | 0.00 | 33.00 | 58.00 | 86.00 | 139.60 | 415.00 |
| Loss | 6.86 | 6.43 | 0.00 | 2.69 | 4.91 | 9.22 | 18.42 | 73.76 |
| Food/Step | 0.56 | 0.77 | 0.00 | 0.24 | 0.33 | 0.56 | 1.55 | 9.60 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 440.04 | 317.41 | -6.2099 | 0.0797 |
| Last 100 | 535.85 | 452.12 | -3.2227 | 0.0423 |
| Last 200 | 543.19 | 454.42 | -0.7984 | 0.0103 |
| Last 500 | 476.43 | 1053.41 | +0.1104 | 0.0002 |
| Last 1000 | 436.59 | 781.99 | +0.1463 | 0.0029 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 6 | 0.4% | 352.8 | 554.9 |
| SnakeCollision | 1440 | 89.5% | 210.6 | 382.1 |
| MaxSteps | 158 | 9.8% | 489.9 | 593.9 |
| BrowserError | 5 | 0.3% | 189.8 | 390.6 |

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 415 | 6,000 | 6.9% |
| Survival | 2000 steps | 1,800 steps | 111.1% |

## Recommendations

Keep training. Monitor for sustained improvement.

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

