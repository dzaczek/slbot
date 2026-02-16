# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-16 20:00:27  
**Total Episodes:** 1499  
**Training Sessions:** 7

## Verdict: LEARNING (Confidence: 65%)

**Goal Feasibility:** POSSIBLE (25-60%)

### Warnings
- Loss very high (11.78) - unstable

### Positive Signals
- Rewards improving: +56.8
- Epsilon low (0.080) - exploiting policy

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 214 | 141.4 | 125.9 | 39.6 | 0.5520 | 0.9% | 80.8% | 17.8% |
| S2 | WALL_AVOID | 400 | 486.0 | 269.1 | 70.2 | 0.4690 | 0.2% | 69.8% | 29.0% |
| S3 | ENEMY_AVOID | 414 | 349.3 | 279.2 | 71.8 | 0.5430 | 0.2% | 99.0% | 0.7% |
| S4 | MASS_MANAGEMENT | 471 | 467.8 | 229.4 | 65.3 | 0.6320 | 0.4% | 99.4% | 0.2% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 393.35 | 663.58 | -20947.60 | 145.34 | 306.55 | 573.60 | 1034.78 | 3071.88 |
| Steps | 238.96 | 252.83 | 1.00 | 63.00 | 162.00 | 333.00 | 642.30 | 2000.00 |
| Food | 64.75 | 43.48 | 0.00 | 32.50 | 58.00 | 86.00 | 137.00 | 415.00 |
| Loss | 6.34 | 5.63 | 0.00 | 2.58 | 4.56 | 8.29 | 17.52 | 40.93 |
| Food/Step | 0.55 | 0.73 | 0.00 | 0.24 | 0.32 | 0.55 | 1.54 | 6.67 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 469.89 | 340.79 | +5.3888 | 0.0521 |
| Last 100 | 329.95 | 2190.33 | +8.4088 | 0.0123 |
| Last 200 | 434.81 | 1582.33 | -0.3678 | 0.0002 |
| Last 500 | 463.57 | 1052.10 | -0.0681 | 0.0001 |
| Last 1000 | 410.62 | 771.62 | +0.1526 | 0.0033 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 6 | 0.4% | 352.8 | 554.9 |
| SnakeCollision | 1330 | 88.7% | 208.8 | 368.8 |
| MaxSteps | 158 | 10.5% | 489.9 | 593.9 |
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

