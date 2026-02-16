# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-16 21:12:41  
**Total Episodes:** 1765  
**Training Sessions:** 7

## Verdict: LEARNING (Confidence: 65%)

**Goal Feasibility:** POSSIBLE (25-60%)

### Warnings
- Loss very high (19.85) - unstable

### Positive Signals
- Rewards improving: +107.6
- Epsilon low (0.080) - exploiting policy

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 214 | 141.4 | 125.9 | 39.6 | 0.5520 | 0.9% | 80.8% | 17.8% |
| S2 | WALL_AVOID | 400 | 486.0 | 269.1 | 70.2 | 0.4690 | 0.2% | 69.8% | 29.0% |
| S3 | ENEMY_AVOID | 414 | 349.3 | 279.2 | 71.8 | 0.5430 | 0.2% | 99.0% | 0.7% |
| S4 | MASS_MANAGEMENT | 737 | 488.3 | 227.8 | 66.0 | 0.6734 | 0.4% | 99.5% | 0.1% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 413.10 | 636.71 | -20947.60 | 156.24 | 325.30 | 586.86 | 1101.42 | 3071.88 |
| Steps | 236.86 | 248.98 | 1.00 | 64.00 | 161.00 | 325.00 | 662.60 | 2000.00 |
| Food | 65.12 | 42.73 | 0.00 | 34.00 | 58.00 | 86.00 | 140.00 | 415.00 |
| Loss | 7.72 | 7.61 | 0.00 | 2.87 | 5.49 | 10.10 | 20.81 | 82.89 |
| Food/Step | 0.58 | 0.83 | 0.00 | 0.24 | 0.33 | 0.56 | 1.58 | 9.60 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 484.71 | 411.84 | -7.1889 | 0.0635 |
| Last 100 | 543.17 | 481.95 | -3.5611 | 0.0455 |
| Last 200 | 492.10 | 411.53 | +0.1553 | 0.0005 |
| Last 500 | 490.98 | 1060.18 | +0.0822 | 0.0001 |
| Last 1000 | 453.09 | 794.10 | +0.1798 | 0.0043 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 7 | 0.4% | 361.3 | 558.1 |
| SnakeCollision | 1595 | 90.4% | 211.4 | 394.6 |
| MaxSteps | 158 | 9.0% | 489.9 | 593.9 |
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

