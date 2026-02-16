# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-16 11:27:39  
**Total Episodes:** 260  
**Training Sessions:** 5

## Verdict: NOT LEARNING (Confidence: 55%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

### Warnings
- Rewards flat: change = 29.6

### Positive Signals
- Positive reward trend (slope=0.2855, R²=0.081)

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 200 | 95.7 | 76.3 | 31.1 | 0.6631 | 10.5% | 88.5% | 1.0% |
| S2 | WALL_AVOID | 60 | 158.6 | 82.8 | 31.8 | 0.5398 | 10.0% | 90.0% | 0.0% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 110.19 | 75.39 | -16.70 | 57.66 | 102.14 | 145.13 | 227.69 | 703.35 |
| Steps | 77.81 | 60.31 | 1.00 | 29.00 | 65.00 | 106.00 | 195.15 | 315.00 |
| Food | 31.25 | 14.77 | 0.00 | 21.00 | 29.00 | 40.00 | 59.00 | 105.00 |
| Loss | 5.48 | 3.64 | 0.33 | 2.89 | 4.69 | 7.05 | 12.57 | 26.28 |
| Food/Step | 0.63 | 0.57 | 0.00 | 0.36 | 0.45 | 0.61 | 2.00 | 3.80 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 157.35 | 106.61 | +2.3596 | 0.1020 |
| Last 100 | 134.64 | 95.61 | +1.2370 | 0.1395 |
| Last 200 | 115.78 | 80.15 | +0.4251 | 0.0938 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 27 | 10.4% | 146.7 | 137.5 |
| SnakeCollision | 231 | 88.8% | 67.8 | 105.7 |
| MaxSteps | 2 | 0.8% | 300.0 | 258.1 |

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 105 | 6,000 | 1.8% |
| Survival | 315 steps | 1,800 steps | 17.5% |

## Recommendations

Fine-tune hyperparameters, increase training duration.

1. Epsilon 0.547 still high. Consider faster decay.

2. Episodes too short. Reduce death penalties or add survival bonus.

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

