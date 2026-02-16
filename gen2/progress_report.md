# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-16 11:17:20  
**Total Episodes:** 213  
**Training Sessions:** 4

## Verdict: NOT LEARNING (Confidence: 45%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

### Warnings
- Rewards flat: change = 5.5

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 200 | 95.7 | 76.3 | 31.1 | 0.6631 | 10.5% | 88.5% | 1.0% |
| S2 | WALL_AVOID | 13 | 144.2 | 81.4 | 29.5 | 0.5887 | 7.7% | 92.3% | 0.0% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 98.62 | 60.65 | -15.00 | 52.65 | 92.51 | 132.37 | 209.69 | 349.67 |
| Steps | 76.63 | 59.65 | 1.00 | 28.00 | 65.00 | 108.00 | 196.20 | 300.00 |
| Food | 30.99 | 14.27 | 0.00 | 21.00 | 29.00 | 40.00 | 59.00 | 76.00 |
| Loss | 4.75 | 2.99 | 0.33 | 2.53 | 4.30 | 6.12 | 9.67 | 26.28 |
| Food/Step | 0.66 | 0.61 | 0.00 | 0.37 | 0.44 | 0.64 | 2.09 | 3.80 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 112.87 | 76.27 | +1.4384 | 0.0741 |
| Last 100 | 99.41 | 66.96 | +0.6426 | 0.0767 |
| Last 200 | 98.49 | 60.23 | +0.1257 | 0.0145 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 22 | 10.3% | 146.6 | 130.2 |
| SnakeCollision | 189 | 88.7% | 66.1 | 93.3 |
| MaxSteps | 2 | 0.9% | 300.0 | 258.1 |

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 76 | 6,000 | 1.3% |
| Survival | 300 steps | 1,800 steps | 16.7% |

## Recommendations

Fine-tune hyperparameters, increase training duration.

1. Epsilon 0.629 still high. Consider faster decay.

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

