# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-16 11:07:01  
**Total Episodes:** 178  
**Training Sessions:** 2

## Verdict: NOT LEARNING (Confidence: 50%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 178 | 95.6 | 75.0 | 31.1 | 0.6660 | 10.1% | 89.3% | 0.6% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 95.60 | 55.95 | -15.00 | 52.06 | 92.44 | 128.29 | 206.23 | 254.50 |
| Steps | 75.02 | 58.36 | 1.00 | 26.50 | 60.00 | 107.50 | 192.05 | 300.00 |
| Food | 31.06 | 14.21 | 0.00 | 21.00 | 29.00 | 41.00 | 58.15 | 76.00 |
| Loss | 4.71 | 2.43 | 0.33 | 2.79 | 4.58 | 6.12 | 9.27 | 11.58 |
| Food/Step | 0.67 | 0.59 | 0.00 | 0.37 | 0.47 | 0.66 | 2.03 | 3.80 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 93.94 | 62.53 | +1.2143 | 0.0785 |
| Last 100 | 98.80 | 58.87 | -0.0867 | 0.0018 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 18 | 10.1% | 146.4 | 140.7 |
| SnakeCollision | 159 | 89.3% | 65.5 | 89.5 |
| MaxSteps | 1 | 0.6% | 300.0 | 254.5 |

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 76 | 6,000 | 1.3% |
| Survival | 300 steps | 1,800 steps | 16.7% |

## Recommendations

Fine-tune hyperparameters, increase training duration.

1. Epsilon 0.705 still high. Consider faster decay.

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

