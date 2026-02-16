# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-16 12:59:13  
**Total Episodes:** 130  
**Training Sessions:** 3

## Verdict: NOT LEARNING (Confidence: 55%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

### Positive Signals
- Food collection improving (slope=0.0799/ep)

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 130 | 100.4 | 83.3 | 30.5 | 0.6745 | 0.8% | 96.2% | 3.1% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 100.41 | 70.05 | -15.00 | 54.09 | 84.85 | 129.71 | 244.74 | 311.49 |
| Steps | 83.25 | 70.51 | 1.00 | 31.00 | 63.50 | 119.25 | 239.30 | 300.00 |
| Food | 30.52 | 16.44 | 0.00 | 21.00 | 27.00 | 37.75 | 61.10 | 85.00 |
| Loss | 6.49 | 8.36 | 0.00 | 1.86 | 3.71 | 5.96 | 22.35 | 40.93 |
| Food/Step | 0.67 | 0.89 | 0.00 | 0.30 | 0.41 | 0.70 | 1.75 | 6.67 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 110.38 | 78.69 | +0.0878 | 0.0003 |
| Last 100 | 106.63 | 72.99 | +0.1412 | 0.0031 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 1 | 0.8% | 262.0 | 299.6 |
| SnakeCollision | 125 | 96.2% | 74.9 | 92.9 |
| MaxSteps | 4 | 3.1% | 300.0 | 285.4 |

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 85 | 6,000 | 1.4% |
| Survival | 300 steps | 1,800 steps | 16.7% |

## Recommendations

Fine-tune hyperparameters, increase training duration.

1. Epsilon 0.690 still high. Consider faster decay.

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

