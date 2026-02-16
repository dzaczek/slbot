# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-16 12:49:00  
**Total Episodes:** 122  
**Training Sessions:** 3

## Verdict: NOT LEARNING (Confidence: 55%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

### Positive Signals
- Food collection improving (slope=0.0708/ep)

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 122 | 98.9 | 81.3 | 30.0 | 0.6727 | 0.8% | 96.7% | 2.5% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 98.86 | 68.74 | -13.90 | 54.09 | 83.13 | 123.81 | 244.48 | 311.49 |
| Steps | 81.25 | 68.28 | 2.00 | 31.00 | 62.00 | 110.50 | 235.05 | 300.00 |
| Food | 30.03 | 15.90 | 1.00 | 21.00 | 26.50 | 36.75 | 59.95 | 85.00 |
| Loss | 6.69 | 8.59 | 0.00 | 1.83 | 3.68 | 7.57 | 23.15 | 40.93 |
| Food/Step | 0.67 | 0.88 | 0.18 | 0.30 | 0.41 | 0.71 | 1.75 | 6.67 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 110.21 | 72.96 | -0.3942 | 0.0061 |
| Last 100 | 104.68 | 72.06 | +0.0872 | 0.0012 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 1 | 0.8% | 262.0 | 299.6 |
| SnakeCollision | 118 | 96.7% | 74.2 | 92.2 |
| MaxSteps | 3 | 2.5% | 300.0 | 293.1 |

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 85 | 6,000 | 1.4% |
| Survival | 300 steps | 1,800 steps | 16.7% |

## Recommendations

Fine-tune hyperparameters, increase training duration.

1. Epsilon 0.723 still high. Consider faster decay.

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

