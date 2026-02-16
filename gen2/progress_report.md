# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-16 12:38:47  
**Total Episodes:** 60  
**Training Sessions:** 2

## Verdict: NOT LEARNING (Confidence: 50%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

### Warnings
- Epsilon very high (0.848) - mostly random

### Positive Signals
- Food collection improving (slope=0.1843/ep)

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 60 | 90.4 | 76.5 | 27.8 | 0.6610 | 0.0% | 96.7% | 3.3% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 90.44 | 66.15 | -9.90 | 49.58 | 71.77 | 114.55 | 241.69 | 299.72 |
| Steps | 76.50 | 65.59 | 2.00 | 29.75 | 62.50 | 100.00 | 189.45 | 300.00 |
| Food | 27.77 | 15.31 | 1.00 | 20.00 | 23.00 | 31.00 | 59.05 | 74.00 |
| Loss | 9.78 | 10.01 | 0.00 | 3.78 | 5.17 | 11.88 | 38.22 | 40.93 |
| Food/Step | 0.66 | 0.87 | 0.20 | 0.31 | 0.41 | 0.65 | 1.78 | 6.33 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 99.79 | 67.93 | +0.0159 | 0.0000 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| SnakeCollision | 58 | 96.7% | 68.8 | 83.8 |
| MaxSteps | 2 | 3.3% | 300.0 | 283.9 |

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 74 | 6,000 | 1.2% |
| Survival | 300 steps | 1,800 steps | 16.7% |

## Recommendations

Fine-tune hyperparameters, increase training duration.

1. Epsilon 0.848 still high. Consider faster decay.

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

### Steps vs Food vs Episode (3D)
![Steps vs Food vs Episode (3D)](chart_18_3d_steps_food_episode.png)

