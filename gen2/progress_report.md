# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-16 10:56:47  
**Total Episodes:** 96  
**Training Sessions:** 1

## Verdict: NOT LEARNING (Confidence: 50%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

### Warnings
- Epsilon very high (0.840) - mostly random

### Positive Signals
- Food collection improving (slope=0.0212/ep)

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 96 | 95.8 | 73.9 | 31.1 | 0.6742 | 8.3% | 90.6% | 1.0% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 95.80 | 53.29 | -9.90 | 57.23 | 91.81 | 130.68 | 203.03 | 254.50 |
| Steps | 73.90 | 57.10 | 2.00 | 28.75 | 64.50 | 102.25 | 183.25 | 300.00 |
| Food | 31.11 | 13.62 | 1.00 | 21.75 | 29.00 | 41.00 | 58.25 | 63.00 |
| Loss | 4.70 | 2.51 | 0.33 | 2.58 | 4.59 | 5.92 | 10.29 | 11.58 |
| Food/Step | 0.67 | 0.61 | 0.20 | 0.38 | 0.47 | 0.62 | 2.00 | 3.80 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 98.89 | 49.49 | +0.7907 | 0.0532 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 8 | 8.3% | 132.2 | 148.3 |
| SnakeCollision | 87 | 90.6% | 65.9 | 89.1 |
| MaxSteps | 1 | 1.0% | 300.0 | 254.5 |

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 63 | 6,000 | 1.1% |
| Survival | 300 steps | 1,800 steps | 16.7% |

## Recommendations

Fine-tune hyperparameters, increase training duration.

1. Epsilon 0.840 still high. Consider faster decay.

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

### Active Agents Over Time
![Active Agents Over Time](chart_15_auto_scaling.png)

### MaxSteps Analysis
![MaxSteps Analysis](chart_16_maxsteps_analysis.png)

### Steps vs Food vs Episode (3D)
![Steps vs Food vs Episode (3D)](chart_18_3d_steps_food_episode.png)

