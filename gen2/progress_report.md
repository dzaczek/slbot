# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-16 12:28:33  
**Total Episodes:** 9  
**Training Sessions:** 1

## Verdict: NOT LEARNING (Confidence: 10%)

**Goal Feasibility:** 

### Critical Issues
- Insufficient data (< 20 episodes)

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 9 | 43.1 | 42.0 | 18.0 | 0.4869 | 0.0% | 100.0% | 0.0% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 43.08 | 24.75 | -9.90 | 26.75 | 49.68 | 53.75 | 71.86 | 72.20 |
| Steps | 42.00 | 25.67 | 2.00 | 31.00 | 38.00 | 45.00 | 85.40 | 99.00 |
| Food | 18.00 | 7.66 | 1.00 | 16.00 | 20.00 | 21.00 | 26.80 | 30.00 |
| Loss | 21.16 | 15.84 | 0.00 | 10.47 | 19.99 | 38.10 | 40.79 | 40.93 |
| Food/Step | 0.49 | 0.15 | 0.27 | 0.31 | 0.50 | 0.64 | 0.67 | 0.68 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| SnakeCollision | 9 | 100.0% | 42.0 | 43.1 |

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 30 | 6,000 | 0.5% |
| Survival | 99 steps | 1,800 steps | 5.5% |

## Recommendations



1. Epsilon 0.957 still high. Consider faster decay.

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

