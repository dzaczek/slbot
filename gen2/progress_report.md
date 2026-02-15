# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-15 17:34:56  
**Total Episodes:** 4  
**Training Sessions:** 1

## Verdict: NOT LEARNING (Confidence: 10%)

**Goal Feasibility:** 

### Critical Issues
- Insufficient data (< 20 episodes)

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 4 | 187.9 | 215.8 | 51.2 | 0.2347 | 25.0% | 75.0% | 0.0% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 187.89 | 61.98 | 111.50 | 149.83 | 179.10 | 217.15 | 268.93 | 281.87 |
| Steps | 215.75 | 47.36 | 177.00 | 177.00 | 196.50 | 235.25 | 281.45 | 293.00 |
| Food | 51.25 | 14.82 | 37.00 | 42.25 | 46.00 | 55.00 | 71.80 | 76.00 |
| Loss | 4.45 | 2.09 | 2.00 | 3.02 | 4.08 | 5.50 | 7.21 | 7.64 |
| Food/Step | 0.23 | 0.02 | 0.21 | 0.22 | 0.24 | 0.25 | 0.26 | 0.26 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 1 | 25.0% | 177.0 | 111.5 |
| SnakeCollision | 3 | 75.0% | 228.7 | 213.4 |

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 76 | 6,000 | 1.3% |
| Survival | 293 steps | 1,800 steps | 16.3% |

## Recommendations



1. Epsilon 0.948 still high. Consider faster decay.

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

