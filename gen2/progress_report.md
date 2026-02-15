# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-15 17:45:05  
**Total Episodes:** 66  
**Training Sessions:** 1

## Verdict: NOT LEARNING (Confidence: 45%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

### Warnings
- Epsilon very high (0.847) - mostly random

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 66 | 127.0 | 98.3 | 38.3 | 0.8279 | 15.2% | 84.8% | 0.0% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 126.95 | 90.67 | -12.90 | 52.97 | 107.68 | 195.28 | 303.49 | 403.17 |
| Steps | 98.26 | 77.26 | 2.00 | 29.25 | 82.00 | 146.25 | 227.00 | 293.00 |
| Food | 38.35 | 22.65 | 0.00 | 22.00 | 31.50 | 50.25 | 77.50 | 111.00 |
| Loss | 4.35 | 2.20 | 1.52 | 2.93 | 4.20 | 5.28 | 8.93 | 11.06 |
| Food/Step | 0.83 | 1.14 | 0.00 | 0.32 | 0.43 | 0.70 | 3.63 | 6.67 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 126.34 | 98.21 | +0.1235 | 0.0003 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 10 | 15.2% | 200.2 | 245.5 |
| SnakeCollision | 56 | 84.8% | 80.1 | 105.8 |

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 111 | 6,000 | 1.8% |
| Survival | 293 steps | 1,800 steps | 16.3% |

## Recommendations

Fine-tune hyperparameters, increase training duration.

1. Epsilon 0.847 still high. Consider faster decay.

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

