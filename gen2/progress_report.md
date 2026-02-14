# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-14 06:37:30  
**Total Episodes:** 5091  
**Training Sessions:** 3

## Verdict: NOT LEARNING (Confidence: 45%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

### Warnings
- Rewards flat: change = 19.9

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 200 | 76.8 | 64.8 | 26.1 | 1.0802 | 0.0% | 0.0% | 0.0% |
| S2 | WALL_AVOID | 1645 | 50.0 | 40.0 | 23.7 | 1.2389 | 0.0% | 0.0% | 0.0% |
| S3 | ENEMY_AVOID | 3246 | 86.9 | 63.6 | 27.6 | 0.7530 | 0.0% | 0.0% | 0.0% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 74.59 | 52.56 | -97.88 | 39.40 | 60.27 | 97.83 | 177.59 | 441.72 |
| Steps | 56.00 | 47.60 | 1.00 | 20.50 | 43.00 | 79.00 | 150.00 | 369.00 |
| Food | 26.28 | 9.14 | 0.00 | 21.00 | 24.00 | 30.00 | 43.00 | 91.00 |
| Loss | 1.75 | 1.21 | 0.00 | 0.99 | 1.53 | 2.26 | 3.74 | 36.36 |
| Food/Step | 0.92 | 1.03 | 0.00 | 0.38 | 0.56 | 0.95 | 3.15 | 7.00 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 84.21 | 49.37 | -0.6735 | 0.0388 |
| Last 100 | 87.51 | 47.89 | -0.1759 | 0.0112 |
| Last 200 | 89.81 | 49.37 | +0.0012 | 0.0000 |
| Last 500 | 83.19 | 47.92 | +0.0440 | 0.0175 |
| Last 1000 | 79.43 | 42.49 | +0.0185 | 0.0158 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 91 | 6,000 | 1.5% |
| Survival | 369 steps | 1,800 steps | 20.5% |

## Recommendations

Fine-tune hyperparameters, increase training duration.

1. Episodes too short. Reduce death penalties or add survival bonus.

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

### Hyperparameter Analysis
![Hyperparameter Analysis](chart_12_hyperparameter_analysis.png)

