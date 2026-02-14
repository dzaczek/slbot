# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-14 05:37:17  
**Total Episodes:** 4539  
**Training Sessions:** 3

## Verdict: NOT LEARNING (Confidence: 45%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

### Warnings
- Rewards flat: change = 24.5

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 200 | 76.8 | 64.8 | 26.1 | 1.0802 | 0.0% | 0.0% | 0.0% |
| S2 | WALL_AVOID | 1645 | 50.0 | 40.0 | 23.7 | 1.2389 | 0.0% | 0.0% | 0.0% |
| S3 | ENEMY_AVOID | 2694 | 87.7 | 64.1 | 27.7 | 0.7508 | 0.0% | 0.0% | 0.0% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 73.59 | 53.10 | -97.88 | 38.33 | 58.79 | 95.79 | 178.98 | 441.72 |
| Steps | 55.42 | 48.27 | 1.00 | 20.00 | 42.00 | 78.00 | 151.00 | 369.00 |
| Food | 26.17 | 9.20 | 0.00 | 21.00 | 24.00 | 30.00 | 44.00 | 91.00 |
| Loss | 1.76 | 1.24 | 0.00 | 0.98 | 1.51 | 2.26 | 3.80 | 36.36 |
| Food/Step | 0.94 | 1.04 | 0.00 | 0.39 | 0.57 | 1.00 | 3.29 | 7.00 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 88.26 | 32.90 | +0.2119 | 0.0086 |
| Last 100 | 80.07 | 30.48 | +0.2034 | 0.0371 |
| Last 200 | 78.62 | 34.38 | +0.0418 | 0.0049 |
| Last 500 | 76.00 | 36.81 | +0.0003 | 0.0000 |
| Last 1000 | 83.23 | 44.59 | -0.0252 | 0.0267 |

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

