# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-14 03:36:52  
**Total Episodes:** 3421  
**Training Sessions:** 3

## Verdict: NOT LEARNING (Confidence: 45%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

### Warnings
- Rewards flat: change = 44.4

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 200 | 76.8 | 64.8 | 26.1 | 1.0802 | 0.0% | 0.0% | 0.0% |
| S2 | WALL_AVOID | 1645 | 50.0 | 40.0 | 23.7 | 1.2389 | 0.0% | 0.0% | 0.0% |
| S3 | ENEMY_AVOID | 1576 | 91.6 | 68.7 | 28.2 | 0.7426 | 0.0% | 0.0% | 0.0% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 70.74 | 55.37 | -97.88 | 35.58 | 53.75 | 89.30 | 184.23 | 441.72 |
| Steps | 54.67 | 51.21 | 1.00 | 18.00 | 39.00 | 75.00 | 157.00 | 369.00 |
| Food | 25.92 | 9.64 | 0.00 | 21.00 | 24.00 | 30.00 | 45.00 | 91.00 |
| Loss | 1.77 | 1.35 | 0.00 | 0.94 | 1.47 | 2.30 | 3.96 | 36.36 |
| Food/Step | 1.00 | 1.09 | 0.00 | 0.39 | 0.59 | 1.06 | 3.50 | 7.00 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 90.28 | 52.16 | -1.5946 | 0.1947 |
| Last 100 | 103.55 | 69.98 | -0.3699 | 0.0233 |
| Last 200 | 95.41 | 62.83 | +0.1172 | 0.0116 |
| Last 500 | 90.79 | 58.64 | +0.0331 | 0.0067 |
| Last 1000 | 89.52 | 58.61 | +0.0048 | 0.0006 |

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

