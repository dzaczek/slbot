# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-14 07:37:43  
**Total Episodes:** 5660  
**Training Sessions:** 3

## Verdict: NOT LEARNING (Confidence: 45%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

### Warnings
- Rewards flat: change = 16.9

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 200 | 76.8 | 64.8 | 26.1 | 1.0802 | 0.0% | 0.0% | 0.0% |
| S2 | WALL_AVOID | 1645 | 50.0 | 40.0 | 23.7 | 1.2389 | 0.0% | 0.0% | 0.0% |
| S3 | ENEMY_AVOID | 3815 | 85.1 | 62.1 | 27.3 | 0.7682 | 0.0% | 0.0% | 0.0% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 74.60 | 51.49 | -97.88 | 39.97 | 61.36 | 97.82 | 174.84 | 441.72 |
| Steps | 55.75 | 46.66 | 1.00 | 21.00 | 43.00 | 79.00 | 147.00 | 369.00 |
| Food | 26.25 | 8.99 | 0.00 | 21.00 | 24.00 | 30.00 | 43.00 | 91.00 |
| Loss | 1.75 | 1.18 | 0.00 | 1.00 | 1.54 | 2.25 | 3.70 | 36.36 |
| Food/Step | 0.92 | 1.02 | 0.00 | 0.38 | 0.55 | 0.95 | 3.14 | 7.00 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 59.98 | 34.07 | +0.8371 | 0.1257 |
| Last 100 | 64.45 | 34.28 | -0.0857 | 0.0052 |
| Last 200 | 71.70 | 41.70 | -0.0911 | 0.0159 |
| Last 500 | 75.13 | 40.90 | -0.0364 | 0.0165 |
| Last 1000 | 79.97 | 44.83 | -0.0212 | 0.0187 |

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

