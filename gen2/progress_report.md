# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-13 22:35:56  
**Total Episodes:** 391  
**Training Sessions:** 2

## Verdict: NOT LEARNING (Confidence: 45%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

### Warnings
- Rewards flat: change = -19.0

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 200 | 76.8 | 64.8 | 26.1 | 1.0802 | 0.0% | 0.0% | 0.0% |
| S2 | WALL_AVOID | 191 | 53.7 | 42.7 | 25.2 | 1.2295 | 0.0% | 0.0% | 0.0% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 65.52 | 46.81 | -97.88 | 39.16 | 54.45 | 92.22 | 147.51 | 296.85 |
| Steps | 54.01 | 57.76 | 1.00 | 16.00 | 34.00 | 74.00 | 153.50 | 300.00 |
| Food | 25.70 | 10.10 | 0.00 | 21.00 | 25.00 | 30.00 | 42.00 | 82.00 |
| Loss | 1.81 | 2.06 | 0.00 | 1.09 | 1.57 | 2.17 | 3.25 | 36.36 |
| Food/Step | 1.15 | 1.25 | 0.00 | 0.41 | 0.69 | 1.18 | 4.00 | 6.67 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 58.87 | 32.61 | +0.6397 | 0.0801 |
| Last 100 | 53.06 | 32.07 | +0.2169 | 0.0381 |
| Last 200 | 56.86 | 36.65 | -0.0766 | 0.0145 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 82 | 6,000 | 1.4% |
| Survival | 300 steps | 1,800 steps | 16.7% |

## Recommendations

Fine-tune hyperparameters, increase training duration.

1. Epsilon 0.617 still high. Consider faster decay.

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

### Hyperparameter Analysis
![Hyperparameter Analysis](chart_12_hyperparameter_analysis.png)

