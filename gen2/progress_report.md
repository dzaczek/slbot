# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-14 14:59:46  
**Total Episodes:** 339  
**Training Sessions:** 1

## Verdict: NOT LEARNING (Confidence: 45%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

### Warnings
- Rewards flat: change = 15.7

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 200 | 99.8 | 76.2 | 30.8 | 0.5736 | 0.0% | 0.0% | 0.0% |
| S2 | WALL_AVOID | 139 | 110.2 | 78.3 | 31.7 | 0.5499 | 0.0% | 0.0% | 0.0% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 104.07 | 69.28 | -16.92 | 58.73 | 90.22 | 125.19 | 244.32 | 461.95 |
| Steps | 77.10 | 52.74 | 2.00 | 39.00 | 64.00 | 103.00 | 192.20 | 291.00 |
| Food | 31.12 | 12.48 | 1.00 | 23.00 | 28.00 | 36.00 | 56.20 | 85.00 |
| Loss | 2.74 | 5.61 | 0.00 | 1.19 | 1.75 | 2.63 | 5.51 | 64.76 |
| Food/Step | 0.56 | 0.50 | 0.25 | 0.35 | 0.44 | 0.60 | 1.18 | 4.75 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 106.79 | 72.40 | -0.0666 | 0.0002 |
| Last 100 | 110.40 | 89.78 | -0.1847 | 0.0035 |
| Last 200 | 106.91 | 79.70 | +0.0902 | 0.0043 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 85 | 6,000 | 1.4% |
| Survival | 291 steps | 1,800 steps | 16.2% |

## Recommendations

Fine-tune hyperparameters, increase training duration.

1. Epsilon 0.408 still high. Consider faster decay.

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

