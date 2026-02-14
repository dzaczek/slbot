# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-14 16:15:28  
**Total Episodes:** 867  
**Training Sessions:** 1

## Verdict: NOT LEARNING (Confidence: 45%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

### Warnings
- Rewards flat: change = 5.6

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 200 | 99.8 | 76.2 | 30.8 | 0.5736 | 0.0% | 0.0% | 0.0% |
| S2 | WALL_AVOID | 667 | 110.2 | 76.7 | 31.8 | 0.6651 | 0.0% | 0.0% | 0.0% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 107.79 | 83.49 | -39.21 | 52.11 | 88.53 | 136.71 | 273.78 | 497.52 |
| Steps | 76.57 | 56.74 | 1.00 | 34.00 | 63.00 | 105.00 | 194.70 | 337.00 |
| Food | 31.54 | 13.97 | 0.00 | 22.00 | 28.00 | 39.00 | 59.70 | 97.00 |
| Loss | 2.93 | 3.77 | 0.00 | 1.48 | 2.35 | 3.52 | 5.98 | 64.76 |
| Food/Step | 0.64 | 0.70 | 0.00 | 0.36 | 0.46 | 0.64 | 1.45 | 6.33 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 116.61 | 103.97 | +1.2911 | 0.0321 |
| Last 100 | 118.44 | 104.18 | +0.1871 | 0.0027 |
| Last 200 | 117.53 | 97.44 | +0.0268 | 0.0003 |
| Last 500 | 111.07 | 92.37 | +0.0295 | 0.0021 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 97 | 6,000 | 1.6% |
| Survival | 337 steps | 1,800 steps | 18.7% |

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

