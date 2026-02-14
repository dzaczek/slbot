# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-14 16:17:48  
**Total Episodes:** 876  
**Training Sessions:** 2

## Verdict: NOT LEARNING (Confidence: 45%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

### Warnings
- Rewards flat: change = 6.8

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 201 | 99.7 | 76.1 | 30.7 | 0.5730 | 0.0% | 0.0% | 0.0% |
| S2 | WALL_AVOID | 675 | 110.8 | 77.2 | 31.9 | 0.6620 | 0.0% | 0.0% | 0.0% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 108.24 | 83.43 | -39.21 | 52.22 | 89.44 | 137.75 | 275.63 | 497.52 |
| Steps | 76.95 | 56.89 | 1.00 | 34.00 | 64.00 | 105.00 | 195.50 | 337.00 |
| Food | 31.64 | 14.02 | 0.00 | 22.00 | 28.00 | 39.00 | 60.00 | 97.00 |
| Loss | 2.94 | 3.76 | 0.00 | 1.49 | 2.35 | 3.53 | 6.00 | 64.76 |
| Food/Step | 0.64 | 0.70 | 0.00 | 0.36 | 0.45 | 0.63 | 1.44 | 6.33 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 127.72 | 102.81 | +1.0837 | 0.0231 |
| Last 100 | 126.37 | 103.58 | +0.0121 | 0.0000 |
| Last 200 | 119.07 | 97.16 | +0.0703 | 0.0017 |
| Last 500 | 112.06 | 92.46 | +0.0348 | 0.0030 |

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

### Active Agents Over Time
![Active Agents Over Time](chart_15_auto_scaling.png)

