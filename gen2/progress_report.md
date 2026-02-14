# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-14 16:24:23  
**Total Episodes:** 923  
**Training Sessions:** 2

## Verdict: NOT LEARNING (Confidence: 45%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

### Warnings
- Rewards flat: change = 9.9

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 248 | 108.3 | 75.4 | 30.8 | 0.5672 | 0.0% | 0.0% | 0.0% |
| S2 | WALL_AVOID | 675 | 110.8 | 77.2 | 31.9 | 0.6620 | 0.0% | 0.0% | 0.0% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 110.13 | 83.27 | -39.21 | 53.36 | 91.95 | 140.79 | 275.84 | 497.52 |
| Steps | 76.74 | 56.36 | 1.00 | 35.00 | 64.00 | 105.00 | 193.90 | 337.00 |
| Food | 31.61 | 13.87 | 0.00 | 22.00 | 28.00 | 39.00 | 59.90 | 97.00 |
| Loss | 3.23 | 4.27 | 0.00 | 1.50 | 2.41 | 3.67 | 7.02 | 64.76 |
| Food/Step | 0.64 | 0.68 | 0.00 | 0.36 | 0.46 | 0.62 | 1.43 | 6.33 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 144.06 | 70.11 | +0.5691 | 0.0137 |
| Last 100 | 137.22 | 89.05 | +0.3797 | 0.0151 |
| Last 200 | 125.91 | 90.80 | +0.2109 | 0.0180 |
| Last 500 | 114.96 | 89.12 | +0.0737 | 0.0142 |

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

### Q-Value & Gradient Analysis
![Q-Value & Gradient Analysis](chart_13_qvalue_gradients.png)

### Action Distribution Analysis
![Action Distribution Analysis](chart_14_action_distribution.png)

### Active Agents Over Time
![Active Agents Over Time](chart_15_auto_scaling.png)

