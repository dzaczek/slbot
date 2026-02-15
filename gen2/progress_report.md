# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-15 03:16:35  
**Total Episodes:** 3865  
**Training Sessions:** 12

## Verdict: LEARNING (Confidence: 60%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

### Warnings
- Loss very high (33.96) - unstable

### Positive Signals
- Rewards improving: +105.0

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 773 | 89.9 | 59.9 | 24.9 | 0.5461 | 0.0% | 0.0% | 0.0% |
| S2 | WALL_AVOID | 688 | 111.7 | 77.5 | 32.1 | 0.6594 | 0.0% | 0.0% | 0.0% |
| S3 | ENEMY_AVOID | 292 | 191.1 | 89.2 | 33.7 | 0.9742 | 0.0% | 0.0% | 0.0% |
| S4 | MASS_MANAGEMENT | 2112 | 230.9 | 90.3 | 36.8 | 0.9087 | 0.0% | 0.0% | 0.0% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 178.50 | 220.42 | -7111.38 | 75.47 | 137.21 | 243.73 | 494.04 | 3256.19 |
| Steps | 81.90 | 129.80 | 1.00 | 26.00 | 61.00 | 110.00 | 209.80 | 5000.00 |
| Food | 33.37 | 19.27 | 0.00 | 21.00 | 30.00 | 43.00 | 70.80 | 128.00 |
| Loss | 20.18 | 70.35 | 0.00 | 2.71 | 8.33 | 20.71 | 43.36 | 996.87 |
| Food/Step | 0.80 | 1.12 | 0.00 | 0.35 | 0.47 | 0.70 | 3.00 | 14.67 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 255.16 | 180.28 | +0.1707 | 0.0002 |
| Last 100 | 241.04 | 156.04 | +0.3179 | 0.0035 |
| Last 200 | 247.64 | 159.90 | -0.1271 | 0.0021 |
| Last 500 | 234.57 | 158.39 | +0.0654 | 0.0035 |
| Last 1000 | 234.78 | 183.66 | +0.0020 | 0.0000 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 128 | 6,000 | 2.1% |
| Survival | 5000 steps | 1,800 steps | 277.8% |

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

