# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-15 17:55:17  
**Total Episodes:** 132  
**Training Sessions:** 3

## Verdict: NOT LEARNING (Confidence: 50%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 132 | 108.7 | 91.1 | 34.4 | 0.6919 | 10.6% | 89.4% | 0.0% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 108.68 | 79.60 | -16.90 | 52.01 | 84.48 | 149.16 | 257.32 | 403.17 |
| Steps | 91.08 | 73.12 | 1.00 | 30.00 | 75.00 | 133.25 | 229.00 | 293.00 |
| Food | 34.45 | 20.21 | 0.00 | 21.00 | 29.00 | 44.25 | 69.45 | 111.00 |
| Loss | 4.85 | 3.09 | 0.95 | 2.93 | 4.22 | 5.86 | 10.52 | 21.62 |
| Food/Step | 0.69 | 0.90 | 0.00 | 0.30 | 0.41 | 0.62 | 2.36 | 6.67 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 94.00 | 59.41 | -0.0601 | 0.0002 |
| Last 100 | 103.77 | 78.85 | -0.4728 | 0.0300 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 14 | 10.6% | 194.0 | 221.4 |
| SnakeCollision | 118 | 89.4% | 78.9 | 95.3 |

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 111 | 6,000 | 1.8% |
| Survival | 293 steps | 1,800 steps | 16.3% |

## Recommendations

Fine-tune hyperparameters, increase training duration.

1. Epsilon 0.678 still high. Consider faster decay.

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

### Goal Progress Over Time
![Goal Progress Over Time](chart_11b_goal_over_time.png)

### Hyperparameter Analysis
![Hyperparameter Analysis](chart_12_hyperparameter_analysis.png)

### Q-Value & Gradient Analysis
![Q-Value & Gradient Analysis](chart_13_qvalue_gradients.png)

### Action Distribution Analysis
![Action Distribution Analysis](chart_14_action_distribution.png)

### Active Agents Over Time
![Active Agents Over Time](chart_15_auto_scaling.png)

### MaxSteps Analysis
![MaxSteps Analysis](chart_16_maxsteps_analysis.png)

### Survival Percentiles
![Survival Percentiles](chart_17_survival_percentiles.png)

