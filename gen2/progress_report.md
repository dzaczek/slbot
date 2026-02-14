# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-14 16:43:15  
**Total Episodes:** 1063  
**Training Sessions:** 2

## Verdict: NOT LEARNING (Confidence: 50%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

### Warnings
- Rewards flat: change = 18.1

### Positive Signals
- Epsilon low (0.137) - exploiting policy

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 388 | 118.3 | 73.7 | 30.2 | 0.6002 | 0.0% | 0.0% | 0.0% |
| S2 | WALL_AVOID | 675 | 110.8 | 77.2 | 31.9 | 0.6620 | 0.0% | 0.0% | 0.0% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 113.53 | 83.07 | -39.21 | 57.15 | 94.67 | 149.29 | 277.55 | 497.52 |
| Steps | 75.93 | 55.81 | 1.00 | 34.00 | 63.00 | 104.00 | 192.00 | 337.00 |
| Food | 31.29 | 13.60 | 0.00 | 22.00 | 28.00 | 38.00 | 58.00 | 97.00 |
| Loss | 3.03 | 4.02 | 0.00 | 1.39 | 2.25 | 3.50 | 6.55 | 64.76 |
| Food/Step | 0.64 | 0.68 | 0.00 | 0.36 | 0.46 | 0.64 | 1.45 | 6.33 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 124.23 | 80.10 | +0.1842 | 0.0011 |
| Last 100 | 141.50 | 81.74 | -0.5785 | 0.0417 |
| Last 200 | 138.97 | 76.16 | -0.0590 | 0.0020 |
| Last 500 | 124.90 | 89.01 | +0.0664 | 0.0116 |
| Last 1000 | 114.26 | 84.37 | +0.0400 | 0.0188 |

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

