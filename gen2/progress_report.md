# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-16 06:09:31  
**Total Episodes:** 6763  
**Training Sessions:** 12

## Verdict: NOT LEARNING (Confidence: 30%)

**Goal Feasibility:** VERY UNLIKELY (<5%)

### Critical Issues
- Rewards DECLINING: -94.3

### Warnings
- Loss very high (11.86) - unstable

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 400 | 99.4 | 103.6 | 30.8 | 0.6613 | 8.0% | 81.5% | 10.5% |
| S2 | WALL_AVOID | 383 | 231.0 | 140.3 | 30.5 | 0.7597 | 12.3% | 70.8% | 17.0% |
| S3 | ENEMY_AVOID | 5980 | 179.7 | 80.4 | 32.1 | 1.0121 | 7.0% | 91.4% | 1.6% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 177.84 | 358.66 | -40.65 | 68.05 | 114.09 | 196.63 | 409.06 | 4390.02 |
| Steps | 85.18 | 131.21 | 1.00 | 23.00 | 55.00 | 101.00 | 219.80 | 1000.00 |
| Food | 31.98 | 18.05 | 0.00 | 21.00 | 28.00 | 42.00 | 66.90 | 135.00 |
| Loss | 10.10 | 6.62 | 0.23 | 5.46 | 9.01 | 13.34 | 21.64 | 119.45 |
| Food/Step | 0.98 | 1.35 | 0.00 | 0.39 | 0.52 | 0.87 | 4.00 | 11.00 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 118.38 | 71.58 | +0.4938 | 0.0099 |
| Last 100 | 106.73 | 69.07 | +0.2923 | 0.0149 |
| Last 200 | 111.44 | 84.85 | -0.0645 | 0.0019 |
| Last 500 | 104.51 | 79.06 | +0.0384 | 0.0049 |
| Last 1000 | 104.47 | 80.58 | +0.0153 | 0.0030 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 498 | 7.4% | 141.1 | 259.5 |
| SnakeCollision | 6063 | 89.6% | 60.3 | 131.4 |
| MaxSteps | 202 | 3.0% | 693.6 | 1371.0 |

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 135 | 6,000 | 2.2% |
| Survival | 1000 steps | 1,800 steps | 55.6% |

## Recommendations

Major changes needed: LR, reward structure, curriculum.

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

