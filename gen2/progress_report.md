# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-15 09:46:35  
**Total Episodes:** 5038  
**Training Sessions:** 14

## Verdict: LEARNING (Confidence: 65%)

**Goal Feasibility:** POSSIBLE (25-60%)

### Positive Signals
- Rewards improving: +140.0

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 1293 | 320.7 | 62.2 | 28.5 | 0.7114 | 0.0% | 0.0% | 0.0% |
| S2 | WALL_AVOID | 1341 | 107.4 | 74.2 | 32.7 | 0.7258 | 0.0% | 0.0% | 0.0% |
| S3 | ENEMY_AVOID | 292 | 191.1 | 89.2 | 33.7 | 0.9742 | 0.0% | 0.0% | 0.0% |
| S4 | MASS_MANAGEMENT | 2112 | 230.9 | 90.3 | 36.8 | 0.9087 | 0.0% | 0.0% | 0.0% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 218.79 | 269.56 | -7111.38 | 73.08 | 140.94 | 285.45 | 705.82 | 3256.19 |
| Steps | 78.78 | 116.76 | 1.00 | 26.00 | 61.00 | 106.00 | 203.00 | 5000.00 |
| Food | 33.40 | 18.75 | 0.00 | 21.00 | 30.00 | 43.00 | 70.00 | 128.00 |
| Loss | 17.09 | 61.97 | 0.00 | 2.88 | 6.71 | 17.00 | 40.52 | 996.87 |
| Food/Step | 0.81 | 1.13 | 0.00 | 0.36 | 0.48 | 0.72 | 3.00 | 16.50 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 128.43 | 108.26 | +0.4156 | 0.0031 |
| Last 100 | 113.58 | 108.11 | +0.6276 | 0.0281 |
| Last 200 | 102.49 | 89.87 | +0.2028 | 0.0170 |
| Last 500 | 103.35 | 95.79 | +0.0368 | 0.0031 |
| Last 1000 | 306.68 | 349.12 | -0.7901 | 0.4268 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 128 | 6,000 | 2.1% |
| Survival | 5000 steps | 1,800 steps | 277.8% |

## Recommendations

Keep training. Monitor for sustained improvement.

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

