# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-15 10:17:15  
**Total Episodes:** 5347  
**Training Sessions:** 14

## Verdict: LEARNING (Confidence: 65%)

**Goal Feasibility:** POSSIBLE (25-60%)

### Positive Signals
- Rewards improving: +121.4

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 1293 | 320.7 | 62.2 | 28.5 | 0.7114 | 14.7% | 85.1% | 0.2% |
| S2 | WALL_AVOID | 1449 | 108.7 | 74.9 | 33.0 | 0.7296 | 9.0% | 91.0% | 0.0% |
| S3 | ENEMY_AVOID | 493 | 183.2 | 77.8 | 32.6 | 1.0150 | 5.1% | 91.5% | 3.4% |
| S4 | MASS_MANAGEMENT | 2112 | 230.9 | 90.3 | 36.8 | 0.9087 | 14.2% | 85.5% | 0.3% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 215.12 | 264.49 | -7111.38 | 72.25 | 139.13 | 279.08 | 692.23 | 3256.19 |
| Steps | 78.19 | 114.40 | 1.00 | 26.00 | 60.00 | 106.00 | 203.00 | 5000.00 |
| Food | 33.37 | 18.78 | 0.00 | 21.00 | 30.00 | 43.00 | 70.00 | 128.00 |
| Loss | 16.37 | 60.22 | 0.00 | 2.85 | 6.25 | 16.13 | 39.61 | 996.87 |
| Food/Step | 0.82 | 1.13 | 0.00 | 0.36 | 0.48 | 0.73 | 3.10 | 16.50 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 223.16 | 216.05 | -1.6691 | 0.0124 |
| Last 100 | 185.52 | 173.43 | +0.7081 | 0.0139 |
| Last 200 | 172.16 | 148.57 | +0.3545 | 0.0190 |
| Last 500 | 134.44 | 131.41 | +0.2481 | 0.0743 |
| Last 1000 | 141.92 | 175.66 | -0.0507 | 0.0069 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 644 | 12.0% | 117.8 | 222.5 |
| SnakeCollision | 4676 | 87.5% | 67.5 | 209.1 |
| MaxSteps | 27 | 0.5% | 977.8 | 1080.7 |

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

