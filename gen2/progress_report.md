# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-15 13:09:38  
**Total Episodes:** 7104  
**Training Sessions:** 15

## Verdict: LEARNING (Confidence: 65%)

**Goal Feasibility:** POSSIBLE (25-60%)

### Warnings
- Loss very high (10.64) - unstable

### Positive Signals
- Rewards improving: +81.1
- Epsilon low (0.132) - exploiting policy

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 1293 | 320.7 | 62.2 | 28.5 | 0.7114 | 14.7% | 85.1% | 0.2% |
| S2 | WALL_AVOID | 1449 | 108.7 | 74.9 | 33.0 | 0.7296 | 9.0% | 91.0% | 0.0% |
| S3 | ENEMY_AVOID | 994 | 210.3 | 81.8 | 33.4 | 0.8921 | 9.0% | 88.8% | 2.2% |
| S4 | MASS_MANAGEMENT | 3368 | 218.0 | 82.0 | 35.6 | 0.9065 | 13.7% | 86.1% | 0.2% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 213.32 | 243.52 | -7111.38 | 78.93 | 147.18 | 277.65 | 650.66 | 3256.19 |
| Steps | 76.89 | 106.21 | 1.00 | 26.00 | 60.00 | 104.00 | 201.00 | 5000.00 |
| Food | 33.47 | 18.67 | 0.00 | 21.00 | 30.00 | 43.00 | 70.00 | 128.00 |
| Loss | 16.30 | 52.53 | 0.00 | 3.53 | 8.46 | 18.01 | 39.23 | 996.87 |
| Food/Step | 0.83 | 1.11 | 0.00 | 0.37 | 0.50 | 0.74 | 3.00 | 16.50 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 247.55 | 203.58 | -0.1884 | 0.0002 |
| Last 100 | 254.26 | 193.40 | -0.3294 | 0.0024 |
| Last 200 | 225.04 | 182.49 | +0.4282 | 0.0184 |
| Last 500 | 208.99 | 154.53 | +0.0922 | 0.0074 |
| Last 1000 | 203.34 | 147.97 | +0.0220 | 0.0018 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 871 | 12.3% | 123.0 | 259.0 |
| SnakeCollision | 6201 | 87.3% | 65.7 | 203.1 |
| MaxSteps | 32 | 0.5% | 981.2 | 947.6 |

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

### MaxSteps Analysis
![MaxSteps Analysis](chart_16_maxsteps_analysis.png)

