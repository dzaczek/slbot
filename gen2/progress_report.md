# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-15 10:51:12  
**Total Episodes:** 5664  
**Training Sessions:** 14

## Verdict: LEARNING (Confidence: 70%)

**Goal Feasibility:** POSSIBLE (25-60%)

### Positive Signals
- Rewards improving: +115.8
- Epsilon low (0.123) - exploiting policy

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 1293 | 320.7 | 62.2 | 28.5 | 0.7114 | 14.7% | 85.1% | 0.2% |
| S2 | WALL_AVOID | 1449 | 108.7 | 74.9 | 33.0 | 0.7296 | 9.0% | 91.0% | 0.0% |
| S3 | ENEMY_AVOID | 810 | 204.7 | 79.1 | 34.0 | 0.9082 | 9.9% | 88.0% | 2.1% |
| S4 | MASS_MANAGEMENT | 2112 | 230.9 | 90.3 | 36.8 | 0.9087 | 14.2% | 85.5% | 0.3% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 216.40 | 261.53 | -7111.38 | 72.94 | 140.18 | 281.25 | 690.80 | 3256.19 |
| Steps | 78.35 | 112.28 | 1.00 | 26.00 | 60.00 | 107.00 | 203.00 | 5000.00 |
| Food | 33.53 | 18.90 | 0.00 | 21.00 | 30.00 | 43.00 | 70.00 | 128.00 |
| Loss | 15.87 | 58.56 | 0.00 | 2.96 | 6.34 | 15.41 | 39.16 | 996.87 |
| Food/Step | 0.82 | 1.11 | 0.00 | 0.36 | 0.48 | 0.73 | 3.00 | 16.50 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 259.11 | 215.21 | -1.8877 | 0.0160 |
| Last 100 | 244.99 | 206.92 | -0.2238 | 0.0010 |
| Last 200 | 228.95 | 192.87 | +0.1886 | 0.0032 |
| Last 500 | 213.64 | 189.06 | +0.1872 | 0.0204 |
| Last 1000 | 162.64 | 162.63 | +0.1880 | 0.1114 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 699 | 12.3% | 120.8 | 240.5 |
| SnakeCollision | 4938 | 87.2% | 67.4 | 208.3 |
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

### MaxSteps Analysis
![MaxSteps Analysis](chart_16_maxsteps_analysis.png)

