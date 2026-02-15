# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-15 12:41:28  
**Total Episodes:** 6844  
**Training Sessions:** 14

## Verdict: LEARNING (Confidence: 60%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

### Warnings
- Loss very high (21.35) - unstable

### Positive Signals
- Rewards improving: +83.5

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 1293 | 320.7 | 62.2 | 28.5 | 0.7114 | 14.7% | 85.1% | 0.2% |
| S2 | WALL_AVOID | 1449 | 108.7 | 74.9 | 33.0 | 0.7296 | 9.0% | 91.0% | 0.0% |
| S3 | ENEMY_AVOID | 835 | 205.8 | 84.8 | 33.9 | 0.8943 | 9.9% | 87.4% | 2.6% |
| S4 | MASS_MANAGEMENT | 3267 | 218.4 | 82.3 | 35.7 | 0.9108 | 13.7% | 86.1% | 0.2% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 212.94 | 245.82 | -7111.38 | 78.63 | 145.89 | 275.87 | 654.16 | 3256.19 |
| Steps | 77.21 | 107.74 | 1.00 | 25.00 | 60.00 | 105.00 | 202.00 | 5000.00 |
| Food | 33.53 | 18.77 | 0.00 | 21.00 | 30.00 | 43.00 | 70.00 | 128.00 |
| Loss | 16.15 | 53.44 | 0.00 | 3.44 | 8.14 | 17.62 | 38.87 | 996.87 |
| Food/Step | 0.83 | 1.11 | 0.00 | 0.37 | 0.49 | 0.74 | 3.00 | 16.50 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 165.51 | 97.61 | +0.7850 | 0.0135 |
| Last 100 | 174.33 | 117.81 | -0.2626 | 0.0041 |
| Last 200 | 186.12 | 129.33 | -0.2033 | 0.0082 |
| Last 500 | 193.21 | 134.34 | -0.0432 | 0.0022 |
| Last 1000 | 195.21 | 138.41 | -0.0065 | 0.0002 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 849 | 12.4% | 123.3 | 256.8 |
| SnakeCollision | 5963 | 87.1% | 65.8 | 202.8 |
| MaxSteps | 32 | 0.5% | 981.2 | 947.6 |

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

