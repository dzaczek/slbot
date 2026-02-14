# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-14 21:46:08  
**Total Episodes:** 1789  
**Training Sessions:** 11

## Verdict: NOT LEARNING (Confidence: 50%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

### Warnings
- Rewards flat: change = 18.0

### Positive Signals
- Epsilon low (0.109) - exploiting policy

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 773 | 89.9 | 59.9 | 24.9 | 0.5461 | 0.0% | 0.0% | 0.0% |
| S2 | WALL_AVOID | 688 | 111.7 | 77.5 | 32.1 | 0.6594 | 0.0% | 0.0% | 0.0% |
| S3 | ENEMY_AVOID | 292 | 191.1 | 89.2 | 33.7 | 0.9742 | 0.0% | 0.0% | 0.0% |
| S4 | MASS_MANAGEMENT | 36 | 242.2 | 96.5 | 40.9 | 0.6129 | 0.0% | 0.0% | 0.0% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 117.88 | 154.46 | -50.00 | 47.04 | 93.16 | 157.61 | 310.50 | 1476.21 |
| Steps | 72.22 | 71.59 | 1.00 | 24.00 | 57.00 | 101.00 | 194.60 | 500.00 |
| Food | 29.44 | 17.21 | 0.00 | 21.00 | 28.00 | 39.00 | 60.00 | 97.00 |
| Loss | 18.88 | 102.28 | 0.00 | 1.48 | 2.57 | 4.30 | 12.60 | 996.87 |
| Food/Step | 0.66 | 0.91 | 0.00 | 0.34 | 0.45 | 0.65 | 2.00 | 14.67 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 426.32 | 435.48 | -16.7362 | 0.3076 |
| Last 100 | 370.50 | 450.75 | +0.8692 | 0.0031 |
| Last 200 | 251.69 | 352.38 | +1.8856 | 0.0954 |
| Last 500 | 146.18 | 253.02 | +0.7565 | 0.1862 |
| Last 1000 | 126.43 | 193.10 | +0.1211 | 0.0328 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 97 | 6,000 | 1.6% |
| Survival | 500 steps | 1,800 steps | 27.8% |

## Recommendations

Fine-tune hyperparameters, increase training duration.

1. No critical issues. Continue training.

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

