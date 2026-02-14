# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-14 13:48:19  
**Total Episodes:** 1116  
**Training Sessions:** 2

## Verdict: NOT LEARNING (Confidence: 50%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

### Warnings
- Rewards flat: change = 9.9

### Positive Signals
- Epsilon low (0.117) - exploiting policy

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 219 | 104.3 | 80.0 | 31.8 | 0.5951 | 0.0% | 0.0% | 0.0% |
| S2 | WALL_AVOID | 431 | 83.2 | 78.1 | 32.2 | 0.6474 | 0.0% | 0.0% | 0.0% |
| S3 | ENEMY_AVOID | 466 | 103.7 | 81.4 | 32.2 | 0.6020 | 0.0% | 0.0% | 0.0% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 95.90 | 66.83 | -25.00 | 48.66 | 79.41 | 130.00 | 226.88 | 511.55 |
| Steps | 79.86 | 59.59 | 1.00 | 33.00 | 65.00 | 113.00 | 195.25 | 376.00 |
| Food | 32.15 | 14.28 | 0.00 | 22.00 | 29.00 | 40.00 | 58.25 | 96.00 |
| Loss | 1.39 | 1.11 | 0.09 | 0.71 | 1.16 | 1.71 | 3.13 | 12.73 |
| Food/Step | 0.62 | 0.59 | 0.00 | 0.35 | 0.44 | 0.62 | 1.51 | 5.67 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 77.01 | 51.78 | -1.0710 | 0.0891 |
| Last 100 | 84.25 | 60.89 | -0.4094 | 0.0377 |
| Last 200 | 105.34 | 79.83 | -0.3991 | 0.0833 |
| Last 500 | 101.88 | 74.20 | +0.0106 | 0.0004 |
| Last 1000 | 94.20 | 66.47 | +0.0250 | 0.0118 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 96 | 6,000 | 1.6% |
| Survival | 376 steps | 1,800 steps | 20.9% |

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

