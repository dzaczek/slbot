# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-13 18:40:34  
**Total Episodes:** 238  
**Training Sessions:** 1

## Verdict: NOT LEARNING (Confidence: 50%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

### Warnings
- Rewards flat: change = -4.3

### Positive Signals
- Food collection improving (slope=0.0253/ep)

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 200 | 78.4 | 50.5 | 26.2 | 0.9286 | 0.0% | 98.0% | 2.0% |
| S2 | WALL_AVOID | 38 | 83.0 | 75.2 | 32.0 | 0.6033 | 0.0% | 100.0% | 0.0% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 79.10 | 41.64 | -12.90 | 52.70 | 73.75 | 100.14 | 150.58 | 249.90 |
| Steps | 54.42 | 40.06 | 2.00 | 24.00 | 46.00 | 73.75 | 127.30 | 219.00 |
| Food | 27.13 | 9.57 | 0.00 | 21.00 | 26.00 | 32.00 | 44.00 | 62.00 |
| Loss | 29.81 | 112.07 | 0.05 | 1.00 | 1.82 | 5.26 | 111.90 | 910.85 |
| Food/Step | 0.88 | 0.95 | 0.00 | 0.42 | 0.54 | 0.83 | 3.00 | 6.67 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 82.72 | 43.78 | +0.0461 | 0.0002 |
| Last 100 | 75.79 | 39.61 | +0.2115 | 0.0237 |
| Last 200 | 80.60 | 40.33 | -0.0411 | 0.0035 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| SnakeCollision | 234 | 98.3% | 52.8 | 77.0 |
| MaxSteps | 4 | 1.7% | 150.0 | 202.6 |

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 62 | 6,000 | 1.0% |
| Survival | 219 steps | 1,800 steps | 12.2% |

## Recommendations

Fine-tune hyperparameters, increase training duration.

1. Epsilon 0.738 still high. Consider faster decay.

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

### Hyperparameter Analysis
![Hyperparameter Analysis](chart_12_hyperparameter_analysis.png)

