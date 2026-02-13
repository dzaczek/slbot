# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-13 18:27:43  
**Total Episodes:** 82  
**Training Sessions:** 1

## Verdict: NOT LEARNING (Confidence: 50%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

### Warnings
- Epsilon very high (0.902) - mostly random

### Positive Signals
- Food collection improving (slope=0.0919/ep)

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 82 | 77.3 | 53.6 | 25.6 | 0.8276 | 0.0% | 96.3% | 3.7% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 77.29 | 44.25 | -12.90 | 52.74 | 73.51 | 100.53 | 152.88 | 216.47 |
| Steps | 53.57 | 36.78 | 2.00 | 26.00 | 54.50 | 71.00 | 126.85 | 150.00 |
| Food | 25.59 | 10.23 | 0.00 | 21.00 | 26.50 | 31.00 | 42.95 | 49.00 |
| Loss | 84.02 | 178.80 | 1.73 | 5.25 | 18.27 | 77.74 | 530.57 | 910.85 |
| Food/Step | 0.83 | 1.07 | 0.00 | 0.39 | 0.49 | 0.75 | 2.96 | 6.67 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 83.14 | 40.49 | -0.7062 | 0.0634 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| SnakeCollision | 79 | 96.3% | 49.9 | 72.6 |
| MaxSteps | 3 | 3.7% | 150.0 | 200.8 |

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 49 | 6,000 | 0.8% |
| Survival | 150 steps | 1,800 steps | 8.3% |

## Recommendations

Fine-tune hyperparameters, increase training duration.

1. Epsilon 0.902 still high. Consider faster decay.

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

