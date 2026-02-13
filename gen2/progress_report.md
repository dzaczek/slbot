# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-13 18:30:48  
**Total Episodes:** 126  
**Training Sessions:** 1

## Verdict: NOT LEARNING (Confidence: 50%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

### Warnings
- Epsilon very high (0.853) - mostly random

### Positive Signals
- Food collection improving (slope=0.0578/ep)

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 126 | 80.9 | 54.4 | 26.5 | 0.8177 | 0.0% | 96.8% | 3.2% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 80.94 | 44.35 | -12.90 | 54.00 | 74.56 | 100.53 | 170.63 | 216.47 |
| Steps | 54.37 | 38.14 | 2.00 | 26.00 | 47.00 | 71.00 | 138.00 | 150.00 |
| Food | 26.45 | 9.85 | 0.00 | 21.00 | 26.00 | 31.00 | 44.00 | 51.00 |
| Loss | 55.32 | 149.47 | 0.58 | 2.23 | 5.12 | 24.43 | 393.58 | 910.85 |
| Food/Step | 0.82 | 0.98 | 0.00 | 0.40 | 0.50 | 0.76 | 2.17 | 6.67 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 86.78 | 41.20 | +0.4062 | 0.0202 |
| Last 100 | 84.88 | 43.48 | +0.0281 | 0.0003 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| SnakeCollision | 122 | 96.8% | 51.2 | 77.0 |
| MaxSteps | 4 | 3.2% | 150.0 | 202.6 |

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 51 | 6,000 | 0.9% |
| Survival | 150 steps | 1,800 steps | 8.3% |

## Recommendations

Fine-tune hyperparameters, increase training duration.

1. Epsilon 0.853 still high. Consider faster decay.

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

