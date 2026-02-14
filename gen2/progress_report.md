# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-14 15:59:55  
**Total Episodes:** 757  
**Training Sessions:** 1

## Verdict: NOT LEARNING (Confidence: 45%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

### Warnings
- Rewards flat: change = 4.6

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 200 | 99.8 | 76.2 | 30.8 | 0.5736 | 0.0% | 0.0% | 0.0% |
| S2 | WALL_AVOID | 557 | 107.8 | 76.0 | 31.7 | 0.6635 | 0.0% | 0.0% | 0.0% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 105.71 | 79.56 | -39.21 | 52.89 | 88.84 | 132.17 | 267.00 | 497.52 |
| Steps | 76.08 | 55.46 | 2.00 | 35.00 | 64.00 | 103.00 | 194.00 | 337.00 |
| Food | 31.42 | 13.62 | 1.00 | 22.00 | 28.00 | 38.00 | 60.00 | 97.00 |
| Loss | 2.85 | 3.95 | 0.00 | 1.42 | 2.23 | 3.33 | 5.80 | 64.76 |
| Food/Step | 0.64 | 0.71 | 0.25 | 0.36 | 0.46 | 0.63 | 1.38 | 6.33 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 96.91 | 56.71 | +0.1466 | 0.0014 |
| Last 100 | 109.11 | 84.65 | -0.1717 | 0.0034 |
| Last 200 | 112.05 | 89.08 | -0.0784 | 0.0026 |
| Last 500 | 107.49 | 86.93 | +0.0020 | 0.0000 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 97 | 6,000 | 1.6% |
| Survival | 337 steps | 1,800 steps | 18.7% |

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

