# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-14 01:36:29  
**Total Episodes:** 2397  
**Training Sessions:** 3

## Verdict: NOT LEARNING (Confidence: 35%)

**Goal Feasibility:** VERY UNLIKELY (<5%)

### Critical Issues
- Very short episodes: avg=50 steps

### Warnings
- Rewards flat: change = 23.1

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 200 | 76.8 | 64.8 | 26.1 | 1.0802 | 0.0% | 0.0% | 0.0% |
| S2 | WALL_AVOID | 1645 | 50.0 | 40.0 | 23.7 | 1.2389 | 0.0% | 0.0% | 0.0% |
| S3 | ENEMY_AVOID | 552 | 94.8 | 72.8 | 28.8 | 0.7330 | 0.0% | 0.0% | 0.0% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 62.56 | 51.69 | -97.88 | 32.01 | 47.66 | 76.80 | 166.06 | 441.72 |
| Steps | 49.62 | 50.52 | 1.00 | 15.00 | 33.00 | 65.00 | 151.20 | 369.00 |
| Food | 25.09 | 9.45 | 0.00 | 21.00 | 23.00 | 28.00 | 43.20 | 91.00 |
| Loss | 1.95 | 1.52 | 0.00 | 0.98 | 1.65 | 2.59 | 4.35 | 36.36 |
| Food/Step | 1.11 | 1.18 | 0.00 | 0.42 | 0.67 | 1.20 | 3.80 | 7.00 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 78.73 | 48.69 | +1.0129 | 0.0901 |
| Last 100 | 78.24 | 55.56 | +0.1114 | 0.0034 |
| Last 200 | 91.61 | 68.43 | -0.2370 | 0.0400 |
| Last 500 | 94.78 | 68.40 | -0.0572 | 0.0146 |
| Last 1000 | 79.91 | 65.37 | +0.0598 | 0.0698 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 91 | 6,000 | 1.5% |
| Survival | 369 steps | 1,800 steps | 20.5% |

## Recommendations

Major changes needed: LR, reward structure, curriculum.

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

