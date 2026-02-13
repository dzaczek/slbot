# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-13 19:40:46  
**Total Episodes:** 872  
**Training Sessions:** 1

## Verdict: NOT LEARNING (Confidence: 35%)

**Goal Feasibility:** VERY UNLIKELY (<5%)

### Critical Issues
- Very short episodes: avg=36 steps

### Warnings
- Rewards flat: change = -2.3

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 872 | 57.4 | 35.7 | 21.8 | 1.3973 | 0.0% | 94.8% | 5.2% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 57.44 | 29.28 | -84.83 | 45.06 | 56.34 | 73.82 | 103.70 | 154.15 |
| Steps | 35.66 | 35.50 | 1.00 | 10.00 | 25.00 | 47.25 | 150.00 | 150.00 |
| Food | 21.79 | 7.28 | 0.00 | 20.00 | 22.00 | 26.00 | 32.45 | 42.00 |
| Loss | 46.37 | 305.09 | 0.00 | 0.31 | 0.73 | 7.00 | 122.58 | 4270.31 |
| Food/Step | 1.40 | 1.43 | 0.00 | 0.52 | 0.85 | 1.57 | 5.00 | 7.00 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 66.75 | 29.47 | -0.7349 | 0.1295 |
| Last 100 | 62.02 | 26.13 | +0.0585 | 0.0042 |
| Last 200 | 58.24 | 27.12 | +0.0226 | 0.0023 |
| Last 500 | 56.94 | 29.84 | +0.0018 | 0.0001 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| SnakeCollision | 827 | 94.8% | 29.4 | 59.8 |
| MaxSteps | 45 | 5.2% | 150.0 | 14.8 |

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 42 | 6,000 | 0.7% |
| Survival | 150 steps | 1,800 steps | 8.3% |

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

