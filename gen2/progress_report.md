# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-13 23:36:08  
**Total Episodes:** 1256  
**Training Sessions:** 2

## Verdict: NOT LEARNING (Confidence: 35%)

**Goal Feasibility:** VERY UNLIKELY (<5%)

### Critical Issues
- Very short episodes: avg=40 steps

### Warnings
- Rewards flat: change = -15.1

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 200 | 76.8 | 64.8 | 26.1 | 1.0802 | 0.0% | 0.0% | 0.0% |
| S2 | WALL_AVOID | 1056 | 45.6 | 35.2 | 22.9 | 1.3124 | 0.0% | 0.0% | 0.0% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 50.60 | 34.73 | -97.88 | 29.95 | 43.30 | 64.15 | 114.72 | 296.85 |
| Steps | 39.88 | 41.05 | 1.00 | 13.00 | 27.00 | 56.00 | 109.00 | 300.00 |
| Food | 23.46 | 7.52 | 0.00 | 20.00 | 23.00 | 26.00 | 37.00 | 82.00 |
| Loss | 2.22 | 1.56 | 0.00 | 1.35 | 2.00 | 2.79 | 4.44 | 36.36 |
| Food/Step | 1.28 | 1.27 | 0.00 | 0.48 | 0.80 | 1.46 | 4.20 | 7.00 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 42.26 | 19.72 | -0.1597 | 0.0137 |
| Last 100 | 37.46 | 18.95 | +0.1322 | 0.0406 |
| Last 200 | 40.51 | 22.57 | -0.0384 | 0.0097 |
| Last 500 | 41.85 | 22.39 | -0.0075 | 0.0024 |
| Last 1000 | 45.17 | 26.32 | -0.0137 | 0.0224 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 82 | 6,000 | 1.4% |
| Survival | 300 steps | 1,800 steps | 16.7% |

## Recommendations

Major changes needed: LR, reward structure, curriculum.

1. Epsilon 0.491 still high. Consider faster decay.

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

