# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-14 00:36:18  
**Total Episodes:** 1911  
**Training Sessions:** 3

## Verdict: NOT LEARNING (Confidence: 35%)

**Goal Feasibility:** VERY UNLIKELY (<5%)

### Critical Issues
- Very short episodes: avg=44 steps

### Warnings
- Rewards flat: change = 2.1

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 200 | 76.8 | 64.8 | 26.1 | 1.0802 | 0.0% | 0.0% | 0.0% |
| S2 | WALL_AVOID | 1645 | 50.0 | 40.0 | 23.7 | 1.2389 | 0.0% | 0.0% | 0.0% |
| S3 | ENEMY_AVOID | 66 | 97.2 | 86.5 | 31.7 | 0.7355 | 0.0% | 0.0% | 0.0% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 54.45 | 43.16 | -97.88 | 30.26 | 43.50 | 66.00 | 134.00 | 441.72 |
| Steps | 44.19 | 46.85 | 1.00 | 13.00 | 29.00 | 59.00 | 128.00 | 369.00 |
| Food | 24.25 | 8.89 | 0.00 | 20.00 | 23.00 | 27.00 | 41.00 | 91.00 |
| Loss | 2.17 | 1.60 | 0.00 | 1.19 | 1.96 | 2.87 | 4.62 | 36.36 |
| Food/Step | 1.20 | 1.24 | 0.00 | 0.44 | 0.75 | 1.36 | 4.00 | 7.00 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 90.69 | 81.33 | +1.3100 | 0.0540 |
| Last 100 | 96.47 | 75.57 | -0.0270 | 0.0001 |
| Last 200 | 104.67 | 76.73 | -0.1177 | 0.0078 |
| Last 500 | 66.85 | 60.52 | +0.1838 | 0.1921 |
| Last 1000 | 54.57 | 47.53 | +0.0610 | 0.1371 |

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

1. Epsilon 0.310 still high. Consider faster decay.

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

