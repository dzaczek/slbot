# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-13 20:40:59  
**Total Episodes:** 1766  
**Training Sessions:** 1

## Verdict: NOT LEARNING (Confidence: 35%)

**Goal Feasibility:** VERY UNLIKELY (<5%)

### Critical Issues
- Very short episodes: avg=36 steps

### Warnings
- Rewards flat: change = 2.9

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 1766 | 59.1 | 36.0 | 22.2 | 1.3869 | 0.0% | 95.7% | 4.3% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 59.06 | 30.16 | -98.91 | 46.34 | 57.22 | 74.40 | 108.59 | 193.13 |
| Steps | 36.05 | 34.79 | 1.00 | 11.00 | 26.00 | 49.00 | 111.00 | 150.00 |
| Food | 22.20 | 7.08 | 0.00 | 20.00 | 22.00 | 26.00 | 33.00 | 51.00 |
| Loss | 22.98 | 215.63 | 0.00 | 0.13 | 0.23 | 0.73 | 85.23 | 4270.31 |
| Food/Step | 1.39 | 1.39 | 0.00 | 0.52 | 0.85 | 1.64 | 4.94 | 7.00 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 65.07 | 25.28 | +0.5059 | 0.0834 |
| Last 100 | 70.06 | 32.08 | -0.0908 | 0.0067 |
| Last 200 | 59.54 | 41.55 | +0.1373 | 0.0364 |
| Last 500 | 59.47 | 32.20 | +0.0185 | 0.0069 |
| Last 1000 | 60.64 | 30.47 | -0.0005 | 0.0000 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| SnakeCollision | 1690 | 95.7% | 30.9 | 61.4 |
| MaxSteps | 76 | 4.3% | 150.0 | 6.6 |

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 51 | 6,000 | 0.9% |
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

