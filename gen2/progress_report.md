# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-13 21:19:59  
**Total Episodes:** 2348  
**Training Sessions:** 1

## Verdict: NOT LEARNING (Confidence: 35%)

**Goal Feasibility:** VERY UNLIKELY (<5%)

### Critical Issues
- Very short episodes: avg=36 steps

### Warnings
- Rewards flat: change = 2.0

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 2348 | 59.5 | 36.0 | 22.3 | 1.3543 | 0.0% | 96.4% | 3.6% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 59.53 | 30.53 | -98.91 | 46.49 | 57.82 | 74.63 | 110.25 | 193.13 |
| Steps | 35.98 | 34.18 | 1.00 | 11.00 | 26.00 | 49.00 | 107.00 | 150.00 |
| Food | 22.31 | 7.13 | 0.00 | 20.00 | 22.00 | 26.00 | 33.00 | 51.00 |
| Loss | 17.32 | 187.26 | 0.00 | 0.11 | 0.18 | 0.45 | 55.72 | 4270.31 |
| Food/Step | 1.35 | 1.35 | 0.00 | 0.52 | 0.85 | 1.58 | 4.75 | 7.00 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 66.51 | 31.17 | +0.0900 | 0.0017 |
| Last 100 | 63.40 | 27.06 | +0.0613 | 0.0043 |
| Last 200 | 63.93 | 28.83 | -0.0319 | 0.0041 |
| Last 500 | 61.59 | 32.49 | +0.0075 | 0.0011 |
| Last 1000 | 60.54 | 32.43 | +0.0053 | 0.0022 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| SnakeCollision | 2264 | 96.4% | 31.8 | 61.6 |
| MaxSteps | 84 | 3.6% | 150.0 | 3.3 |

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

