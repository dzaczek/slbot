# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-13 17:40:34  
**Total Episodes:** 8033  
**Training Sessions:** 1

## Verdict: NOT LEARNING (Confidence: 35%)

**Goal Feasibility:** VERY UNLIKELY (<5%)

### Critical Issues
- Very short episodes: avg=42 steps

### Warnings
- Rewards flat: change = 1.9

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 100 | 53.7 | 29.7 | 20.7 | 1.2942 | 0.0% | 95.0% | 5.0% |
| S2 | WALL_AVOID | 6784 | 51.0 | 42.2 | 24.4 | 1.1771 | 0.0% | 99.7% | 0.3% |
| S3 | ENEMY_AVOID | 1149 | 55.3 | 41.7 | 24.2 | 1.2021 | 0.0% | 100.0% | 0.0% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 51.61 | 33.59 | -31.88 | 30.03 | 43.67 | 65.86 | 116.64 | 269.43 |
| Steps | 42.00 | 36.33 | 1.00 | 14.00 | 32.00 | 59.00 | 115.00 | 208.00 |
| Food | 24.32 | 7.62 | 0.00 | 21.00 | 23.00 | 28.00 | 38.00 | 64.00 |
| Loss | 1.88 | 20.68 | 0.00 | 0.07 | 0.12 | 0.21 | 1.34 | 504.77 |
| Food/Step | 1.18 | 1.25 | 0.00 | 0.46 | 0.70 | 1.25 | 4.20 | 9.50 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 57.42 | 53.75 | -1.1240 | 0.0911 |
| Last 100 | 59.09 | 50.87 | -0.0356 | 0.0004 |
| Last 200 | 57.81 | 44.00 | +0.0314 | 0.0017 |
| Last 500 | 57.84 | 41.16 | -0.0127 | 0.0020 |
| Last 1000 | 57.32 | 39.34 | +0.0029 | 0.0005 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| SnakeCollision | 8009 | 99.7% | 41.6 | 51.2 |
| MaxSteps | 24 | 0.3% | 179.2 | 194.0 |

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 64 | 6,000 | 1.1% |
| Survival | 208 steps | 1,800 steps | 11.6% |

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

