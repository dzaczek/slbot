# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-13 17:16:25  
**Total Episodes:** 7695  
**Training Sessions:** 1

## Verdict: NOT LEARNING (Confidence: 35%)

**Goal Feasibility:** VERY UNLIKELY (<5%)

### Critical Issues
- Very short episodes: avg=42 steps

### Warnings
- Rewards flat: change = 1.6

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 100 | 53.7 | 29.7 | 20.7 | 1.2942 | 0.0% | 95.0% | 5.0% |
| S2 | WALL_AVOID | 6784 | 51.0 | 42.2 | 24.4 | 1.1771 | 0.0% | 99.7% | 0.3% |
| S3 | ENEMY_AVOID | 811 | 55.3 | 42.8 | 24.2 | 1.1677 | 0.0% | 100.0% | 0.0% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 51.45 | 33.30 | -30.00 | 29.97 | 43.56 | 65.59 | 116.15 | 269.43 |
| Steps | 42.12 | 36.41 | 1.00 | 15.00 | 32.00 | 60.00 | 116.00 | 200.00 |
| Food | 24.33 | 7.59 | 0.00 | 21.00 | 23.00 | 28.00 | 38.00 | 64.00 |
| Loss | 1.95 | 21.12 | 0.00 | 0.07 | 0.12 | 0.21 | 1.44 | 504.77 |
| Food/Step | 1.18 | 1.24 | 0.00 | 0.46 | 0.70 | 1.25 | 4.20 | 9.50 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 57.16 | 38.35 | +0.8078 | 0.0924 |
| Last 100 | 57.60 | 37.42 | +0.0846 | 0.0043 |
| Last 200 | 61.90 | 44.05 | -0.0584 | 0.0059 |
| Last 500 | 59.63 | 40.72 | +0.0180 | 0.0041 |
| Last 1000 | 56.58 | 38.78 | +0.0106 | 0.0062 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| SnakeCollision | 7671 | 99.7% | 41.7 | 51.0 |
| MaxSteps | 24 | 0.3% | 179.2 | 194.0 |

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 64 | 6,000 | 1.1% |
| Survival | 200 steps | 1,800 steps | 11.1% |

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

### Metric Correlations
![Metric Correlations](chart_05_correlations.png)

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

