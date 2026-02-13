# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-13 17:14:18  
**Total Episodes:** 7669  
**Training Sessions:** 1

## Verdict: NOT LEARNING (Confidence: 35%)

**Goal Feasibility:** VERY UNLIKELY (<5%)

### Critical Issues
- Very short episodes: avg=42 steps

### Warnings
- Rewards flat: change = 1.4

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 100 | 53.7 | 29.7 | 20.7 | 1.2942 | 0.0% | 95.0% | 5.0% |
| S2 | WALL_AVOID | 6784 | 51.0 | 42.2 | 24.4 | 1.1771 | 0.0% | 99.7% | 0.3% |
| S3 | ENEMY_AVOID | 785 | 54.9 | 42.6 | 24.2 | 1.1705 | 0.0% | 100.0% | 0.0% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 51.40 | 33.23 | -30.00 | 29.97 | 43.55 | 65.47 | 115.91 | 269.43 |
| Steps | 42.10 | 36.39 | 1.00 | 15.00 | 32.00 | 59.00 | 116.00 | 200.00 |
| Food | 24.33 | 7.58 | 0.00 | 21.00 | 23.00 | 28.00 | 38.00 | 64.00 |
| Loss | 1.96 | 21.16 | 0.00 | 0.07 | 0.12 | 0.21 | 1.44 | 504.77 |
| Food/Step | 1.18 | 1.25 | 0.00 | 0.46 | 0.70 | 1.25 | 4.20 | 9.50 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 53.12 | 27.96 | -0.2054 | 0.0112 |
| Last 100 | 55.31 | 37.95 | -0.1185 | 0.0081 |
| Last 200 | 60.29 | 42.75 | -0.0534 | 0.0052 |
| Last 500 | 60.04 | 40.64 | +0.0039 | 0.0002 |
| Last 1000 | 55.84 | 38.33 | +0.0116 | 0.0077 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| SnakeCollision | 7645 | 99.7% | 41.7 | 50.9 |
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

