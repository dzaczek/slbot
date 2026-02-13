# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-13 22:19:31  
**Total Episodes:** 160  
**Training Sessions:** 2

## Verdict: NOT LEARNING (Confidence: 55%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

### Positive Signals
- Food collection improving (slope=0.0190/ep)

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 160 | 73.6 | 67.7 | 25.5 | 1.0308 | 0.0% | 0.0% | 0.0% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 73.61 | 56.29 | -97.88 | 48.58 | 68.42 | 104.56 | 160.44 | 296.85 |
| Steps | 67.74 | 74.84 | 1.00 | 16.75 | 45.00 | 86.50 | 300.00 | 300.00 |
| Food | 25.53 | 12.23 | 0.00 | 21.00 | 25.50 | 32.00 | 44.10 | 82.00 |
| Loss | 2.00 | 3.04 | 0.00 | 1.12 | 1.55 | 2.23 | 3.27 | 36.36 |
| Food/Step | 1.03 | 1.18 | 0.00 | 0.39 | 0.58 | 1.09 | 3.81 | 6.67 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 82.72 | 53.34 | -1.5787 | 0.1824 |
| Last 100 | 74.62 | 45.62 | +0.0277 | 0.0003 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 82 | 6,000 | 1.4% |
| Survival | 300 steps | 1,800 steps | 16.7% |

## Recommendations

Fine-tune hyperparameters, increase training duration.

1. Epsilon 0.779 still high. Consider faster decay.

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

