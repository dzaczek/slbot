# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-13 22:08:20  
**Total Episodes:** 12  
**Training Sessions:** 2

## Verdict: NOT LEARNING (Confidence: 10%)

**Goal Feasibility:** 

### Critical Issues
- Insufficient data (< 20 episodes)

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 12 | 5.2 | 212.9 | 14.2 | 0.1735 | 0.0% | 0.0% | 0.0% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 5.25 | 115.01 | -97.88 | -80.42 | -47.45 | 57.64 | 211.41 | 296.85 |
| Steps | 212.92 | 118.32 | 1.00 | 89.00 | 300.00 | 300.00 | 300.00 | 300.00 |
| Food | 14.17 | 24.16 | 0.00 | 0.00 | 0.00 | 23.75 | 56.15 | 82.00 |
| Loss | 5.54 | 10.11 | 0.00 | 0.25 | 0.25 | 8.30 | 22.36 | 36.36 |
| Food/Step | 0.17 | 0.29 | 0.00 | 0.00 | 0.00 | 0.33 | 0.67 | 1.00 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 82 | 6,000 | 1.4% |
| Survival | 300 steps | 1,800 steps | 16.7% |

## Recommendations



1. Epsilon 0.941 still high. Consider faster decay.

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

### Q-Value & Gradient Analysis
![Q-Value & Gradient Analysis](chart_13_qvalue_gradients.png)

### Action Distribution Analysis
![Action Distribution Analysis](chart_14_action_distribution.png)

