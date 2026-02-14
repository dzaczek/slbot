# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-14 21:10:33  
**Total Episodes:** 1370  
**Training Sessions:** 5

## Verdict: NOT LEARNING (Confidence: 5%)

**Goal Feasibility:** IMPOSSIBLE with current setup

### Critical Issues
- CRITICAL: Learning Rate = 0.0 - Model CANNOT learn!

### Warnings
- Rewards flat: change = -13.9
- Loss very high (40.84) - unstable

### Positive Signals
- Epsilon low (0.146) - exploiting policy

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 695 | 86.2 | 56.7 | 23.5 | 0.4863 | 0.0% | 0.0% | 0.0% |
| S2 | WALL_AVOID | 675 | 110.8 | 77.2 | 31.9 | 0.6620 | 0.0% | 0.0% | 0.0% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 98.29 | 93.42 | -50.00 | 43.77 | 89.12 | 144.47 | 274.27 | 497.52 |
| Steps | 66.81 | 57.20 | 1.00 | 24.00 | 55.00 | 96.00 | 185.55 | 337.00 |
| Food | 27.68 | 15.98 | 0.00 | 21.00 | 27.00 | 36.00 | 56.00 | 97.00 |
| Loss | 23.06 | 116.54 | 0.00 | 1.29 | 2.20 | 3.56 | 19.74 | 996.87 |
| Food/Step | 0.57 | 0.67 | 0.00 | 0.33 | 0.43 | 0.60 | 1.40 | 6.33 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | -31.80 | 65.32 | +1.3054 | 0.0832 |
| Last 100 | -40.90 | 47.07 | +0.4362 | 0.0716 |
| Last 200 | -7.36 | 79.37 | -0.7630 | 0.3081 |
| Last 500 | 81.63 | 106.68 | -0.4769 | 0.4163 |
| Last 1000 | 96.72 | 100.75 | -0.1019 | 0.0853 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 97 | 6,000 | 1.6% |
| Survival | 337 steps | 1,800 steps | 18.7% |

## Recommendations

Fix critical issues: LR, rewards, episode length.

1. URGENT: LR=0. Reset learning rate or set min_lr floor.

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

### Q-Value & Gradient Analysis
![Q-Value & Gradient Analysis](chart_13_qvalue_gradients.png)

### Action Distribution Analysis
![Action Distribution Analysis](chart_14_action_distribution.png)

### Active Agents Over Time
![Active Agents Over Time](chart_15_auto_scaling.png)

