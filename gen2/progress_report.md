# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-16 13:50:05  
**Total Episodes:** 150  
**Training Sessions:** 4

## Verdict: NOT LEARNING (Confidence: 55%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

### Positive Signals
- Food collection improving (slope=0.0672/ep)

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 150 | 100.9 | 85.3 | 31.0 | 0.6493 | 1.3% | 94.7% | 3.3% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 100.94 | 71.13 | -15.00 | 54.09 | 83.13 | 129.71 | 246.65 | 320.47 |
| Steps | 85.29 | 74.51 | 1.00 | 31.00 | 62.00 | 119.25 | 260.65 | 300.00 |
| Food | 30.97 | 17.09 | 0.00 | 21.00 | 27.00 | 37.75 | 71.55 | 85.00 |
| Loss | 5.70 | 8.04 | 0.00 | 1.21 | 2.96 | 5.43 | 20.98 | 40.93 |
| Food/Step | 0.65 | 0.83 | 0.00 | 0.30 | 0.41 | 0.67 | 1.75 | 6.67 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 102.44 | 75.64 | +0.3308 | 0.0040 |
| Last 100 | 104.81 | 71.49 | +0.1819 | 0.0054 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 2 | 1.3% | 277.0 | 251.0 |
| SnakeCollision | 142 | 94.7% | 75.4 | 92.3 |
| MaxSteps | 5 | 3.3% | 300.0 | 292.5 |
| BrowserError | 1 | 0.7% | 36.0 | 72.1 |

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 85 | 6,000 | 1.4% |
| Survival | 300 steps | 1,800 steps | 16.7% |

## Recommendations

Fine-tune hyperparameters, increase training duration.

1. Epsilon 0.595 still high. Consider faster decay.

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

### Goal Progress Over Time
![Goal Progress Over Time](chart_11b_goal_over_time.png)

### Hyperparameter Analysis
![Hyperparameter Analysis](chart_12_hyperparameter_analysis.png)

### Q-Value & Gradient Analysis
![Q-Value & Gradient Analysis](chart_13_qvalue_gradients.png)

### Action Distribution Analysis
![Action Distribution Analysis](chart_14_action_distribution.png)

### Active Agents Over Time
![Active Agents Over Time](chart_15_auto_scaling.png)

### MaxSteps Analysis
![MaxSteps Analysis](chart_16_maxsteps_analysis.png)

### Survival Percentiles
![Survival Percentiles](chart_17_survival_percentiles.png)

### Steps vs Food vs Episode (3D)
![Steps vs Food vs Episode (3D)](chart_18_3d_steps_food_episode.png)

