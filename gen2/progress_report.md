# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-16 11:48:04  
**Total Episodes:** 301  
**Training Sessions:** 7

## Verdict: NOT LEARNING (Confidence: 50%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

### Warnings
- Rewards flat: change = 43.6
- Loss very high (14.30) - unstable

### Positive Signals
- Positive reward trend (slope=0.2926, R²=0.105)

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 200 | 95.7 | 76.3 | 31.1 | 0.6631 | 10.5% | 88.5% | 1.0% |
| S2 | WALL_AVOID | 101 | 157.5 | 83.1 | 31.3 | 0.5085 | 9.9% | 90.1% | 0.0% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 116.42 | 78.64 | -16.70 | 61.62 | 106.59 | 154.91 | 241.40 | 703.35 |
| Steps | 78.60 | 60.32 | 1.00 | 35.00 | 65.00 | 106.00 | 198.00 | 315.00 |
| Food | 31.17 | 14.71 | 0.00 | 21.00 | 29.00 | 39.00 | 59.00 | 105.00 |
| Loss | 6.97 | 6.36 | 0.00 | 3.09 | 5.23 | 8.07 | 18.50 | 50.43 |
| Food/Step | 0.61 | 0.54 | 0.00 | 0.35 | 0.44 | 0.59 | 1.82 | 3.80 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 172.66 | 115.45 | -1.4727 | 0.0339 |
| Last 100 | 158.64 | 97.13 | +0.2199 | 0.0043 |
| Last 200 | 127.33 | 86.61 | +0.5052 | 0.1134 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 31 | 10.3% | 152.0 | 146.2 |
| SnakeCollision | 268 | 89.0% | 68.5 | 111.9 |
| MaxSteps | 2 | 0.7% | 300.0 | 258.1 |

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 105 | 6,000 | 1.8% |
| Survival | 315 steps | 1,800 steps | 17.5% |

## Recommendations

Fine-tune hyperparameters, increase training duration.

1. Epsilon 0.484 still high. Consider faster decay.

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

