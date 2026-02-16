# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-16 14:00:13  
**Total Episodes:** 190  
**Training Sessions:** 5

## Verdict: LEARNING (Confidence: 75%)

**Goal Feasibility:** POSSIBLE (25-60%)

### Positive Signals
- Positive reward trend (slope=0.6943, R²=0.160)
- Episodes getting longer (slope=0.788/ep)
- Food collection improving (slope=0.1579/ep)

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 190 | 123.8 | 110.3 | 35.8 | 0.5854 | 1.1% | 85.8% | 12.6% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 123.77 | 95.15 | -144.80 | 57.66 | 89.46 | 176.26 | 323.38 | 378.45 |
| Steps | 110.27 | 95.52 | 1.00 | 32.25 | 81.00 | 161.50 | 300.00 | 300.00 |
| Food | 35.85 | 21.23 | 0.00 | 22.00 | 28.50 | 48.00 | 78.65 | 91.00 |
| Loss | 5.03 | 7.29 | 0.00 | 1.25 | 2.72 | 4.60 | 19.65 | 40.93 |
| Food/Step | 0.59 | 0.76 | 0.00 | 0.28 | 0.38 | 0.59 | 1.59 | 6.67 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 187.15 | 121.21 | +2.9705 | 0.1251 |
| Last 100 | 150.22 | 109.79 | +1.4152 | 0.1385 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 2 | 1.1% | 277.0 | 251.0 |
| SnakeCollision | 163 | 85.8% | 80.7 | 99.6 |
| MaxSteps | 24 | 12.6% | 300.0 | 279.5 |
| BrowserError | 1 | 0.5% | 36.0 | 72.1 |

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 91 | 6,000 | 1.5% |
| Survival | 300 steps | 1,800 steps | 16.7% |

## Recommendations

Keep training. Monitor for sustained improvement.

1. Epsilon 0.479 still high. Consider faster decay.

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

