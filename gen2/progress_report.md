# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-16 14:30:55  
**Total Episodes:** 326  
**Training Sessions:** 5

## Verdict: LEARNING (Confidence: 90%)

**Goal Feasibility:** LIKELY (>60%)

### Positive Signals
- Rewards improving: +335.2
- Positive reward trend (slope=1.8613, R²=0.390)
- Episodes getting longer (slope=0.935/ep)
- Food collection improving (slope=0.1992/ep)

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 214 | 141.4 | 125.9 | 39.6 | 0.5520 | 0.9% | 80.8% | 17.8% |
| S2 | WALL_AVOID | 112 | 532.2 | 286.1 | 75.2 | 0.3431 | 0.0% | 63.4% | 33.0% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 275.71 | 280.62 | -144.80 | 72.46 | 169.97 | 367.02 | 929.14 | 1203.89 |
| Steps | 180.94 | 154.03 | 1.00 | 55.00 | 140.00 | 300.00 | 500.00 | 500.00 |
| Food | 51.80 | 35.15 | 0.00 | 23.00 | 40.50 | 75.00 | 126.00 | 163.00 |
| Loss | 4.01 | 5.78 | 0.00 | 1.30 | 2.57 | 4.18 | 13.44 | 40.93 |
| Food/Step | 0.48 | 0.62 | 0.00 | 0.26 | 0.31 | 0.46 | 1.14 | 6.67 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 514.10 | 272.05 | +4.3866 | 0.0541 |
| Last 100 | 523.30 | 324.37 | -0.3302 | 0.0009 |
| Last 200 | 386.37 | 306.09 | +2.5661 | 0.2343 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 2 | 0.6% | 277.0 | 251.0 |
| SnakeCollision | 244 | 74.8% | 113.0 | 174.9 |
| MaxSteps | 75 | 23.0% | 398.7 | 596.6 |
| BrowserError | 5 | 1.5% | 189.8 | 390.6 |

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 163 | 6,000 | 2.7% |
| Survival | 500 steps | 1,800 steps | 27.8% |

## Recommendations

Training looks healthy. Continue and monitor.

1. No critical issues. Continue training.

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

