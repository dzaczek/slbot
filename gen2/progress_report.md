# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-16 14:10:23  
**Total Episodes:** 245  
**Training Sessions:** 5

## Verdict: LEARNING (Confidence: 90%)

**Goal Feasibility:** LIKELY (>60%)

### Positive Signals
- Rewards improving: +203.7
- Positive reward trend (slope=1.9037, R²=0.341)
- Episodes getting longer (slope=1.110/ep)
- Food collection improving (slope=0.2722/ep)

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 214 | 141.4 | 125.9 | 39.6 | 0.5520 | 0.9% | 80.8% | 17.8% |
| S2 | WALL_AVOID | 31 | 613.0 | 305.3 | 90.0 | 0.3815 | 0.0% | 58.1% | 41.9% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 201.11 | 230.64 | -144.80 | 65.13 | 117.52 | 262.45 | 769.97 | 1203.89 |
| Steps | 148.61 | 131.60 | 1.00 | 42.00 | 100.00 | 259.00 | 479.40 | 500.00 |
| Food | 45.95 | 32.64 | 0.00 | 22.00 | 33.00 | 67.00 | 119.20 | 163.00 |
| Loss | 4.26 | 6.60 | 0.00 | 1.12 | 2.15 | 4.27 | 16.01 | 40.93 |
| Food/Step | 0.53 | 0.68 | 0.00 | 0.27 | 0.33 | 0.50 | 1.17 | 6.67 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 492.92 | 340.16 | +9.9128 | 0.1769 |
| Last 100 | 346.13 | 295.55 | +5.9086 | 0.3330 |
| Last 200 | 225.58 | 246.39 | +2.5740 | 0.3638 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 2 | 0.8% | 277.0 | 251.0 |
| SnakeCollision | 191 | 78.0% | 93.8 | 129.9 |
| MaxSteps | 51 | 20.8% | 351.0 | 468.3 |
| BrowserError | 1 | 0.4% | 36.0 | 72.1 |

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 163 | 6,000 | 2.7% |
| Survival | 500 steps | 1,800 steps | 27.8% |

## Recommendations

Training looks healthy. Continue and monitor.

1. Epsilon 0.351 still high. Consider faster decay.

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

