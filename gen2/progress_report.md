# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-16 16:13:44  
**Total Episodes:** 667  
**Training Sessions:** 7

## Verdict: LEARNING (Confidence: 95%)

**Goal Feasibility:** LIKELY (>60%)

### Positive Signals
- Rewards improving: +163.7
- Positive reward trend (slope=0.5288, R²=0.092)
- Episodes getting longer (slope=0.307/ep)
- Food collection improving (slope=0.0649/ep)
- Epsilon low (0.089) - exploiting policy

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 214 | 141.4 | 125.9 | 39.6 | 0.5520 | 0.9% | 80.8% | 17.8% |
| S2 | WALL_AVOID | 339 | 493.7 | 273.5 | 70.8 | 0.4592 | 0.0% | 68.4% | 30.4% |
| S3 | ENEMY_AVOID | 114 | 390.8 | 383.2 | 91.7 | 0.5205 | 0.0% | 97.4% | 2.6% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 363.07 | 335.79 | -144.80 | 107.91 | 256.50 | 553.91 | 939.30 | 2629.01 |
| Steps | 244.87 | 261.68 | 1.00 | 65.50 | 165.00 | 343.50 | 500.00 | 2000.00 |
| Food | 64.35 | 49.58 | 0.00 | 28.50 | 55.00 | 91.00 | 133.70 | 415.00 |
| Loss | 4.60 | 5.69 | 0.00 | 1.75 | 3.01 | 4.68 | 15.54 | 40.93 |
| Food/Step | 0.50 | 0.72 | 0.00 | 0.24 | 0.30 | 0.47 | 1.14 | 6.67 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 493.44 | 264.99 | -1.3642 | 0.0055 |
| Last 100 | 478.30 | 260.36 | +0.2073 | 0.0005 |
| Last 200 | 356.40 | 265.31 | +2.0176 | 0.1928 |
| Last 500 | 447.40 | 346.08 | -0.1124 | 0.0022 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 2 | 0.3% | 277.0 | 251.0 |
| SnakeCollision | 516 | 77.4% | 180.1 | 261.0 |
| MaxSteps | 144 | 21.6% | 478.5 | 729.4 |
| BrowserError | 5 | 0.7% | 189.8 | 390.6 |

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 415 | 6,000 | 6.9% |
| Survival | 2000 steps | 1,800 steps | 111.1% |

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

