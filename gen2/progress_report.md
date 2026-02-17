# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-17 12:57:20  
**Total Episodes:** 3304  
**Training Sessions:** 12

## Verdict: NOT LEARNING (Confidence: 40%)

**Goal Feasibility:** VERY UNLIKELY (<5%)

### Critical Issues
- Rewards DECLINING: -1716.8
- Negative reward trend (slope=-1.1630)

### Positive Signals
- Episodes getting longer (slope=0.160/ep)
- Epsilon low (0.080) - exploiting policy

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 349 | 173.1 | 144.5 | 48.6 | 0.5765 | 0.6% | 77.7% | 21.5% |
| S2 | WALL_AVOID | 400 | 486.0 | 269.1 | 70.2 | 0.4690 | 0.2% | 69.8% | 29.0% |
| S3 | ENEMY_AVOID | 714 | 16.7 | 276.3 | 64.5 | 0.6791 | 0.1% | 96.8% | 3.1% |
| S4 | MASS_MANAGEMENT | 1841 | -958.3 | 440.3 | 59.1 | 0.6278 | 0.2% | 87.2% | 12.6% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | -453.24 | 3828.14 | -35299.32 | 118.47 | 290.46 | 510.81 | 1101.92 | 4632.57 |
| Steps | 352.90 | 522.81 | 1.00 | 65.00 | 163.00 | 360.25 | 2000.00 | 2000.00 |
| Food | 60.53 | 42.50 | 0.00 | 30.00 | 55.00 | 81.00 | 134.00 | 418.00 |
| Loss | 8.60 | 10.70 | 0.00 | 2.97 | 5.73 | 10.26 | 24.21 | 159.03 |
| Food/Step | 0.61 | 0.97 | 0.00 | 0.23 | 0.34 | 0.58 | 1.90 | 11.00 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 214.19 | 141.45 | -2.0879 | 0.0454 |
| Last 100 | 219.53 | 140.12 | -0.8136 | 0.0281 |
| Last 200 | -2677.88 | 5782.65 | +54.4073 | 0.2951 |
| Last 500 | -3557.51 | 6874.82 | -0.4742 | 0.0001 |
| Last 1000 | -2259.07 | 6148.83 | -4.2600 | 0.0400 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 7 | 0.2% | 361.3 | 558.1 |
| SnakeCollision | 2846 | 86.1% | 201.3 | 401.1 |
| MaxSteps | 445 | 13.5% | 1322.5 | -5930.9 |
| InvalidFrame | 1 | 0.0% | 1197.0 | -5534.5 |
| BrowserError | 5 | 0.2% | 189.8 | 390.6 |

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 418 | 6,000 | 7.0% |
| Survival | 2000 steps | 1,800 steps | 111.1% |

## Recommendations

Major changes needed: LR, reward structure, curriculum.

1. Average reward negative. Reduce penalties or boost food_reward.

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

