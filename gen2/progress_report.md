# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-16 09:23:39  
**Total Episodes:** 8233  
**Training Sessions:** 15

## Verdict: NOT LEARNING (Confidence: 40%)

**Goal Feasibility:** VERY UNLIKELY (<5%)

### Critical Issues
- Rewards DECLINING: -88.7

### Positive Signals
- Epsilon low (0.086) - exploiting policy

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 400 | 99.4 | 103.6 | 30.8 | 0.6613 | 8.0% | 81.5% | 10.5% |
| S2 | WALL_AVOID | 383 | 231.0 | 140.3 | 30.5 | 0.7597 | 12.3% | 70.8% | 17.0% |
| S3 | ENEMY_AVOID | 7450 | 168.7 | 77.9 | 31.9 | 0.9783 | 5.7% | 92.9% | 1.4% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 168.24 | 327.71 | -40.65 | 68.13 | 113.04 | 190.48 | 384.88 | 4390.02 |
| Steps | 82.09 | 123.89 | 1.00 | 23.00 | 55.00 | 99.00 | 207.00 | 1000.00 |
| Food | 31.76 | 17.58 | 0.00 | 21.00 | 28.00 | 41.00 | 65.00 | 135.00 |
| Loss | 9.55 | 6.81 | 0.00 | 4.87 | 8.23 | 12.51 | 21.22 | 119.45 |
| Food/Step | 0.95 | 1.31 | 0.00 | 0.39 | 0.52 | 0.86 | 3.73 | 11.00 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 134.80 | 95.70 | +0.0257 | 0.0000 |
| Last 100 | 134.27 | 90.93 | -0.0545 | 0.0003 |
| Last 200 | 138.62 | 90.23 | -0.0683 | 0.0019 |
| Last 500 | 137.33 | 100.26 | +0.0487 | 0.0049 |
| Last 1000 | 127.83 | 89.88 | +0.0296 | 0.0091 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 502 | 6.1% | 141.0 | 259.3 |
| SnakeCollision | 7515 | 91.3% | 60.9 | 129.8 |
| MaxSteps | 209 | 2.5% | 703.8 | 1337.2 |
| BrowserError | 7 | 0.1% | 2.7 | 6.0 |

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 135 | 6,000 | 2.2% |
| Survival | 1000 steps | 1,800 steps | 55.6% |

## Recommendations

Major changes needed: LR, reward structure, curriculum.

1. Episodes too short. Reduce death penalties or add survival bonus.

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

