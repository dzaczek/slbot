# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-16 19:19:10  
**Total Episodes:** 1359  
**Training Sessions:** 7

## Verdict: LEARNING (Confidence: 70%)

**Goal Feasibility:** POSSIBLE (25-60%)

### Positive Signals
- Rewards improving: +52.0
- Epsilon low (0.080) - exploiting policy

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 214 | 141.4 | 125.9 | 39.6 | 0.5520 | 0.9% | 80.8% | 17.8% |
| S2 | WALL_AVOID | 400 | 486.0 | 269.1 | 70.2 | 0.4690 | 0.2% | 69.8% | 29.0% |
| S3 | ENEMY_AVOID | 414 | 349.3 | 279.2 | 71.8 | 0.5430 | 0.2% | 99.0% | 0.7% |
| S4 | MASS_MANAGEMENT | 331 | 499.3 | 217.2 | 64.7 | 0.6344 | 0.0% | 100.0% | 0.0% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 393.35 | 356.27 | -144.80 | 133.32 | 296.71 | 554.68 | 1011.94 | 3071.88 |
| Steps | 236.98 | 249.13 | 1.00 | 63.00 | 159.00 | 329.50 | 624.60 | 2000.00 |
| Food | 64.52 | 43.98 | 0.00 | 32.00 | 57.00 | 86.00 | 135.10 | 415.00 |
| Loss | 5.88 | 5.42 | 0.00 | 2.46 | 4.25 | 7.48 | 16.33 | 40.93 |
| Food/Step | 0.54 | 0.72 | 0.00 | 0.24 | 0.32 | 0.55 | 1.50 | 6.67 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 586.43 | 467.34 | -1.9843 | 0.0038 |
| Last 100 | 551.75 | 465.62 | +0.2577 | 0.0003 |
| Last 200 | 515.58 | 446.88 | +0.8146 | 0.0111 |
| Last 500 | 458.73 | 403.55 | +0.3880 | 0.0193 |
| Last 1000 | 424.09 | 370.48 | +0.1031 | 0.0065 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 4 | 0.3% | 216.0 | 248.8 |
| SnakeCollision | 1193 | 87.8% | 205.2 | 349.4 |
| MaxSteps | 157 | 11.6% | 480.3 | 731.1 |
| BrowserError | 5 | 0.4% | 189.8 | 390.6 |

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 415 | 6,000 | 6.9% |
| Survival | 2000 steps | 1,800 steps | 111.1% |

## Recommendations

Keep training. Monitor for sustained improvement.

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

