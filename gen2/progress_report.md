# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-16 22:04:25  
**Total Episodes:** 1962  
**Training Sessions:** 7

## Verdict: LEARNING (Confidence: 65%)

**Goal Feasibility:** POSSIBLE (25-60%)

### Warnings
- Loss very high (13.69) - unstable

### Positive Signals
- Rewards improving: +121.9
- Epsilon low (0.080) - exploiting policy

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 214 | 141.4 | 125.9 | 39.6 | 0.5520 | 0.9% | 80.8% | 17.8% |
| S2 | WALL_AVOID | 400 | 486.0 | 269.1 | 70.2 | 0.4690 | 0.2% | 69.8% | 29.0% |
| S3 | ENEMY_AVOID | 414 | 349.3 | 279.2 | 71.8 | 0.5430 | 0.2% | 99.0% | 0.7% |
| S4 | MASS_MANAGEMENT | 934 | 489.5 | 224.5 | 65.7 | 0.6706 | 0.3% | 99.6% | 0.1% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 421.24 | 618.17 | -20947.60 | 162.56 | 331.19 | 589.60 | 1152.40 | 3071.88 |
| Steps | 234.36 | 245.21 | 1.00 | 64.00 | 157.00 | 320.00 | 664.90 | 2000.00 |
| Food | 65.07 | 42.36 | 0.00 | 34.00 | 58.00 | 85.00 | 141.00 | 415.00 |
| Loss | 8.47 | 8.11 | 0.00 | 3.11 | 5.99 | 11.48 | 22.75 | 82.89 |
| Food/Step | 0.59 | 0.85 | 0.00 | 0.24 | 0.34 | 0.58 | 1.57 | 9.60 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 574.42 | 419.34 | +5.8638 | 0.0407 |
| Last 100 | 528.77 | 414.48 | +1.8964 | 0.0174 |
| Last 200 | 492.40 | 406.70 | +0.7473 | 0.0113 |
| Last 500 | 511.01 | 422.97 | -0.0933 | 0.0010 |
| Last 1000 | 480.07 | 801.24 | +0.0830 | 0.0009 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 7 | 0.4% | 361.3 | 558.1 |
| SnakeCollision | 1792 | 91.3% | 211.5 | 405.6 |
| MaxSteps | 158 | 8.1% | 489.9 | 593.9 |
| BrowserError | 5 | 0.3% | 189.8 | 390.6 |

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

