# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-16 10:36:34  
**Total Episodes:** 8890  
**Training Sessions:** 15

## Verdict: NOT LEARNING (Confidence: 35%)

**Goal Feasibility:** VERY UNLIKELY (<5%)

### Critical Issues
- Rewards DECLINING: -82.5

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 400 | 99.4 | 103.6 | 30.8 | 0.6613 | 8.0% | 81.5% | 10.5% |
| S2 | WALL_AVOID | 383 | 231.0 | 140.3 | 30.5 | 0.7597 | 12.3% | 70.8% | 17.0% |
| S3 | ENEMY_AVOID | 8107 | 166.1 | 77.2 | 32.0 | 0.9783 | 5.7% | 92.9% | 1.3% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 165.87 | 316.65 | -40.65 | 68.07 | 113.25 | 190.54 | 377.26 | 4390.02 |
| Steps | 81.11 | 120.33 | 1.00 | 23.00 | 55.00 | 99.00 | 205.00 | 1000.00 |
| Food | 31.85 | 17.62 | 0.00 | 21.00 | 28.00 | 41.00 | 65.00 | 135.00 |
| Loss | 9.34 | 6.66 | 0.00 | 4.79 | 7.98 | 12.20 | 20.69 | 119.45 |
| Food/Step | 0.95 | 1.31 | 0.00 | 0.39 | 0.52 | 0.86 | 3.80 | 11.00 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 145.46 | 113.58 | -0.7954 | 0.0102 |
| Last 100 | 151.41 | 111.77 | -0.5065 | 0.0171 |
| Last 200 | 146.06 | 108.46 | +0.0128 | 0.0000 |
| Last 500 | 143.79 | 103.36 | +0.0294 | 0.0017 |
| Last 1000 | 138.55 | 102.00 | +0.0075 | 0.0005 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 545 | 6.1% | 139.9 | 256.2 |
| SnakeCollision | 8129 | 91.4% | 61.2 | 129.8 |
| MaxSteps | 209 | 2.4% | 703.8 | 1337.2 |
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

