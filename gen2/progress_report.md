# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-16 09:54:56  
**Total Episodes:** 8540  
**Training Sessions:** 15

## Verdict: NOT LEARNING (Confidence: 35%)

**Goal Feasibility:** VERY UNLIKELY (<5%)

### Critical Issues
- Rewards DECLINING: -86.3

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 400 | 99.4 | 103.6 | 30.8 | 0.6613 | 8.0% | 81.5% | 10.5% |
| S2 | WALL_AVOID | 383 | 231.0 | 140.3 | 30.5 | 0.7597 | 12.3% | 70.8% | 17.0% |
| S3 | ENEMY_AVOID | 7757 | 167.0 | 77.4 | 31.9 | 0.9801 | 5.7% | 92.9% | 1.3% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 166.70 | 322.35 | -40.65 | 68.08 | 112.99 | 190.07 | 380.86 | 4390.02 |
| Steps | 81.41 | 122.11 | 1.00 | 23.00 | 55.00 | 99.00 | 205.00 | 1000.00 |
| Food | 31.74 | 17.56 | 0.00 | 21.00 | 28.00 | 41.00 | 65.00 | 135.00 |
| Loss | 9.45 | 6.74 | 0.00 | 4.84 | 8.07 | 12.36 | 20.93 | 119.45 |
| Food/Step | 0.96 | 1.31 | 0.00 | 0.39 | 0.52 | 0.86 | 3.80 | 11.00 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 146.87 | 99.15 | +0.0077 | 0.0000 |
| Last 100 | 145.65 | 97.66 | +0.0423 | 0.0002 |
| Last 200 | 131.80 | 95.38 | +0.2393 | 0.0210 |
| Last 500 | 130.53 | 92.83 | -0.0107 | 0.0003 |
| Last 1000 | 127.45 | 94.97 | +0.0266 | 0.0065 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 523 | 6.1% | 140.5 | 257.1 |
| SnakeCollision | 7801 | 91.3% | 60.8 | 129.4 |
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

