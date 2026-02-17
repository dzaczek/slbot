# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-17 20:07:53  
**Total Episodes:** 4989  
**Training Sessions:** 19

## Verdict: NOT LEARNING (Confidence: 35%)

**Goal Feasibility:** VERY UNLIKELY (<5%)

### Critical Issues
- Rewards DECLINING: -993.8

### Warnings
- Reward in plateau (< 1% change over 200 eps)

### Positive Signals
- Epsilon low (0.090) - exploiting policy

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 2034 | 211.1 | 218.3 | 79.1 | 0.5371 | 0.1% | 63.3% | 36.5% |
| S2 | WALL_AVOID | 400 | 486.0 | 269.1 | 70.2 | 0.4690 | 0.2% | 69.8% | 29.0% |
| S3 | ENEMY_AVOID | 714 | 16.7 | 276.3 | 64.5 | 0.6791 | 0.1% | 96.8% | 3.1% |
| S4 | MASS_MANAGEMENT | 1841 | -958.3 | 440.3 | 59.1 | 0.6278 | 0.2% | 87.2% | 12.6% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | -226.21 | 3133.51 | -35299.32 | 98.85 | 251.69 | 429.15 | 938.00 | 4632.57 |
| Steps | 312.62 | 437.08 | 1.00 | 80.00 | 191.00 | 300.00 | 2000.00 | 2000.00 |
| Food | 68.95 | 44.87 | 0.00 | 35.00 | 62.00 | 97.00 | 146.00 | 418.00 |
| Loss | 6.17 | 9.44 | 0.00 | 1.19 | 3.20 | 7.67 | 19.88 | 159.03 |
| Food/Step | 0.59 | 0.88 | 0.00 | 0.28 | 0.37 | 0.53 | 1.57 | 11.00 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 185.43 | 178.08 | -1.1791 | 0.0091 |
| Last 100 | 225.85 | 206.99 | -1.8214 | 0.0645 |
| Last 200 | 226.15 | 225.48 | -0.2078 | 0.0028 |
| Last 500 | 289.26 | 259.46 | -0.3531 | 0.0386 |
| Last 1000 | 239.12 | 215.23 | +0.1059 | 0.0202 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 8 | 0.2% | 343.6 | 515.3 |
| SnakeCollision | 3862 | 77.4% | 191.6 | 332.5 |
| MaxSteps | 1113 | 22.3% | 732.0 | -2168.2 |
| InvalidFrame | 1 | 0.0% | 1197.0 | -5534.5 |
| BrowserError | 5 | 0.1% | 189.8 | 390.6 |

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

## AI Supervisor — Recent Changes

| Time | Episode | Changes | Reasoning |
|------|---------|---------|----------|
| 02-17 13:20 | 3400 | `death_snake`=-70.0, `enemy_proximity_penalty`=2.0 | The agent is dying primarily from snake collisions (51.6%) in stage 1, which should focus on food collection. The high r... |
| 02-17 15:00 | 3600 | `death_snake`=-45.0 | The agent shows strong improvement (reward trend +1272) but has a critical problem: 63.6% deaths from snake collisions i... |
| 02-17 15:33 | 3800 | `enemy_proximity_penalty`=2.0 | The agent is dying primarily from snake collisions (57.4%) in stage 1, which should focus on food collection. The high r... |
| 02-17 16:05 | 4000 | `death_snake`=-65.0, `enemy_proximity_penalty`=2.5 | The agent is performing well in Stage 1 with good food collection (82.39 avg) and reasonable survival. However, 52.8% sn... |
| 02-17 16:39 | 4200 | `epsilon_target`=0.35, `death_snake`=-45.0 | The agent is showing good food collection (81.77 avg) but concerning signs: reward trend is negative (-16.16), nearly 50... |
| 02-17 17:14 | 4400 | `death_snake`=-60.0, `epsilon_target`=0.3 | The agent is showing good learning progress with improving rewards and steps, but has concerning death patterns - 45.8% ... |
| 02-17 18:54 | 4600 | `death_snake`=-75.0, `enemy_proximity_penalty`=2.0 | The agent is performing well in Stage 1 with strong reward trend (+152) and good food collection (89.75 avg). However, 5... |
| 02-17 19:42 | 4800 | `death_snake`=-85.0, `enemy_proximity_penalty`=1.5 | The agent is dying to snake collisions 67.8% of the time in Stage 1, which should focus on food collection. The high sna... |

**Total consultations:** 8  
**Most adjusted:** `death_snake` (7x), `enemy_proximity_penalty` (5x), `epsilon_target` (2x)

