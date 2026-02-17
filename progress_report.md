# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-17 22:53:42  
**Total Episodes:** 5688  
**Training Sessions:** 20

## Verdict: NOT LEARNING (Confidence: 40%)

**Goal Feasibility:** VERY UNLIKELY (<5%)

### Critical Issues
- Rewards DECLINING: -501.0

### Positive Signals
- Epsilon low (0.091) - exploiting policy

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 2414 | 224.2 | 231.7 | 83.3 | 0.5339 | 0.1% | 65.8% | 34.0% |
| S2 | WALL_AVOID | 719 | 525.2 | 281.3 | 84.3 | 0.4580 | 0.1% | 70.8% | 28.5% |
| S3 | ENEMY_AVOID | 714 | 16.7 | 276.3 | 64.5 | 0.6791 | 0.1% | 96.8% | 3.1% |
| S4 | MASS_MANAGEMENT | 1841 | -958.3 | 440.3 | 59.1 | 0.6278 | 0.2% | 87.2% | 12.6% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | -146.53 | 2944.73 | -35299.32 | 104.84 | 260.93 | 449.35 | 960.98 | 4632.57 |
| Steps | 311.10 | 414.64 | 1.00 | 85.00 | 201.00 | 315.00 | 1128.05 | 2000.00 |
| Food | 73.26 | 47.58 | 0.00 | 37.00 | 65.00 | 101.00 | 170.00 | 418.00 |
| Loss | 5.65 | 8.96 | 0.00 | 1.14 | 2.73 | 6.81 | 18.88 | 159.03 |
| Food/Step | 0.57 | 0.86 | 0.00 | 0.29 | 0.36 | 0.51 | 1.50 | 11.00 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 575.84 | 326.03 | +2.2236 | 0.0097 |
| Last 100 | 490.44 | 333.85 | +3.0360 | 0.0689 |
| Last 200 | 535.24 | 322.01 | -0.1470 | 0.0007 |
| Last 500 | 489.49 | 342.93 | +0.5515 | 0.0539 |
| Last 1000 | 366.10 | 320.12 | +0.4409 | 0.1581 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 8 | 0.1% | 343.6 | 515.3 |
| SnakeCollision | 4394 | 77.3% | 195.4 | 327.7 |
| MaxSteps | 1280 | 22.5% | 707.8 | -1776.6 |
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

### Steps vs Food vs Episode (3D rotating)
![Steps vs Food vs Episode (3D rotating)](img/3d_training.gif)

### Steps vs Food vs Episode (3D static)
![Steps vs Food vs Episode (3D static)](chart_18_3d_steps_food_episode.png)

### Steps vs Reward vs Episode — Bubble (3D rotating)
![Steps vs Reward vs Episode — Bubble (3D rotating)](img/bubble_training.gif)

### Steps vs Reward vs Episode — Bubble (3D static)
![Steps vs Reward vs Episode — Bubble (3D static)](chart_19_bubble_training.png)

## AI Supervisor — Recent Changes

| Time | Episode | Changes | Reasoning |
|------|---------|---------|----------|
| 02-17 13:20 | 3400 | `death_snake`=-70.0, `enemy_proximity_penalty`=2.0 | The agent is dying primarily from snake collisions (51.6%) in stage 1, which should focus on food collection. The high reward trend (+6925) shows improvement, but the negative Q-values and high collision rate suggest the death penalties need strengthening to discourage risky behavior. |
| 02-17 15:00 | 3600 | `death_snake`=-45.0 | The agent shows strong improvement (reward trend +1272) but has a critical problem: 63.6% deaths from snake collisions in Stage 1, which should focus on food collection. The high sharp turn usage (40.5%) suggests erratic movement. Need to reduce snake collision penalty to allow more exploration and learning. |
| 02-17 15:33 | 3800 | `enemy_proximity_penalty`=2.0 | The agent is dying primarily from snake collisions (57.4%) in stage 1, which should focus on food collection. The high reward trend (+59.5) shows good learning progress, but the agent needs stronger penalties for risky behavior. Increasing enemy proximity penalty will discourage getting too close to other snakes. |
| 02-17 16:05 | 4000 | `death_snake`=-65.0, `enemy_proximity_penalty`=2.5 | The agent is performing well in Stage 1 with good food collection (82.39 avg) and reasonable survival. However, 52.8% snake collision deaths suggest the agent needs stronger penalties for risky behavior. The high sharp turn usage (40.6%) and low straight movement (7.6%) indicate erratic movement patterns that lead to self-collisions. |
| 02-17 16:39 | 4200 | `epsilon_target`=0.35, `death_snake`=-45.0 | The agent is showing good food collection (81.77 avg) but concerning signs: reward trend is negative (-16.16), nearly 50% snake collision deaths, and very low exploration (epsilon 0.08). The high sharp turn usage (39.7%) suggests erratic movement patterns that lead to self-collisions. |
| 02-17 17:14 | 4400 | `death_snake`=-60.0, `epsilon_target`=0.3 | The agent is showing good learning progress with improving rewards and steps, but has concerning death patterns - 45.8% snake deaths in stage 1 suggests the agent is being too aggressive with sharp turns (39.6%) and needs better survival incentives. The high action entropy (0.899) indicates good exploration but may need slightly more focus. |
| 02-17 18:54 | 4600 | `death_snake`=-75.0, `enemy_proximity_penalty`=2.0 | The agent is performing well in Stage 1 with strong reward trend (+152) and good food collection (89.75 avg). However, 55.6% snake collision deaths suggest the agent is being too aggressive or not learning proper avoidance. The high sharp turn usage (38.8%) indicates erratic movement patterns that may contribute to self-collisions. |
| 02-17 19:42 | 4800 | `death_snake`=-85.0, `enemy_proximity_penalty`=1.5 | The agent is dying to snake collisions 67.8% of the time in Stage 1, which should focus on food collection. The high snake death rate and preference for sharp turns (39.3%) suggests the agent is being too aggressive. Need to increase snake death penalty and reduce enemy penalties to encourage safer food-seeking behavior. |
| 02-17 20:25 | 5000 | `death_snake`=-95.0 | The agent is dying 80.8% from snake collisions in stage 1, which should focus on food collection. The high sharp turn usage (40.6%) and declining reward trend suggest the agent is struggling with basic navigation. The death penalty for snake collision needs to be more severe to discourage this behavior. |
| 02-17 21:15 | 5200 | `death_snake`=-100, `epsilon_target`=0.35 | The agent has a severe snake collision problem (81.2% death rate) while in Stage 1, which should focus on food collection. The high death rate from self-collision suggests the agent needs stronger penalties for risky maneuvers and potentially higher exploration to learn safer paths. |
| 02-17 22:01 | 5400 | `enemy_proximity_penalty`=2.2, `enemy_approach_penalty`=1.5 | The agent is dying from snake collisions 80.2% of the time in Stage 2 (Wall Avoid), which suggests insufficient penalty for enemy proximity. The high sharp turn usage (38.9%) indicates reactive rather than proactive avoidance. Increasing enemy proximity penalties should encourage better positioning. |
| 02-17 22:48 | 5600 | `enemy_proximity_penalty`=2.8, `enemy_approach_penalty`=1.8, `lr`=0.00015 | Snake collisions dominate deaths at 74.8%, indicating the agent struggles with enemy avoidance in Stage 2. The high reward trend (+174) shows learning progress, but enemy proximity penalties need strengthening. Also, the low learning rate may be limiting adaptation speed given the stable loss trend. |

**Total consultations:** 12  
**Most adjusted:** `death_snake` (9x), `enemy_proximity_penalty` (7x), `epsilon_target` (3x), `enemy_approach_penalty` (2x), `lr` (1x)

