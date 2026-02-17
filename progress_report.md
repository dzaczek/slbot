# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-18 00:56:03  
**Total Episodes:** 6063  
**Training Sessions:** 20

## Verdict: NOT LEARNING (Confidence: 55%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

### Warnings
- Rewards flat: change = 24.0

### Positive Signals
- Food collection improving (slope=0.0101/ep)
- Epsilon low (0.080) - exploiting policy

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 2414 | 224.2 | 231.7 | 83.3 | 0.5339 | 0.1% | 65.8% | 34.0% |
| S2 | WALL_AVOID | 800 | 528.6 | 286.7 | 87.2 | 0.4505 | 0.1% | 70.5% | 28.9% |
| S3 | ENEMY_AVOID | 1008 | 208.9 | 322.6 | 85.9 | 0.6263 | 0.1% | 97.2% | 2.7% |
| S4 | MASS_MANAGEMENT | 1841 | -958.3 | 440.3 | 59.1 | 0.6278 | 0.2% | 87.2% | 12.6% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | -97.24 | 2864.12 | -35299.32 | 108.92 | 269.18 | 465.70 | 1017.66 | 4632.57 |
| Steps | 317.42 | 413.30 | 1.00 | 87.00 | 207.00 | 338.00 | 1153.90 | 2000.00 |
| Food | 76.91 | 54.47 | 0.00 | 38.00 | 66.00 | 104.00 | 181.00 | 551.00 |
| Loss | 5.55 | 8.70 | 0.00 | 1.18 | 2.90 | 6.56 | 18.30 | 159.03 |
| Food/Step | 0.57 | 0.85 | 0.00 | 0.30 | 0.36 | 0.50 | 1.45 | 11.25 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 1018.37 | 1058.03 | -15.2166 | 0.0431 |
| Last 100 | 977.72 | 1055.28 | -3.0012 | 0.0067 |
| Last 200 | 742.35 | 885.66 | +4.2449 | 0.0766 |
| Last 500 | 615.08 | 639.61 | +0.9416 | 0.0451 |
| Last 1000 | 519.82 | 521.14 | +0.5289 | 0.0858 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 8 | 0.1% | 343.6 | 515.3 |
| SnakeCollision | 4738 | 78.1% | 209.1 | 346.0 |
| MaxSteps | 1311 | 21.6% | 708.6 | -1700.7 |
| InvalidFrame | 1 | 0.0% | 1197.0 | -5534.5 |
| BrowserError | 5 | 0.1% | 189.8 | 390.6 |

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 551 | 6,000 | 9.2% |
| Survival | 2000 steps | 1,800 steps | 111.1% |

## Recommendations

Fine-tune hyperparameters, increase training duration.

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
| 02-17 23:42 | 5800 | `enemy_proximity_penalty`=3.0, `enemy_approach_penalty`=2.0, `death_snake`=-40.0 | Snake collision deaths at 77% indicate severe enemy avoidance issues in stage 3. The agent is being too aggressive with sharp turns (37.7%) and U-turns (19.2%) but still dying to enemies. Need to increase enemy penalties and reduce death penalties to encourage more cautious behavior. |

**Total consultations:** 13  
**Most adjusted:** `death_snake` (10x), `enemy_proximity_penalty` (8x), `epsilon_target` (3x), `enemy_approach_penalty` (3x), `lr` (1x)

