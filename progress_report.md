# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-18 06:01:48  
**Total Episodes:** 7089  
**Training Sessions:** 20

## Verdict: LEARNING (Confidence: 70%)

**Goal Feasibility:** POSSIBLE (25-60%)

### Warnings
- Loss very high (11.47) - unstable

### Positive Signals
- Rewards improving: +888.5
- Food collection improving (slope=0.0109/ep)
- Epsilon low (0.081) - exploiting policy

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 2414 | 224.2 | 231.7 | 83.3 | 0.5339 | 0.1% | 65.8% | 34.0% |
| S2 | WALL_AVOID | 800 | 528.6 | 286.7 | 87.2 | 0.4505 | 0.1% | 70.5% | 28.9% |
| S3 | ENEMY_AVOID | 1214 | 286.5 | 336.0 | 93.2 | 0.6197 | 0.1% | 97.6% | 2.3% |
| S4 | MASS_MANAGEMENT | 2661 | -406.1 | 417.7 | 77.3 | 0.6297 | 0.1% | 90.8% | 9.1% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 32.65 | 2693.69 | -35299.32 | 113.40 | 284.58 | 520.63 | 1290.56 | 6098.01 |
| Steps | 325.60 | 411.56 | 1.00 | 88.00 | 210.00 | 367.00 | 1202.40 | 2000.00 |
| Food | 83.18 | 66.48 | 0.00 | 40.00 | 68.00 | 107.00 | 198.00 | 560.00 |
| Loss | 6.09 | 8.49 | 0.00 | 1.35 | 3.58 | 7.65 | 19.18 | 159.03 |
| Food/Step | 0.58 | 0.87 | 0.00 | 0.30 | 0.36 | 0.50 | 1.49 | 11.25 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 1035.90 | 1223.92 | -29.1980 | 0.1185 |
| Last 100 | 1097.70 | 1322.30 | +0.1787 | 0.0000 |
| Last 200 | 861.53 | 1051.74 | +4.2337 | 0.0540 |
| Last 500 | 918.74 | 1084.30 | +0.1066 | 0.0002 |
| Last 1000 | 809.29 | 986.25 | +0.2759 | 0.0065 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 8 | 0.1% | 343.6 | 515.3 |
| SnakeCollision | 5754 | 81.2% | 235.4 | 418.1 |
| MaxSteps | 1321 | 18.6% | 718.4 | -1646.4 |
| InvalidFrame | 1 | 0.0% | 1197.0 | -5534.5 |
| BrowserError | 5 | 0.1% | 189.8 | 390.6 |

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 560 | 6,000 | 9.3% |
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
| 02-18 00:57 | 6000 | `enemy_proximity_penalty`=3.0, `enemy_approach_penalty`=2.0, `death_snake`=-60.0 | Stage 3 shows concerning 86% snake collision deaths despite good reward trends. The agent is taking too many sharp turns (37.9%) and U-turns (18.8%), suggesting aggressive/erratic behavior. Increasing enemy penalties should encourage more cautious play around other snakes. |
| 02-18 01:59 | 6200 | `enemy_proximity_penalty`=3.0, `enemy_approach_penalty`=2.0, `death_snake`=-75.0 | The agent is dying almost exclusively to snake collisions (98.8%) despite being in stage 4. The high sharp turn usage (37.5%) and U-turn usage (18.9%) suggests erratic movement patterns. The enemy proximity and approach penalties need to be increased to teach better collision avoidance. |
| 02-18 03:02 | 6400 | `enemy_proximity_penalty`=3.0, `enemy_approach_penalty`=2.0, `death_snake`=-85.0, `lr`=0.0001 | The agent has extremely high snake collision deaths (98.4%) with declining reward/steps trends and rising loss, indicating poor enemy avoidance despite being in stage 4. The high sharp turn usage (37.6%) and U-turns (19.1%) suggest erratic behavior around enemies. Need to increase enemy penalties and reduce learning rate to stabilize training. |
| 02-18 03:50 | 6600 | `enemy_proximity_penalty`=3.0, `enemy_approach_penalty`=2.0, `death_snake`=-95.0, `epsilon_target`=0.15 | The agent is suffering from severe snake collision deaths (99.6%) with declining reward and step trends, indicating it's becoming more reckless over time. The high sharp turn usage (37.6%) and U-turns (18.7%) suggest erratic behavior. Need to increase enemy penalties and reduce exploration to stabilize the learned policy. |
| 02-18 04:56 | 6800 | `enemy_proximity_penalty`=3.0, `enemy_approach_penalty`=2.0, `death_snake`=-100 | The agent is dying almost exclusively to snake collisions (99.4%) in stage 4, indicating poor enemy avoidance despite reasonable reward trends. The high sharp turn usage (36.7%) and U-turns (19.3%) suggest erratic movement patterns. Need to increase enemy penalties to encourage safer play. |
| 02-18 05:54 | 7000 | `enemy_proximity_penalty`=3.0, `enemy_approach_penalty`=2.0, `death_snake`=-100, `gamma`=0.97 | The agent has extremely high snake collision deaths (98.8%) and declining reward trend, indicating poor enemy avoidance despite being in stage 4. The high action entropy and frequent sharp turns suggest the agent is making erratic movements. I'll increase enemy penalties to discourage risky behavior and raise gamma to improve long-term planning for survival. |

**Total consultations:** 19  
**Most adjusted:** `death_snake` (16x), `enemy_proximity_penalty` (14x), `enemy_approach_penalty` (9x), `epsilon_target` (4x), `lr` (2x), `gamma` (1x)

