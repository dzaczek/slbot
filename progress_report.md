# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-18 18:27:13  
**Total Episodes:** 9357  
**Training Sessions:** 21

## Verdict: LEARNING (Confidence: 75%)

**Goal Feasibility:** POSSIBLE (25-60%)

### Positive Signals
- Rewards improving: +1033.0
- Food collection improving (slope=0.0103/ep)
- Epsilon low (0.080) - exploiting policy

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 2414 | 224.2 | 231.7 | 83.3 | 0.5339 | 0.1% | 65.8% | 34.0% |
| S2 | WALL_AVOID | 800 | 528.6 | 286.7 | 87.2 | 0.4505 | 0.1% | 70.5% | 28.9% |
| S3 | ENEMY_AVOID | 1214 | 286.5 | 336.0 | 93.2 | 0.6197 | 0.1% | 97.6% | 2.3% |
| S4 | MASS_MANAGEMENT | 4929 | 226.3 | 415.7 | 101.9 | 0.5989 | 0.1% | 94.2% | 5.6% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 259.41 | 2442.89 | -35299.32 | 130.98 | 322.26 | 675.44 | 1957.58 | 6245.81 |
| Steps | 346.88 | 415.64 | 1.00 | 92.00 | 223.00 | 427.00 | 1240.20 | 2000.00 |
| Food | 94.70 | 82.57 | 0.00 | 43.00 | 72.00 | 116.00 | 249.00 | 569.00 |
| Loss | 7.39 | 8.97 | 0.00 | 1.83 | 5.05 | 9.86 | 21.36 | 204.95 |
| Food/Step | 0.57 | 0.86 | 0.00 | 0.30 | 0.35 | 0.49 | 1.45 | 11.25 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 1232.24 | 1111.72 | -6.3610 | 0.0068 |
| Last 100 | 1161.54 | 997.03 | +0.9875 | 0.0008 |
| Last 200 | 1070.71 | 1076.52 | +1.2730 | 0.0047 |
| Last 500 | 999.92 | 1119.35 | +0.1107 | 0.0002 |
| Last 1000 | 1020.81 | 1213.27 | +0.0543 | 0.0002 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| Wall | 11 | 0.1% | 546.8 | 1185.4 |
| SnakeCollision | 7981 | 85.3% | 277.7 | 549.7 |
| MaxSteps | 1357 | 14.5% | 752.4 | -1451.8 |
| InvalidFrame | 1 | 0.0% | 1197.0 | -5534.5 |
| BrowserError | 7 | 0.1% | 191.3 | 377.7 |

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 569 | 6,000 | 9.5% |
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
| 02-18 06:59 | 7200 | `death_snake`=-100, `enemy_proximity_penalty`=3.0 | The agent is performing well with high rewards and good progression, but 98% snake collision deaths indicate extreme risk-taking behavior. The high use of sharp turns (39.5%) and U-turns (20.4%) suggests aggressive maneuvering that leads to crashes. Need to increase enemy collision penalty to encourage more cautious play. |
| 02-18 08:09 | 7400 | `enemy_proximity_penalty`=3.0, `enemy_approach_penalty`=1.5, `death_snake`=-100 | The agent is suffering from extremely high snake collision deaths (98%) in stage 4, indicating poor enemy avoidance despite being in the final curriculum stage. The reward trend is declining and action distribution shows excessive sharp turns and U-turns, suggesting panic responses. Need to increase enemy penalties to improve survival instincts. |
| 02-18 09:17 | 7600 | `death_snake`=-100, `epsilon_target`=0.15 | The agent is in stage 4 but dying almost exclusively to snake collisions (98.4%). The high sharp turn usage (42.8%) and declining reward trend suggest the agent is making erratic movements. The death penalty for snake collisions should be increased to discourage this behavior, and epsilon should be lowered since exploration is currently too high for this advanced stage. |
| 02-18 10:28 | 7800 | `enemy_proximity_penalty`=2.5, `enemy_approach_penalty`=1.8, `death_snake`=-100 | The agent has a severe snake collision problem (98% deaths) despite being in stage 4. The high sharp turn usage (43.9%) and U-turns (18.8%) suggest erratic movement patterns. The enemy proximity penalties are insufficient to teach proper avoidance behavior. |
| 02-18 11:20 | 8000 | `enemy_proximity_penalty`=3.0, `enemy_approach_penalty`=2.0, `death_snake`=-100 | The agent is dying almost exclusively to snake collisions (99%) in stage 4, with declining reward trend despite high food consumption. The high sharp turn usage (44.1%) suggests aggressive maneuvering that leads to collisions. Need to increase enemy penalties to teach safer navigation. |
| 02-18 12:14 | 8200 | `enemy_proximity_penalty`=3.0, `enemy_approach_penalty`=2.0, `death_snake`=-100 | The agent is dying almost exclusively to snake collisions (99.4%) and showing declining performance trends. With high sharp turns and U-turns (65.3% combined), it's being overly aggressive and not learning proper enemy avoidance. The enemy proximity penalty needs strengthening to teach better evasion. |
| 02-18 13:16 | 8400 | `enemy_proximity_penalty`=2.5, `enemy_approach_penalty`=1.5 | Snake collision deaths at 99% indicate the agent is being overly aggressive with sharp turns and U-turns (66% combined). The high reward trend shows learning progress, but the agent needs to balance aggression with survival. Reducing enemy proximity penalties will encourage more cautious behavior around other snakes. |
| 02-18 14:20 | 8600 | `enemy_proximity_penalty`=3.0, `enemy_approach_penalty`=2.0, `death_snake`=-65.0 | The agent is heavily dying from snake collisions (97.4%) and using sharp turns excessively (45.6%), indicating poor enemy avoidance. The high reward trend shows learning progress, but enemy collision penalties need strengthening to improve survival skills. |
| 02-18 15:35 | 8800 | `enemy_proximity_penalty`=3.0, `enemy_approach_penalty`=2.0 | The agent is dying almost exclusively from snake collisions (96.2%) with very few wall deaths, indicating good wall avoidance but poor enemy avoidance. The high sharp turn usage (46.3%) and U-turn frequency (20.8%) suggests panic responses. Need to increase enemy proximity penalties to encourage more cautious behavior around other snakes. |
| 02-18 16:42 | 9000 | `enemy_proximity_penalty`=3.0, `enemy_approach_penalty`=2.0, `death_snake`=-75.0 | Stage 4 agent shows excellent wall avoidance (only 0.2% wall deaths) but 97.4% snake collision deaths indicate severe enemy collision issues. The high sharp turn usage (48.5%) and U-turns (20.2%) suggest panic responses. Need to increase enemy penalties to encourage safer navigation. |

**Total consultations:** 29  
**Most adjusted:** `death_snake` (24x), `enemy_proximity_penalty` (23x), `enemy_approach_penalty` (17x), `epsilon_target` (5x), `lr` (2x), `gamma` (1x)

