# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-03-27 21:10:26  
## Training Summary

| Scope | Total Episodes | Best Food | Best Survival | Current Stage |
|-------|----------------|-----------|---------------|---------------|
| **Global (All UIDs)** | 101092 | 4532 | 99999 steps | - |
| **Current Chain** | 11 | 237 | 927 steps | MASS_MANAGEMENT |

## Verdict: NOT LEARNING (Confidence: 10%)

**Goal Feasibility:** 

### Critical Issues
- Insufficient data (< 20 episodes)

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Avg PkLen | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-----------|-------|--------|----------|
| S4 | MASS_MANAGEMENT | 11 | 451.4 | 402.0 | 113.5 | 31.3 | 0.3128 | 0.0% | 100.0% | 0.0% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 451.44 | 302.87 | 67.35 | 207.53 | 362.09 | 650.25 | 951.29 | 1012.98 |
| Steps | 402.00 | 303.12 | 50.00 | 141.50 | 361.00 | 592.00 | 919.00 | 927.00 |
| Food | 113.45 | 75.10 | 21.00 | 39.50 | 108.00 | 165.50 | 232.00 | 237.00 |
| PeakLength | 31.27 | 4.92 | 25.00 | 27.00 | 31.00 | 35.00 | 38.50 | 39.00 |
| Loss | 51.54 | 98.75 | 0.00 | 5.31 | 14.15 | 38.10 | 217.20 | 355.31 |
| Food/Step | 0.31 | 0.06 | 0.24 | 0.27 | 0.29 | 0.35 | 0.42 | 0.42 |

## Q-Value & Gradient Health

| Metric | Last | Avg (50) | Min | Max | Trend |
|--------|------|----------|-----|-----|-------|
| Q Mean | -64.9982 | -20.6564 | -64.9982 | 40.1742 | DOWN (slope=-4.0422) |
| Q Max | 458.5070 | 1116.1786 | 0.0000 | 2088.9019 | DOWN (slope=-48.5374) |
| TD Error | 145.3874 | 322.4705 | 0.0000 | 854.9417 | DOWN (slope=-27.0559) |
| Grad Norm | 480.1998 | 769.5013 | 0.0000 | 2199.2097 | DOWN (slope=-44.6669) |

## Action Distribution

| Action | Overall % | Last 100 % | First 100 % | Change |
|--------|----------|-----------|------------|--------|
| Straight | 0.6% | 0.6% | 0.6% | 0.0% |
| Gentle | 25.6% | 25.6% | 25.6% | 0.0% |
| Medium | 18.3% | 18.3% | 18.3% | 0.0% |
| Sharp | 29.2% | 29.2% | 29.2% | 0.0% |
| U-turn | 20.9% | 20.9% | 20.9% | 0.0% |
| Boost | 5.4% | 5.4% | 5.4% | 0.0% |

**Action Entropy (last 100):** 2.07 / 2.58 bits (80% diversity)

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|
| SnakeCollision | 11 | 100.0% | 402.0 | 451.4 |

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 237 | 6,000 | 4.0% |
| Survival | 927 steps | 1,800 steps | 51.5% |

## Recommendations



1. No critical issues. Continue training.

## Charts

### Interactive 3D Charts
Open in browser for zoom, rotate, and hover details:

- [Steps vs Food vs Episode (Interactive 3D)](charts/chart_18_interactive.html)
- [Steps vs Reward vs Episode (Interactive 3D)](charts/chart_19_interactive.html)

### Main Dashboard
![Main Dashboard](charts/chart_01_dashboard.png)

### Stage Progression
![Stage Progression](charts/chart_02_stage_progression.png)

### Per-Stage Distributions
![Per-Stage Distributions](charts/chart_03_stage_distributions.png)

### Hyperparameter Tracking
![Hyperparameter Tracking](charts/chart_04_hyperparameters.png)

### Metric Correlations (Scatter)
![Metric Correlations (Scatter)](charts/chart_05_correlations.png)

### Correlation Heatmap & Rankings
![Correlation Heatmap & Rankings](charts/chart_05b_correlation_heatmap.png)

### Performance Percentile Bands
![Performance Percentile Bands](charts/chart_06_performance_bands.png)

### Death Analysis
![Death Analysis](charts/chart_07_death_analysis.png)

### Food Efficiency
![Food Efficiency](charts/chart_08_food_efficiency.png)

### Reward Distributions
![Reward Distributions](charts/chart_09_reward_distributions.png)

### Learning Detection
![Learning Detection](charts/chart_10_learning_detection.png)

### Goal Progress
![Goal Progress](charts/chart_11_goal_gauges.png)

### Goal Progress Over Time
![Goal Progress Over Time](charts/chart_11b_goal_over_time.png)

### Hyperparameter Analysis
![Hyperparameter Analysis](charts/chart_12_hyperparameter_analysis.png)

### Q-Value & Gradient Analysis
![Q-Value & Gradient Analysis](charts/chart_13_qvalue_gradients.png)

### Action Distribution Analysis
![Action Distribution Analysis](charts/chart_14_action_distribution.png)

### Active Agents Over Time
![Active Agents Over Time](charts/chart_15_auto_scaling.png)

### MaxSteps Analysis
![MaxSteps Analysis](charts/chart_16_maxsteps_analysis.png)

### Survival Percentiles
![Survival Percentiles](charts/chart_17_survival_percentiles.png)

### 3D Steps vs Food (Full History - Rotating)
![3D Steps vs Food (Full History - Rotating)](charts/chart_18_3d_steps_food_full.gif)

### 3D Steps vs Food (Full History - Static)
![3D Steps vs Food (Full History - Static)](charts/chart_18_3d_steps_food_full.png)

### 3D Steps vs Food (Recent 10k - Rotating)
![3D Steps vs Food (Recent 10k - Rotating)](charts/chart_18_3d_steps_food_recent_10k.gif)

### 3D Steps vs Food (Recent 10k - Static)
![3D Steps vs Food (Recent 10k - Static)](charts/chart_18_3d_steps_food_recent_10k.png)

### 3D Steps vs Food (Recent 3k - Rotating)
![3D Steps vs Food (Recent 3k - Rotating)](charts/chart_18_3d_steps_food_recent_3k.gif)

### 3D Steps vs Food (Recent 3k - Static)
![3D Steps vs Food (Recent 3k - Static)](charts/chart_18_3d_steps_food_recent_3k.png)

### Steps vs Reward vs Episode — Bubble (3D rotating)
![Steps vs Reward vs Episode — Bubble (3D rotating)](charts/chart_19_bubble_training.gif)

### Steps vs Reward vs Episode — Bubble (3D static)
![Steps vs Reward vs Episode — Bubble (3D static)](charts/chart_19_bubble_training.png)

### Snake Size (Peak Length) Analysis
![Snake Size (Peak Length) Analysis](charts/chart_20_peak_length.png)

## AI Supervisor — Recent Changes

| Time | Episode | Changes | Reasoning |
|------|---------|---------|----------|
| 02-19 00:43 | 10200 | `death_snake`=-75.0, `enemy_proximity_penalty`=3.0, `enemy_approach_penalty`=2.0 | Agent is in Stage 4 with 97.4% snake collision deaths, indicating severe struggle with enemy avoidance. The high sharp turn usage (50.4%) and U-turns (17.4%) suggest panic behavior. Need to increase enemy penalties and reduce death penalty to allow more exploration of survival strategies. |
| 02-19 06:47 | 10400 | `enemy_proximity_penalty`=2.5, `enemy_approach_penalty`=2.0, `death_snake`=-60.0 | Agent is dying 96% from snake collisions in Stage 4 (Mass Management), indicating poor enemy avoidance. The declining reward trend and high sharp turn usage suggest panic responses. Need to strengthen enemy avoidance penalties and reduce death penalties to encourage learning rather than punishing exploration. |
| 02-19 08:01 | 10600 | `enemy_proximity_penalty`=3.0, `enemy_approach_penalty`=2.0, `lr`=7.5e-05 | Stage 4 shows concerning signs: 96.8% snake collision deaths indicate poor enemy avoidance, declining reward trend suggests learning plateau, and excessive sharp turns (53.3%) suggest panic responses. Need to strengthen enemy avoidance penalties and reduce learning rate for more stable training. |
| 02-19 09:23 | 10800 | `enemy_proximity_penalty`=3.0, `enemy_approach_penalty`=2.0, `death_snake`=-75.0 | The agent has extremely high snake collision deaths (96.8%) which is the primary issue. Despite good reward trends, it's dying to enemies constantly. In Stage 4 (Mass Management), the agent should learn defensive positioning but is failing catastrophically. Need to significantly increase enemy penalties and reduce enemy approach penalty to make the agent more cautious around other snakes. |
| 02-19 10:29 | 11000 | `enemy_proximity_penalty`=3.0, `death_snake`=-60.0, `enemy_approach_penalty`=2.0 | The agent is overwhelmingly dying to snake collisions (98.6%) and showing declining performance trends. The high sharp turn usage (53.3%) suggests erratic movement patterns. Need to increase enemy proximity penalties and reduce death penalty to encourage safer play while maintaining exploration. |
| 02-19 11:50 | 11200 | `enemy_proximity_penalty`=2.5, `enemy_approach_penalty`=2.0 | The agent is dominated by snake collisions (98.4%) in Stage 4 (Mass Management) and shows excessive sharp turning (54.1%). The high enemy_proximity_penalty (3.0) may be causing overcautious behavior leading to poor positioning. Reducing it slightly and increasing enemy_approach_penalty should encourage better strategic positioning while maintaining survival instincts. |
| 02-19 13:05 | 11400 | `enemy_proximity_penalty`=3.0, `enemy_approach_penalty`=1.5, `death_snake`=-65.0 | The agent has a severe snake collision problem (97.4% deaths) despite being in Stage 4. The high sharp turn usage (53.9%) and low straight movement (4.4%) suggests erratic behavior. Need to increase enemy proximity penalties and reduce enemy approach penalty to encourage more cautious behavior around other snakes. |
| 02-19 14:30 | 11600 | `enemy_proximity_penalty`=2.0, `enemy_approach_penalty`=0.5, `length_bonus`=0.35 | Stage 4 agent is dying 96.8% to snake collisions despite decent survival time (544 steps). The high sharp turn usage (54.4%) suggests panicked movement. Need to reduce enemy proximity penalties to encourage calmer navigation and increase length bonus to promote defensive growth. |
| 02-19 15:25 | 11800 | `enemy_proximity_penalty`=2.5, `enemy_approach_penalty`=1.0 | Agent is in Stage 4 with 97.4% snake collision deaths, indicating poor enemy avoidance despite good food collection (133 avg). The high sharp turn usage (42.8%) and rising loss trend suggest the agent is struggling with enemy proximity decisions. Increasing enemy proximity penalty and reducing enemy approach penalty will help discourage risky enemy encounters. |
| 02-19 16:24 | 12000 | `enemy_proximity_penalty`=3.0, `enemy_approach_penalty`=1.5, `death_snake`=-70.0, `lr`=5e-05 | The agent is dying almost exclusively to enemy snakes (98.6%) in Stage 4, indicating poor enemy avoidance. The high loss trend and Q-value instability suggest learning issues. Need to increase enemy penalties and reduce learning rate for stability. |
| 02-19 17:43 | 12200 | `enemy_proximity_penalty`=2.5, `enemy_approach_penalty`=2.0 | The agent is dying almost exclusively to snake collisions (98.6%) in Stage 4, indicating poor enemy avoidance. The high reward trend shows it's learning to survive longer and collect food, but needs stronger penalties for approaching enemies to reduce collision deaths. |
| 02-19 19:00 | 12400 | `enemy_proximity_penalty`=3.0, `enemy_approach_penalty`=2.0, `death_snake`=-65.0 | The agent is dying almost exclusively to snake collisions (96.8%) in Stage 4, indicating poor enemy avoidance despite good food collection (133 avg). The high reward trend shows learning progress, but enemy proximity penalties need strengthening to reduce collision deaths. |
| 02-19 20:33 | 12600 | `lr`=7e-05, `gamma`=0.975 | The agent shows concerning declining trends with reward dropping by 3668 and steps by 90. Food efficiency (0.2858) is healthy but the negative trends suggest the agent may be overexploring or struggling with value function stability. The high loss trend (24.48) indicates training instability. I'll reduce learning rate for stability and slightly increase gamma for better long-term planning in Stage 4. |
| 02-19 21:28 | 12800 | `enemy_proximity_penalty`=0.6, `gamma`=0.985 | The agent shows concerning regression with significant drops in reward (-3654), steps (-150), and Q-values (-134) despite healthy food collection. The high sharp turn usage (50.2%) and declining survival suggests the agent is becoming overly reactive. Reducing enemy proximity penalty and increasing gamma should help stabilize longer-term planning. |
| 02-19 22:11 | 13000 | `food_reward`=12.0, `lr`=7.5e-05 | The agent is showing concerning negative trends across all key metrics despite healthy food efficiency. The sharp decline in reward (-284), steps (-126), and Q-values (-40) suggests the agent is becoming less effective over time. With 98.4% snake deaths being normal, the issue is likely insufficient reward signal strength. Boosting food_reward will strengthen positive reinforcement, while reducing learning rate will help stabilize the declining loss trend. |
| 02-19 23:15 | 13200 | `lr`=6e-05 | Training shows excellent progress with strong reward trend (+584) and healthy food efficiency (0.32). The agent is in Stage 4 and performing well. Loss is trending up slightly which suggests learning rate could be reduced for more stable training. The high sharp turn usage (43.7%) indicates good tactical maneuvering. |
| 02-20 00:35 | 13400 | `lr`=5e-05 | Training shows excellent progress with strong reward and survival improvements. The agent is performing well in Stage 4 with healthy food efficiency (0.30 food/step) and good episode length (419 steps). However, the high loss trend (+13.57) and rising Q-values suggest some training instability that could benefit from a small learning rate reduction. |
| 02-20 01:49 | 13600 | `lr`=3.5e-05, `gamma`=0.975 | The agent shows declining reward (-220) and steps (-37) trends with rising loss (+19.9), indicating training instability. The high sharp turn usage (56%) and U-turns (21%) suggest frantic movement patterns. The loss is quite high at 69.9, indicating the learning rate may be too aggressive for this stage. |
| 02-20 02:46 | 13800 | `wall_proximity_penalty`=0.3, `survival`=0.3, `lr`=2.5e-05 | The agent shows concerning trends: reward declining by -608 and steps dropping by -151, indicating deteriorating performance. Food efficiency remains healthy at 0.29, but high loss (82.5) and sharp turn dominance (63.4%) suggest the agent is becoming overly defensive. Reducing wall proximity penalty and boosting survival reward should encourage more confident exploration. |
| 02-20 03:44 | 14000 | `food_reward`=12.0, `length_bonus`=0.08 | Training shows concerning negative trends across all metrics despite healthy food intake. The sharp turn dominance (65.1%) and declining rewards suggest the agent is becoming overly defensive. In Stage 4 (Mass Management), we need to encourage more growth-oriented behavior by boosting food rewards and reducing excessive turning penalties. |
| 02-20 04:43 | 14200 | `length_bonus`=0.12 | Training shows strong performance with excellent food efficiency (0.304 food/step) and good reward trend (+454). However, steps are declining (-19.8) and the agent is overusing sharp turns (62.6%) which may indicate defensive play. Since we're in Stage 4 (Mass Management), we should encourage growth through higher length bonus. |
| 02-20 05:46 | 14400 | `lr`=2e-05 | Training is progressing very well with strong positive trends in reward (+1202), steps (+89), and Q-values. Food efficiency is healthy at 0.30. The high loss (83) and rising loss trend (+13.5) suggests learning rate may be too aggressive for current progress. Conservative LR reduction will stabilize training while maintaining momentum. |
| 02-20 07:05 | 14600 | `enemy_proximity_penalty`=1.5 | The agent shows concerning signs: reward trend is strongly negative (-419) despite stable steps, and loss remains high (79.6). The sharp turn bias (48.9%) suggests the snake is making erratic movements, possibly due to conflicting reward signals. Reducing enemy_proximity_penalty should help the snake navigate more smoothly around other snakes while maintaining food collection efficiency. |
| 02-20 08:11 | 14800 | `lr`=7.5e-05, `gamma`=0.975 | The agent shows declining performance with reward trend -1699 and steps trend -37, indicating instability. High loss (55.8) and declining Q-values suggest learning rate may be too high. Food efficiency remains healthy at 0.31, so focus on stabilizing training rather than reward structure changes. |
| 02-20 09:10 | 15000 | `length_bonus`=0.08, `food_reward`=10.0 | The agent is in Stage 4 with declining performance trends across all metrics (reward -84, steps -40, Q-values -34). Despite healthy food intake (0.315 food/step), the decreasing survival time and reward suggest the agent is struggling with mass management and survival. The high sharp turn usage (52.8%) indicates defensive behavior. I'll boost length_bonus to encourage growth as a survival strategy and slightly increase food_reward to strengthen the growth incentive. |
| 02-20 09:58 | 15200 | `lr`=6e-05, `gamma`=0.975 | The agent shows strong food efficiency (0.315 food/step) and improving rewards (+319 trend), but steps are declining (-67 trend) and loss is increasing (+6.85 trend). The high sharp turn usage (47.6%) suggests reactive movement patterns. Slightly reducing learning rate should stabilize loss while maintaining progress, and increasing gamma will encourage longer-term planning to counter the declining step trend. |
| 02-20 10:52 | 15400 | `lr`=4.5e-05 | Training shows excellent progress with strong reward trend (+254), healthy food efficiency (0.32), and good survival duration (308 steps). Loss is increasing which suggests learning rate might be too high for this mature stage. The slight decline in steps (-11) combined with rising loss indicates potential instability from overly aggressive learning. |
| 02-20 13:18 | 15800 | `lr`=3.5e-05 | Training shows excellent progress with strong positive trends in reward (+720), steps (+101), and Q-values (+37). Food efficiency is healthy at 0.30 food/step. The 99% snake collision death rate is normal for slither.io. However, the loss trend (+4.8) indicates some instability, and the high loss value (63.6) suggests the learning rate may be too aggressive for the current stage. |
| 02-20 14:23 | 16000 | `lr`=2.5e-05, `survival`=0.35 | The agent shows concerning trends: reward declining (-197), steps declining (-34), and loss increasing (+6.2). Despite healthy food efficiency (0.3027), the declining survival suggests the agent is becoming less effective at avoiding fatal situations. The high sharp turn usage (58.8%) indicates reactive rather than proactive movement. Reducing learning rate should stabilize training, while increasing survival reward will emphasize staying alive longer. |
| 02-20 15:31 | 16200 | `lr`=2e-05, `gamma`=0.98 | Agent is in Stage 4 with declining reward and steps trends despite healthy food collection (0.30 food/step). The high sharp turn usage (59.3%) and rising loss suggest the agent is becoming overly reactive. Need to stabilize learning with lower learning rate and encourage longer-term thinking with higher gamma. |
| 02-20 16:40 | 16400 | `lr`=1.5e-05 | Training shows excellent progress with strong reward trend (+262), healthy food efficiency (0.30), and good survival time (441 steps). Loss trend is rising slightly which suggests learning rate may be too high. The agent is in Stage 4 (Mass Management) and performing well overall - only a minor lr reduction needed. |
| 02-20 18:10 | 16600 | `lr`=1.2e-05 | Training is progressing very well with strong positive trends in reward (+615), steps (+86), and Q-values. Food efficiency is healthy at 0.30. However, loss is trending upward (+8.7) indicating some training instability. The agent is in Stage 4 (Mass Management) and showing good survival with 493 avg steps. A slight learning rate reduction will help stabilize training while maintaining progress. |
| 02-21 16:18 | 16800 | `epsilon_target`=0.25, `gamma`=0.975 | Training is performing well with healthy food intake (0.33 food/step), improving rewards (+110), and stable learning. However, the agent is making excessive sharp turns (46.3%) and U-turns (11.2%), indicating inefficient movement. The high action entropy (0.809) suggests good exploration but possibly too much randomness. Slightly reducing epsilon and increasing gamma will encourage more strategic behavior in Stage 4. |
| 02-21 17:49 | 17000 | `enemy_approach_penalty`=0.6 | The agent is performing excellently with very high food efficiency (0.43) and strong reward growth (+1362). However, the declining steps trend (-119) and Q-value trend (-59) suggest the agent may be becoming overly aggressive or taking unnecessary risks. The high sharp turn usage (47%) indicates active maneuvering. Reducing enemy approach penalty will encourage more calculated aggression appropriate for Stage 4. |
| 02-21 19:20 | 17200 | `lr`=7.5e-05, `boost_penalty`=0.2, `length_bonus`=0.05 | The agent is performing exceptionally well with food efficiency at 0.57 (well above healthy range) and positive reward trend. However, steps are declining while loss is increasing, suggesting overfitting. The very low boost usage (1.5%) and high sharp turns indicate overcautious behavior that may hurt survival in Stage 4. |
| 02-21 20:51 | 17400 | `food_reward`=6.5, `boost_penalty`=-0.1 | The agent is performing exceptionally well with very high food efficiency (0.61 vs healthy 0.25-0.40) and strong positive reward trends. However, the extremely high food/step ratio suggests the agent may be too focused on food collection at the expense of strategic positioning. The low boost usage (0.8%) and high sharp turn usage (56.2%) indicate overly cautious behavior. Since this is Stage 4 (Mass Management), we should slightly reduce food motivation and encourage more dynamic movement. |
| 02-21 22:00 | 17600 | `food_reward`=4.5, `food_shaping`=0.15 | The agent is collecting too much food per step (0.58 vs healthy 0.25-0.40) but reward and steps are declining, suggesting it's becoming overly food-focused at the expense of survival strategy. The high sharp turn usage (53.5%) indicates frantic movement patterns. Reducing food_reward and food_shaping will encourage more strategic play. |
| 02-21 23:52 | 17800 | `food_reward`=3.0, `food_shaping`=0.08, `length_bonus`=0.08 | The agent has excellent food efficiency (0.57 vs healthy 0.25-0.40) but reward is declining (-1437) despite stable survival time. The high food intake suggests over-aggressive food seeking that may compromise survival positioning. Need to reduce food incentives and encourage more conservative play in Stage 4. |
| 02-22 01:20 | 18000 | `food_reward`=2.5, `length_bonus`=0.12 | The agent is performing exceptionally well with food/step at 0.54 (well above healthy range), positive reward trend, and good survival time. The high food efficiency suggests food_reward may be too high, causing over-focus on food collection. In Stage 4 (Mass Management), we should encourage more strategic play by slightly reducing food incentive and increasing length bonus to reward sustained growth. |
| 02-22 02:52 | 18200 | `food_reward`=2.0 | The agent is performing exceptionally well with very high food efficiency (0.51 vs healthy 0.25-0.40) and strong reward trend (+1481). However, the declining steps trend (-139) and rising loss suggest the agent may be becoming overaggressive in food collection, leading to riskier behavior. Reducing food_reward slightly will encourage more balanced risk-taking. |
| 02-22 06:35 | 18600 | `lr`=7.5e-05 | The agent is performing exceptionally well with very high food efficiency (0.45 vs healthy 0.25-0.40) and strong positive trends in reward (+485), steps (+72), and Q-values. The 97.8% snake collision death rate is completely normal for slither.io. Loss is trending upward slightly, suggesting learning rate may be too high for this mature stage. |
| 02-22 08:29 | 18800 | `boost_penalty`=0.0, `enemy_proximity_penalty`=1.5, `length_bonus`=0.08 | The agent shows concerning trends with declining reward (-607), steps (-26.6), and Q-values despite excellent food efficiency (0.465). The high sharp turn usage (45.7%) and low boost usage (1.0%) suggests overly defensive play in Stage 4 (Mass Management). Need to encourage more confident movement and boost usage while stabilizing the declining performance. |
| 02-22 10:06 | 19000 | `lr`=7.5e-05, `gamma`=0.975 | The agent shows excellent food efficiency (0.50 food/step, well above healthy range) but concerning declining trends in reward (-2044) and steps (-83). The high loss (99.97) with upward trend (+36.9) suggests learning instability. Reducing learning rate should stabilize training while slight gamma increase will help value estimation for longer sequences. |
| 02-22 11:13 | 19200 | `lr`=5e-05, `gamma`=0.97 | Food efficiency is excellent (0.57) but all metrics are declining - reward down 505, steps down 76, suggesting the agent is becoming less effective despite good food collection. The high loss (100+) and declining Q-values indicate training instability. Reducing learning rate should stabilize training while slightly lowering gamma will help focus on immediate rewards during this unstable period. |
| 02-22 12:18 | 19400 | `food_reward`=5.0, `length_bonus`=0.08, `boost_penalty`=0.0 | The agent is collecting food extremely well (0.64 food/step vs healthy 0.25-0.40) but reward trend is declining significantly (-974). This suggests the reward structure may be saturated or unbalanced. The very low boost usage (0.7%) and declining performance indicate the agent needs recalibration for Stage 4's mass management focus. |
| 02-22 14:58 | 20000 | `food_reward`=3.0, `survival`=0.35, `length_bonus`=0.12 | The agent is performing exceptionally well with food collection (0.75 food/step vs 0.25-0.40 healthy range), but reward and steps are declining significantly. This suggests the agent is becoming overly aggressive in food seeking, leading to more collisions. Need to reduce food incentives and improve survival focus. |
| 02-22 17:54 | 20500 | `food_reward`=2.0 | The agent is performing exceptionally well with food/step at 0.72 (nearly double the healthy range), strong positive reward trend (+3390), and good survival time (238 steps). The extremely high food efficiency suggests food_reward may be too high, causing the agent to prioritize food over survival strategy. Reducing food_reward will encourage more balanced behavior while maintaining growth. |
| 02-22 21:44 | 21000 | `food_reward`=1.5, `food_shaping`=0.2, `length_bonus`=0.08 | Food efficiency is extremely high (0.62 vs healthy 0.25-0.40) suggesting overtuned food rewards are dominating all other signals. The negative reward trend (-3038) indicates deteriorating performance despite good food collection. Need to rebalance rewards to allow other behaviors like strategic positioning. |
| 02-23 02:35 | 21500 | `food_reward`=1.2, `length_bonus`=0.12 | The agent is performing exceptionally well with food/step at 0.53 (well above healthy range), strong positive reward trend (+5351), and good survival time. However, the extremely high food efficiency suggests the agent may be playing too conservatively in Stage 4 (Mass Management). Since it's surviving well and collecting abundant food, we should encourage more strategic risk-taking by reducing food rewards and slightly increasing length bonus to promote growth-focused behavior appropriate for this stage. |
| 02-23 07:06 | 22000 | `lr`=7.5e-05, `gamma`=0.975 | The agent shows excellent food efficiency (0.42 food/step) but concerning trends: reward and steps declining while loss increasing significantly. The high loss trend (115.24) suggests learning instability. Reducing learning rate should stabilize training, while slightly increasing gamma will help the agent better value long-term survival in this advanced stage. |
| 02-23 08:28 | 22500 | `lr`=5e-05, `boost_penalty`=-0.2 | The agent shows concerning negative trends across all key metrics despite healthy food intake. The sharp decline in reward (-4751), steps (-339), and Q-values (-79) suggests the agent may be stuck in suboptimal behaviors. The extremely low boost usage (0.8%) and high sharp turns (60%) indicate overly conservative play. Reducing learning rate for stability and encouraging more aggressive play with negative boost penalty. |
| 02-23 10:15 | 23000 | `lr`=3.5e-05, `gamma`=0.975 | Despite declining reward trend, the agent shows healthy fundamentals: excellent food efficiency (0.35 food/step), good survival duration (292 steps), and normal death patterns. The loss is increasing and Q-values declining, suggesting learning instability. Reducing learning rate should stabilize training, while a small gamma increase will help value longer-term planning in Stage 4. |
| 02-23 11:41 | 23500 | `lr`=2.5e-05, `boost_penalty`=0.2 | The agent shows excellent food efficiency (0.47 food/step, well above healthy range) and reasonable survival (224 steps avg), but reward trend is declining significantly (-454). The high loss trend (+46) and declining Q-values suggest learning instability. The very low boost usage (0.7%) in Stage 4 indicates the agent is overly conservative. Reducing learning rate should stabilize training, and decreasing boost penalty will encourage more aggressive play appropriate for mass management stage. |
| 02-23 12:50 | 24000 | `food_reward`=5.0, `food_shaping`=0.1, `length_bonus`=0.05 | Food intake is extremely high (0.54 vs healthy 0.25-0.40) indicating over-optimization for food collection. The snake is turning too aggressively (76% sharp/uturn actions) and steps are declining despite reward growth. Need to reduce food emphasis and encourage more strategic movement. |
| 02-23 14:13 | 24500 | `food_reward`=3.5, `gamma`=0.985 | The agent is collecting food extremely well (0.55 food/step vs healthy 0.25-0.40) but reward trend is strongly negative (-593), indicating overconsumption without strategic value. The high sharp turn usage (57.6%) suggests erratic movement. Need to reduce food reward to encourage more strategic behavior and increase gamma for better long-term planning in Stage 4. |
| 02-23 15:34 | 25000 | `food_reward`=2.5, `gamma`=0.99, `lr`=2e-05 | Food efficiency is extremely high (0.53 vs healthy 0.25-0.40) indicating overemphasis on food collection. The snake is turning too aggressively (64% sharp turns, 7.6% U-turns) likely due to excessive food reward. Loss is trending upward significantly (+408) suggesting instability. Need to reduce food_reward and increase gamma for better long-term planning in Stage 4. |
| 02-23 16:52 | 25500 | `lr`=1.5e-05, `enemy_proximity_penalty`=1.0, `enemy_approach_penalty`=0.6 | The agent shows excellent food efficiency (0.445 food/step) but concerning negative trends across all metrics. The high sharp turn usage (59.9%) and declining performance suggests the agent may be over-aggressive or panicking. Reducing learning rate for stability and slightly decreasing enemy penalties to allow calmer movement patterns. |
| 02-24 06:57 | 26500 |  | Training is exceptionally healthy with strong positive trends across all metrics. Food efficiency is excellent at 0.32/step, survival is improving (458 avg steps), and reward is trending strongly upward (+438). The high sharp turn usage (44%) suggests good tactical maneuvering. No adjustments needed - the agent is learning effectively in Stage 4. |
| 02-24 10:27 | 27000 | `food_reward`=15.0, `food_shaping`=0.15, `lr`=1e-05 | Food efficiency is critically low at 0.1065 (needs 0.25-0.40). Despite good survival length (789 steps), the snake isn't eating enough to grow. The declining reward trend and very high loss indicate the agent needs stronger food motivation. Boosting food_reward and adding food_shaping will help guide the snake to food sources. |
| 02-24 18:33 | 27500 | `food_reward`=25.0, `lr`=5e-06 | Food efficiency is critically low at 0.0931 (should be 0.25-0.40), indicating the snake isn't eating enough to sustain growth. Despite positive reward trend, the very high loss values and explosive Q-value growth suggest training instability. Need to boost food rewards and reduce learning rate to stabilize. |
| 02-24 21:22 | 27500 | `food_reward`=30.0, `lr`=2e-06, `food_shaping`=0.5 | Food efficiency is critically low at 0.1079 (healthy: 0.25-0.40), indicating the snake isn't eating enough to grow and survive. Despite positive reward trend, the extremely high loss values and declining steps suggest instability. Need to boost food motivation and reduce learning rate for stability. |
| 02-25 08:05 | 28000 |  | Training shows excellent progress with strong reward trend (+3500), good food efficiency (0.30), and improving survival (421 avg steps). The loss is decreasing significantly, indicating stable learning. The high sharp turn usage (44.9%) suggests good evasive behavior. No adjustments needed - the agent is learning effectively in Stage 4. |
| 02-25 10:57 | 28500 | `gamma`=0.975, `lr`=2.5e-06 | The agent shows declining performance with negative reward and steps trends, but food efficiency remains healthy at 0.32. The high sharp turn usage (44.7%) suggests the agent is reactive rather than strategic. Slightly increasing gamma will encourage longer-term planning, and a small lr boost will help adapt to the declining trends while maintaining stability. |
| 02-25 14:48 | 29000 |  | Training shows strong positive trends across all metrics with healthy food efficiency (0.30 food/step). The snake is surviving well (594 steps avg) and learning effectively. Loss is increasing but this is normal as Q-values grow. No adjustments needed - the current parameters are working well. |
| 02-25 18:11 | 29500 | `enemy_proximity_penalty`=1.0, `length_bonus`=0.05 | The agent shows concerning signs: negative reward and steps trends indicate declining performance, while food efficiency remains healthy at 0.30. The high sharp turn usage (51.2%) suggests over-reactive movement. Slightly reducing enemy proximity penalty and boosting length bonus should encourage more strategic positioning while maintaining food collection. |
| 02-26 07:15 | 30000 | `boost_penalty`=-0.1, `enemy_approach_penalty`=0.5, `length_bonus`=0.1 | The agent shows excellent food collection (0.30 food/step) and decent survival (530 steps avg), but reward trend is significantly declining (-1240) despite stable other metrics. The high sharp turn usage (47.8%) and very low boost usage (2.8%) suggests overly cautious behavior. In Stage 5, the agent should be more aggressive and use boost strategically. |
| 02-26 09:47 | 30500 | `lr`=7.5e-05 | Training shows concerning trends: reward dropping by -768, steps decreasing by -67, and loss increasing by 22. The agent appears to be destabilizing despite healthy food intake. The high loss and declining performance suggest the learning rate may be too aggressive for this stage of training. |
| 02-26 12:54 | 31000 | `lr`=6e-05 | Training is very healthy with strong positive trends in reward (+753), steps (+113), and food efficiency (0.276). The high loss trend (+33) suggests learning instability, so reducing learning rate slightly will help stabilize convergence while maintaining progress. |
| 02-26 16:05 | 31500 | `lr`=4.5e-05, `target_update_freq`=2000 | The agent shows concerning signs: reward trend is negative (-18.2) despite stable steps, loss is increasing (+38.9), and Q-values are rising sharply (+37.1) suggesting overestimation. The high sharp turn usage (38.2%) indicates erratic movement patterns. Need to stabilize learning with reduced learning rate and target Q-value growth. |
| 02-26 19:02 | 32000 | `epsilon_target`=0.35, `lr`=6e-05 | Stage 5 shows declining performance with negative reward and steps trends, plus rising loss. The agent is over-exploring (epsilon too low at 0.08) and may need higher learning rate to adapt faster. Food efficiency remains healthy at 0.27. |
| 02-26 22:20 | 32500 |  | Training is progressing very well with strong positive trends in reward (+458) and steps (+77), healthy food efficiency (0.27), and decreasing loss. The agent is in Stage 5 and showing good exploration with high action entropy. No adjustments needed at this time. |
| 02-27 07:28 | 33000 | `lr`=4.5e-05, `survival`=0.5 | Training shows concerning regression with reward dropping -935 and steps decreasing -84. The agent has good food efficiency (0.27) but is dying earlier and performing worse. High loss trend (+255) and decreasing Q-values suggest learning instability. Reducing learning rate should stabilize training, and slightly increasing survival reward will encourage longer episodes. |
| 02-27 11:04 | 33600 |  | Training is progressing excellently with strong positive reward and steps trends, healthy food efficiency, and decreasing loss. The agent is in Stage 5 and showing mature behavior with good action diversity. No changes needed as all metrics indicate stable, improving performance. |
| 02-27 14:48 | 34200 |  | Training shows excellent progress with strong reward trend (+237), good survival time (563 steps), and healthy food intake (0.264 food/step). The snake is learning well in Stage 5. Loss is decreasing and Q-values are stable. High sharp turn usage (41.6%) suggests good maneuverability. No changes needed as all metrics indicate healthy learning. |
| 02-27 18:18 | 34800 | `lr`=3.5e-05, `food_shaping`=0.075, `gamma`=0.995 | Training shows declining performance with reward and steps trending downward while loss increases. The agent is overusing sharp turns (38.6%) and U-turns, suggesting poor pathfinding. High action entropy indicates the policy is becoming less confident. Need to stabilize learning and improve decision quality. |
| 02-27 21:00 | 35400 | `lr`=2.5e-05, `gamma`=0.997 | The agent is in Stage 5 with declining performance - reward dropping by 269 points and survival time decreasing. Loss is increasing significantly (234 trend) while Q-values are growing, indicating potential overfitting or instability. The learning rate needs reduction to stabilize training, and slight gamma increase will help with longer-term planning in this advanced stage. |
| 02-28 14:42 | 36000 |  | Training is progressing very well with strong positive trends in reward (+456), steps (+117), and decreasing loss. The agent is in Stage 5 with healthy food efficiency (0.292) and good survival. The high sharp turn usage (49%) suggests active maneuvering around enemies. Since performance is improving steadily, no changes are needed. |
| 03-02 09:26 | 36600 | `gamma`=0.998, `lr`=2e-05 | Training is progressing very well with strong positive trends in reward (+576), steps (+128), and Q-values. Food efficiency is healthy at 0.30. The high loss value (372) suggests the network is still learning actively. Since we're in Stage 5 (Mastery Survival), I'll make a small gamma increase to encourage longer-term planning and slightly reduce learning rate to stabilize the high loss. |
| 03-02 13:08 | 37200 | `lr`=1.5e-05, `survival`=0.45, `length_bonus`=0.08 | Training shows concerning signs: reward and steps are declining (-311 reward trend, -80 steps trend) while loss is exploding (996 increase). The high loss suggests learning instability. The agent's food efficiency remains healthy (0.30), but shorter episodes indicate reduced survival ability. Need to stabilize learning with lower learning rate and slightly increase survival incentives. |
| 03-02 16:01 | 37800 | `lr`=1e-05 | The agent shows declining reward (-254) and steps (-112) trends with rising loss (+748), indicating training instability. Food efficiency remains healthy at 0.309. The high loss trend suggests learning rate is too high for current convergence stage. Reducing lr should stabilize training and help recover performance. |
| 03-02 19:22 | 38400 |  | Training is very healthy with strong positive trends in reward (+830), steps (+175), and declining loss. Food efficiency is excellent at 0.304. The high sharp turn usage (36.4%) suggests good navigational learning. No changes needed as the agent is progressing well in Stage 5. |
| 03-04 10:59 | 39000 | `lr`=5e-06, `gamma`=0.995 | The agent shows concerning negative reward and steps trends (-477.7 reward, -126.7 steps) indicating deteriorating performance. The extremely high loss values and Q-values suggest training instability. The learning rate needs to be reduced to stabilize training, and gamma should be increased for better long-term planning in Stage 5. |
| 03-04 11:51 | 39000 | `lr`=2e-06 | The training shows concerning instability with massive loss values (101M) and exploding Q-values (1.5M), indicating gradient explosion. The negative reward and steps trends suggest performance degradation. The learning rate needs immediate reduction to stabilize training. |
| 03-04 16:29 | 39600 | `food_reward`=15.0, `lr`=1e-06, `survival`=0.6 | The agent is severely underperforming with food/step at 0.11 (well below healthy 0.25-0.40) and negative reward trend. Despite good survival time, it's not eating efficiently. The extremely high loss values and Q-values suggest reward signal issues. Need to boost food incentives and stabilize learning. |
| 03-04 18:33 | 39600 | `food_reward`=22.0, `lr`=5e-06, `food_shaping`=0.08 | The agent shows critically low food efficiency (0.12 vs healthy 0.25-0.40) and declining steps despite positive reward trend. The high loss values and Q-value instability suggest learning rate is too low for the current dynamics. Need to boost food rewards and increase learning rate. |
| 03-04 20:33 | 40200 | `lr`=1e-06, `epsilon_target`=0.15 | Training shows excellent food efficiency (0.363 food/step) and strong reward improvement (+257), but loss values are extremely high and unstable. The learning rate needs significant reduction to stabilize training. Also lowering epsilon_target since exploration is already very low at 0.08. |
| 03-05 07:13 | 39900 | `food_reward`=15.0, `lr`=1e-06, `food_shaping`=0.15 | The agent has very poor food efficiency (0.13 vs healthy 0.25-0.40) and strongly negative reward trend, indicating it's not learning to eat effectively. The high loss and Q-values suggest training instability. Need to boost food collection incentives and stabilize learning. |
| 03-05 09:58 | 40600 | `lr`=1e-06, `target_update_freq`=3000 | The agent shows concerning instability: loss has exploded 53K higher, Q-values dropped 342 points, and steps decreased by 42 despite good food efficiency (0.32). This suggests the neural network is becoming unstable. The extremely high loss indicates gradient explosion or learning rate issues. |
| 03-05 12:27 | 41300 | `lr`=1e-06, `epsilon_target`=0.35 | Training shows positive trends with healthy food efficiency (0.3043) and improving rewards (+106.8). However, the extremely high loss (371k) and volatile loss trend (+150k) indicates learning instability. The very low epsilon (0.08) may be causing insufficient exploration in Stage 5. Reducing learning rate should stabilize training, while increasing epsilon target will improve exploration for this complex stage. |
| 03-05 15:16 | 42000 |  | Training shows strong performance with healthy food efficiency (0.303), good survival duration (267 steps), and positive reward/steps trends indicating continued learning. The high loss values and negative Q-value trend suggest the agent is still actively learning complex behaviors in Stage 5. No adjustments needed as all key metrics are in healthy ranges. |
| 03-06 05:45 | 42700 | `lr`=1e-06, `survival`=0.6, `length_bonus`=0.08 | The agent is showing concerning trends with reward and steps declining significantly (-822 reward, -80 steps) while loss is exploding (+106k trend). The high loss suggests training instability. Food efficiency remains healthy at 0.31, so the core eating behavior is intact. Need to stabilize training with lower learning rate and improve survival incentives. |
| 03-06 09:23 | 43400 | `lr`=1e-06 | Training is progressing well with strong positive reward trend (+613) and good food efficiency (0.30 food/step). The high loss values and declining Q-values suggest the learning rate may be too aggressive for the current complexity. Reducing lr slightly will stabilize learning while maintaining progress. |
| 03-06 13:53 | 44100 | `lr`=1e-06 | Training shows excellent fundamentals - healthy food efficiency (0.31), strong positive reward/steps trends, and appropriate death distribution. However, the loss is extremely high (12K+) and trending upward, indicating training instability. The learning rate needs to be reduced to stabilize gradient updates. |
| 03-06 17:55 | 44800 | `epsilon_target`=0.35, `boost_penalty`=0.0 | The agent shows concerning trends with declining reward (-415), steps (-87), and Q-values (-604) despite good food efficiency (0.34 food/step). High boost usage (48.5%) and very low epsilon (0.08) suggest over-exploitation. Need to increase exploration and reduce boost penalty to allow more strategic movement. |
| 03-07 12:34 | 46000 | `lr`=7.5e-05 | Training shows excellent progress with positive reward/steps trends and healthy food efficiency (0.32). However, the extremely high loss (26k) and volatile Q-values suggest learning instability. Reducing learning rate will stabilize training while maintaining the strong performance trajectory. |
| 03-10 14:47 | 48000 | `boost_penalty`=0.35 | Training is progressing very well with strong positive trends in reward (+649) and steps (+293), excellent food efficiency (0.338), and decreasing loss. However, the extremely high boost usage (44.7%) suggests the agent may be over-relying on boosting. Since this is Stage 5 (Mastery Survival), I'll slightly increase boost penalty to encourage more strategic movement while maintaining the positive learning trajectory. |
| 03-11 12:05 | 51000 | `boost_penalty`=0.6, `survival`=0.25 | The agent shows concerning declining performance with negative reward and steps trends despite healthy food collection. High boost usage (41.6%) and declining survival suggest over-aggressive behavior. Reducing boost penalty to discourage excessive boosting while slightly increasing survival reward should help stabilize performance. |
| 03-12 10:48 | 54000 | `boost_penalty`=1.2, `survival`=0.35, `gamma`=0.97 | The agent shows concerning trends: reward and steps are declining (-477 reward, -138 steps) despite healthy food efficiency (0.34 food/step). High boost usage (40%) and declining survival suggest the snake is being too aggressive. Need to discourage boosting and encourage more cautious behavior. |
| 03-13 11:26 | 57000 | `boost_penalty`=1.5, `lr`=8e-05 | The agent shows declining performance with negative reward and steps trends despite healthy food efficiency (0.33). High boost usage (39.8%) and declining Q-values suggest the agent is becoming overly aggressive. Reducing boost penalty to discourage excessive boosting and slightly lowering learning rate to stabilize the declining loss trend. |
| 03-14 07:27 | 60000 | `boost_penalty`=1.8, `survival`=0.15 | The agent shows concerning negative trends in reward (-424) and steps (-107) despite healthy food efficiency (0.384). High boost usage (45%) and declining performance suggest the agent is becoming overly aggressive. Reducing boost penalty and increasing survival reward should encourage more cautious, sustainable gameplay. |

**Total consultations:** 131  
**Most adjusted:** `lr` (57x), `enemy_proximity_penalty` (42x), `death_snake` (33x), `enemy_approach_penalty` (33x), `food_reward` (22x), `gamma` (20x), `length_bonus` (17x), `boost_penalty` (13x), `epsilon_target` (10x), `survival` (10x), `food_shaping` (9x), `target_update_freq` (2x), `wall_proximity_penalty` (1x)

