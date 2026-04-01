# Karpathy Mod Experiment Report
*Generated: 2026-04-01 17:35:09*

## Summary

| Metric | Value |
|--------|-------|
| Total experiments | 110 |
| Keep rate | 5.5% |
| Kept | 6 |
| Discarded | 104 |
| Inconclusive | 0 |
| Best improvement | +1.93% |
| Worst regression | -8.72% |
| Avg improvement | -0.27% |
| Trend | **IMPROVING** |

## Strategy Effectiveness

| Strategy | Count | Kept | Keep Rate | Avg Improvement | Best |
|----------|------:|-----:|----------:|----------------:|-----:|
| explore | 38 | 3 | 7.9% | -0.58% | +1.93% |
| radical | 34 | 2 | 5.9% | -0.04% | +0.11% |
| tweak | 38 | 1 | 2.6% | -0.16% | +0.06% |

## Stage Performance

| Stage | Name | Count | Kept | Keep Rate | Avg Improvement |
|------:|------|------:|-----:|----------:|----------------:|
| S1 | FOOD_VECTOR | 20 | 1 | 5.0% | +0.08% |
| S2 | WALL_AVOID | 11 | 0 | 0.0% | -0.95% |
| S3 | ENEMY_AVOID | 17 | 0 | 0.0% | -1.04% |
| S4 | MASS_MANAGEMENT | 23 | 1 | 4.3% | -0.04% |
| S5 | MASTERY_SURVIVAL | 25 | 4 | 16.0% | -0.02% |
| S6 | APEX_PREDATOR | 14 | 0 | 0.0% | -0.09% |

## Top 5 Best Kept Experiments

| Round | Improvement | Strategy | Stage | Description |
|------:|------------:|----------|------:|-------------|
| R7310 | +1.93% | explore | S1 | [explore] S1/FOOD_VECTOR: death_wall: -15 -> -13.415340727031698; food_shaping:  |
| R7302 | +0.12% | explore | S5 | [explore] S5/MASTERY_SURVIVAL: enemy_alert_dist: 2000 -> 2051; mass_loss_penalty |
| R7306 | +0.11% | explore | S5 | [explore] S5/MASTERY_SURVIVAL: food_reward: 8.0000 -> 6.8031; survival: 0.3323 - |
| R7309 | +0.11% | radical | S5 | [radical] S5/MASTERY_SURVIVAL: gamma: 0.9700 -> 0.9770; food_shaping: 0.2500 ->  |
| R7316 | +0.09% | radical | S4 | [radical] S4/MASS_MANAGEMENT: gamma: 0.9700 -> 0.9840; food_shaping: 0.3000 -> 0 |

## Top 5 Worst Discarded Experiments

| Round | Improvement | Strategy | Stage | Description |
|------:|------------:|----------|------:|-------------|
| R7317 | -8.72% | explore | S3 | [explore] S3/ENEMY_AVOID: starvation_penalty: 0.0080 -> 0.0114; food_reward: 5.0 |
| R7304 | -8.57% | explore | S3 | [explore] S3/ENEMY_AVOID: gamma: 0.9500 -> 0.9291; enemy_proximity_penalty: 1.50 |
| R7308 | -5.96% | explore | S2 | [explore] S2/WALL_AVOID: food_shaping: 0.1500 -> 0.0965; survival_escalation: 0. |
| R7318 | -4.41% | tweak | S2 | [tweak] S2/WALL_AVOID: gamma: 0.9300 -> 0.9399 |
| R7270 | -0.63% | tweak | S6 | [tweak] S6/APEX_PREDATOR: enemy_zone_control_reward: 0.0600 -> 0.0645 |

## Charts

### Experiment Overview Dashboard
![Experiment Overview Dashboard](charts/karpathy_overview.png)

### Strategy Effectiveness
![Strategy Effectiveness](charts/karpathy_strategy.png)

### Stage Analysis
![Stage Analysis](charts/karpathy_stages.png)

### Metric Evolution
![Metric Evolution](charts/karpathy_metrics.png)

### Parameter Impact Analysis
![Parameter Impact Analysis](charts/karpathy_parameters.png)

### Improvement Distribution
![Improvement Distribution](charts/karpathy_distribution.png)

### Strategy x Stage Heatmap
![Strategy x Stage Heatmap](charts/karpathy_heatmap.png)

### Interactive Explorer
[Open Experiment Explorer](charts/karpathy_experiment_explorer.html)

## Trend Assessment

**IMPROVING**

Keep rate is increasing over time. The mutation system is learning what works.
