# Karpathy Mod Experiment Report
*Generated: 2026-04-03 13:00:40*

## Summary

| Metric | Value |
|--------|-------|
| Total experiments | 123 |
| Keep rate | 14.6% |
| Kept | 18 |
| Discarded | 105 |
| Inconclusive | 0 |
| Best improvement | +4.41% |
| Worst regression | -8.72% |
| Avg improvement | -0.21% |
| Trend | **IMPROVING** |

## Strategy Effectiveness

| Strategy | Count | Kept | Keep Rate | Avg Improvement | Best |
|----------|------:|-----:|----------:|----------------:|-----:|
| explore | 48 | 13 | 27.1% | -0.31% | +2.08% |
| radical | 36 | 3 | 8.3% | -0.27% | +0.33% |
| tweak | 39 | 2 | 5.1% | -0.04% | +4.41% |

## Stage Performance

| Stage | Name | Count | Kept | Keep Rate | Avg Improvement |
|------:|------|------:|-----:|----------:|----------------:|
| S1 | FOOD_VECTOR | 23 | 4 | 17.4% | +0.26% |
| S2 | WALL_AVOID | 12 | 1 | 8.3% | -0.77% |
| S3 | ENEMY_AVOID | 20 | 2 | 10.0% | -1.05% |
| S4 | MASS_MANAGEMENT | 26 | 4 | 15.4% | -0.00% |
| S5 | MASTERY_SURVIVAL | 28 | 7 | 25.0% | -0.01% |
| S6 | APEX_PREDATOR | 14 | 0 | 0.0% | -0.09% |

## Top 5 Best Kept Experiments

| Round | Improvement | Strategy | Stage | Description |
|------:|------------:|----------|------:|-------------|
| R7334 | +4.41% | tweak | S3 | [tweak] S3/ENEMY_AVOID: enemy_approach_penalty: 0.5000 -> 0.5727 |
| R7323 | +2.08% | explore | S1 | [explore] S1/FOOD_VECTOR: food_reward: 3.5666 -> 3.8172; death_snake: -15 -> -17 |
| R7310 | +1.93% | explore | S1 | [explore] S1/FOOD_VECTOR: death_wall: -15 -> -13.415340727031698; food_shaping:  |
| R7324 | +1.67% | explore | S1 | [explore] S1/FOOD_VECTOR: max_steps: 600 -> 707; death_snake: -17.2502 -> -18.21 |
| R7331 | +1.15% | explore | S2 | [explore] S2/WALL_AVOID: starvation_penalty: 0.0050 -> 0.0074; death_wall: -40 - |

## Top 5 Worst Discarded Experiments

| Round | Improvement | Strategy | Stage | Description |
|------:|------------:|----------|------:|-------------|
| R7317 | -8.72% | explore | S3 | [explore] S3/ENEMY_AVOID: starvation_penalty: 0.0080 -> 0.0114; food_reward: 5.0 |
| R7322 | -8.71% | radical | S3 | [radical] S3/ENEMY_AVOID: starvation_grace_steps: 60 -> 74; death_wall: -40 -> - |
| R7304 | -8.57% | explore | S3 | [explore] S3/ENEMY_AVOID: gamma: 0.9500 -> 0.9291; enemy_proximity_penalty: 1.50 |
| R7308 | -5.96% | explore | S2 | [explore] S2/WALL_AVOID: food_shaping: 0.1500 -> 0.0965; survival_escalation: 0. |
| R7318 | -4.41% | tweak | S2 | [tweak] S2/WALL_AVOID: gamma: 0.9300 -> 0.9399 |

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
