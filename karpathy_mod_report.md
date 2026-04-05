# Karpathy Mod Experiment Report
*Generated: 2026-04-05 11:58:54*

## Summary

| Metric | Value |
|--------|-------|
| Total experiments | 318 |
| Keep rate | 66.4% |
| Kept | 211 |
| Discarded | 107 |
| Inconclusive | 0 |
| Best improvement | +4.41% |
| Worst regression | -8.72% |
| Avg improvement | +0.73% |
| Trend | **IMPROVING** |

## Strategy Effectiveness

| Strategy | Count | Kept | Keep Rate | Avg Improvement | Best |
|----------|------:|-----:|----------:|----------------:|-----:|
| radical | 107 | 73 | 68.2% | +0.72% | +4.28% |
| explore | 110 | 75 | 68.2% | +0.56% | +4.28% |
| tweak | 101 | 63 | 62.4% | +0.91% | +4.41% |

## Stage Performance

| Stage | Name | Count | Kept | Keep Rate | Avg Improvement |
|------:|------|------:|-----:|----------:|----------------:|
| S1 | FOOD_VECTOR | 76 | 57 | 75.0% | +0.50% |
| S2 | WALL_AVOID | 56 | 45 | 80.4% | +0.67% |
| S3 | ENEMY_AVOID | 59 | 41 | 69.5% | +2.47% |
| S4 | MASS_MANAGEMENT | 51 | 29 | 56.9% | +0.14% |
| S5 | MASTERY_SURVIVAL | 62 | 39 | 62.9% | +0.06% |
| S6 | APEX_PREDATOR | 14 | 0 | 0.0% | -0.09% |

## Top 5 Best Kept Experiments

| Round | Improvement | Strategy | Stage | Description |
|------:|------------:|----------|------:|-------------|
| R7334 | +4.41% | tweak | S3 | [tweak] S3/ENEMY_AVOID: enemy_approach_penalty: 0.5000 -> 0.5727 |
| R7342 | +4.28% | tweak | S3 | [tweak] S3/ENEMY_AVOID: gamma: 0.9500 -> 0.9368 |
| R7349 | +4.28% | explore | S3 | [explore] S3/ENEMY_AVOID: death_snake: -40 -> -39.59706296492766; enemy_proximit |
| R7354 | +4.28% | tweak | S3 | [tweak] S3/ENEMY_AVOID: starvation_penalty: 0.0089 -> 0.0087 |
| R7358 | +4.28% | tweak | S3 | [tweak] S3/ENEMY_AVOID: food_shaping: 0.1000 -> 0.1044 |

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
