# Karpathy Mod Experiment Report
*Generated: 2026-03-30 11:47:34*

## Summary

| Metric | Value |
|--------|-------|
| Total experiments | 96 |
| Keep rate | 0.0% |
| Kept | 0 |
| Discarded | 96 |
| Inconclusive | 0 |
| Best improvement | +0.00% |
| Worst regression | -0.63% |
| Avg improvement | -0.04% |
| Trend | **PLATEAUING** |

## Strategy Effectiveness

| Strategy | Count | Kept | Keep Rate | Avg Improvement | Best |
|----------|------:|-----:|----------:|----------------:|-----:|
| explore | 31 | 0 | 0.0% | -0.03% | +0.00% |
| radical | 31 | 0 | 0.0% | -0.05% | +0.00% |
| tweak | 34 | 0 | 0.0% | -0.05% | +0.00% |

## Stage Performance

| Stage | Name | Count | Kept | Keep Rate | Avg Improvement |
|------:|------|------:|-----:|----------:|----------------:|
| S1 | FOOD_VECTOR | 19 | 0 | 0.0% | -0.02% |
| S2 | WALL_AVOID | 9 | 0 | 0.0% | -0.01% |
| S3 | ENEMY_AVOID | 14 | 0 | 0.0% | -0.02% |
| S4 | MASS_MANAGEMENT | 19 | 0 | 0.0% | -0.06% |
| S5 | MASTERY_SURVIVAL | 21 | 0 | 0.0% | -0.05% |
| S6 | APEX_PREDATOR | 14 | 0 | 0.0% | -0.09% |

## Top 5 Best Kept Experiments

| Round | Improvement | Strategy | Stage | Description |
|------:|------------:|----------|------:|-------------|

## Top 5 Worst Discarded Experiments

| Round | Improvement | Strategy | Stage | Description |
|------:|------------:|----------|------:|-------------|
| R7270 | -0.63% | tweak | S6 | [tweak] S6/APEX_PREDATOR: enemy_zone_control_reward: 0.0600 -> 0.0645 |
| R7270 | -0.61% | radical | S6 | [radical] S6/APEX_PREDATOR: max_steps: 99999 -> 10000; death_snake: -30 -> -25.8 |
| R7270 | -0.60% | explore | S5 | [explore] S5/MASTERY_SURVIVAL: death_snake: -50 -> -57.592492888353505; enemy_al |
| R7270 | -0.60% | radical | S4 | [radical] S4/MASS_MANAGEMENT: death_snake: -50 -> -37.885784361432904; death_wal |
| R7278 | -0.20% | tweak | S4 | [tweak] S4/MASS_MANAGEMENT: length_bonus: 0.0200 -> 0.0246 |

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

**PLATEAUING**

Keep rate is stable. Consider trying more explore/radical strategies to escape local optimum.
