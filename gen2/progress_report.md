# Slither.io Bot - Training Progress Report

**Generated:** 2026-02-12 21:54:56  
**Total Episodes:** 20  
**Training Sessions:** 1

## Verdict: LEARNING (Confidence: 70%)

**Goal Feasibility:** POSSIBLE (25-60% chance) with continued training

### Warnings
- Epsilon very high (0.984) - still mostly random

### Positive Signals
- Positive reward trend (slope=14.1545, R²=0.207)
- Episodes getting longer (slope=2.696/ep)
- Food collection improving (slope=1.1880/ep)

## Key Statistics

| Metric | Mean | Std | Min | Max | P50 | P95 |
|--------|------|-----|-----|-----|-----|-----|
| Reward | 136.01 | 179.52 | -32.56 | 762.50 | 82.11 | 402.20 |
| Steps | 56.80 | 42.23 | 4.00 | 183.00 | 50.00 | 123.15 |
| Food | 22.10 | 11.54 | 3.00 | 41.00 | 21.00 | 41.00 |
| Loss | 250.12 | 94.78 | 56.42 | 392.24 | 236.07 | 383.85 |

## Goal Progress

| Target | Current Best | Goal | Progress |
|--------|-------------|------|----------|
| Points | 41 | 6,000 | 0.7% |
| Survival | 6.1 min | 60 min | 10.2% |

## Session History

| # | Style | Episodes | Avg Reward | Avg Steps |
|---|-------|----------|------------|----------|
| 1 | Standard (Curriculum) | 1-20 | 136.0 | 57 |

## Recommendations

Keep training. Monitor for sustained improvement.

1. Epsilon is 0.984 after 20 episodes. Consider faster decay (eps_decay=50000) or lower eps_start if resuming.

2. Average episode too short. Consider:
     - Reducing death penalties to avoid discouraging exploration
     - Adding survival bonus to incentivize staying alive

## Charts

![Overview](training_progress_overview.png)

![Learning Detection](training_learning_detection.png)

![Goal Progress](training_goal_progress.png)

