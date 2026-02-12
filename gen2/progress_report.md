# Slither.io Bot - Training Progress Report

**Generated:** 2026-02-12 16:31:20  
**Total Episodes:** 86  
**Training Sessions:** 1

## Verdict: NOT LEARNING (Confidence: 50%)

**Goal Feasibility:** UNLIKELY (5-25% chance) without tuning

### Warnings
- Epsilon very high (0.866) - still mostly random

### Positive Signals
- Loss decreasing (model converging)

## Key Statistics

| Metric | Mean | Std | Min | Max | P50 | P95 |
|--------|------|-----|-----|-----|-----|-----|
| Reward | 718.53 | 784.57 | -31.58 | 3106.91 | 362.48 | 2544.06 |
| Steps | 149.63 | 103.74 | 5.00 | 400.00 | 117.50 | 360.75 |
| Food | 49.84 | 27.70 | 3.00 | 122.00 | 39.00 | 102.75 |
| Loss | 0.14 | 0.39 | 0.01 | 3.35 | 0.05 | 0.42 |

## Goal Progress

| Target | Current Best | Goal | Progress |
|--------|-------------|------|----------|
| Points | 122 | 6,000 | 2.0% |
| Survival | 13.3 min | 60 min | 22.2% |

## Session History

| # | Style | Episodes | Avg Reward | Avg Steps |
|---|-------|----------|------------|----------|
| 1 | Unknown | 1-86 | 718.5 | 150 |

## Recommendations

Some learning signals present but not strong enough.
  1. Fine-tune hyperparameters
  2. Increase training duration significantly
  3. Consider curriculum adjustments

1. Epsilon is 0.866 after 86 episodes. Consider faster decay (eps_decay=50000) or lower eps_start if resuming.

## Charts

![Overview](training_progress_overview.png)

![Learning Detection](training_learning_detection.png)

![Goal Progress](training_goal_progress.png)

