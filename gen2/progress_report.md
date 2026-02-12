# Slither.io Bot - Training Progress Report

**Generated:** 2026-02-12 16:41:24  
**Total Episodes:** 175  
**Training Sessions:** 2

## Verdict: NOT LEARNING (Confidence: 45%)

**Goal Feasibility:** UNLIKELY (5-25% chance) without tuning

### Warnings
- Epsilon very high (0.811) - still mostly random

## Key Statistics

| Metric | Mean | Std | Min | Max | P50 | P95 |
|--------|------|-----|-----|-----|-----|-----|
| Reward | 645.54 | 589.94 | -31.58 | 3106.91 | 490.90 | 1829.70 |
| Steps | 106.73 | 94.32 | 1.00 | 400.00 | 75.00 | 302.00 |
| Food | 39.33 | 24.16 | 0.00 | 122.00 | 31.00 | 86.60 |
| Loss | 0.23 | 0.31 | 0.00 | 3.35 | 0.14 | 0.63 |

## Goal Progress

| Target | Current Best | Goal | Progress |
|--------|-------------|------|----------|
| Points | 122 | 6,000 | 2.0% |
| Survival | 13.3 min | 60 min | 22.2% |

## Session History

| # | Style | Episodes | Avg Reward | Avg Steps |
|---|-------|----------|------------|----------|
| 1 | Unknown | 1-86 | 718.5 | 150 |
| 2 | Aggressive (Hunter) | 87-175 | 575.0 | 65 |

## Recommendations

Some learning signals present but not strong enough.
  1. Fine-tune hyperparameters
  2. Increase training duration significantly
  3. Consider curriculum adjustments

1. Epsilon is 0.811 after 175 episodes. Consider faster decay (eps_decay=50000) or lower eps_start if resuming.

2. Average episode too short. Consider:
     - Reducing death penalties to avoid discouraging exploration
     - Adding survival bonus to incentivize staying alive

## Charts

![Overview](training_progress_overview.png)

![Learning Detection](training_learning_detection.png)

![Goal Progress](training_goal_progress.png)

