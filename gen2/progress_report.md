# Slither.io Bot - Training Progress Report

**Generated:** 2026-02-12 22:04:59  
**Total Episodes:** 120  
**Training Sessions:** 1

## Verdict: NOT LEARNING (Confidence: 55%)

**Goal Feasibility:** UNLIKELY (5-25% chance) without tuning

### Warnings
- Epsilon very high (0.902) - still mostly random

### Positive Signals
- Food collection improving (slope=0.0547/ep)
- Loss decreasing (model converging)

## Key Statistics

| Metric | Mean | Std | Min | Max | P50 | P95 |
|--------|------|-----|-----|-----|-----|-----|
| Reward | 248.01 | 356.97 | -32.56 | 1996.81 | 104.39 | 980.32 |
| Steps | 75.83 | 66.25 | 4.00 | 318.00 | 55.00 | 211.05 |
| Food | 30.25 | 14.52 | 3.00 | 77.00 | 25.50 | 58.10 |
| Loss | 55.43 | 97.18 | 1.26 | 392.24 | 9.04 | 274.99 |

## Goal Progress

| Target | Current Best | Goal | Progress |
|--------|-------------|------|----------|
| Points | 77 | 6,000 | 1.3% |
| Survival | 10.6 min | 60 min | 17.7% |

## Session History

| # | Style | Episodes | Avg Reward | Avg Steps |
|---|-------|----------|------------|----------|
| 1 | Standard (Curriculum) | 1-120 | 248.0 | 76 |

## Recommendations

Some learning signals present but not strong enough.
  1. Fine-tune hyperparameters
  2. Increase training duration significantly
  3. Consider curriculum adjustments

1. Epsilon is 0.902 after 120 episodes. Consider faster decay (eps_decay=50000) or lower eps_start if resuming.

2. Average episode too short. Consider:
     - Reducing death penalties to avoid discouraging exploration
     - Adding survival bonus to incentivize staying alive

## Charts

![Overview](training_progress_overview.png)

![Learning Detection](training_learning_detection.png)

![Goal Progress](training_goal_progress.png)

