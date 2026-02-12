# Slither.io Bot - Training Progress Report

**Generated:** 2026-02-12 17:40:17  
**Total Episodes:** 120  
**Training Sessions:** 2

## Verdict: NOT LEARNING (Confidence: 50%)

**Goal Feasibility:** UNLIKELY (5-25% chance) without tuning

### Warnings
- Loss very high (53.31) - training unstable
- Epsilon very high (0.925) - still mostly random

### Positive Signals
- Food collection improving (slope=0.0259/ep)
- Loss decreasing (model converging)

## Key Statistics

| Metric | Mean | Std | Min | Max | P50 | P95 |
|--------|------|-----|-----|-----|-----|-----|
| Reward | 145.03 | 204.97 | -28.48 | 1104.08 | 74.22 | 530.98 |
| Steps | 56.02 | 47.44 | 2.00 | 225.00 | 43.00 | 146.45 |
| Food | 27.04 | 9.40 | 4.00 | 62.00 | 25.00 | 45.05 |
| Loss | 80.53 | 61.35 | 0.00 | 588.59 | 73.34 | 146.84 |

## Goal Progress

| Target | Current Best | Goal | Progress |
|--------|-------------|------|----------|
| Points | 62 | 6,000 | 1.0% |
| Survival | 7.5 min | 60 min | 12.5% |

## Session History

| # | Style | Episodes | Avg Reward | Avg Steps |
|---|-------|----------|------------|----------|
| 1 | Unknown | 1-8 | 253.5 | 86 |
| 2 | Standard (Curriculum) | 9-120 | 137.3 | 54 |

## Recommendations

Some learning signals present but not strong enough.
  1. Fine-tune hyperparameters
  2. Increase training duration significantly
  3. Consider curriculum adjustments

1. Epsilon is 0.925 after 120 episodes. Consider faster decay (eps_decay=50000) or lower eps_start if resuming.

2. Average episode too short. Consider:
     - Reducing death penalties to avoid discouraging exploration
     - Adding survival bonus to incentivize staying alive

## Charts

![Overview](training_progress_overview.png)

![Learning Detection](training_learning_detection.png)

![Goal Progress](training_goal_progress.png)

