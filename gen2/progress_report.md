# Slither.io Bot - Training Progress Report

**Generated:** 2026-02-12 18:00:25  
**Total Episodes:** 345  
**Training Sessions:** 2

## Verdict: NOT LEARNING (Confidence: 45%)

**Goal Feasibility:** UNLIKELY (5-25% chance) without tuning

### Warnings
- Rewards flat: change = -33.8 between halves
- Loss very high (41.08) - training unstable

### Positive Signals
- Loss decreasing (model converging)

## Key Statistics

| Metric | Mean | Std | Min | Max | P50 | P95 |
|--------|------|-----|-----|-----|-----|-----|
| Reward | 120.44 | 171.93 | -38.52 | 1104.08 | 63.41 | 486.17 |
| Steps | 50.60 | 41.73 | 2.00 | 225.00 | 40.00 | 139.00 |
| Food | 26.11 | 8.48 | 0.00 | 62.00 | 24.00 | 41.80 |
| Loss | 51.92 | 44.72 | 0.00 | 588.59 | 40.77 | 116.10 |

## Goal Progress

| Target | Current Best | Goal | Progress |
|--------|-------------|------|----------|
| Points | 62 | 6,000 | 1.0% |
| Survival | 7.5 min | 60 min | 12.5% |

## Session History

| # | Style | Episodes | Avg Reward | Avg Steps |
|---|-------|----------|------------|----------|
| 1 | Unknown | 1-8 | 253.5 | 86 |
| 2 | Standard (Curriculum) | 9-345 | 117.3 | 50 |

## Recommendations

Some learning signals present but not strong enough.
  1. Fine-tune hyperparameters
  2. Increase training duration significantly
  3. Consider curriculum adjustments

1. Average episode too short. Consider:
     - Reducing death penalties to avoid discouraging exploration
     - Adding survival bonus to incentivize staying alive

## Charts

![Overview](training_progress_overview.png)

![Learning Detection](training_learning_detection.png)

![Goal Progress](training_goal_progress.png)

