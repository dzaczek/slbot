# Slither.io Bot - Training Progress Report

**Generated:** 2026-02-12 22:35:11  
**Total Episodes:** 430  
**Training Sessions:** 1

## Verdict: NOT LEARNING (Confidence: 50%)

**Goal Feasibility:** UNLIKELY (5-25% chance) without tuning

### Warnings
- Rewards flat: change = -18.0 between halves

### Positive Signals
- Loss decreasing (model converging)

## Key Statistics

| Metric | Mean | Std | Min | Max | P50 | P95 |
|--------|------|-----|-----|-----|-----|-----|
| Reward | 183.64 | 257.98 | -38.58 | 1996.81 | 90.59 | 682.12 |
| Steps | 64.60 | 53.72 | 2.00 | 318.00 | 50.00 | 169.55 |
| Food | 27.86 | 11.11 | 0.00 | 77.00 | 25.00 | 48.00 |
| Loss | 19.24 | 56.11 | 0.86 | 392.24 | 5.54 | 75.34 |

## Goal Progress

| Target | Current Best | Goal | Progress |
|--------|-------------|------|----------|
| Points | 77 | 6,000 | 1.3% |
| Survival | 10.6 min | 60 min | 17.7% |

## Session History

| # | Style | Episodes | Avg Reward | Avg Steps |
|---|-------|----------|------------|----------|
| 1 | Standard (Curriculum) | 1-430 | 183.6 | 65 |

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

