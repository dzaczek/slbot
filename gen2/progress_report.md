# Slither.io Bot - Training Progress Report

**Generated:** 2026-02-12 22:55:19  
**Total Episodes:** 631  
**Training Sessions:** 1

## Verdict: NOT LEARNING (Confidence: 50%)

**Goal Feasibility:** UNLIKELY (5-25% chance) without tuning

### Warnings
- Rewards flat: change = -20.7 between halves

### Positive Signals
- Loss decreasing (model converging)

## Key Statistics

| Metric | Mean | Std | Min | Max | P50 | P95 |
|--------|------|-----|-----|-----|-----|-----|
| Reward | 179.32 | 262.70 | -38.58 | 2242.28 | 82.25 | 678.93 |
| Steps | 63.21 | 54.00 | 2.00 | 340.00 | 46.00 | 169.00 |
| Food | 27.72 | 10.85 | 0.00 | 77.00 | 25.00 | 47.50 |
| Loss | 15.60 | 46.69 | 0.86 | 392.24 | 5.92 | 40.98 |

## Goal Progress

| Target | Current Best | Goal | Progress |
|--------|-------------|------|----------|
| Points | 77 | 6,000 | 1.3% |
| Survival | 11.3 min | 60 min | 18.9% |

## Session History

| # | Style | Episodes | Avg Reward | Avg Steps |
|---|-------|----------|------------|----------|
| 1 | Standard (Curriculum) | 1-631 | 179.3 | 63 |

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

