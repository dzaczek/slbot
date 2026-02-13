# Slither.io Bot - Training Progress Report

**Generated:** 2026-02-13 01:26:18  
**Total Episodes:** 2125  
**Training Sessions:** 1

## Verdict: NOT LEARNING (Confidence: 45%)

**Goal Feasibility:** UNLIKELY (5-25% chance) without tuning

### Warnings
- Rewards flat: change = -14.9 between halves

## Key Statistics

| Metric | Mean | Std | Min | Max | P50 | P95 |
|--------|------|-----|-----|-----|-----|-----|
| Reward | 189.34 | 257.43 | -38.58 | 2242.28 | 92.32 | 701.15 |
| Steps | 65.91 | 53.96 | 2.00 | 340.00 | 51.00 | 171.00 |
| Food | 28.33 | 10.68 | 0.00 | 84.00 | 26.00 | 48.00 |
| Loss | 11.06 | 25.99 | 0.47 | 392.24 | 7.47 | 20.20 |

## Goal Progress

| Target | Current Best | Goal | Progress |
|--------|-------------|------|----------|
| Points | 84 | 6,000 | 1.4% |
| Survival | 11.3 min | 60 min | 18.9% |

## Session History

| # | Style | Episodes | Avg Reward | Avg Steps |
|---|-------|----------|------------|----------|
| 1 | Standard (Curriculum) | 1-2125 | 189.3 | 66 |

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

