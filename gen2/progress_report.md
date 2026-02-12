# Slither.io Bot - Training Progress Report

**Generated:** 2026-02-12 23:55:42  
**Total Episodes:** 1184  
**Training Sessions:** 1

## Verdict: NOT LEARNING (Confidence: 45%)

**Goal Feasibility:** UNLIKELY (5-25% chance) without tuning

### Warnings
- Rewards flat: change = 38.3 between halves

## Key Statistics

| Metric | Mean | Std | Min | Max | P50 | P95 |
|--------|------|-----|-----|-----|-----|-----|
| Reward | 201.90 | 273.67 | -38.58 | 2242.28 | 102.97 | 719.66 |
| Steps | 68.62 | 55.86 | 2.00 | 340.00 | 54.00 | 174.85 |
| Food | 28.75 | 11.13 | 0.00 | 84.00 | 26.00 | 49.00 |
| Loss | 12.34 | 34.42 | 0.53 | 392.24 | 6.68 | 19.71 |

## Goal Progress

| Target | Current Best | Goal | Progress |
|--------|-------------|------|----------|
| Points | 84 | 6,000 | 1.4% |
| Survival | 11.3 min | 60 min | 18.9% |

## Session History

| # | Style | Episodes | Avg Reward | Avg Steps |
|---|-------|----------|------------|----------|
| 1 | Standard (Curriculum) | 1-1184 | 201.9 | 69 |

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

