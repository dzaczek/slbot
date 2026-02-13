# Slither.io Bot - Training Progress Report

**Generated:** 2026-02-13 01:16:14  
**Total Episodes:** 2011  
**Training Sessions:** 1

## Verdict: NOT LEARNING (Confidence: 45%)

**Goal Feasibility:** UNLIKELY (5-25% chance) without tuning

### Warnings
- Rewards flat: change = -6.5 between halves

## Key Statistics

| Metric | Mean | Std | Min | Max | P50 | P95 |
|--------|------|-----|-----|-----|-----|-----|
| Reward | 190.60 | 260.19 | -38.58 | 2242.28 | 92.30 | 710.05 |
| Steps | 66.08 | 54.39 | 2.00 | 340.00 | 51.00 | 172.50 |
| Food | 28.34 | 10.75 | 0.00 | 84.00 | 25.00 | 48.50 |
| Loss | 11.08 | 26.66 | 0.47 | 392.24 | 7.37 | 19.68 |

## Goal Progress

| Target | Current Best | Goal | Progress |
|--------|-------------|------|----------|
| Points | 84 | 6,000 | 1.4% |
| Survival | 11.3 min | 60 min | 18.9% |

## Session History

| # | Style | Episodes | Avg Reward | Avg Steps |
|---|-------|----------|------------|----------|
| 1 | Standard (Curriculum) | 1-2011 | 190.6 | 66 |

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

