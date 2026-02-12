# Slither.io Bot - Training Progress Report

**Generated:** 2026-02-12 15:02:09  
**Total Episodes:** 3955  
**Training Sessions:** 1

## Verdict: NOT LEARNING (Confidence: 45%)

**Goal Feasibility:** UNLIKELY (5-25% chance) without tuning

### Warnings
- Rewards flat: change = -2.9 between halves

## Key Statistics

| Metric | Mean | Std | Min | Max | P50 | P95 |
|--------|------|-----|-----|-----|-----|-----|
| Reward | 210.56 | 307.87 | -40.00 | 3087.36 | 101.51 | 796.23 |
| Steps | 69.13 | 59.39 | 1.00 | 400.00 | 53.00 | 186.00 |
| Food | 30.13 | 12.97 | 0.00 | 106.00 | 27.00 | 55.00 |
| Loss | 2.75 | 2.46 | 0.01 | 18.59 | 2.11 | 7.65 |

## Goal Progress

| Target | Current Best | Goal | Progress |
|--------|-------------|------|----------|
| Points | 106 | 6,000 | 1.8% |
| Survival | 13.3 min | 60 min | 22.2% |

## Session History

| # | Style | Episodes | Avg Reward | Avg Steps |
|---|-------|----------|------------|----------|
| 1 | Standard (Curriculum) | 1-3955 | 210.6 | 69 |

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

