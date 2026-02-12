# Slither.io Bot - Training Progress Report

**Generated:** 2026-02-12 12:30:26  
**Total Episodes:** 2472  
**Training Sessions:** 1

## Verdict: NOT LEARNING (Confidence: 45%)

**Goal Feasibility:** UNLIKELY (5-25% chance) without tuning

### Warnings
- Rewards flat: change = 38.2 between halves

## Key Statistics

| Metric | Mean | Std | Min | Max | P50 | P95 |
|--------|------|-----|-----|-----|-----|-----|
| Reward | 205.48 | 314.19 | -40.00 | 3087.36 | 92.27 | 822.31 |
| Steps | 67.38 | 60.10 | 1.00 | 400.00 | 50.00 | 189.00 |
| Food | 29.73 | 13.10 | 0.00 | 106.00 | 26.00 | 56.00 |
| Loss | 1.72 | 1.71 | 0.01 | 10.93 | 1.18 | 5.19 |

## Goal Progress

| Target | Current Best | Goal | Progress |
|--------|-------------|------|----------|
| Points | 106 | 6,000 | 1.8% |
| Survival | 13.3 min | 60 min | 22.2% |

## Session History

| # | Style | Episodes | Avg Reward | Avg Steps |
|---|-------|----------|------------|----------|
| 1 | Standard (Curriculum) | 1-2472 | 205.5 | 67 |

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

