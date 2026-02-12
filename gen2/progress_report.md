# Slither.io Bot - Training Progress Report

**Generated:** 2026-02-12 22:25:07  
**Total Episodes:** 325  
**Training Sessions:** 1

## Verdict: NOT LEARNING (Confidence: 50%)

**Goal Feasibility:** UNLIKELY (5-25% chance) without tuning

### Warnings
- Rewards flat: change = -43.6 between halves

### Positive Signals
- Loss decreasing (model converging)

## Key Statistics

| Metric | Mean | Std | Min | Max | P50 | P95 |
|--------|------|-----|-----|-----|-----|-----|
| Reward | 187.95 | 271.63 | -38.58 | 1996.81 | 91.72 | 788.52 |
| Steps | 65.17 | 55.26 | 2.00 | 318.00 | 51.00 | 185.40 |
| Food | 27.93 | 11.52 | 0.00 | 77.00 | 25.00 | 50.80 |
| Loss | 23.78 | 63.86 | 1.07 | 392.24 | 5.87 | 181.56 |

## Goal Progress

| Target | Current Best | Goal | Progress |
|--------|-------------|------|----------|
| Points | 77 | 6,000 | 1.3% |
| Survival | 10.6 min | 60 min | 17.7% |

## Session History

| # | Style | Episodes | Avg Reward | Avg Steps |
|---|-------|----------|------------|----------|
| 1 | Standard (Curriculum) | 1-325 | 187.9 | 65 |

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

