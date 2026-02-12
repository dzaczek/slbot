# Slither.io Bot - Training Progress Report

**Generated:** 2026-02-12 23:25:30  
**Total Episodes:** 918  
**Training Sessions:** 1

## Verdict: NOT LEARNING (Confidence: 50%)

**Goal Feasibility:** UNLIKELY (5-25% chance) without tuning

### Warnings
- Rewards flat: change = 11.4 between halves

### Positive Signals
- Loss decreasing (model converging)

## Key Statistics

| Metric | Mean | Std | Min | Max | P50 | P95 |
|--------|------|-----|-----|-----|-----|-----|
| Reward | 189.10 | 264.02 | -38.58 | 2242.28 | 88.65 | 684.89 |
| Steps | 65.61 | 54.82 | 2.00 | 340.00 | 49.00 | 170.15 |
| Food | 28.10 | 10.93 | 0.00 | 77.00 | 25.00 | 49.00 |
| Loss | 13.43 | 38.95 | 0.53 | 392.24 | 6.36 | 22.16 |

## Goal Progress

| Target | Current Best | Goal | Progress |
|--------|-------------|------|----------|
| Points | 77 | 6,000 | 1.3% |
| Survival | 11.3 min | 60 min | 18.9% |

## Session History

| # | Style | Episodes | Avg Reward | Avg Steps |
|---|-------|----------|------------|----------|
| 1 | Standard (Curriculum) | 1-918 | 189.1 | 66 |

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

