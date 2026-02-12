# Slither.io Bot - Training Progress Report

**Generated:** 2026-02-12 23:15:27  
**Total Episodes:** 829  
**Training Sessions:** 1

## Verdict: NOT LEARNING (Confidence: 50%)

**Goal Feasibility:** UNLIKELY (5-25% chance) without tuning

### Warnings
- Rewards flat: change = -6.2 between halves

### Positive Signals
- Loss decreasing (model converging)

## Key Statistics

| Metric | Mean | Std | Min | Max | P50 | P95 |
|--------|------|-----|-----|-----|-----|-----|
| Reward | 181.64 | 261.33 | -38.58 | 2242.28 | 83.00 | 679.00 |
| Steps | 63.90 | 53.99 | 2.00 | 340.00 | 48.00 | 169.00 |
| Food | 27.79 | 10.74 | 0.00 | 77.00 | 25.00 | 47.60 |
| Loss | 13.98 | 40.93 | 0.53 | 392.24 | 6.23 | 24.88 |

## Goal Progress

| Target | Current Best | Goal | Progress |
|--------|-------------|------|----------|
| Points | 77 | 6,000 | 1.3% |
| Survival | 11.3 min | 60 min | 18.9% |

## Session History

| # | Style | Episodes | Avg Reward | Avg Steps |
|---|-------|----------|------------|----------|
| 1 | Standard (Curriculum) | 1-829 | 181.6 | 64 |

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

