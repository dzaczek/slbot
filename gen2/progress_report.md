# Slither.io Bot - Training Progress Report

**Generated:** 2026-02-13 00:05:46  
**Total Episodes:** 1270  
**Training Sessions:** 1

## Verdict: LEARNING (Confidence: 60%)

**Goal Feasibility:** UNLIKELY (5-25% chance) without tuning

### Warnings
- Loss very high (10.57) - training unstable

### Positive Signals
- Rewards improving: +51.7 (2nd half vs 1st half)

## Key Statistics

| Metric | Mean | Std | Min | Max | P50 | P95 |
|--------|------|-----|-----|-----|-----|-----|
| Reward | 207.73 | 275.70 | -38.58 | 2242.28 | 105.44 | 754.01 |
| Steps | 69.94 | 56.42 | 2.00 | 340.00 | 55.00 | 180.00 |
| Food | 29.00 | 11.18 | 0.00 | 84.00 | 26.00 | 50.00 |
| Loss | 12.14 | 33.28 | 0.53 | 392.24 | 6.79 | 20.22 |

## Goal Progress

| Target | Current Best | Goal | Progress |
|--------|-------------|------|----------|
| Points | 84 | 6,000 | 1.4% |
| Survival | 11.3 min | 60 min | 18.9% |

## Session History

| # | Style | Episodes | Avg Reward | Avg Steps |
|---|-------|----------|------------|----------|
| 1 | Standard (Curriculum) | 1-1270 | 207.7 | 70 |

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

