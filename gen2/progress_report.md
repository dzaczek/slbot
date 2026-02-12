# Slither.io Bot - Training Progress Report

**Generated:** 2026-02-13 00:56:06  
**Total Episodes:** 1773  
**Training Sessions:** 1

## Verdict: NOT LEARNING (Confidence: 45%)

**Goal Feasibility:** UNLIKELY (5-25% chance) without tuning

### Warnings
- Rewards flat: change = 19.7 between halves

## Key Statistics

| Metric | Mean | Std | Min | Max | P50 | P95 |
|--------|------|-----|-----|-----|-----|-----|
| Reward | 200.34 | 267.76 | -38.58 | 2242.28 | 102.45 | 720.53 |
| Steps | 68.36 | 55.35 | 2.00 | 340.00 | 54.00 | 176.00 |
| Food | 28.75 | 10.97 | 0.00 | 84.00 | 26.00 | 49.00 |
| Loss | 11.26 | 28.33 | 0.47 | 392.24 | 7.15 | 19.82 |

## Goal Progress

| Target | Current Best | Goal | Progress |
|--------|-------------|------|----------|
| Points | 84 | 6,000 | 1.4% |
| Survival | 11.3 min | 60 min | 18.9% |

## Session History

| # | Style | Episodes | Avg Reward | Avg Steps |
|---|-------|----------|------------|----------|
| 1 | Standard (Curriculum) | 1-1773 | 200.3 | 68 |

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

