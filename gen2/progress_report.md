# Slither.io Bot - Training Progress Report

**Generated:** 2026-02-12 22:45:15  
**Total Episodes:** 517  
**Training Sessions:** 1

## Verdict: NOT LEARNING (Confidence: 50%)

**Goal Feasibility:** UNLIKELY (5-25% chance) without tuning

### Warnings
- Rewards flat: change = 6.7 between halves

### Positive Signals
- Loss decreasing (model converging)

## Key Statistics

| Metric | Mean | Std | Min | Max | P50 | P95 |
|--------|------|-----|-----|-----|-----|-----|
| Reward | 197.84 | 280.11 | -38.58 | 2242.28 | 93.48 | 766.90 |
| Steps | 67.25 | 56.50 | 2.00 | 340.00 | 53.00 | 183.20 |
| Food | 28.40 | 11.46 | 0.00 | 77.00 | 26.00 | 49.20 |
| Loss | 17.22 | 51.39 | 0.86 | 392.24 | 5.68 | 59.18 |

## Goal Progress

| Target | Current Best | Goal | Progress |
|--------|-------------|------|----------|
| Points | 77 | 6,000 | 1.3% |
| Survival | 11.3 min | 60 min | 18.9% |

## Session History

| # | Style | Episodes | Avg Reward | Avg Steps |
|---|-------|----------|------------|----------|
| 1 | Standard (Curriculum) | 1-517 | 197.8 | 67 |

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

