# Slither.io Bot - Training Progress Report

**Generated:** 2026-02-13 01:06:10  
**Total Episodes:** 1888  
**Training Sessions:** 1

## Verdict: NOT LEARNING (Confidence: 40%)

**Goal Feasibility:** VERY UNLIKELY (<5% chance)

### Warnings
- Rewards flat: change = 7.1 between halves
- Loss very high (10.13) - training unstable

## Key Statistics

| Metric | Mean | Std | Min | Max | P50 | P95 |
|--------|------|-----|-----|-----|-----|-----|
| Reward | 195.59 | 263.41 | -38.58 | 2242.28 | 98.64 | 716.86 |
| Steps | 67.31 | 54.77 | 2.00 | 340.00 | 53.00 | 174.00 |
| Food | 28.54 | 10.83 | 0.00 | 84.00 | 26.00 | 49.00 |
| Loss | 11.20 | 27.49 | 0.47 | 392.24 | 7.25 | 20.23 |

## Goal Progress

| Target | Current Best | Goal | Progress |
|--------|-------------|------|----------|
| Points | 84 | 6,000 | 1.4% |
| Survival | 11.3 min | 60 min | 18.9% |

## Session History

| # | Style | Episodes | Avg Reward | Avg Steps |
|---|-------|----------|------------|----------|
| 1 | Standard (Curriculum) | 1-1888 | 195.6 | 67 |

## Recommendations

Significant issues detected. Major changes needed:
  1. Fix learning rate and optimizer state
  2. Simplify reward structure
  3. Ensure episodes can last long enough to learn from

1. Average episode too short. Consider:
     - Reducing death penalties to avoid discouraging exploration
     - Adding survival bonus to incentivize staying alive

## Charts

![Overview](training_progress_overview.png)

![Learning Detection](training_learning_detection.png)

![Goal Progress](training_goal_progress.png)

