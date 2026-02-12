# Slither.io Bot - Training Progress Report

**Generated:** 2026-02-13 00:46:02  
**Total Episodes:** 1661  
**Training Sessions:** 1

## Verdict: NOT LEARNING (Confidence: 40%)

**Goal Feasibility:** VERY UNLIKELY (<5% chance)

### Warnings
- Rewards flat: change = 48.3 between halves
- Loss very high (10.46) - training unstable

## Key Statistics

| Metric | Mean | Std | Min | Max | P50 | P95 |
|--------|------|-----|-----|-----|-----|-----|
| Reward | 206.16 | 272.45 | -38.58 | 2242.28 | 105.74 | 731.56 |
| Steps | 69.66 | 56.02 | 2.00 | 340.00 | 55.00 | 177.00 |
| Food | 28.96 | 11.09 | 0.00 | 84.00 | 26.00 | 50.00 |
| Loss | 11.41 | 29.24 | 0.47 | 392.24 | 7.11 | 20.24 |

## Goal Progress

| Target | Current Best | Goal | Progress |
|--------|-------------|------|----------|
| Points | 84 | 6,000 | 1.4% |
| Survival | 11.3 min | 60 min | 18.9% |

## Session History

| # | Style | Episodes | Avg Reward | Avg Steps |
|---|-------|----------|------------|----------|
| 1 | Standard (Curriculum) | 1-1661 | 206.2 | 70 |

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

