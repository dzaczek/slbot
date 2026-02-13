# Slither.io Bot - Training Progress Report

**Generated:** 2026-02-13 08:06:38  
**Total Episodes:** 376  
**Training Sessions:** 1

## Verdict: NOT LEARNING (Confidence: 45%)

**Goal Feasibility:** UNLIKELY (5-25% chance) without tuning

### Critical Issues
- Very short episodes: avg=33 steps (dying too fast)

### Warnings
- Rewards flat: change = -1.9 between halves

### Positive Signals
- Food collection improving (slope=0.0110/ep)
- Loss decreasing (model converging)

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 100 | 53.7 | 29.7 | 20.7 | 0.0% | 95.0% | 5.0% |
| S2 | WALL_AVOID | 276 | 45.4 | 34.6 | 23.7 | 0.0% | 100.0% | 0.0% |

## Key Statistics

| Metric | Mean | Std | Min | Max | P50 | P95 |
|--------|------|-----|-----|-----|-----|-----|
| Reward | 47.61 | 30.88 | -16.90 | 187.71 | 41.48 | 106.28 |
| Steps | 33.30 | 29.53 | 2.00 | 162.00 | 25.00 | 98.25 |
| Food | 22.90 | 7.69 | 0.00 | 50.00 | 22.00 | 36.00 |
| Loss | 36.47 | 88.75 | 0.25 | 504.77 | 6.40 | 237.83 |

## Goal Progress

| Target | Current Best | Goal | Progress |
|--------|-------------|------|----------|
| Points | 50 | 6,000 | 0.8% |
| Survival | 5.4 min | 60 min | 9.0% |

## Session History

| # | Style | Episodes | Avg Reward | Avg Steps |
|---|-------|----------|------------|----------|
| 1 | Standard (Curriculum) | 1-376 | 47.6 | 33 |

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

![Stage Progression](training_stage_progression.png)

![Learning Detection](training_learning_detection.png)

![Goal Progress](training_goal_progress.png)

