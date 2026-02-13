# Slither.io Bot - Training Progress Report

**Generated:** 2026-02-13 08:16:42  
**Total Episodes:** 519  
**Training Sessions:** 1

## Verdict: NOT LEARNING (Confidence: 40%)

**Goal Feasibility:** VERY UNLIKELY (<5% chance)

### Critical Issues
- Very short episodes: avg=35 steps (dying too fast)

### Warnings
- Rewards flat: change = -3.2 between halves

### Positive Signals
- Loss decreasing (model converging)

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 100 | 53.7 | 29.7 | 20.7 | 0.0% | 95.0% | 5.0% |
| S2 | WALL_AVOID | 419 | 46.2 | 35.7 | 23.7 | 0.0% | 100.0% | 0.0% |

## Key Statistics

| Metric | Mean | Std | Min | Max | P50 | P95 |
|--------|------|-----|-----|-----|-----|-----|
| Reward | 47.65 | 30.23 | -16.90 | 187.71 | 41.83 | 104.03 |
| Steps | 34.54 | 29.56 | 1.00 | 162.00 | 27.00 | 99.10 |
| Food | 23.15 | 7.76 | 0.00 | 50.00 | 23.00 | 37.00 |
| Loss | 26.74 | 77.17 | 0.16 | 504.77 | 3.57 | 161.06 |

## Goal Progress

| Target | Current Best | Goal | Progress |
|--------|-------------|------|----------|
| Points | 50 | 6,000 | 0.8% |
| Survival | 5.4 min | 60 min | 9.0% |

## Session History

| # | Style | Episodes | Avg Reward | Avg Steps |
|---|-------|----------|------------|----------|
| 1 | Standard (Curriculum) | 1-519 | 47.6 | 35 |

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

![Stage Progression](training_stage_progression.png)

![Learning Detection](training_learning_detection.png)

![Goal Progress](training_goal_progress.png)

