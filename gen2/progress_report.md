# Slither.io Bot - Training Progress Report

**Generated:** 2026-02-13 16:02:50  
**Total Episodes:** 6755  
**Training Sessions:** 1

## Verdict: NOT LEARNING (Confidence: 35%)

**Goal Feasibility:** VERY UNLIKELY (<5% chance)

### Critical Issues
- Very short episodes: avg=42 steps (dying too fast)

### Warnings
- Rewards flat: change = 1.3 between halves

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 100 | 53.7 | 29.7 | 20.7 | 0.0% | 95.0% | 5.0% |
| S2 | WALL_AVOID | 6655 | 50.7 | 41.9 | 24.4 | 0.0% | 99.7% | 0.3% |

## Key Statistics

| Metric | Mean | Std | Min | Max | P50 | P95 |
|--------|------|-----|-----|-----|-----|-----|
| Reward | 50.77 | 32.40 | -16.90 | 269.43 | 42.82 | 114.24 |
| Steps | 41.76 | 36.17 | 1.00 | 200.00 | 32.00 | 114.00 |
| Food | 24.31 | 7.59 | 0.00 | 64.00 | 23.00 | 38.00 |
| Loss | 2.20 | 22.53 | 0.00 | 504.77 | 0.12 | 1.89 |

## Goal Progress

| Target | Current Best | Goal | Progress |
|--------|-------------|------|----------|
| Points | 64 | 6,000 | 1.1% |
| Survival | 6.7 min | 60 min | 11.1% |

## Session History

| # | Style | Episodes | Avg Reward | Avg Steps |
|---|-------|----------|------------|----------|
| 1 | Standard (Curriculum) | 1-6755 | 50.8 | 42 |

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

