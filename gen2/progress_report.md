# Slither.io Bot - Training Progress Report

**Generated:** 2026-02-13 03:27:07  
**Total Episodes:** 3727  
**Training Sessions:** 1

## Verdict: NOT LEARNING (Confidence: 35%)

**Goal Feasibility:** VERY UNLIKELY (<5% chance)

### Critical Issues
- Rewards DECLINING: -121.1 (getting worse)

## Key Statistics

| Metric | Mean | Std | Min | Max | P50 | P95 |
|--------|------|-----|-----|-----|-----|-----|
| Reward | 135.95 | 214.74 | -38.58 | 2242.28 | 57.91 | 568.04 |
| Steps | 53.06 | 47.32 | 2.00 | 340.00 | 38.00 | 152.00 |
| Food | 26.13 | 9.44 | 0.00 | 84.00 | 24.00 | 44.00 |
| Loss | 10.53 | 19.99 | 0.47 | 392.24 | 7.99 | 20.58 |

## Goal Progress

| Target | Current Best | Goal | Progress |
|--------|-------------|------|----------|
| Points | 84 | 6,000 | 1.4% |
| Survival | 11.3 min | 60 min | 18.9% |

## Session History

| # | Style | Episodes | Avg Reward | Avg Steps |
|---|-------|----------|------------|----------|
| 1 | Standard (Curriculum) | 1-3727 | 136.0 | 53 |

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

