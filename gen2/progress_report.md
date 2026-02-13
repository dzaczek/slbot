# Slither.io Bot - Training Progress Report

**Generated:** 2026-02-13 03:17:03  
**Total Episodes:** 3592  
**Training Sessions:** 1

## Verdict: NOT LEARNING (Confidence: 35%)

**Goal Feasibility:** VERY UNLIKELY (<5% chance)

### Critical Issues
- Rewards DECLINING: -123.0 (getting worse)

## Key Statistics

| Metric | Mean | Std | Min | Max | P50 | P95 |
|--------|------|-----|-----|-----|-----|-----|
| Reward | 138.85 | 217.83 | -38.58 | 2242.28 | 59.50 | 578.54 |
| Steps | 53.71 | 47.89 | 2.00 | 340.00 | 39.00 | 154.00 |
| Food | 26.23 | 9.55 | 0.00 | 84.00 | 24.00 | 45.00 |
| Loss | 10.57 | 20.34 | 0.47 | 392.24 | 7.98 | 20.74 |

## Goal Progress

| Target | Current Best | Goal | Progress |
|--------|-------------|------|----------|
| Points | 84 | 6,000 | 1.4% |
| Survival | 11.3 min | 60 min | 18.9% |

## Session History

| # | Style | Episodes | Avg Reward | Avg Steps |
|---|-------|----------|------------|----------|
| 1 | Standard (Curriculum) | 1-3592 | 138.9 | 54 |

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

