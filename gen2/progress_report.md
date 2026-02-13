# Slither.io Bot - Training Progress Report

**Generated:** 2026-02-13 02:06:34  
**Total Episodes:** 2644  
**Training Sessions:** 1

## Verdict: NOT LEARNING (Confidence: 35%)

**Goal Feasibility:** VERY UNLIKELY (<5% chance)

### Critical Issues
- Rewards DECLINING: -90.2 (getting worse)

## Key Statistics

| Metric | Mean | Std | Min | Max | P50 | P95 |
|--------|------|-----|-----|-----|-----|-----|
| Reward | 166.46 | 240.68 | -38.58 | 2242.28 | 75.35 | 634.62 |
| Steps | 60.46 | 51.47 | 2.00 | 340.00 | 44.00 | 163.00 |
| Food | 27.39 | 10.22 | 0.00 | 84.00 | 25.00 | 47.00 |
| Loss | 11.20 | 23.49 | 0.47 | 392.24 | 7.95 | 21.76 |

## Goal Progress

| Target | Current Best | Goal | Progress |
|--------|-------------|------|----------|
| Points | 84 | 6,000 | 1.4% |
| Survival | 11.3 min | 60 min | 18.9% |

## Session History

| # | Style | Episodes | Avg Reward | Avg Steps |
|---|-------|----------|------------|----------|
| 1 | Standard (Curriculum) | 1-2644 | 166.5 | 60 |

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

