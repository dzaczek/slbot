# Slither.io Bot - Training Progress Report

**Generated:** 2026-02-13 02:16:38  
**Total Episodes:** 2766  
**Training Sessions:** 1

## Verdict: NOT LEARNING (Confidence: 35%)

**Goal Feasibility:** VERY UNLIKELY (<5% chance)

### Critical Issues
- Rewards DECLINING: -102.2 (getting worse)

## Key Statistics

| Metric | Mean | Std | Min | Max | P50 | P95 |
|--------|------|-----|-----|-----|-----|-----|
| Reward | 162.37 | 236.77 | -38.58 | 2242.28 | 73.09 | 620.04 |
| Steps | 59.57 | 50.78 | 2.00 | 340.00 | 43.50 | 161.00 |
| Food | 27.25 | 10.07 | 0.00 | 84.00 | 25.00 | 47.00 |
| Loss | 11.15 | 23.00 | 0.47 | 392.24 | 8.01 | 21.67 |

## Goal Progress

| Target | Current Best | Goal | Progress |
|--------|-------------|------|----------|
| Points | 84 | 6,000 | 1.4% |
| Survival | 11.3 min | 60 min | 18.9% |

## Session History

| # | Style | Episodes | Avg Reward | Avg Steps |
|---|-------|----------|------------|----------|
| 1 | Standard (Curriculum) | 1-2766 | 162.4 | 60 |

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

