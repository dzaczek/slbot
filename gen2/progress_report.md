# Slither.io Bot - Training Progress Report

**Generated:** 2026-02-13 02:26:42  
**Total Episodes:** 2906  
**Training Sessions:** 1

## Verdict: NOT LEARNING (Confidence: 35%)

**Goal Feasibility:** VERY UNLIKELY (<5% chance)

### Critical Issues
- Rewards DECLINING: -115.4 (getting worse)

## Key Statistics

| Metric | Mean | Std | Min | Max | P50 | P95 |
|--------|------|-----|-----|-----|-----|-----|
| Reward | 156.40 | 233.04 | -38.58 | 2242.28 | 68.73 | 610.66 |
| Steps | 58.01 | 50.28 | 2.00 | 340.00 | 42.00 | 159.00 |
| Food | 26.98 | 9.99 | 0.00 | 84.00 | 24.00 | 46.00 |
| Loss | 11.06 | 22.47 | 0.47 | 392.24 | 8.02 | 21.65 |

## Goal Progress

| Target | Current Best | Goal | Progress |
|--------|-------------|------|----------|
| Points | 84 | 6,000 | 1.4% |
| Survival | 11.3 min | 60 min | 18.9% |

## Session History

| # | Style | Episodes | Avg Reward | Avg Steps |
|---|-------|----------|------------|----------|
| 1 | Standard (Curriculum) | 1-2906 | 156.4 | 58 |

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

