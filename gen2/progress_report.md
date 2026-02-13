# Slither.io Bot - Training Progress Report

**Generated:** 2026-02-13 02:56:54  
**Total Episodes:** 3325  
**Training Sessions:** 1

## Verdict: NOT LEARNING (Confidence: 35%)

**Goal Feasibility:** VERY UNLIKELY (<5% chance)

### Critical Issues
- Rewards DECLINING: -123.3 (getting worse)

## Key Statistics

| Metric | Mean | Std | Min | Max | P50 | P95 |
|--------|------|-----|-----|-----|-----|-----|
| Reward | 144.41 | 223.84 | -38.58 | 2242.28 | 60.51 | 593.75 |
| Steps | 54.97 | 48.92 | 2.00 | 340.00 | 39.00 | 156.00 |
| Food | 26.46 | 9.73 | 0.00 | 84.00 | 24.00 | 45.00 |
| Loss | 10.75 | 21.10 | 0.47 | 392.24 | 8.00 | 21.08 |

## Goal Progress

| Target | Current Best | Goal | Progress |
|--------|-------------|------|----------|
| Points | 84 | 6,000 | 1.4% |
| Survival | 11.3 min | 60 min | 18.9% |

## Session History

| # | Style | Episodes | Avg Reward | Avg Steps |
|---|-------|----------|------------|----------|
| 1 | Standard (Curriculum) | 1-3325 | 144.4 | 55 |

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

