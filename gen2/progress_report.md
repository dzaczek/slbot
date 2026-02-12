# Slither.io Bot - Training Progress Report

**Generated:** 2026-02-12 08:38:33  
**Total Episodes:** 112  
**Training Sessions:** 1

## Verdict: NOT LEARNING (Confidence: 40%)

**Goal Feasibility:** VERY UNLIKELY (<5% chance)

### Critical Issues
- Negative reward trend (slope=-2.0447, R²=0.105)

### Warnings
- Epsilon very high (0.876) - still mostly random

### Positive Signals
- Loss decreasing (model converging)

## Key Statistics

| Metric | Mean | Std | Min | Max | P50 | P95 |
|--------|------|-----|-----|-----|-----|-----|
| Reward | 177.08 | 203.87 | -23.99 | 989.70 | 124.58 | 558.75 |
| Steps | 64.36 | 49.27 | 3.00 | 212.00 | 61.50 | 150.05 |
| Food | 29.51 | 11.60 | 11.00 | 66.00 | 27.00 | 48.80 |
| Loss | 0.30 | 0.41 | 0.01 | 2.90 | 0.15 | 1.13 |

## Goal Progress

| Target | Current Best | Goal | Progress |
|--------|-------------|------|----------|
| Points | 66 | 6,000 | 1.1% |
| Survival | 7.1 min | 60 min | 11.8% |

## Session History

| # | Style | Episodes | Avg Reward | Avg Steps |
|---|-------|----------|------------|----------|
| 1 | Standard (Curriculum) | 1-112 | 177.1 | 64 |

## Recommendations

Significant issues detected. Major changes needed:
  1. Fix learning rate and optimizer state
  2. Simplify reward structure
  3. Ensure episodes can last long enough to learn from

1. Epsilon is 0.876 after 112 episodes. Consider faster decay (eps_decay=50000) or lower eps_start if resuming.

2. Average episode too short. Consider:
     - Reducing death penalties to avoid discouraging exploration
     - Adding survival bonus to incentivize staying alive

## Charts

![Overview](training_progress_overview.png)

![Learning Detection](training_learning_detection.png)

![Goal Progress](training_goal_progress.png)

