# Slither.io Bot - Training Progress Report

**Generated:** 2026-02-13 07:58:38  
**Total Episodes:** 256  
**Training Sessions:** 1

## Verdict: NOT LEARNING (Confidence: 45%)

**Goal Feasibility:** UNLIKELY (5-25% chance) without tuning

### Critical Issues
- Very short episodes: avg=34 steps (dying too fast)

### Warnings
- Rewards flat: change = -1.2 between halves
- Epsilon very high (0.819) - still mostly random
- Reward in plateau (change < 1% over last 200 episodes)

### Positive Signals
- Episodes getting longer (slope=0.078/ep)
- Food collection improving (slope=0.0298/ep)
- Loss decreasing (model converging)

## Key Statistics

| Metric | Mean | Std | Min | Max | P50 | P95 |
|--------|------|-----|-----|-----|-----|-----|
| Reward | 49.48 | 32.45 | -16.90 | 187.71 | 43.11 | 115.60 |
| Steps | 33.92 | 29.79 | 2.00 | 162.00 | 26.00 | 99.25 |
| Food | 22.75 | 8.22 | 0.00 | 50.00 | 22.00 | 36.25 |
| Loss | 52.40 | 103.79 | 0.47 | 504.77 | 12.68 | 344.51 |

## Goal Progress

| Target | Current Best | Goal | Progress |
|--------|-------------|------|----------|
| Points | 50 | 6,000 | 0.8% |
| Survival | 5.4 min | 60 min | 9.0% |

## Session History

| # | Style | Episodes | Avg Reward | Avg Steps |
|---|-------|----------|------------|----------|
| 1 | Standard (Curriculum) | 1-256 | 49.5 | 34 |

## Recommendations

Some learning signals present but not strong enough.
  1. Fine-tune hyperparameters
  2. Increase training duration significantly
  3. Consider curriculum adjustments

1. Epsilon is 0.819 after 256 episodes. Consider faster decay (eps_decay=50000) or lower eps_start if resuming.

2. Average episode too short. Consider:
     - Reducing death penalties to avoid discouraging exploration
     - Adding survival bonus to incentivize staying alive

## Charts

![Overview](training_progress_overview.png)

![Learning Detection](training_learning_detection.png)

![Goal Progress](training_goal_progress.png)

