# Slither.io Bot - Training Progress Report

**Generated:** 2026-02-10 17:55:40  
**Total Episodes:** 16202  
**Training Sessions:** 7

## Verdict: NOT LEARNING (Confidence: 40%)

**Goal Feasibility:** VERY UNLIKELY (<5% chance)

### Warnings
- Rewards flat: change = 36.8 between halves
- Epsilon moderate (0.499) - still significant random exploration
- Multiple training restarts detected (7 sessions) - fragmented learning

## Key Statistics

| Metric | Mean | Std | Min | Max | P50 | P95 |
|--------|------|-----|-----|-----|-----|-----|
| Reward | 461.97 | 170.74 | -61.43 | 2129.92 | 432.63 | 765.81 |
| Steps | 52.29 | 44.55 | 1.00 | 357.00 | 40.00 | 139.00 |
| Food | 26.26 | 9.54 | 0.00 | 114.00 | 24.00 | 44.00 |
| Loss | 2.10 | 1.62 | 0.00 | 45.93 | 1.80 | 4.48 |

## Goal Progress

| Target | Current Best | Goal | Progress |
|--------|-------------|------|----------|
| Points | 114 | 6,000 | 1.9% |
| Survival | 11.9 min | 60 min | 19.8% |

## Session History

| # | Style | Episodes | Avg Reward | Avg Steps |
|---|-------|----------|------------|----------|
| 1 | Unknown | 1-238 | 150.4 | 66 |
| 2 | Standard (Curriculum) | 239-307 | 535.0 | 68 |
| 3 | Aggressive (Hunter) | 301-1069 | 478.0 | 53 |
| 4 | Aggressive (Hunter) | 1070-1094 | 990.6 | 158 |
| 5 | Aggressive (Hunter) | 1095-8480 | 451.3 | 48 |
| 6 | Aggressive (Hunter) | 8451-15975 | 485.2 | 55 |
| 7 | Explorer (Anti-Float) | 15976-16165 | 184.4 | 51 |

## Recommendations

Significant issues detected. Major changes needed:
  1. Fix learning rate and optimizer state
  2. Simplify reward structure
  3. Ensure episodes can last long enough to learn from

1. Epsilon is 0.499 after 16202 episodes. Consider faster decay (eps_decay=50000) or lower eps_start if resuming.

2. Average episode too short. Consider:
     - Reducing death penalties to avoid discouraging exploration
     - Adding survival bonus to incentivize staying alive

3. Multiple session restarts detected. Each restart disrupts learning continuity. Try to maintain consistent training runs of 10,000+ episodes.

## Charts

![Overview](training_progress_overview.png)

![Learning Detection](training_learning_detection.png)

![Goal Progress](training_goal_progress.png)

