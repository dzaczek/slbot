# Slither.io Bot - Training Progress Report

**Generated:** 2026-02-11 06:15:22  
**Total Episodes:** 22430  
**Training Sessions:** 8

## Verdict: NOT LEARNING (Confidence: 20%)

**Goal Feasibility:** IMPOSSIBLE with current setup

### Critical Issues
- Rewards DECLINING: -208.2 (getting worse)
- Very short episodes: avg=49 steps (dying too fast)

### Warnings
- Multiple training restarts detected (8 sessions) - fragmented learning

## Key Statistics

| Metric | Mean | Std | Min | Max | P50 | P95 |
|--------|------|-----|-----|-----|-----|-----|
| Reward | 358.57 | 217.96 | -4215.93 | 2129.92 | 386.56 | 707.08 |
| Steps | 49.24 | 69.59 | 1.00 | 800.00 | 33.00 | 133.00 |
| Food | 24.79 | 9.50 | 0.00 | 114.00 | 23.00 | 42.00 |
| Loss | 1.53 | 1.59 | 0.00 | 45.85 | 1.22 | 4.07 |

## Goal Progress

| Target | Current Best | Goal | Progress |
|--------|-------------|------|----------|
| Points | 114 | 6,000 | 1.9% |
| Survival | 26.7 min | 60 min | 44.4% |

## Session History

| # | Style | Episodes | Avg Reward | Avg Steps |
|---|-------|----------|------------|----------|
| 1 | Unknown | 975-1069 | 461.4 | 45 |
| 2 | Unknown | 1070-1094 | 990.6 | 158 |
| 3 | Aggressive (Hunter) | 1095-8480 | 451.3 | 48 |
| 4 | Aggressive (Hunter) | 8451-15975 | 485.2 | 55 |
| 5 | Aggressive (Hunter) | 15976-16165 | 184.4 | 51 |
| 6 | Standard (Curriculum) | 1-87 | 129.2 | 51 |
| 7 | Standard (Curriculum) | 1-270 | 177.5 | 79 |
| 8 | Standard (Curriculum) | 1-6852 | 130.6 | 42 |

## Recommendations

Training is fundamentally broken. Fix critical issues first:
  1. Restore learning rate > 0 (check LR scheduler)
  2. Consider resetting epsilon to allow fresh exploration
  3. Review reward shaping - excessive penalties prevent learning

1. Average episode too short. Consider:
     - Reducing death penalties to avoid discouraging exploration
     - Adding survival bonus to incentivize staying alive

2. Multiple session restarts detected. Each restart disrupts learning continuity. Try to maintain consistent training runs of 10,000+ episodes.

## Charts

![Overview](training_progress_overview.png)

![Learning Detection](training_learning_detection.png)

![Goal Progress](training_goal_progress.png)

