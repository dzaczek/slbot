# Slither.io Bot - Training Progress Report

**Generated:** 2026-02-12 17:30:12  
**Total Episodes:** 8  
**Training Sessions:** 1

## Verdict: NOT LEARNING (Confidence: 10%)

**Goal Feasibility:** 

### Critical Issues
- Insufficient data (< 20 episodes)

## Key Statistics

| Metric | Mean | Std | Min | Max | P50 | P95 |
|--------|------|-----|-----|-----|-----|-----|
| Reward | 253.47 | 192.85 | 32.55 | 594.00 | 247.23 | 547.18 |
| Steps | 85.75 | 44.06 | 30.00 | 155.00 | 90.00 | 147.30 |
| Food | 32.88 | 13.14 | 16.00 | 50.00 | 35.50 | 49.65 |
| Loss | 122.41 | 88.75 | 0.00 | 290.42 | 129.83 | 256.41 |

## Goal Progress

| Target | Current Best | Goal | Progress |
|--------|-------------|------|----------|
| Points | 50 | 6,000 | 0.8% |
| Survival | 5.2 min | 60 min | 8.6% |

## Session History

| # | Style | Episodes | Avg Reward | Avg Steps |
|---|-------|----------|------------|----------|
| 1 | Unknown | 1-8 | 253.5 | 86 |

## Recommendations



1. Epsilon is 0.992 after 8 episodes. Consider faster decay (eps_decay=50000) or lower eps_start if resuming.

2. Average episode too short. Consider:
     - Reducing death penalties to avoid discouraging exploration
     - Adding survival bonus to incentivize staying alive

## Charts

![Overview](training_progress_overview.png)

![Learning Detection](training_learning_detection.png)

![Goal Progress](training_goal_progress.png)

