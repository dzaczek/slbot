# Slither.io Bot - Training Progress Report v3

**Generated:** 2026-02-14 21:12:58  
**Total Episodes:** 1386  
**Training Sessions:** 6

## Verdict: NOT LEARNING (Confidence: 50%)

**Goal Feasibility:** UNLIKELY (5-25%) without tuning

### Warnings
- Rewards flat: change = -13.4

### Positive Signals
- Epsilon low (0.141) - exploiting policy

## Curriculum Stage Breakdown

| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |
|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|
| S1 | FOOD_VECTOR | 711 | 87.6 | 58.3 | 23.8 | 0.4862 | 0.0% | 0.0% | 0.0% |
| S2 | WALL_AVOID | 675 | 110.8 | 77.2 | 31.9 | 0.6620 | 0.0% | 0.0% | 0.0% |

## Key Statistics

| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |
|--------|------|-----|-----|-----|--------|-----|-----|-----|
| Reward | 98.87 | 93.26 | -50.00 | 44.59 | 90.28 | 145.37 | 271.99 | 497.52 |
| Steps | 67.50 | 57.63 | 1.00 | 24.00 | 56.00 | 96.00 | 187.75 | 337.00 |
| Food | 27.76 | 15.98 | 0.00 | 21.00 | 27.00 | 36.75 | 56.00 | 97.00 |
| Loss | 22.85 | 115.88 | 0.00 | 1.30 | 2.20 | 3.58 | 19.79 | 996.87 |
| Food/Step | 0.57 | 0.67 | 0.00 | 0.33 | 0.43 | 0.60 | 1.40 | 6.33 |

## Windowed Trend Analysis

| Window | Mean Reward | Std | Slope | R² |
|--------|-----------|-----|-------|----|
| Last 50 | 31.78 | 108.30 | +4.9710 | 0.4387 |
| Last 100 | -9.11 | 86.82 | +1.8480 | 0.3776 |
| Last 200 | -5.84 | 83.26 | -0.0673 | 0.0022 |
| Last 500 | 81.60 | 106.75 | -0.4250 | 0.3302 |
| Last 1000 | 96.62 | 99.50 | -0.0914 | 0.0703 |

## Death Cause Analysis

| Cause | Count | % | Avg Steps | Avg Reward |
|-------|-------|---|-----------|------------|

## Goal Progress

| Target | Best | Goal | Progress |
|--------|------|------|----------|
| Points | 97 | 6,000 | 1.6% |
| Survival | 337 steps | 1,800 steps | 18.7% |

## Recommendations

Fine-tune hyperparameters, increase training duration.

1. Episodes too short. Reduce death penalties or add survival bonus.

## Charts

### Main Dashboard
![Main Dashboard](chart_01_dashboard.png)

### Stage Progression
![Stage Progression](chart_02_stage_progression.png)

### Per-Stage Distributions
![Per-Stage Distributions](chart_03_stage_distributions.png)

### Hyperparameter Tracking
![Hyperparameter Tracking](chart_04_hyperparameters.png)

### Metric Correlations (Scatter)
![Metric Correlations (Scatter)](chart_05_correlations.png)

### Correlation Heatmap & Rankings
![Correlation Heatmap & Rankings](chart_05b_correlation_heatmap.png)

### Performance Percentile Bands
![Performance Percentile Bands](chart_06_performance_bands.png)

### Death Analysis
![Death Analysis](chart_07_death_analysis.png)

### Food Efficiency
![Food Efficiency](chart_08_food_efficiency.png)

### Reward Distributions
![Reward Distributions](chart_09_reward_distributions.png)

### Learning Detection
![Learning Detection](chart_10_learning_detection.png)

### Goal Progress
![Goal Progress](chart_11_goal_gauges.png)

### Hyperparameter Analysis
![Hyperparameter Analysis](chart_12_hyperparameter_analysis.png)

### Q-Value & Gradient Analysis
![Q-Value & Gradient Analysis](chart_13_qvalue_gradients.png)

### Action Distribution Analysis
![Action Distribution Analysis](chart_14_action_distribution.png)

### Active Agents Over Time
![Active Agents Over Time](chart_15_auto_scaling.png)

