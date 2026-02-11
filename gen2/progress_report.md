# Slither.io Bot Gen2 - Training Progress Report

**Updated:** 2026-02-11
**Architecture:** Dueling DQN + Double DQN + Prioritized Experience Replay
**Observation:** 84x84 3-channel egocentric matrix, 4-frame stack (12 channels)
**Actions:** 6 (straight, left/right small, left/right big, boost)

---

## Previous Training Analysis (5000+ episodes)

**Verdict: NOT LEARNING** - Agent was stuck with median 40-50 steps per episode.

| Problem | Root Cause |
|---------|-----------|
| 55% deaths = SnakeCollision, 40% = Wall | Agent never learned obstacle avoidance |
| Curriculum backwards | Stage 1 = EAT (passed in 101 ep), Stage 2 = SURVIVE (stuck 4800+ ep) |
| Food reward dominated | 80% of reward was food; death still "profitable" (reward ~175 in 50 steps) |
| tanh crushed penalties | -30 and -100 both mapped to ~-1.0, no differentiation |
| LR frozen at 1e-6 | Scheduler reduced it, recovery bug (`<` instead of `<=`) |
| North-up observation | Contradicted relative actions - same situation yielded different training data |
| Epsilon decayed too slow | `steps_done += 1` per batch instead of per-agent |

---

## Fixes Applied (2026-02-10 — 2026-02-11)

### Observation & Perception
- **Egocentric rotation**: Snake heading always = "up" in matrix. Eliminates north-up contradiction
- **Enemy body sampling**: Doubled density (ptsLen/40), trim reduced 25%->15%
- **COLLISION_BUFFER / WALL_BUFFER**: 40 -> 120 (frame_skip=4 compensation)

### Reward System Redesign: SURVIVE First
- **Curriculum reversed**: Stage 1=SURVIVE, Stage 2=EAT, Stage 3=GROW
- **Escalating survival reward**: `survival * (1 + escalation * steps_in_episode)`
  - Stage 1: step 1 -> +0.5, step 50 -> +0.75, step 100 -> +1.0, step 200 -> +1.5
- **Food minimized in Stage 1**: 1.0 (was 8.0) with no shaping
- **Equal death penalties in Stage 1**: wall=-200, snake=-200 (both equally dangerous)
- **Proximity penalties active from start**: wall=0.3, enemy=0.3

### Reward Normalization
- **tanh -> linear clamp [-5, 5]**: `clamp(reward / scale, -5, 5)`
  - Now: -30 -> -3.0, -100 -> -5.0 (was both ~-1.0 with tanh)
  - Survival 0.5 -> 0.05, food 8.0 -> 0.8

### Training Stability
- **LR recovery fix**: `<` -> `<=` catches exactly min_lr
- **Stagnation handler**: Now also resets LR to initial value (was only boosting epsilon)
- **steps_done += num_agents**: Epsilon decays per game-step, not per batch
- **eps_decay**: 100000 -> 50000 (faster initial exploration reduction)
- **prev_length fallback**: 10 instead of 0 (prevents phantom food reward on reset)

### Infrastructure
- **UID system**: YYYYMMDD-8hex format, parent_uid lineage tracking
- **CSV columns**: UID, ParentUID added for run traceability

---

## Curriculum Stages (New)

| Stage | Name | Goal | Food Reward | Survival | Death Penalty | Promote Condition |
|-------|------|------|------------|----------|---------------|-------------------|
| 1 | SURVIVE | Don't die | 1.0 | 0.5 + escalation | -200 (both) | avg_steps >= 80 over 100 ep |
| 2 | EAT | Find food | 8.0 | 0.2 + escalation | wall=-200, snake=-100 | food_per_step >= 0.15 over 100 ep |
| 3 | GROW | Full game | 10.0 | 0.1 + escalation | wall=-100, snake=-30 | Terminal stage |

---

## Expected Behavior After Fixes

1. Agent starts in Stage 1 (SURVIVE), not EAT
2. Survival reward grows over time - longer episodes become increasingly rewarding
3. Death penalties are clearly differentiated (-3.0 vs -5.0 normalized)
4. LR starts at 1e-4 (not frozen at 1e-6)
5. Egocentric view: same spatial situation always produces same observation
6. Promotion from Stage 1 requires avg 80+ steps (was auto-promoting at 101 episodes)

## Charts

![Overview](training_progress_overview.png)

![Learning Detection](training_learning_detection.png)

![Goal Progress](training_goal_progress.png)

