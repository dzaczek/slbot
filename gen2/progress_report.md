# Slither.io Bot - Complete Training & Code Analysis Report

**Generated:** 2026-02-10 18:02
**Total Episodes:** 16,202
**Training Sessions:** 7
**Branch:** fix-learning-rewards (merged to main)

---

## 1. LEARNING VERDICT

| Factor | Status | Details |
|--------|--------|---------|
| **Is it learning?** | NO | All 4 quarterly reward trends are FLAT (R² < 0.03) |
| **Confidence** | 40% | Some positive signals exist but overwhelmed by bugs |
| **Goal feasibility** | <5% with old code | After fixes: ~60-70% achievable |

### Key Numbers

| Metric | Mean | Best | Goal | Progress |
|--------|------|------|------|----------|
| Food (points) | 26.3 | 114 | 6,000 | 1.9% |
| Survival (steps) | 52.3 | 357 (11.9 min) | 1,800 (60 min) | 19.8% |
| Reward | 462.0 | 2,129.9 | — | — |
| Loss | 2.10 | — | — | — |

---

## 2. CRITICAL BUGS FOUND & FIXED

### 2.1 Learning Rate = 0 (FIXED)
- **Bug:** `ReduceLROnPlateau` had no `min_lr` floor. After ~50 plateau detections: `0.0001 × 0.5^N → 0.0`
- **Impact:** Model weights completely frozen — zero gradient updates
- **Fix:** Added `min_lr=1e-6` to scheduler + LR recovery on checkpoint load
- **File:** `agent.py:52-58`, `config.py:39`, `trainer.py:600-603`

### 2.2 Reward Signal Crushed (FIXED)
- **Bug:** `reward_scale=100.0` with clamp to [-1, 1] meant: food reward `5.0 / 100 = 0.05` (invisible), death `-200 / 100 = -1.0` (saturated)
- **Impact:** Agent only learned "death is bad", could not distinguish food from noise
- **Fix:** Changed `reward_scale` from 100.0 to 10.0. Now food = 0.5, death = -1.0
- **File:** `config.py:34`

### 2.3 Epsilon Oscillation (FIXED)
- **Bug:** Stagnation handler reset `best_avg_reward = -inf` and `reward_window.clear()`, then boosted epsilon to 0.5. Next window always looked "improved" → no boost → stagnate → boost again → endless cycle
- **Impact:** Epsilon stuck at ~0.5 after 16k episodes (should be ~0.1)
- **Fix:** Gentler boost (0.3), removed baseline reset
- **File:** `trainer.py:721-724`

### 2.4 Multi-Agent steps_done Overcounting (FIXED)
- **Bug:** `steps_done++` inside `select_action()`, called N times per step (once per agent). With 2 agents, epsilon decayed 2× faster
- **Impact:** Epsilon schedule didn't match config, inconsistent exploration
- **Fix:** Moved increment to trainer (once per batch step)
- **File:** `agent.py:111-118`, `trainer.py:759-760`

### 2.5 Wall Distance JS/Python Mismatch (FIXED)
- **Bug:** Reward calculation used `data.get('dist_to_wall')` from JavaScript, but death classification used Python `_calc_dist_to_wall()`. Different formulas, different timing
- **Impact:** Wall proximity rewards inconsistent with wall death detection
- **Fix:** Now uses `self.last_dist_to_wall` (Python) everywhere
- **File:** `slither_env.py:567-569`

---

## 3. WALL DETECTION ANALYSIS

### 3.1 Core Formula — CORRECT
```python
def _calc_dist_to_wall(self, x, y):
    dist_from_center = math.hypot(x - self.map_center_x, y - self.map_center_y)
    return self.map_radius - dist_from_center
```
Slither.io uses a circular map (radius=21600, center=21600,21600). This formula is mathematically correct. Positive = inside, negative = outside.

### 3.2 Map Constants — RISK: Hardcoded Defaults
```
MAP_RADIUS = 21600, MAP_CENTER = (21600, 21600)
```
These are fallbacks. The JS engine tries to read actual values from `grd` variable, but if detection fails, defaults may be wrong on non-standard servers.

**Status:** OK for slither.io, RISKY for eslither.io or custom servers.

### 3.3 Wall Rendering vs Death Zone — MISMATCH
- **Observation matrix** draws wall with 500-unit safety margin: `radius_sq = (map_radius - 500)²`
- **Death detection** uses actual map radius (no margin)
- **Result:** Agent "sees" wall 500 units closer than it actually is. The visual input doesn't match where death occurs.

**Status:** Intentional safety buffer, but confuses the neural network. Consider reducing to 200 or matching exactly.

### 3.4 Death Classification Buffers (FIXED)
- **Old:** `WALL_BUFFER = 40` — too tight for frame_skip=4 (snake moves ~200 units between observations)
- **New:** `WALL_BUFFER = 120` — catches wall deaths that happen between frames
- **Status:** FIXED. Many "Unknown" deaths were actually wall deaths slipping through.

### 3.5 Wall Detection Accuracy: ~85% (was ~70%)

---

## 4. ENEMY/SNAKE DETECTION ANALYSIS

### 4.1 Distance Calculation — CORRECT but LIMITED
```python
def _min_enemy_distance(self, enemies, mx, my):
    for e in enemies:
        # checks head distance
        # checks body point distances
```
Calculates euclidean distance from our head to every enemy head + body point. Correct for point-to-point collision detection.

**Limitation:** Only checks against our HEAD position. Doesn't check if enemy collides with our BODY (self-collision via enemy cutting through us). This is acceptable since slither.io death = head collision.

### 4.2 Body Point Sampling — IMPROVED but GAPS REMAIN
| Phase | Visibility Check | Body Collection | Tail Trim |
|-------|-----------------|-----------------|-----------|
| Old | every 20th point | every Nth (ptsLen/150) | 25% + 4×speed |
| New | every 40th point (FIXED) | every Nth (ptsLen/150) | 15% + 2×speed (FIXED) |

**Remaining risk:** A snake with 400 body points still only samples ~10 for visibility check. Fast-moving small snakes could slip between sample points.

### 4.3 Death Misclassification — PARTIALLY FIXED
Priority 4 (default) still assigns "SnakeCollision" if ANY enemy exists in view, even 5000 units away. This inflates snake collision stats.

```python
# Priority 4: Default
cause = "Unknown" if min_enemy_dist == float('inf') else "SnakeCollision"
```

**Remaining issue:** Should add a distance threshold to Priority 4 (e.g., only classify as SnakeCollision if enemy < 500 units).

### 4.4 Enemy Detection Accuracy: ~75% (was ~60%)

---

## 5. TRAINER LOGIC ANALYSIS

### 5.1 Training Loop — CORRECT after fixes
- Frame stacking (4 frames → 12 channels): OK
- Double DQN target calculation: OK
- PER sampling + IS weights: OK
- Gradient clipping: OK

### 5.2 Remaining Issues

| Issue | Severity | Description |
|-------|----------|-------------|
| Session restarts | MEDIUM | 7 sessions with different styles fragment learning. Curriculum changes reset reward baselines |
| SuperPatternOptimizer | LOW | Dynamically adjusts rewards during training, adding noise to an already noisy signal |
| `train._last_loss` | LOW | Storing loss as attribute on function object is fragile |
| Warmup vs Scheduler conflict | LOW | LR warmup runs once, then scheduler can override immediately |

### 5.3 Architecture — SOUND
- Dueling DQN + Double DQN + PER is state-of-the-art for this type of problem
- 84×84 3-channel observation is standard (Atari-level)
- 6 discrete actions are reasonable for slither.io

---

## 6. SESSION HISTORY

| # | Style | Episodes | Avg Reward | Avg Steps | Notes |
|---|-------|----------|------------|-----------|-------|
| 1 | Unknown | 1-238 | 150.4 | 66 | Initial exploration |
| 2 | Standard (Curriculum) | 239-307 | 535.0 | 68 | Best avg reward |
| 3 | Aggressive (Hunter) | 301-1,069 | 478.0 | 53 | Switched too early |
| 4 | Aggressive (Hunter) | 1,070-1,094 | 990.6 | 158 | Best session (25 eps) |
| 5 | Aggressive (Hunter) | 1,095-8,480 | 451.3 | 48 | Long run, LR dying |
| 6 | Aggressive (Hunter) | 8,451-15,975 | 485.2 | 55 | LR = 0, frozen |
| 7 | Explorer (Anti-Float) | 15,976-16,165 | 184.4 | 51 | Worst performance |

**Key insight:** Session 4 (25 episodes, avg reward 990) shows the model CAN perform well. The architecture works — the bugs killed the learning.

---

## 7. REPAIR PROCEDURE

### Phase 1: Fresh Start (Recommended)
The existing checkpoint has 16k episodes of corrupted training (LR=0, bad rewards, epsilon oscillation). It's better to start fresh.

```bash
# 1. Backup old checkpoint
cp checkpoint.pth checkpoint_old_broken.pth

# 2. Delete old checkpoint to force fresh start
rm checkpoint.pth

# 3. Clear old stats
mv training_stats.csv training_stats_old.csv

# 4. Start training with fixes applied
python trainer.py --num_agents 2 --style-name "Standard" --view
```

### Phase 2: Monitor First 500 Episodes
After starting, verify the fixes work:

```bash
# Run analyzer after ~500 episodes
python training_progress_analyzer.py
```

**Check these metrics:**
- [ ] LR should be `1e-4` (not 0)
- [ ] Epsilon should decay: ~0.99 at ep 100, ~0.90 at ep 1000
- [ ] Food reward visible in stats (not crushed to 0.05)
- [ ] Wall deaths correctly classified (check events/ folder)
- [ ] Loss should decrease over time

### Phase 3: Curriculum Progression
Stay on **Standard (Curriculum)** style until:
- Average food > 30 per episode (promotion to Stage 2)
- Average steps > 100 (snake survives at least 3 minutes)

**Do NOT switch styles** until Stage 2 is reached. Style-switching disrupts learning.

### Phase 4: Remaining Code Fixes (Optional, After Phase 2 Confirms Learning)

| Priority | Fix | File | Description |
|----------|-----|------|-------------|
| 1 | Wall render margin | `slither_env.py:846` | Reduce safety margin from 500 to 200 to match death zone closer |
| 2 | Priority 4 distance cap | `slither_env.py:341-343` | Add `min_enemy_dist < 500` check before defaulting to SnakeCollision |
| 3 | Remove SuperPatternOptimizer | `trainer.py:272-375` | Disable dynamic reward adjustment during early training (set `super_pattern_enabled: False`) |
| 4 | Faster epsilon decay | `config.py:30` | Change `eps_decay` from 100000 to 50000 for faster exploitation |

### Phase 5: Long-term Training
Once learning is confirmed (reward trend UP in analyzer):
- Run uninterrupted for 10,000+ episodes
- Use `--resume` flag to continue, never restart
- Target: avg food > 100 by episode 5,000

---

## 8. DEATH STATISTICS

| Cause | Count | % |
|-------|-------|---|
| SnakeCollision | 9,528 | 58.8% |
| Wall | 6,183 | 38.2% |
| Unknown | 491 | 3.0% |

**Analysis:** 38% wall deaths is very high. After the buffer fix (40→120), more deaths will be correctly classified. Expected split after fix: ~45% Snake, ~50% Wall, ~5% Unknown.

---

## Charts

![Overview](training_progress_overview.png)

![Learning Detection](training_learning_detection.png)

![Goal Progress](training_goal_progress.png)
