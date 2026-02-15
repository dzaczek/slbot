# Training Operations & Tuning Guide

A practical guide for monitoring training, diagnosing problems, and adjusting parameters. Written from experience — every recommendation here comes from a failure that taught us something.

## Reading the Training Logs

Every episode produces a log line like this:

```
Ep 5031 | S3:ENEMY_AVOID | Rw: 264.00 | St: 142 | Fd: 52 (0.366/st) | Eps: 0.122 | L: 4.17 | Q: 18.93/31.75 | SnakeCollision | Pos:(38113,36118) Wall:7388 Enemy:439
```

Here's what matters:

| Field | Meaning | What to watch |
|-------|---------|---------------|
| `S3:ENEMY_AVOID` | Current curriculum stage | Should progress S1→S2→S3→S4 over time |
| `Rw: 264.00` | Total episode reward | Higher is better, but context matters (see below) |
| `St: 142` | Steps survived | **The most important metric.** More steps = longer survival |
| `Fd: 52 (0.366/st)` | Food eaten (efficiency) | Efficiency above 0.3/st is good |
| `Eps: 0.122` | Epsilon (exploration rate) | Should decay toward 0.08 over time |
| `L: 4.17` | Training loss | Stable 1-10 is healthy. Above 50 is a problem |
| `Q: 18.93/31.75` | Q-value mean/max | Should grow slowly. If max > 3x mean, values are diverging |
| `SnakeCollision` | Death cause | Tells you what killed the bot |
| `Wall:7388` | Distance to wall at death | Below 500 means the bot was near the edge |
| `Enemy:439` | Distance to nearest enemy at death | Below 200 means the bot was very close |

The **Death Stats** line appears every 10 episodes:
```
Death Stats: {'Wall': 120, 'SnakeCollision': 1171, 'InvalidFrame': 0, 'BrowserError': 0, 'MaxSteps': 5}
```

These are cumulative counts for the current run. Calculate percentages to see what's killing the bot.

## Key Metrics Per Stage

### Stage 1: FOOD_VECTOR

**Target:** Bot learns to navigate toward food and eat it.

| Metric | Bad | OK | Good |
|--------|-----|-----|------|
| avg_steps | <30 | 30-60 | >60 |
| avg_food | <3 | 3-10 | >10 |
| food/step | <0.1 | 0.1-0.3 | >0.3 |

**Promotion requires:** avg_food >= 5 AND avg_steps >= 50 (over 200 episodes).

**Common problems:**

| Symptom | Cause | Fix |
|---------|-------|-----|
| Bot spins in circles, food stays at 0 | Food shaping too weak, bot never accidentally eats | Increase `food_shaping` (0.5→1.0) or `food_reward` (3→5) |
| Bot eats but dies instantly (steps < 10) | Spawning near walls/enemies, bad luck | Normal for S1 — should improve with more episodes |
| Epsilon not decaying | `steps_done` not incrementing properly | Check that `steps_done += 1` happens per batch step in trainer |
| Loss is NaN or > 1000 | Learning rate too high or reward scale wrong | Reduce `lr` (1e-4→5e-5) or check `reward_scale` |

### Stage 2: WALL_AVOID

**Target:** Bot stays away from the circular map boundary.

| Metric | Bad | OK | Good |
|--------|-----|-----|------|
| avg_steps | <50 | 50-80 | >80 |
| wall_death % | >30% | 10-30% | <10% |
| wall distance at death | <500 | 500-2000 | >2000 |

**Promotion requires:** avg_steps >= 55 AND wall_death_rate < 12% (over 200 episodes).

**Common problems:**

| Symptom | Cause | Fix |
|---------|-------|-----|
| Wall deaths stay above 30% | `wall_proximity_penalty` too weak | Increase from 1.5→2.0 or increase `wall_alert_dist` (2500→3000) |
| Bot avoids walls but stops eating | Wall penalty dominates food reward | Increase `food_reward` (2.0→3.0) to rebalance |
| Bot hugs the center and never explores | Wall penalty too strong, bot afraid of edges | Decrease `wall_proximity_penalty` (1.5→0.8) |
| Stuck at avg_steps ~70, won't reach threshold | Snake deaths cut episodes short before wall skill matters | Lower `promote_threshold` (see our fix: 80→55) |
| Q-values growing fast (>30 mean) | Gamma too high for current skill level | Reduce `gamma` (0.93→0.90) |

### Stage 3: ENEMY_AVOID

**Target:** Bot detects and dodges enemy snakes while maintaining food collection.

| Metric | Bad | OK | Good |
|--------|-----|-----|------|
| avg_steps | <60 | 60-150 | >150 |
| snake_death % | >90% | 70-90% | <70% |
| enemy distance at death | <100 | 100-500 | >500 |
| food/step | <0.2 | 0.2-0.4 | >0.4 |

**Promotion requires:** avg_steps >= 200 (over 300 episodes). This is intentionally high — the bot needs extensive practice here.

**Common problems:**

| Symptom | Cause | Fix |
|---------|-------|-----|
| Snake deaths stay at ~100% | `enemy_proximity_penalty` too weak or `enemy_alert_dist` too small | Increase `enemy_proximity_penalty` (→1.5+) and `enemy_alert_dist` (→1500+) |
| Bot avoids enemies but wall deaths spike to 20%+ | Bot "panics" and runs from enemies straight into walls | Increase `wall_proximity_penalty` (0.5→0.8). The bot needs balanced fear |
| Steps aren't growing despite fewer snake deaths | Episodes are short for other reasons (wall, timeouts) | Check `max_steps` is high enough (1000+) and `survival` reward is strong (1.0+) |
| Bot stops eating food (food/step drops below 0.2) | Enemy penalties overwhelm food reward | Increase `food_reward` (5.0→7.0). The bot must stay motivated to eat |
| Q-values exploding (mean > 40) | Gamma 0.97 is too aggressive for current learning state | Option 1: Reduce `gamma` (→0.95). Option 2: Reduce `survival_escalation` (→0.005) |
| Loss spikes above 30-50 regularly | Network struggling with conflicting signals (eat vs avoid) | Reduce `grad_clip` (1.0→0.5) for smoother updates. Consider reducing `lr` |
| Enemy distance at death is always <100 | Bot doesn't react until enemy is on top of it | Increase `enemy_alert_dist` (→2000) so the penalty kicks in earlier |
| Bot promoted too fast to S4 | Threshold too low or lucky outlier episodes inflated avg | Increase `promote_threshold` and `promote_window` (we use 200/300) |

### Stage 4: MASS_MANAGEMENT

**Target:** Optimize for long survival and growth. Balance risk vs reward.

| Metric | Bad | OK | Good |
|--------|-----|-----|------|
| avg_steps | <100 | 100-300 | >300 |
| avg_reward | <200 | 200-500 | >500 |
| MaxSteps % | 0% | 1-5% | >5% |

**No promotion** — this is the final stage.

**Common problems:**

| Symptom | Cause | Fix |
|---------|-------|-----|
| Q-values explode (mean > 50, max > 100) | `length_bonus` creates compounding reward that grows with snake length | Reduce `length_bonus` (0.02→0.005) or remove it entirely (0.0) |
| Loss becomes unstable (spikes to 50-100) | Q-value divergence causes large TD-errors | Reduce `gamma` (0.97→0.95), increase `target_update_freq`, or reduce `lr` |
| Steps DECREASE compared to S3 | S4's relaxed enemy penalties (0.8 vs 1.5) let the bot take risks again | Keep S3's `enemy_proximity_penalty` level (1.5) in S4 |
| Bot regresses on wall avoidance | `death_wall` decreased in S4 (-35 vs -40 in S3) | Keep death_wall at -40 |
| Everything was fine then suddenly collapsed | Classic DQN instability — one bad batch poisons the network | Load a backup checkpoint from before the collapse |

## Parameter Reference

### Reward parameters (in `styles.py`)

These control what the bot gets rewarded or punished for. They are the primary tuning knobs.

| Parameter | What it does | Range | Notes |
|-----------|-------------|-------|-------|
| `food_reward` | Reward per food unit eaten | 1-20 | Primary positive signal. Too high and the bot ignores dangers |
| `food_shaping` | Reward for moving toward nearest food | 0-1 | High in S1 (0.5) to bootstrap learning, low later (0.1) |
| `survival` | Per-step reward for being alive | 0-2 | Key parameter for teaching survival. Higher = bot values staying alive |
| `survival_escalation` | How much survival reward grows per step | 0-0.02 | Makes later steps in an episode worth more than early ones |
| `death_wall` | One-time penalty for hitting the wall | -50 to -10 | Must be large enough that the bot cares about walls |
| `death_snake` | One-time penalty for hitting an enemy | -50 to -10 | Must be large enough relative to food_reward. If food_reward=20 and death_snake=-10, the bot earns 400-1200 food but only loses 10 for dying — it will ignore enemies |
| `wall_proximity_penalty` | Per-step penalty for being near the wall | 0-3 | Linear ramp: zero at `wall_alert_dist`, maximum at the wall |
| `wall_alert_dist` | Distance where wall penalty starts | 1000-3000 | Map radius is ~24500. 2000 means penalty starts at ~8% from edge |
| `enemy_proximity_penalty` | Per-step penalty for being near enemies | 0-3 | Same linear ramp as wall_proximity_penalty |
| `enemy_approach_penalty` | Extra penalty for getting CLOSER to an enemy | 0-1 | Only fires when the enemy distance is decreasing AND within alert range |
| `enemy_alert_dist` | Distance where enemy penalty starts | 500-2000 | Larger = bot starts dodging earlier. Smaller = bot only reacts to close threats |
| `boost_penalty` | Penalty for using the boost action | 0-1 | Prevents the bot from wasting length on boost in early stages |
| `straight_penalty` | Penalty for going straight | 0-0.2 | Forces exploration. Only used in Explorer style |
| `length_bonus` | Per-step reward proportional to snake length | 0-0.05 | Dangerous — creates compounding reward. Only in S4, use sparingly |
| `gamma` | Discount factor for future rewards | 0.8-0.99 | Higher = bot plans further ahead. Lower = faster learning but shortsighted |
| `max_steps` | Episode length limit | 100-5000 | Prevents infinite episodes. Should increase with stage |

### Training parameters (in `config.py`)

These control the learning algorithm. Change these when the network isn't learning properly.

| Parameter | Default | What it does | When to change |
|-----------|---------|-------------|----------------|
| `lr` | 1e-4 | Learning rate | Decrease if loss is unstable or Q-values explode |
| `batch_size` | 64 | Transitions per training step | Increase to 128 if training is noisy |
| `grad_clip` | 0.5 | Maximum gradient norm | Decrease if loss spikes. Increase if learning is too slow |
| `eps_start` | 1.0 | Initial exploration rate | Always 1.0 for new training |
| `eps_end` | 0.08 | Final exploration rate | Lower (0.05) for more exploitation. Higher (0.15) if stuck |
| `eps_decay` | 8000 | Steps for epsilon to halve | Lower = faster exploitation. Higher = more exploration |
| `target_update_freq` | 1000 | Steps between target network updates | Increase if Q-values are unstable (→2000-5000) |
| `reward_scale` | 1.0 | Divides raw rewards before clamping | Was 10.0 in early versions (crushed signal). Keep at 1.0 |
| `capacity` | 100000 | Replay buffer size | Larger = more diverse training data but more RAM |
| `alpha` | 0.6 | PER prioritization strength | Higher = more focus on surprising transitions |
| `beta_start` | 0.4 | PER importance sampling correction | Anneals to 1.0. Lower start = more biased early sampling |

### Promotion parameters (in `styles.py`)

These control when the curriculum advances to the next stage.

| Parameter | What it does | Guidance |
|-----------|-------------|----------|
| `promote_metric` | What to measure (`avg_steps`, `compound`, or `None`) | `compound` for S1 (must eat AND survive), `avg_steps` for S2/S3 |
| `promote_threshold` | Target value for the metric | Set to a level that proves the bot actually learned the skill |
| `promote_conditions` | Dict of conditions for `compound` metric | S1 uses `avg_food >= 5` AND `avg_steps >= 50` |
| `promote_wall_death_max` | Max wall death rate to allow promotion | Only in S2. Prevents promotion if the bot still hits walls |
| `promote_window` | Number of episodes to average over | Larger = more confident promotion but slower. 200-300 is good |

## Decision Flowchart

When training isn't going well, work through these questions in order:

### 1. Is the bot dying too fast? (avg_steps < 50)

**Check death causes:**
- >50% Wall → increase `wall_proximity_penalty` and `wall_alert_dist`
- >80% SnakeCollision → increase `enemy_proximity_penalty` and `enemy_alert_dist`
- Mixed → increase `survival` reward to incentivize staying alive regardless of cause

### 2. Is the bot not eating? (food/step < 0.15)

**Check if penalties are too strong:**
- The bot may be so afraid of walls/enemies that it avoids food near them
- Reduce penalties slightly or increase `food_reward`
- Check `food_shaping` — in early stages this helps guide the bot toward food

### 3. Is loss unstable? (regular spikes above 30)

**Stabilize the network:**
- Reduce `grad_clip` (1.0→0.5→0.3)
- Reduce `lr` (1e-4→5e-5)
- Increase `target_update_freq` (1000→2000)
- Check if reward values are in a reasonable range (death penalties within [-50, 0])

### 4. Are Q-values exploding? (mean > 40 or max > 100)

**This is the most common failure mode.** It means the network is overestimating future rewards.

Causes and fixes:
- `gamma` too high → reduce by 0.02 (e.g., 0.97→0.95)
- `survival_escalation` too high → creates exponentially growing reward predictions
- `length_bonus` > 0 → creates compounding reward. Reduce or remove
- Reward scale/clamp misconfigured → check that rewards are divided by `reward_scale` and clamped to [-10, +10]

### 5. Is the bot stuck (no improvement for 500+ episodes)?

**Check epsilon:** If it's already low (< 0.1), the bot may be stuck in a local optimum.
- The stagnation handler will boost epsilon to 0.3 after `adaptive_eps_patience` episodes
- If that's not helping, consider loading a backup checkpoint from when training was improving

**Check if the stage is wrong:**
- If the bot hasn't learned S2 skills but is in S3, it will fail at everything
- Go back to the appropriate stage with `--stage N`

### 6. Is the bot promoting too fast?

**Increase promotion requirements:**
- Raise `promote_threshold` so the bot must genuinely master the skill
- Increase `promote_window` so a few lucky episodes don't trigger promotion
- Add compound conditions (e.g., `promote_wall_death_max` in S2)

## Reward Balance Rules of Thumb

1. **Death penalty should be >= 5x the average per-step reward.** If the bot earns ~5 reward per step and lives 100 steps (total 500), a death penalty of -10 is meaningless. It needs to be at least -25 to matter.

2. **Per-step penalties should be < food_reward.** If enemy_proximity_penalty (1.5) is close to food_reward (2.0), the bot will avoid enemies rather than eat nearby food. Keep penalties below ~50% of food_reward unless you specifically want avoidance behavior.

3. **Gamma determines the planning horizon.** At gamma=0.97, the bot effectively looks ahead ~33 steps (1/(1-0.97)). At gamma=0.85, it's ~7 steps. Match gamma to the time scale of the skill being learned:
   - Food collection: 5-10 steps ahead → gamma 0.85-0.90
   - Wall avoidance: 10-20 steps ahead → gamma 0.90-0.95
   - Enemy avoidance: 20-50 steps ahead → gamma 0.95-0.97

4. **Survival escalation is powerful but dangerous.** At escalation=0.015 and survival=1.2, the per-step reward at step 200 is 1.2 * (1 + 0.015 * 200) = 4.8. Make sure this doesn't exceed your food reward or the bot will learn to survive by avoiding everything instead of eating.

5. **Keep total per-step reward in [-5, +5] after scaling.** The clamping range is [-10, +10] but most rewards should be well within that. If rewards regularly hit the clamp, the network loses gradient information.

## SuperPattern Optimizer

Active from Stage 3 onwards. Automatically adjusts reward parameters based on death statistics over a rolling window.

**What it does:**
- If wall death rate > 55%: increases `wall_proximity_penalty` (up to cap)
- If snake death rate > 55%: increases `enemy_proximity_penalty` (up to cap)
- If food rate is too low: increases `food_reward` (up to cap of 15.0)
- If death rates drop below 38%: relaxes penalties back toward base values

**When to disable:** If you're trying to maintain stable reward parameters for controlled experiments, set `super_pattern_enabled: false` in config.py.

## Backup and Recovery

Checkpoints are saved every 50 episodes to `checkpoint.pth`. Best models are saved to `backup_models/` whenever the rolling average reward improves.

**If checkpoint.pth is corrupted** (common when training is killed during a save):
```bash
# Find the best backup
ls -lt backup_models/ | head -5

# Copy it as the main checkpoint
cp backup_models/best_model_XXXXX.pth checkpoint.pth
```

Note: backup models only contain network weights and optimizer state — they don't store epsilon or curriculum stage. Use `--stage N` to set the stage manually when resuming from a backup.

**If training has regressed** (metrics getting worse):
```bash
# Find a backup from when metrics were good
ls backup_models/ | grep "rw[0-9]*" | sort -t_ -k4 -n

# Restore and restart
cp backup_models/best_model_XXXXX_epN_rwM.pth checkpoint.pth
python trainer.py --resume --stage 3
```

## Analysis Workflow

After every major training session, run the analyzer to see trends:

```bash
# Full analysis of latest run
python training_progress_analyzer.py --latest

# Quick check (no charts)
python training_progress_analyzer.py --latest --no-charts
```

**Key charts to check:**
- **Chart 01 (Dashboard)**: Overall reward and step trends. Are they going up?
- **Chart 07 (Death Analysis)**: What's killing the bot? Is the cause distribution changing?
- **Chart 13 (Q-values)**: Are Q-values stable or exploding?
- **Chart 14 (Actions)**: Is the bot using all actions or stuck on one?
- **Chart 16 (MaxSteps)**: Is the bot hitting the episode length limit? If yes, increase `max_steps`

**Quick health check from the terminal:**
```bash
# Last 50 episodes summary
tail -50 training_stats.csv | awk -F',' '{s+=$4; f+=$12; n++; if($10=="SnakeCollision") sc++; if($10=="Wall") w++} END {printf "steps=%.0f food=%.0f snake=%.0f%% wall=%.0f%%\n", s/n, f/n, sc/n*100, w/n*100}'
```
