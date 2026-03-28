# Karpathy Mod — Research Program for SlitherBot

## Mission

Autonomously optimize SlitherBot reward configuration through systematic experimentation.
The system edits `styles.py` (reward parameters) while keeping all other code frozen.
Each experiment trains for a fixed budget of episodes, measures a single composite score,
and decides: keep or revert.

## The Loop

```
while True:
    1. HYPOTHESIZE — Generate mutation(s) to reward params
    2. MUTATE      — Apply to styles.py in isolated worktree
    3. TRAIN       — Run trainer for N episodes (fixed budget)
    4. EVALUATE    — Compute score from training_stats.csv
    5. DECIDE      — Keep (improve) or Revert (discard)
    6. LOG         — Record everything in karpathy_mod_results.tsv
```

## Rules

### What can be mutated
- Any numerical reward parameter in `styles.py` stages 1-6
- Parameters: gamma, food_reward, food_shaping, survival, death penalties,
  enemy penalties, boost penalties, max_steps, etc.
- See `MUTABLE_PARAMS` in `karpathy_mod_mutator.py` for full list with ranges

### What is frozen (NEVER edit)
- `trainer.py` — training loop, optimization, model updates
- `slither_env.py` — environment, observation, reward computation logic
- `agent.py` — DQN agent, replay buffer, action selection
- `model.py` — neural network architecture
- `config.py` — model/optimizer/buffer hyperparameters
- `browser_engine.py` — browser automation

### Evaluation metric
Primary score = `avg_steps * 1.0 + avg_peak_length * 0.5 + avg_food * 0.3 - snake_death_rate * 0.5`

A mutation is KEPT only if:
1. Composite score improves by > 2%
2. avg_steps also improves (no regression in survival)
3. snake_death_rate doesn't increase by > 15%

### Mutation strategies

| Strategy   | Params changed | Intensity | When to use                    |
|------------|---------------|-----------|--------------------------------|
| `tweak`    | 1             | Low       | Fine-tuning, most of the time  |
| `explore`  | 2-3           | Medium    | Searching for better regions   |
| `radical`  | 3-5           | High      | Breaking out of local optima   |
| `targeted` | Group         | Medium    | Focus on enemy/food/survival   |
| `crossover`| Blend         | Medium    | Transfer good params between stages |

Distribution: 50% tweak, 33% explore, 17% radical/targeted/crossover

### Multi-agent parallel experiments
When `--parallel N` is used:
- N independent mutations are generated (diversified strategies)
- Each runs in its own git worktree with its own styles.py
- Each has its own trainer process and training_stats.csv
- After budget, ALL are evaluated against the SAME baseline
- Only the BEST one wins (if it beats baseline)
- This is NOT ensemble — it's competitive evolution

### Budget guidelines
- `--budget 300` — Quick exploration (30-60 min per round)
- `--budget 500` — Standard (recommended, 1-2 hrs)
- `--budget 1000` — Thorough (3-4 hrs, higher confidence)

### Safety
- Original `styles.py` is preserved in git history (every change is a commit)
- Failed experiments are automatically reverted
- `--cleanup` removes all worktrees
- Ctrl+C at any time → graceful shutdown, state saved
- `karpathy_mod_state.json` tracks progress across restarts

## Parameter Groups (for targeted mutations)

### Survival group
Controls how long the snake lives:
- `survival`, `survival_escalation` — per-step survival reward
- `death_wall`, `death_snake` — death penalties
- `max_steps` — episode length limit
- `starvation_penalty`, `starvation_grace_steps`

### Enemy group
Controls snake-enemy interaction:
- `enemy_alert_dist` — detection range
- `enemy_proximity_penalty` — penalty for being near enemies
- `enemy_approach_penalty` — penalty for moving toward enemies
- `death_snake` — death penalty for snake collision

### Food group
Controls eating behavior:
- `food_reward`, `food_shaping` — eating rewards
- `length_bonus` — growth reward
- `contest_food_reward` — reward for contested food
- `starvation_penalty` — penalty for not eating

### Risk group
Controls caution vs aggression:
- `boost_penalty` — cost of boosting
- `mass_loss_penalty` — cost of losing mass
- `wall_proximity_penalty` — penalty near walls
- `enemy_proximity_penalty` — penalty near enemies

## Interpreting results

### karpathy_mod_results.tsv columns
- `timestamp` — when experiment finished
- `round` — round number
- `experiment_id` — unique ID
- `strategy` — mutation strategy used
- `stage` — which curriculum stage was mutated
- `decision` — keep/discard
- `improvement_pct` — % change in composite score
- `baseline_score` / `experiment_score` — raw scores
- `avg_steps`, `avg_food`, `peak_length`, `snake_death_rate` — key metrics
- `description` — what was changed

### What to look for
1. Which strategies produce the most "keep" decisions?
2. Which parameter groups have the most impact?
3. Are improvements plateauing? → Try more radical strategies
4. Is snake_death_rate climbing? → Focus on enemy group

## Analyzer — `karpathy_mod_analyzer.py`

Visual analytics dashboard for experiment results. Reads `karpathy_mod_results.tsv`
and generates charts, terminal report, and markdown report.

### Usage

```bash
python karpathy_mod_analyzer.py                    # full analysis
python karpathy_mod_analyzer.py --no-charts        # text report only
python karpathy_mod_analyzer.py --no-report        # skip markdown
python karpathy_mod_analyzer.py --tsv PATH         # custom TSV path
python karpathy_mod_analyzer.py --last N           # only last N rounds
```

### Generated charts (saved to `charts/`)

| File | Description |
|------|-------------|
| `karpathy_overview.png` | 2×2 dashboard: score timeline, improvement bars, decision pie, cumulative keep rate |
| `karpathy_strategy.png` | Per-strategy effectiveness: keep rate, avg improvement, experiment count |
| `karpathy_stages.png` | Per-stage analysis: experiment count, keep rate, avg improvement |
| `karpathy_metrics.png` | Metric evolution for kept experiments: avg_steps, avg_food, peak_length, snake_death_rate |
| `karpathy_parameters.png` | Which reward parameters lead to keep vs discard decisions |
| `karpathy_distribution.png` | Histogram of improvement % colored by decision |
| `karpathy_heatmap.png` | Strategy × Stage heatmap with avg improvement and counts |
| `karpathy_experiment_explorer.html` | Interactive Plotly scatter — hover for full experiment details |

### Terminal report sections
1. **Summary** — total rounds, keep rate, best/worst experiments
2. **Strategy leaderboard** — table with keep rate and avg improvement per strategy
3. **Stage performance** — per-stage statistics
4. **Top 5 best** kept experiments with descriptions
5. **Top 5 worst** discarded experiments
6. **Trend assessment** — improving, plateauing, or regressing
7. **Recommendations** — what to try next

### Markdown report
Generated as `karpathy_mod_report.md` with all tables and embedded chart images.

## File inventory

| File | Purpose |
|------|---------|
| `karpathy_mod_program.md` | This document — research program specification |
| `karpathy_mod_runner.py` | Main orchestrator — mutation loop, worktree management, training |
| `karpathy_mod_mutator.py` | Mutation engine — parameter selection, strategies, serialization |
| `karpathy_mod_evaluator.py` | Metrics computation and experiment comparison |
| `karpathy_mod_analyzer.py` | Results visualization and reporting |
| `karpathy_mod_results.tsv` | Experiment log (auto-generated, gitignored) |
| `karpathy_mod_state.json` | Persistent state across restarts (gitignored) |
