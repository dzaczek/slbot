"""
Karpathy-style mutation engine for SlitherBot reward configs.

Two mutation strategies:
1. Rule-based: random parameter tweaks within bounded ranges
2. LLM-powered: creative mutations via Claude/OpenAI API (optional)
"""

import copy
import random
import math
import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple


# === Mutable parameter definitions ===
# Each entry: (min, max, step_pct) where step_pct = max % change per mutation
MUTABLE_PARAMS = {
    "gamma":                    (0.80, 0.995, 0.03),
    "food_reward":              (1.0,  25.0,  0.30),
    "food_shaping":             (0.0,  1.0,   0.40),
    "survival":                 (0.0,  2.0,   0.30),
    "survival_escalation":      (0.0,  0.01,  0.50),
    "death_wall":               (-100, -5,    0.25),
    "death_snake":              (-100, -5,    0.25),
    "length_bonus":             (0.0,  0.5,   0.50),
    "wall_proximity_penalty":   (0.0,  3.0,   0.40),
    "enemy_alert_dist":         (500,  4000,  0.25),
    "enemy_proximity_penalty":  (0.0,  5.0,   0.40),
    "enemy_approach_penalty":   (0.0,  3.0,   0.40),
    "boost_penalty":            (0.0,  15.0,  0.40),
    "mass_loss_penalty":        (0.0,  20.0,  0.40),
    "starvation_penalty":       (0.0,  0.5,   0.50),
    "starvation_grace_steps":   (5,    200,   0.30),
    "max_steps":                (100,  10000, 0.20),
    "contest_food_reward":      (0.0,  5.0,   0.50),
    "enemy_zone_control_reward":(0.0,  0.5,   0.50),
    "kill_opportunity_reward":  (0.0,  30.0,  0.40),
}

# Parameters that should stay integers
INTEGER_PARAMS = {"max_steps", "starvation_grace_steps", "enemy_alert_dist"}

# Parameters we never touch (structural, not reward-related)
FROZEN_PARAMS = {"name", "type", "description", "promote_metric",
                 "promote_threshold", "promote_window", "promote_conditions",
                 "promote_wall_death_max"}


def generate_experiment_id(mutation_desc: str) -> str:
    """Short unique ID for an experiment."""
    h = hashlib.md5(f"{mutation_desc}{random.random()}".encode()).hexdigest()[:8]
    return h


def mutate_value(key: str, current: float, intensity: float = 1.0) -> float:
    """Mutate a single parameter value within its allowed range."""
    if key not in MUTABLE_PARAMS:
        return current

    lo, hi, step_pct = MUTABLE_PARAMS[key]
    max_delta = abs(current) * step_pct * intensity
    if max_delta < 1e-6:
        max_delta = (hi - lo) * 0.05 * intensity

    delta = random.uniform(-max_delta, max_delta)
    new_val = current + delta
    new_val = max(lo, min(hi, new_val))

    if key in INTEGER_PARAMS:
        new_val = int(round(new_val))

    return new_val


def mutate_stage_config(stage_config: Dict[str, Any],
                        num_mutations: int = 2,
                        intensity: float = 1.0) -> Tuple[Dict[str, Any], List[str]]:
    """
    Apply random mutations to a stage config dict.
    Returns (mutated_config, list_of_change_descriptions).
    """
    config = copy.deepcopy(stage_config)
    mutable_keys = [k for k in config if k in MUTABLE_PARAMS]

    if not mutable_keys:
        return config, ["no mutable params"]

    num_mutations = min(num_mutations, len(mutable_keys))
    targets = random.sample(mutable_keys, num_mutations)
    changes = []

    for key in targets:
        old_val = config[key]
        new_val = mutate_value(key, old_val, intensity)
        if old_val != new_val:
            config[key] = new_val
            if isinstance(old_val, float):
                changes.append(f"{key}: {old_val:.4f} -> {new_val:.4f}")
            else:
                changes.append(f"{key}: {old_val} -> {new_val}")

    if not changes:
        changes.append("no effective changes")

    return config, changes


class Mutator:
    """
    Generates mutations for SlitherBot reward configs.

    Supports multiple strategies:
    - 'tweak': small parameter adjustments (1-2 params, low intensity)
    - 'explore': bigger changes (2-4 params, medium intensity)
    - 'radical': large changes (3-5 params, high intensity)
    - 'targeted': focus on a specific parameter group
    - 'crossover': blend two stage configs
    """

    STRATEGIES = ['tweak', 'tweak', 'tweak', 'explore', 'explore', 'radical']

    # Parameter groups for targeted mutations
    PARAM_GROUPS = {
        'survival': ['survival', 'survival_escalation', 'death_wall', 'death_snake',
                      'max_steps', 'starvation_penalty', 'starvation_grace_steps'],
        'enemy':    ['enemy_alert_dist', 'enemy_proximity_penalty',
                      'enemy_approach_penalty', 'death_snake'],
        'food':     ['food_reward', 'food_shaping', 'length_bonus',
                      'contest_food_reward', 'starvation_penalty'],
        'risk':     ['boost_penalty', 'mass_loss_penalty',
                      'wall_proximity_penalty', 'enemy_proximity_penalty'],
    }

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

    def generate_mutation(self,
                          styles_dict: Dict,
                          target_stage: int = 0,
                          strategy: Optional[str] = None) -> Dict:
        """
        Generate a single mutation proposal.

        Returns dict with:
            experiment_id: str
            strategy: str
            target_stage: int
            changes: list of descriptions
            mutated_stages: dict of stage_num -> mutated_config
            original_stages: dict of stage_num -> original_config (for rollback)
        """
        stages = styles_dict.get("stages", {})
        if not stages:
            raise ValueError("No stages found in styles_dict")

        if strategy is None:
            strategy = random.choice(self.STRATEGIES)

        # Pick target stage
        if target_stage > 0 and target_stage in stages:
            stage_num = target_stage
        else:
            stage_num = random.choice(list(stages.keys()))

        original = copy.deepcopy(stages[stage_num])

        if strategy == 'tweak':
            mutated, changes = mutate_stage_config(original, num_mutations=1, intensity=0.5)
        elif strategy == 'explore':
            mutated, changes = mutate_stage_config(original, num_mutations=3, intensity=1.0)
        elif strategy == 'radical':
            mutated, changes = mutate_stage_config(original, num_mutations=5, intensity=1.5)
        elif strategy == 'targeted':
            mutated, changes = self._targeted_mutation(original, intensity=1.0)
        elif strategy == 'crossover':
            mutated, changes = self._crossover_mutation(stages, stage_num)
        else:
            mutated, changes = mutate_stage_config(original, num_mutations=2, intensity=1.0)

        stage_name = original.get('name', f'S{stage_num}')
        desc = f"[{strategy}] S{stage_num}/{stage_name}: " + "; ".join(changes)
        exp_id = generate_experiment_id(desc)

        return {
            "experiment_id": exp_id,
            "strategy": strategy,
            "target_stage": stage_num,
            "description": desc,
            "changes": changes,
            "mutated_stages": {stage_num: mutated},
            "original_stages": {stage_num: original},
        }

    def generate_batch(self,
                       styles_dict: Dict,
                       count: int = 4,
                       target_stage: int = 0) -> List[Dict]:
        """Generate multiple independent mutations for parallel experiments."""
        mutations = []
        strategies_used = set()

        for i in range(count):
            # Diversify strategies across the batch
            if len(strategies_used) < len(set(self.STRATEGIES)):
                remaining = [s for s in set(self.STRATEGIES) if s not in strategies_used]
                strategy = random.choice(remaining)
            else:
                strategy = random.choice(self.STRATEGIES)

            mutation = self.generate_mutation(styles_dict, target_stage, strategy)
            mutations.append(mutation)
            strategies_used.add(strategy)

        return mutations

    def _targeted_mutation(self, config: Dict, intensity: float) -> Tuple[Dict, List[str]]:
        """Mutate parameters from the same functional group."""
        config = copy.deepcopy(config)
        group_name = random.choice(list(self.PARAM_GROUPS.keys()))
        group_keys = [k for k in self.PARAM_GROUPS[group_name] if k in config]

        if not group_keys:
            return mutate_stage_config(config, num_mutations=2, intensity=intensity)

        changes = []
        for key in group_keys:
            old_val = config[key]
            new_val = mutate_value(key, old_val, intensity)
            if old_val != new_val:
                config[key] = new_val
                if isinstance(old_val, float):
                    changes.append(f"{key}: {old_val:.4f} -> {new_val:.4f}")
                else:
                    changes.append(f"{key}: {old_val} -> {new_val}")

        if not changes:
            changes.append(f"targeted/{group_name}: no effective changes")
        else:
            changes.insert(0, f"group={group_name}")

        return config, changes

    def _crossover_mutation(self, stages: Dict, target_stage: int) -> Tuple[Dict, List[str]]:
        """Blend parameters from another stage into the target."""
        other_stages = [s for s in stages if s != target_stage]
        if not other_stages:
            return mutate_stage_config(copy.deepcopy(stages[target_stage]), 2, 1.0)

        donor_num = random.choice(other_stages)
        target = copy.deepcopy(stages[target_stage])
        donor = stages[donor_num]

        blend_ratio = random.uniform(0.2, 0.5)
        blendable = [k for k in target if k in donor and k in MUTABLE_PARAMS]
        num_blend = max(1, len(blendable) // 3)
        blend_keys = random.sample(blendable, min(num_blend, len(blendable)))

        changes = [f"crossover from S{donor_num} (ratio={blend_ratio:.2f})"]
        for key in blend_keys:
            old_val = target[key]
            donor_val = donor[key]
            new_val = old_val * (1 - blend_ratio) + donor_val * blend_ratio

            lo, hi, _ = MUTABLE_PARAMS[key]
            new_val = max(lo, min(hi, new_val))
            if key in INTEGER_PARAMS:
                new_val = int(round(new_val))

            target[key] = new_val
            if isinstance(old_val, float):
                changes.append(f"{key}: {old_val:.4f} -> {new_val:.4f}")
            else:
                changes.append(f"{key}: {old_val} -> {new_val}")

        return target, changes


def apply_mutation_to_styles(styles: Dict, mutation: Dict) -> Dict:
    """Apply a mutation dict to a full STYLES structure. Returns modified copy."""
    styles = copy.deepcopy(styles)
    curriculum = styles.get("Standard (Curriculum)", {})
    stages = curriculum.get("stages", {})

    for stage_num, mutated_config in mutation["mutated_stages"].items():
        if stage_num in stages:
            # Preserve frozen params from original
            original = stages[stage_num]
            for key in FROZEN_PARAMS:
                if key in original:
                    mutated_config[key] = original[key]
            stages[stage_num] = mutated_config

    return styles


def styles_dict_to_python(styles: Dict) -> str:
    """Serialize STYLES dict back to valid Python source code."""
    lines = [
        '',
        '"""',
        'Reward definitions for different learning styles.',
        '"""',
        '',
        'STYLES = '
    ]
    # Use json for the dict, then fix Python-specific syntax
    raw = json.dumps(styles, indent=4, default=str)
    # Fix JSON -> Python: null->None, true->True, false->False
    raw = raw.replace(": null", ": None")
    raw = raw.replace(": true", ": True")
    raw = raw.replace(": false", ": False")
    # Fix integer keys for stages (JSON forces string keys)
    import re
    raw = re.sub(r'"(\d+)":', r'\1:', raw)
    lines[-1] += raw + '\n'
    return '\n'.join(lines)


if __name__ == "__main__":
    # Quick test
    from styles import STYLES
    m = Mutator(seed=42)
    curriculum = STYLES["Standard (Curriculum)"]

    print("=== Single mutation ===")
    mut = m.generate_mutation(curriculum, target_stage=5)
    print(f"  ID: {mut['experiment_id']}")
    print(f"  Desc: {mut['description']}")

    print("\n=== Batch of 4 ===")
    batch = m.generate_batch(curriculum, count=4, target_stage=5)
    for i, mut in enumerate(batch):
        print(f"  [{i}] {mut['strategy']:10s} {mut['description']}")
