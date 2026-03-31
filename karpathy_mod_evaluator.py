"""
Karpathy-style experiment evaluator for SlitherBot.

Reads training_stats.csv, computes metrics over episode windows,
and performs statistical comparison between baseline and experiment.
"""

import csv
import os
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ExperimentMetrics:
    """Aggregated metrics from a training window."""
    num_episodes: int = 0
    avg_steps: float = 0.0
    avg_reward: float = 0.0
    avg_food: float = 0.0
    avg_peak_length: float = 0.0
    avg_loss: float = 0.0
    avg_q_mean: float = 0.0
    wall_death_rate: float = 0.0
    snake_death_rate: float = 0.0
    max_steps_rate: float = 0.0
    avg_reflex_rate: float = 0.0
    median_steps: float = 0.0
    p90_steps: float = 0.0

    def score(self, weights: Optional[Dict[str, float]] = None, baseline: Optional['ExperimentMetrics'] = None) -> float:
        """
        Compute a single scalar score for this experiment.
        Higher = better.

        If baseline is provided, use 'lite normalization':
        Score = Sum over metrics [ (value / baseline_value) * weight ]
        This ensures each metric contributes proportionally to its importance weight
        regardless of its raw scale.
        """
        if weights is None:
            weights = {
                'avg_steps': 1.0,
                'avg_peak_length': 0.5,
                'avg_food': 0.3,
                'snake_death_rate': -0.5,  # lower is better
                'wall_death_rate': -0.2,   # lower is better
                'avg_reward': 0.2,
            }

        if baseline:
            score = 0.0
            for key, w in weights.items():
                val = getattr(self, key, 0.0)
                b_val = getattr(baseline, key, 0.0)
                if abs(b_val) > 1e-6:
                    # Contribution is relative to baseline
                    score += (val / b_val) * w
                else:
                    # Fallback for zero baseline: use raw if possible, or 1.0
                    score += (val if abs(val) < 2.0 else 1.0) * w
            return score
        else:
            # Raw score (original behavior for legacy compatibility)
            score = 0.0
            for key, w in weights.items():
                val = getattr(self, key, 0.0)
                score += val * w
            return score

    def to_dict(self) -> Dict:
        return {
            'num_episodes': self.num_episodes,
            'avg_steps': round(self.avg_steps, 1),
            'avg_reward': round(self.avg_reward, 2),
            'avg_food': round(self.avg_food, 1),
            'avg_peak_length': round(self.avg_peak_length, 1),
            'avg_loss': round(self.avg_loss, 4),
            'avg_q_mean': round(self.avg_q_mean, 2),
            'wall_death_rate': round(self.wall_death_rate, 4),
            'snake_death_rate': round(self.snake_death_rate, 4),
            'max_steps_rate': round(self.max_steps_rate, 4),
            'avg_reflex_rate': round(self.avg_reflex_rate, 4),
            'median_steps': round(self.median_steps, 1),
            'p90_steps': round(self.p90_steps, 1),
        }


def read_csv_tail(csv_path: str, n_episodes: int, stage_filter: Optional[int] = None) -> List[Dict]:
    """Read the last N rows from training_stats.csv, optionally filtering by Stage."""
    if not os.path.exists(csv_path):
        return []

    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        # Buffer last N matching rows
        from collections import deque
        buffer = deque(maxlen=n_episodes)
        for row in reader:
            if stage_filter is not None:
                try:
                    row_stage = int(row.get('Stage', 0))
                    if row_stage != stage_filter:
                        continue
                except (ValueError, TypeError):
                    continue
            buffer.append(row)
        rows = list(buffer)

    return rows


def compute_metrics(rows: List[Dict], warmup: int = 0) -> ExperimentMetrics:
    """
    Compute aggregated metrics from CSV rows.

    Args:
        rows: List of CSV row dicts
        warmup: Number of initial rows to skip (transition period after config change)
    """
    if warmup > 0:
        rows = rows[warmup:]

    if not rows:
        return ExperimentMetrics()

    m = ExperimentMetrics(num_episodes=len(rows))

    steps_list = []
    rewards = []
    foods = []
    peaks = []
    losses = []
    q_means = []
    reflex_rates = []
    wall_deaths = 0
    snake_deaths = 0
    max_steps_deaths = 0

    for row in rows:
        steps = _float(row.get('Steps', 0))
        steps_list.append(steps)
        rewards.append(_float(row.get('Reward', 0)))
        foods.append(_float(row.get('Food', 0)))
        peaks.append(_float(row.get('PeakLength', 0)))

        loss = _float(row.get('Loss', 0))
        if loss > 0:
            losses.append(loss)

        qm = _float(row.get('QMean', 0))
        if qm != 0:
            q_means.append(qm)

        rr = _float(row.get('ReflexRate', 0))
        reflex_rates.append(rr)

        cause = row.get('Cause', '')
        if cause == 'Wall':
            wall_deaths += 1
        elif cause == 'SnakeCollision':
            snake_deaths += 1
        elif cause == 'MaxSteps':
            max_steps_deaths += 1

    n = len(rows)
    m.avg_steps = sum(steps_list) / n
    m.avg_reward = sum(rewards) / n
    m.avg_food = sum(foods) / n
    m.avg_peak_length = sum(peaks) / n
    m.avg_loss = sum(losses) / len(losses) if losses else 0
    m.avg_q_mean = sum(q_means) / len(q_means) if q_means else 0
    m.wall_death_rate = wall_deaths / n
    m.snake_death_rate = snake_deaths / n
    m.max_steps_rate = max_steps_deaths / n
    m.avg_reflex_rate = sum(reflex_rates) / n

    steps_sorted = sorted(steps_list)
    m.median_steps = steps_sorted[n // 2]
    m.p90_steps = steps_sorted[int(n * 0.9)]

    return m


def compare_experiments(baseline: ExperimentMetrics,
                        experiment: ExperimentMetrics,
                        min_improvement: float = 0.005) -> Dict:
    """
    Compare experiment against baseline using lite normalization.

    Returns dict with:
        decision: 'keep' | 'discard' | 'inconclusive'
        improvement_pct: float
        baseline_score: float
        experiment_score: float
        details: str
    """
    # Use baseline to normalize scores
    b_score = baseline.score(baseline=baseline)
    e_score = experiment.score(baseline=baseline)

    if abs(b_score) < 1e-6:
        improvement = (e_score - b_score)
    else:
        improvement = (e_score - b_score) / abs(b_score)

    # Statistical significance check via Mann-Whitney-like heuristic
    # (simplified: require both score improvement AND avg_steps improvement)
    # Note: avg_steps improvement threshold is now relative
    steps_improved = experiment.avg_steps > baseline.avg_steps * (1 + min_improvement * 0.2)
    score_improved = improvement > min_improvement

    # Check for regression in critical metrics
    death_rate_regression = (experiment.snake_death_rate > baseline.snake_death_rate * 1.10
                             and baseline.snake_death_rate > 0.1)

    # Wall death regression check
    wall_death_regression = (experiment.wall_death_rate > baseline.wall_death_rate * 1.15
                             and baseline.wall_death_rate > 0.05)

    if score_improved and (steps_improved or improvement > min_improvement * 3) and not death_rate_regression and not wall_death_regression:
        decision = 'keep'
    elif improvement < -min_improvement:
        decision = 'discard'
    else:
        decision = 'inconclusive'

    details_parts = [
        f"score: {b_score:.2f} -> {e_score:.2f} ({improvement:+.2%})",
        f"steps: {baseline.avg_steps:.0f} -> {experiment.avg_steps:.0f}",
        f"food: {baseline.avg_food:.1f} -> {experiment.avg_food:.1f}",
        f"snake_death: {baseline.snake_death_rate:.1%} -> {experiment.snake_death_rate:.1%}",
        f"peak_len: {baseline.avg_peak_length:.0f} -> {experiment.avg_peak_length:.0f}",
    ]
    if death_rate_regression:
        details_parts.append("WARN: snake death")
    if wall_death_regression:
        details_parts.append("WARN: wall death")

    return {
        'decision': decision,
        'improvement_pct': improvement,
        'baseline_score': b_score,
        'experiment_score': e_score,
        'details': " | ".join(details_parts),
    }


def count_csv_rows(csv_path: str) -> int:
    """Count total rows in CSV (excluding header)."""
    if not os.path.exists(csv_path):
        return 0
    with open(csv_path, 'r') as f:
        return sum(1 for _ in f) - 1  # subtract header


def wait_for_episodes(csv_path: str, target_count: int,
                      poll_interval: float = 10.0,
                      timeout: float = 3600.0) -> int:
    """
    Block until CSV has at least target_count rows.
    Returns actual row count.
    """
    import time
    start = time.time()
    while True:
        current = count_csv_rows(csv_path)
        if current >= target_count:
            return current
        if time.time() - start > timeout:
            return current
        time.sleep(poll_interval)


def _float(val, default=0.0) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


if __name__ == "__main__":
    # Quick test with current training data
    csv_path = os.path.join(os.path.dirname(__file__), 'training_stats.csv')
    rows = read_csv_tail(csv_path, 1000)
    if rows:
        metrics = compute_metrics(rows, warmup=100)
        print("=== Last 1000 episodes (warmup=100) ===")
        for k, v in metrics.to_dict().items():
            print(f"  {k:20s}: {v}")
        print(f"  {'score':20s}: {metrics.score():.1f}")
    else:
        print("No training data found.")
