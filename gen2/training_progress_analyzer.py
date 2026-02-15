#!/usr/bin/env python3
"""
=============================================================================
  SLITHER.IO BOT - DEEP TRAINING PROGRESS ANALYZER v3
=============================================================================
  Full data-analytics dashboard for AI model learning progress.
  Parses train.log + training_stats.csv.

  Generates 10+ professional charts covering every metric in the CSV.

  Usage:
    python training_progress_analyzer.py
    python training_progress_analyzer.py --latest
    python training_progress_analyzer.py --no-charts   # text report only
=============================================================================
"""

import re
import os
import sys
import argparse
import warnings
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════
#  TERMINAL COLORS
# ═══════════════════════════════════════════════════════

class C:
    R = '\033[0m'
    B = '\033[1m'
    DIM = '\033[2m'
    RED = '\033[91m'
    GRN = '\033[92m'
    YEL = '\033[93m'
    BLU = '\033[94m'
    MAG = '\033[95m'
    CYN = '\033[96m'
    WHT = '\033[97m'
    BG_RED = '\033[41m'
    BG_GRN = '\033[42m'
    BG_YEL = '\033[43m'
    BG_BLU = '\033[44m'

def c(text, *colors):
    return ''.join(colors) + str(text) + C.R

def bar(value, max_val, width=30, fill_char='\u2588', empty_char='\u2591', color=C.CYN):
    if max_val <= 0:
        return empty_char * width
    ratio = min(max(value / max_val, 0), 1.0)
    filled = int(ratio * width)
    return c(fill_char * filled, color) + c(empty_char * (width - filled), C.DIM)


# ═══════════════════════════════════════════════════════
#  DATA STRUCTURES
# ═══════════════════════════════════════════════════════

STAGE_NAMES = {1: 'FOOD_VECTOR', 2: 'WALL_AVOID', 3: 'ENEMY_AVOID', 4: 'MASS_MANAGEMENT'}
STAGE_COLORS_HEX = {1: '#ff6666', 2: '#ffaa00', 3: '#00ccff', 4: '#00ff88'}
CAUSE_NAMES = ['Wall', 'SnakeCollision', 'MaxSteps', 'InvalidFrame', 'BrowserError']
CAUSE_COLORS = {'Wall': '#ff4444', 'SnakeCollision': '#ffaa00', 'MaxSteps': '#00ccff',
                'InvalidFrame': '#888888', 'BrowserError': '#ff8888'}

STAGE_PROMOTE = {
    1: ('compound', {'avg_food': 5, 'avg_steps': 50}, 200),
    2: ('avg_steps', 80, 200),
    3: ('avg_steps', 120, 200),
    4: (None, None, None),
}

@dataclass
class Episode:
    number: int
    stage: int
    stage_name: str
    reward: float
    steps: int
    food: int
    food_per_step: float
    epsilon: float
    loss: float
    cause: str
    wall_dist: Optional[int] = None
    enemy_dist: Optional[int] = None
    timestamp: Optional[datetime] = None
    style: str = ''
    lr: float = 0.0
    beta: float = 0.0
    uid: str = ''
    parent_uid: str = ''
    # New training metrics (alpha-4)
    q_mean: float = 0.0
    q_max: float = 0.0
    td_error: float = 0.0
    grad_norm: float = 0.0
    act_straight: float = 0.0
    act_gentle: float = 0.0
    act_medium: float = 0.0
    act_sharp: float = 0.0
    act_uturn: float = 0.0
    act_boost: float = 0.0
    num_agents: int = 0
    global_idx: int = 0

@dataclass
class TrainingSession:
    style: str
    start_episode: int
    end_episode: int
    episodes: List[Episode] = field(default_factory=list)
    agents: int = 1
    lr_config: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


# ═══════════════════════════════════════════════════════
#  LOG PARSER
# ═══════════════════════════════════════════════════════

EP_PATTERN = re.compile(
    r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - INFO - '
    r'Ep (\d+) \| S(\d+):(\w+) \| '
    r'Rw: ([\-\d.]+) \| St: (\d+) \| Fd: (\d+) \(([\d.]+)/st\) \| '
    r'Eps: ([\d.]+) \| L: ([\d.]+) \| '
    r'(?:Q: [\d.\-]+/[\d.\-]+ \| )?'
    r'(\w+)'
)

WALL_ENEMY_PATTERN = re.compile(r'Wall:(\d+)\s*(?:Enemy:(\d+))?')
CONFIG_STYLE_PATTERN = re.compile(r'Style: (.+)')
CONFIG_AGENTS_PATTERN = re.compile(r'Agents: (?:AUTO|(\d+))')
CONFIG_LR_PATTERN = re.compile(r'LR: ([\d.e\-]+)')


def parse_log(log_path: str) -> Tuple[List[Episode], List[TrainingSession]]:
    episodes = []
    sessions = []
    current_style = "Unknown"
    current_agents = 1
    current_lr = 0.0
    current_session_episodes = []
    session_start_time = None

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m_style = CONFIG_STYLE_PATTERN.search(line)
            if m_style:
                if current_session_episodes:
                    sessions.append(TrainingSession(
                        style=current_style,
                        start_episode=current_session_episodes[0].number,
                        end_episode=current_session_episodes[-1].number,
                        episodes=current_session_episodes,
                        agents=current_agents,
                        lr_config=current_lr,
                        start_time=session_start_time,
                        end_time=current_session_episodes[-1].timestamp,
                    ))
                current_style = m_style.group(1).strip()
                current_session_episodes = []
                session_start_time = None
                continue

            m_agents = CONFIG_AGENTS_PATTERN.search(line)
            if m_agents:
                current_agents = int(m_agents.group(1)) if m_agents.group(1) else 1
                continue

            m_lr = CONFIG_LR_PATTERN.search(line)
            if m_lr:
                current_lr = float(m_lr.group(1))
                continue

            m = EP_PATTERN.search(line)
            if m:
                ts = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
                wall_dist = None
                enemy_dist = None
                m_we = WALL_ENEMY_PATTERN.search(line)
                if m_we:
                    wall_dist = int(m_we.group(1))
                    if m_we.group(2):
                        enemy_dist = int(m_we.group(2))

                ep = Episode(
                    number=int(m.group(2)), stage=int(m.group(3)),
                    stage_name=m.group(4), reward=float(m.group(5)),
                    steps=int(m.group(6)), food=int(m.group(7)),
                    food_per_step=float(m.group(8)), epsilon=float(m.group(9)),
                    loss=float(m.group(10)), cause=m.group(11),
                    wall_dist=wall_dist, enemy_dist=enemy_dist,
                    timestamp=ts, style=current_style,
                )
                episodes.append(ep)
                current_session_episodes.append(ep)
                if session_start_time is None:
                    session_start_time = ts

    if current_session_episodes:
        sessions.append(TrainingSession(
            style=current_style,
            start_episode=current_session_episodes[0].number,
            end_episode=current_session_episodes[-1].number,
            episodes=current_session_episodes,
            agents=current_agents, lr_config=current_lr,
            start_time=session_start_time,
            end_time=current_session_episodes[-1].timestamp if current_session_episodes else None,
        ))
    return episodes, sessions


def _safe_float(parts, idx, default=0.0):
    try:
        return float(parts[idx]) if idx < len(parts) and parts[idx] else default
    except (ValueError, IndexError):
        return default

def _safe_int(parts, idx, default=0):
    try:
        return int(parts[idx]) if idx < len(parts) and parts[idx] else default
    except (ValueError, IndexError):
        return default

def parse_csv(csv_path: str) -> List[Episode]:
    episodes = []
    try:
        with open(csv_path, 'r') as f:
            header = f.readline().strip().split(',')
            has_uid = header[0] == 'UID'
            has_new_metrics = len(header) >= 22  # alpha-4 extended CSV
            for line in f:
                parts = line.strip().split(',')
                if has_uid:
                    if len(parts) < 12:
                        continue
                    ep = Episode(
                        uid=parts[0], parent_uid=parts[1],
                        number=int(parts[2]), steps=int(parts[3]),
                        reward=float(parts[4]), epsilon=float(parts[5]),
                        loss=float(parts[6]), beta=float(parts[7]),
                        lr=float(parts[8]), cause=parts[9],
                        stage=int(parts[10]), food=int(parts[11]),
                        stage_name='',
                        food_per_step=int(parts[11]) / max(int(parts[3]), 1),
                        # New metrics (backward-compatible: default 0.0 if missing)
                        q_mean=_safe_float(parts, 12),
                        q_max=_safe_float(parts, 13),
                        td_error=_safe_float(parts, 14),
                        grad_norm=_safe_float(parts, 15),
                        act_straight=_safe_float(parts, 16),
                        act_gentle=_safe_float(parts, 17),
                        act_medium=_safe_float(parts, 18),
                        act_sharp=_safe_float(parts, 19),
                        act_uturn=_safe_float(parts, 20),
                        act_boost=_safe_float(parts, 21),
                        num_agents=_safe_int(parts, 22),
                    )
                else:
                    if len(parts) < 10:
                        continue
                    ep = Episode(
                        number=int(parts[0]), steps=int(parts[1]),
                        reward=float(parts[2]), epsilon=float(parts[3]),
                        loss=float(parts[4]), beta=float(parts[5]),
                        lr=float(parts[6]), cause=parts[7],
                        stage=int(parts[8]), food=int(parts[9]),
                        stage_name='',
                        food_per_step=int(parts[9]) / max(int(parts[1]), 1),
                    )
                episodes.append(ep)
    except Exception as e:
        print(c(f"  Warning: Could not parse CSV: {e}", C.YEL))
    return episodes


def discover_uids(episodes: List[Episode]) -> List[Tuple[str, str, int, int, int]]:
    """Return [(uid, parent_uid, count, first_ep, last_ep), ...]"""
    uid_map = OrderedDict()
    for ep in episodes:
        uid = ep.uid or 'unknown'
        if uid not in uid_map:
            uid_map[uid] = {'parent': ep.parent_uid or '', 'count': 0,
                            'first': ep.number, 'last': ep.number}
        uid_map[uid]['count'] += 1
        uid_map[uid]['last'] = max(uid_map[uid]['last'], ep.number)
        uid_map[uid]['first'] = min(uid_map[uid]['first'], ep.number)
    return [(uid, info['parent'], info['count'], info['first'], info['last'])
            for uid, info in uid_map.items()]


def _print_uid_tree(uids: List[Tuple[str, str, int, int, int]]):
    """Print a tree visualization of UID lineage."""
    # Build parent→children mapping
    uid_info = {u[0]: u for u in uids}
    children = defaultdict(list)
    roots = []
    for uid, parent, count, first, last in uids:
        if parent and parent in uid_info:
            children[parent].append(uid)
        else:
            roots.append(uid)

    def _print_node(uid, prefix='', is_last=True):
        info = uid_info[uid]
        connector = '\u2514\u2500\u2500 ' if is_last else '\u251c\u2500\u2500 '
        line_prefix = prefix + connector if prefix else '    '
        _, _, count, first, last = info
        uid_short = uid.split('-', 1)[-1] if '-' in uid else uid[:8]
        print(f'{line_prefix}{c(uid_short, C.WHT)}  eps {first}-{last}  ({count} eps)')
        kids = sorted(children.get(uid, []), key=lambda u: uid_info[u][3])
        for i, child in enumerate(kids):
            ext = '    ' if is_last else '\u2502   '
            _print_node(child, prefix + (ext if prefix else '    '), i == len(kids) - 1)

    print(c('  UID Tree:', C.CYN, C.B))
    for i, root in enumerate(roots):
        _print_node(root)


def _get_ancestor_chain(target_uid: str, episodes: List[Episode]) -> set:
    """Get set of UIDs forming the ancestor chain from root to target_uid."""
    # Build uid→parent mapping
    uid_parent = {}
    for ep in episodes:
        uid = ep.uid or 'unknown'
        if uid not in uid_parent:
            uid_parent[uid] = ep.parent_uid or ''
    # Walk up from target
    chain = {target_uid}
    current = target_uid
    while current in uid_parent and uid_parent[current] and uid_parent[current] in uid_parent:
        current = uid_parent[current]
        chain.add(current)
    return chain


def _find_longest_chain(uids: List[Tuple[str, str, int, int, int]]) -> set:
    """Find the longest lineage chain (root→deepest leaf with most episodes)."""
    uid_info = {u[0]: u for u in uids}
    children = defaultdict(list)
    roots = []
    for uid, parent, count, first, last in uids:
        if parent and parent in uid_info:
            children[parent].append(uid)
        else:
            roots.append(uid)

    # DFS to find the chain with most total episodes
    def _chain_episodes(uid):
        """Return (total_eps, chain_uids) for the best path from uid to a leaf."""
        info = uid_info[uid]
        total = info[2]  # count
        best_sub_total = 0
        best_sub_chain = []
        for child in children.get(uid, []):
            sub_total, sub_chain = _chain_episodes(child)
            if sub_total > best_sub_total:
                best_sub_total = sub_total
                best_sub_chain = sub_chain
        return total + best_sub_total, [uid] + best_sub_chain

    best_total = 0
    best_chain = []
    for root in roots:
        total, chain = _chain_episodes(root)
        if total > best_total:
            best_total = total
            best_chain = chain
    return set(best_chain)


def build_lineage_chain(episodes: List[Episode]) -> List[Episode]:
    """
    Sort episodes into lineage-aware global order and assign global_idx.

    - Groups by UID, builds parent→child tree
    - Walks tree depth-first: root first, children after parent's last episode
    - Each episode gets a sequential global_idx
    - Returns the reordered list
    """
    if not episodes or not any(e.uid for e in episodes):
        # No UID data — just assign sequential indices
        for i, ep in enumerate(episodes):
            ep.global_idx = i
        return episodes

    # Group episodes by UID
    uid_episodes = defaultdict(list)
    for ep in episodes:
        uid_episodes[ep.uid or 'unknown'].append(ep)

    # Sort each group by episode number
    for uid in uid_episodes:
        uid_episodes[uid].sort(key=lambda e: e.number)

    # Build parent→children tree
    uid_parent = {}
    for uid, eps in uid_episodes.items():
        parent = eps[0].parent_uid or ''
        uid_parent[uid] = parent

    children = defaultdict(list)
    roots = []
    for uid, parent in uid_parent.items():
        if parent and parent in uid_episodes:
            children[parent].append(uid)
        else:
            roots.append(uid)

    # Sort roots by first episode number
    roots.sort(key=lambda u: uid_episodes[u][0].number)

    # DFS walk: emit episodes in lineage order
    result = []

    def _walk(uid):
        result.extend(uid_episodes[uid])
        # Sort children by their first episode number
        kids = sorted(children.get(uid, []), key=lambda u: uid_episodes[u][0].number)
        for child in kids:
            _walk(child)

    for root in roots:
        _walk(root)

    # Assign global indices
    for i, ep in enumerate(result):
        ep.global_idx = i

    return result


def get_branch_points(episodes: List[Episode]) -> List[Tuple[int, str]]:
    """Return list of (global_idx, new_uid_short) where UID changes."""
    if not episodes:
        return []
    points = []
    prev_uid = episodes[0].uid
    for ep in episodes:
        if ep.uid != prev_uid:
            uid_short = ep.uid.split('-', 1)[-1] if ep.uid and '-' in ep.uid else (ep.uid[:8] if ep.uid else '?')
            points.append((ep.global_idx, uid_short))
            prev_uid = ep.uid
    return points


# ═══════════════════════════════════════════════════════
#  STATISTICAL ANALYSIS
# ═══════════════════════════════════════════════════════

def moving_average(data, window):
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')

def exponential_ma(data, span):
    alpha = 2 / (span + 1)
    ema = np.zeros(len(data))
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema

def rolling_std(data, window):
    if len(data) < window:
        return np.zeros(len(data))
    result = np.zeros(len(data))
    for i in range(window - 1, len(data)):
        result[i] = np.std(data[i - window + 1:i + 1])
    return result

def rolling_percentile(data, window, pct):
    if len(data) < window:
        return np.full(len(data), np.nan)
    result = np.full(len(data), np.nan)
    for i in range(window - 1, len(data)):
        result[i] = np.percentile(data[i - window + 1:i + 1], pct)
    return result

def linear_trend(data):
    if len(data) < 10:
        return 0.0, 0.0
    x = np.arange(len(data))
    coeffs = np.polyfit(x, data, 1)
    slope = coeffs[0]
    predicted = np.polyval(coeffs, x)
    ss_res = np.sum((data - predicted) ** 2)
    ss_tot = np.sum((data - np.mean(data)) ** 2)
    r_sq = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    return slope, r_sq

def segment_trends(data, n_segments=4):
    if len(data) < n_segments * 10:
        return []
    seg_size = len(data) // n_segments
    results = []
    for i in range(n_segments):
        start = i * seg_size
        end = start + seg_size if i < n_segments - 1 else len(data)
        seg = data[start:end]
        slope, r2 = linear_trend(seg)
        results.append({
            'segment': i + 1, 'start_idx': start, 'end_idx': end,
            'mean': np.mean(seg), 'std': np.std(seg),
            'slope': slope, 'r2': r2,
        })
    return results

def detect_plateau(data, window=100, threshold=0.01):
    if len(data) < window * 2:
        return False, 0
    recent = data[-window:]
    earlier = data[-window * 2:-window]
    recent_mean = np.mean(recent)
    earlier_mean = np.mean(earlier)
    if earlier_mean == 0:
        return True, 0
    change_pct = abs(recent_mean - earlier_mean) / abs(earlier_mean)
    return change_pct < threshold, change_pct

def compute_percentiles(data):
    if len(data) == 0:
        return {}
    return {
        'p5': np.percentile(data, 5), 'p25': np.percentile(data, 25),
        'p50': np.percentile(data, 50), 'p75': np.percentile(data, 75),
        'p95': np.percentile(data, 95), 'mean': np.mean(data),
        'std': np.std(data), 'min': np.min(data), 'max': np.max(data),
    }


# ═══════════════════════════════════════════════════════
#  LEARNING DETECTION ENGINE
# ═══════════════════════════════════════════════════════

@dataclass
class LearningVerdict:
    is_learning: bool
    confidence: float
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    positives: List[str] = field(default_factory=list)
    recommendation: str = ''
    goal_feasibility: str = ''
    estimated_episodes_to_goal: Optional[int] = None


def assess_learning(episodes, csv_episodes, sessions):
    verdict = LearningVerdict(is_learning=False, confidence=0.0)
    score = 50

    if len(episodes) < 20:
        verdict.issues.append("Insufficient data (< 20 episodes)")
        verdict.confidence = 0.1
        return verdict

    rewards = np.array([e.reward for e in episodes])
    steps = np.array([e.steps for e in episodes])
    food = np.array([e.food for e in episodes])
    losses = np.array([e.loss for e in episodes])
    epsilons = np.array([e.epsilon for e in episodes])

    if csv_episodes:
        current_lr = csv_episodes[-1].lr
        if current_lr == 0.0:
            verdict.issues.append("CRITICAL: Learning Rate = 0.0 - Model CANNOT learn!")
            score -= 40
        elif current_lr < 1e-7:
            verdict.warnings.append(f"Learning rate extremely low: {current_lr:.2e}")
            score -= 15

    n_recent = min(len(rewards), 100)
    recent_rewards = rewards[-n_recent:]

    if len(rewards) > 200:
        first_half = rewards[:len(rewards) // 2]
        second_half = rewards[len(rewards) // 2:]
        improvement = np.mean(second_half) - np.mean(first_half)
        if improvement > 50:
            verdict.positives.append(f"Rewards improving: +{improvement:.1f}")
            score += 15
        elif improvement < -50:
            verdict.issues.append(f"Rewards DECLINING: {improvement:.1f}")
            score -= 15
        else:
            verdict.warnings.append(f"Rewards flat: change = {improvement:.1f}")
            score -= 5

    # Q-value analysis (alpha-4 metrics)
    q_means = np.array([e.q_mean for e in episodes])
    if np.any(q_means != 0):
        q_slope, q_r2 = linear_trend(q_means)
        if q_slope > 0.01 and q_r2 > 0.05:
            verdict.positives.append(f"Q-values increasing (slope={q_slope:.4f}, R²={q_r2:.3f})")
            score += 10
        elif q_slope < -0.01 and q_r2 > 0.05:
            verdict.warnings.append(f"Q-values declining (slope={q_slope:.4f})")
            score -= 5
        recent_q = q_means[-min(50, len(q_means)):]
        if np.std(recent_q) < 0.01 and np.mean(np.abs(recent_q)) > 0:
            verdict.warnings.append(f"Q-values collapsed to constant ({np.mean(recent_q):.3f})")
            score -= 10

    # Action diversity check
    act_straights = np.array([e.act_straight for e in episodes])
    if np.any(act_straights != 0):
        recent_straight = np.mean(act_straights[-min(100, len(act_straights)):])
        if recent_straight > 0.5:
            verdict.warnings.append(f"Action dominated by straight ({recent_straight*100:.0f}%) — poor policy diversity")
            score -= 5

    reward_slope, reward_r2 = linear_trend(rewards)
    if reward_slope > 0.1 and reward_r2 > 0.05:
        verdict.positives.append(f"Positive reward trend (slope={reward_slope:.4f}, R²={reward_r2:.3f})")
        score += 10
    elif reward_slope < -0.1 and reward_r2 > 0.05:
        verdict.issues.append(f"Negative reward trend (slope={reward_slope:.4f})")
        score -= 10

    steps_slope, steps_r2 = linear_trend(steps.astype(float))
    if steps_slope > 0.05 and steps_r2 > 0.03:
        verdict.positives.append(f"Episodes getting longer (slope={steps_slope:.3f}/ep)")
        score += 10

    avg_steps = np.mean(steps)
    if avg_steps < 50:
        verdict.issues.append(f"Very short episodes: avg={avg_steps:.0f} steps")
        score -= 10
    elif avg_steps > 500:
        verdict.positives.append(f"Good episode length: avg={avg_steps:.0f} steps")
        score += 5

    food_slope, food_r2 = linear_trend(food.astype(float))
    if food_slope > 0.01:
        verdict.positives.append(f"Food collection improving (slope={food_slope:.4f}/ep)")
        score += 5

    if len(losses) > 50:
        recent_loss_mean = np.mean(losses[-50:])
        if recent_loss_mean < 0.01:
            verdict.warnings.append(f"Loss near zero ({recent_loss_mean:.6f}) - possible collapse")
            score -= 10
        elif recent_loss_mean > 10:
            verdict.warnings.append(f"Loss very high ({recent_loss_mean:.2f}) - unstable")
            score -= 5

    current_eps = epsilons[-1]
    if current_eps > 0.8:
        verdict.warnings.append(f"Epsilon very high ({current_eps:.3f}) - mostly random")
        score -= 5
    elif current_eps < 0.15:
        verdict.positives.append(f"Epsilon low ({current_eps:.3f}) - exploiting policy")
        score += 5

    causes = [e.cause for e in episodes[-min(200, len(episodes)):]]
    wall_pct = sum(1 for ca in causes if ca == 'Wall') / len(causes) * 100
    if wall_pct > 60:
        verdict.issues.append(f"Wall deaths dominant ({wall_pct:.0f}%)")
        score -= 10

    is_plateau, _ = detect_plateau(rewards)
    if is_plateau:
        verdict.warnings.append("Reward in plateau (< 1% change over 200 eps)")
        score -= 5

    score = max(0, min(100, score))
    verdict.confidence = score / 100.0
    verdict.is_learning = score > 55

    if score < 25:
        verdict.goal_feasibility = "IMPOSSIBLE with current setup"
        verdict.recommendation = "Fix critical issues: LR, rewards, episode length."
    elif score < 45:
        verdict.goal_feasibility = "VERY UNLIKELY (<5%)"
        verdict.recommendation = "Major changes needed: LR, reward structure, curriculum."
    elif score < 65:
        verdict.goal_feasibility = "UNLIKELY (5-25%) without tuning"
        verdict.recommendation = "Fine-tune hyperparameters, increase training duration."
    elif score < 80:
        verdict.goal_feasibility = "POSSIBLE (25-60%)"
        verdict.recommendation = "Keep training. Monitor for sustained improvement."
    else:
        verdict.goal_feasibility = "LIKELY (>60%)"
        verdict.recommendation = "Training looks healthy. Continue and monitor."

    if reward_slope > 0.1:
        remaining = 6000 - np.mean(recent_rewards)
        if remaining > 0:
            verdict.estimated_episodes_to_goal = int(remaining / reward_slope)

    return verdict


# ═══════════════════════════════════════════════════════
#  REPORT PRINTER (TERMINAL)
# ═══════════════════════════════════════════════════════

W = 78

def sep(char='\u2500', color=C.DIM):
    print(c(char * W, color))

def header(text, color=C.CYN):
    print()
    sep('\u2550', color)
    print(c(f'  {text}', color, C.B))
    sep('\u2550', color)

def section(text, color=C.BLU):
    print()
    print(c(f'  [{text}]', color, C.B))
    sep('\u2500', C.DIM)


def print_full_report(episodes, csv_episodes, sessions, verdict):
    rewards = np.array([e.reward for e in episodes])
    steps = np.array([e.steps for e in episodes])
    food = np.array([e.food for e in episodes])
    losses = np.array([e.loss for e in episodes])
    epsilons = np.array([e.epsilon for e in episodes])

    print()
    print(c('=' * W, C.CYN, C.B))
    print(c('  SLITHER.IO BOT - DEEP TRAINING PROGRESS ANALYSIS v3', C.CYN, C.B))
    print(c('  ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'), C.DIM))
    print(c('=' * W, C.CYN, C.B))

    # === VERDICT ===
    section('LEARNING VERDICT', C.MAG)
    status = c(' LEARNING ', C.B, C.BG_GRN) if verdict.is_learning else c(' NOT LEARNING ', C.B, C.BG_RED)
    confidence_bar = bar(verdict.confidence, 1.0, 30, color=C.GRN if verdict.confidence > 0.5 else C.RED)
    print(f'  Status:     {status}')
    print(f'  Confidence: {confidence_bar} {verdict.confidence * 100:.0f}%')
    print(f'  Goal:       {c(verdict.goal_feasibility, C.YEL, C.B)}')
    if verdict.estimated_episodes_to_goal:
        print(f'  ETA:        ~{verdict.estimated_episodes_to_goal:,} episodes')

    for label, items, color in [('CRITICAL ISSUES:', verdict.issues, C.RED),
                                 ('WARNINGS:', verdict.warnings, C.YEL),
                                 ('POSITIVE SIGNALS:', verdict.positives, C.GRN)]:
        if items:
            print()
            sym = '\u2717' if color == C.RED else '\u26a0' if color == C.YEL else '\u2713'
            print(c(f'  {label}', color, C.B))
            for item in items:
                print(c(f'    {sym} {item}', color))

    # === OVERVIEW ===
    section('TRAINING OVERVIEW')
    total_episodes = len(episodes)
    hours = 0
    if episodes[0].timestamp and episodes[-1].timestamp:
        hours = (episodes[-1].timestamp - episodes[0].timestamp).total_seconds() / 3600

    print(f'  Total Episodes:    {c(total_episodes, C.WHT, C.B)}')
    print(f'  Episode Range:     {episodes[0].number} -> {episodes[-1].number}')
    print(f'  Training Time:     {hours:.1f} hours')
    print(f'  Sessions:          {len(sessions)}')

    # === CURRICULUM STAGE ANALYSIS ===
    stage_episodes = defaultdict(list)
    for e in episodes:
        stage_episodes[e.stage].append(e)

    if stage_episodes:
        section('CURRICULUM STAGE ANALYSIS')
        current_stage = max(stage_episodes.keys())
        print(f'  Current Stage: {c(f"S{current_stage}", C.CYN, C.B)} ({c(STAGE_NAMES.get(current_stage, "?"), C.CYN)})')
        print()
        print(f'  {"Stage":<6} {"Name":<17} {"Eps":<8} {"Avg Rw":<10} {"Avg St":<10} '
              f'{"Avg Fd":<10} {"Wall%":<7} {"Snake%":<8} {"MaxSt%":<8}')
        sep('\u2500', C.DIM)

        for sn in sorted(stage_episodes.keys()):
            eps_list = stage_episodes[sn]
            s_causes = [e.cause for e in eps_list]
            n = len(eps_list)
            wp = sum(1 for x in s_causes if x == 'Wall') / n * 100
            sp = sum(1 for x in s_causes if x == 'SnakeCollision') / n * 100
            mp = sum(1 for x in s_causes if x == 'MaxSteps') / n * 100
            name = STAGE_NAMES.get(sn, '?')
            color = C.CYN if sn == current_stage else C.DIM
            print(f'  {c(f"S{sn}", color):<14} {c(name[:16], color):<25} '
                  f'{n:<8} {np.mean([e.reward for e in eps_list]):>8.1f}  '
                  f'{np.mean([e.steps for e in eps_list]):>8.1f}  '
                  f'{np.mean([e.food for e in eps_list]):>8.1f}  '
                  f'{wp:>5.1f}% {sp:>6.1f}% {mp:>6.1f}%')

        # Promotion progress
        metric_name, threshold, window = STAGE_PROMOTE.get(current_stage, (None, None, None))
        if metric_name and threshold:
            cur_eps = stage_episodes[current_stage]
            recent = cur_eps[-min(len(cur_eps), window or 100):]
            print()

            if metric_name == 'compound':
                # threshold is a dict of conditions
                conditions = threshold
                min_pct = 100
                for cond_name, cond_val in conditions.items():
                    if cond_name == 'avg_food':
                        val = np.mean([e.food for e in recent])
                    elif cond_name == 'avg_steps':
                        val = np.mean([e.steps for e in recent])
                    else:
                        continue
                    pct = min(val / cond_val * 100, 100)
                    min_pct = min(min_pct, pct)
                    ok = C.GRN if pct >= 100 else C.YEL
                    print(f'  Promote [{cond_name}]: {c(f"{val:.1f}", ok)} / {cond_val} '
                          f'({len(recent)}/{window} window)  {bar(pct, 100, 30)} {pct:.0f}%')
                print(f'  Overall:   {bar(min_pct, 100, 40)} {min_pct:.0f}%')
            else:
                if metric_name == 'avg_food':
                    val = np.mean([e.food for e in recent])
                else:
                    val = np.mean([e.steps for e in recent])
                pct = min(val / threshold * 100, 100)
                print(f'  Promotion: {c(metric_name, C.YEL)} = {c(f"{val:.1f}", C.WHT, C.B)} / {threshold} '
                      f'({len(recent)}/{window} window)')
                print(f'  Progress:  {bar(pct, 100, 40)} {pct:.0f}%')

            if current_stage == 2:
                wd = sum(1 for e in recent if e.cause == 'Wall') / len(recent) * 100
                wc = C.GRN if wd < 3 else C.RED
                print(f'  Wall death: {c(f"{wd:.1f}%", wc)} (need < 3%)')

    # === REWARD ANALYSIS ===
    section('REWARD ANALYSIS')
    rp = compute_percentiles(rewards)
    print(f'  Mean:   {rp["mean"]:>10.2f}   Std:    {rp["std"]:>10.2f}')
    print(f'  Median: {rp["p50"]:>10.2f}   Min:    {rp["min"]:>10.2f}')
    print(f'  P95:    {rp["p95"]:>10.2f}   Max:    {rp["max"]:>10.2f}')
    print()
    print(f'  {"Window":<10} {"Mean":<11} {"Std":<11} {"Best":<11} {"Trend":<12}')
    sep('\u2500', C.DIM)
    for w in [50, 100, 500, 1000]:
        if len(rewards) >= w:
            wr = rewards[-w:]
            sl, r2 = linear_trend(wr)
            t = c("\u2197", C.GRN) if sl > 0.1 else (c("\u2198", C.RED) if sl < -0.1 else c("\u2192", C.DIM))
            print(f'  Last {w:<4} {np.mean(wr):>9.2f}  {np.std(wr):>9.2f}  {np.max(wr):>9.2f}  {t} ({sl:+.3f})')

    # === SURVIVAL ===
    section('SURVIVAL ANALYSIS')
    sp_p = compute_percentiles(steps.astype(float))
    STEP_DUR = 2.0
    print(f'  Mean Steps: {sp_p["mean"]:>8.1f}   Max: {sp_p["max"]:>8.0f}   P95: {sp_p["p95"]:>8.0f}')
    avg_surv_s = sp_p["mean"] * STEP_DUR
    avg_surv_m = avg_surv_s / 60
    print(f'  Est. avg survival: {c(f"{avg_surv_s:.0f}s ({avg_surv_m:.1f}min)", C.YEL)}')
    goal_pct = sp_p["max"] * STEP_DUR / 3600 * 100
    print(f'  Goal progress:     {bar(goal_pct, 100, 30)} {goal_pct:.1f}%')

    # === FOOD ===
    section('FOOD COLLECTION')
    fp = compute_percentiles(food.astype(float))
    fps = compute_percentiles(np.array([e.food_per_step for e in episodes]))
    print(f'  Mean Food/Ep:   {fp["mean"]:>8.1f}   Max: {int(fp["max"])}')
    print(f'  Mean Food/Step: {fps["mean"]:>8.4f}   Max: {fps["max"]:.4f}')
    food_pct = int(fp["max"]) / 6000 * 100
    print(f'  Goal progress:  {bar(food_pct, 100, 30)} {food_pct:.1f}%')

    # === DEATH ANALYSIS ===
    section('DEATH CAUSE ANALYSIS')
    cause_counts = defaultdict(int)
    for e in episodes:
        cause_counts[e.cause] += 1
    total = len(episodes)
    for cause, count in sorted(cause_counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / total * 100
        cc = C.RED if cause == 'Wall' else C.YEL if cause == 'SnakeCollision' else C.CYN if cause == 'MaxSteps' else C.DIM
        print(f'  {cause:<18} {count:>5} ({pct:>5.1f}%)  {bar(pct, 100, 25, color=cc)}')

    # === TRAINING HEALTH ===
    section('TRAINING HEALTH')
    current_eps = epsilons[-1]
    eps_color = C.RED if current_eps > 0.5 else C.YEL if current_eps > 0.2 else C.GRN
    print(f'  Epsilon:    {c(f"{current_eps:.4f}", eps_color)}  {bar(1 - current_eps, 1.0, 20, color=C.GRN)} exploitation')

    if csv_episodes:
        cur_lr = csv_episodes[-1].lr
        lr_c = C.RED if cur_lr == 0 else C.YEL if cur_lr < 1e-6 else C.GRN
        print(f'  Learn Rate: {c(f"{cur_lr:.2e}", lr_c)}')
        print(f'  Beta (PER): {csv_episodes[-1].beta:.4f}')

    print(f'  Last Loss:  {losses[-1]:.4f}')
    if len(losses) > 50:
        print(f'  Avg Loss (50): {np.mean(losses[-50:]):.4f}')

    # === Q-VALUE & GRADIENT HEALTH ===
    q_means = np.array([e.q_mean for e in episodes])
    q_maxes = np.array([e.q_max for e in episodes])
    td_errors = np.array([e.td_error for e in episodes])
    grad_norms = np.array([e.grad_norm for e in episodes])

    has_q = np.any(q_means != 0) or np.any(q_maxes != 0)
    if has_q:
        section('Q-VALUE & GRADIENT ANALYSIS')
        n_recent = min(50, len(q_means))
        print(f'  Q-value (mean):  last={q_means[-1]:.4f}  avg50={np.mean(q_means[-n_recent:]):.4f}  range=[{np.min(q_means):.2f}, {np.max(q_means):.2f}]')
        print(f'  Q-value (max):   last={q_maxes[-1]:.4f}  avg50={np.mean(q_maxes[-n_recent:]):.4f}  range=[{np.min(q_maxes):.2f}, {np.max(q_maxes):.2f}]')
        print(f'  TD-error (mean): last={td_errors[-1]:.4f}  avg50={np.mean(td_errors[-n_recent:]):.4f}')
        print(f'  Grad norm:       last={grad_norms[-1]:.4f}  avg50={np.mean(grad_norms[-n_recent:]):.4f}  max={np.max(grad_norms):.2f}')

        q_slope, q_r2 = linear_trend(q_means)
        q_trend = c("\u2197 UP", C.GRN) if q_slope > 0.01 else (c("\u2198 DOWN", C.RED) if q_slope < -0.01 else c("\u2192 FLAT", C.YEL))
        print(f'  Q trend:         {q_trend} (slope={q_slope:.4f}, R²={q_r2:.3f})')

    # === ACTION DISTRIBUTION ===
    act_data = {
        'Straight': np.array([e.act_straight for e in episodes]),
        'Gentle': np.array([e.act_gentle for e in episodes]),
        'Medium': np.array([e.act_medium for e in episodes]),
        'Sharp': np.array([e.act_sharp for e in episodes]),
        'U-turn': np.array([e.act_uturn for e in episodes]),
        'Boost': np.array([e.act_boost for e in episodes]),
    }
    has_act = any(np.any(v != 0) for v in act_data.values())
    if has_act:
        section('ACTION DISTRIBUTION (Last 100 eps)')
        n_act = min(100, len(episodes))
        for name, arr in act_data.items():
            avg_pct = np.mean(arr[-n_act:]) * 100
            color = C.RED if name == 'Straight' and avg_pct > 50 else C.CYN
            print(f'  {name:<10} {bar(avg_pct, 100, 25, color=color)} {avg_pct:>5.1f}%')

    # === SEGMENTED TREND ===
    section('SEGMENTED TREND ANALYSIS')
    segments = segment_trends(rewards, 4)
    if segments:
        print(f'  {"Quarter":<10} {"Mean Rw":<12} {"Std":<10} {"Trend":<14} {"R\u00b2":<8}')
        sep('\u2500', C.DIM)
        for seg in segments:
            t = c("\u2197 UP", C.GRN) if seg['slope'] > 0.5 else (c("\u2198 DOWN", C.RED) if seg['slope'] < -0.5 else c("\u2192 FLAT", C.YEL))
            print(f'  Q{seg["segment"]}        {seg["mean"]:>10.1f}  {seg["std"]:>8.1f}  {t:<22} {seg["r2"]:.4f}')

    # === RECOMMENDATIONS ===
    section('RECOMMENDATIONS', C.MAG)
    print(c(f'  {verdict.recommendation}', C.WHT))
    print()
    recs = generate_specific_recommendations(episodes, csv_episodes, sessions, verdict)
    for i, rec in enumerate(recs, 1):
        print(c(f'  {i}. {rec}', C.CYN))
    sep('\u2550', C.CYN)
    print()


def generate_specific_recommendations(episodes, csv_episodes, sessions, verdict):
    recs = []
    if csv_episodes and csv_episodes[-1].lr == 0:
        recs.append("URGENT: LR=0. Reset learning rate or set min_lr floor.")
    if episodes[-1].epsilon > 0.3:
        recs.append(f"Epsilon {episodes[-1].epsilon:.3f} still high. Consider faster decay.")
    steps_arr = np.array([e.steps for e in episodes[-100:]])
    if np.mean(steps_arr) < 100:
        recs.append("Episodes too short. Reduce death penalties or add survival bonus.")
    causes = [e.cause for e in episodes[-200:]]
    wall_pct = sum(1 for x in causes if x == 'Wall') / len(causes)
    if wall_pct > 0.5:
        recs.append(f"Wall deaths {wall_pct*100:.0f}%. Increase wall_proximity_penalty.")
    if np.mean([e.reward for e in episodes]) < 0:
        recs.append("Average reward negative. Reduce penalties or boost food_reward.")
    if not recs:
        recs.append("No critical issues. Continue training.")
    return recs


# ═══════════════════════════════════════════════════════
#  CHART GENERATION (PROFESSIONAL ANALYTICS)
# ═══════════════════════════════════════════════════════

def _setup_matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import matplotlib.ticker as mticker
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': '#0d1117',
        'axes.facecolor': '#161b22',
        'axes.edgecolor': '#30363d',
        'axes.labelcolor': '#c9d1d9',
        'xtick.color': '#8b949e',
        'ytick.color': '#8b949e',
        'text.color': '#c9d1d9',
        'grid.color': '#21262d',
        'grid.alpha': 0.6,
        'legend.facecolor': '#161b22',
        'legend.edgecolor': '#30363d',
        'font.size': 10,
    })
    return plt, GridSpec, mticker


def _add_stage_bands(ax, ep_nums, stages, branch_points=None):
    """Add colored background bands per curriculum stage + branch markers."""
    prev = stages[0]
    start = ep_nums[0]
    for i in range(1, len(stages)):
        if stages[i] != prev or i == len(stages) - 1:
            end = ep_nums[i]
            col = STAGE_COLORS_HEX.get(prev, '#888')
            ax.axvspan(start, end, alpha=0.06, color=col)
            start = ep_nums[i]
            prev = stages[i]
    if branch_points:
        _add_branch_markers(ax, branch_points)


def _add_branch_markers(ax, branch_points, y_top=None):
    """Add vertical dotted lines and UID labels at branch points."""
    for gidx, uid_short in branch_points:
        ax.axvline(gidx, color='#8b949e', linestyle=':', linewidth=0.8, alpha=0.6)
        if y_top is not None:
            ax.text(gidx, y_top, uid_short, fontsize=6, color='#8b949e',
                    rotation=90, va='top', ha='right', alpha=0.7)


def _setup_ep_axis(ax, ep_nums, orig_ep_nums, label='Episode (global)'):
    """Set x-axis label and add custom tick formatter showing original episode numbers."""
    ax.set_xlabel(label)
    if len(ep_nums) > 0 and len(orig_ep_nums) == len(ep_nums):
        # Build mapping: global_idx → original ep number for ticks
        import matplotlib.ticker as mticker_local

        idx_to_orig = dict(zip(ep_nums, orig_ep_nums))

        def _fmt(x, pos):
            if int(x) in idx_to_orig:
                return str(idx_to_orig[int(x)])
            return str(int(x))

        ax.xaxis.set_major_formatter(mticker_local.FuncFormatter(_fmt))


def generate_charts(episodes, csv_episodes, sessions, verdict, output_dir):
    try:
        plt, GridSpec, mticker = _setup_matplotlib()
    except ImportError:
        print(c("  matplotlib not installed, skipping charts.", C.YEL))
        return

    # Extract arrays — use global_idx for x-axis
    ep_nums = np.array([e.global_idx for e in episodes])
    orig_ep_nums = np.array([e.number for e in episodes])
    branch_points = get_branch_points(episodes)
    rewards = np.array([e.reward for e in episodes])
    steps = np.array([e.steps for e in episodes], dtype=float)
    food = np.array([e.food for e in episodes], dtype=float)
    losses = np.array([e.loss for e in episodes])
    epsilons = np.array([e.epsilon for e in episodes])
    stages_arr = np.array([e.stage for e in episodes])
    causes_arr = np.array([e.cause for e in episodes])
    food_per_step = np.array([e.food_per_step for e in episodes])

    # CSV-only fields
    lr_arr = np.array([e.lr for e in episodes]) if episodes[0].lr > 0 or any(e.lr > 0 for e in episodes) else None
    beta_arr = np.array([e.beta for e in episodes]) if any(e.beta > 0 for e in episodes) else None

    N = len(episodes)
    sma_w = min(50, N // 3) if N > 30 else max(3, N // 3)

    def _save(fig, name):
        path = os.path.join(output_dir, name)
        fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close(fig)
        print(c(f'  Chart saved: {name}', C.GRN))

    # ──────────────────────────────────────────────────
    # CHART 1: MAIN DASHBOARD (6 panels)
    # ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 16))
    fig.suptitle('SLITHER.IO BOT - TRAINING DASHBOARD', fontsize=20, fontweight='bold', y=0.98)
    gs = GridSpec(3, 2, figure=fig, hspace=0.32, wspace=0.25)

    # 1a. Reward with Bollinger bands
    ax = fig.add_subplot(gs[0, 0])
    _add_stage_bands(ax, ep_nums, stages_arr, branch_points)
    ax.plot(ep_nums, rewards, alpha=0.12, color='#484f58', linewidth=0.5)
    if N > sma_w:
        sma = moving_average(rewards, sma_w)
        x_sma = ep_nums[sma_w - 1:]
        rstd = rolling_std(rewards, sma_w)[sma_w - 1:]
        ax.fill_between(x_sma, sma - 2 * rstd, sma + 2 * rstd, alpha=0.12, color='#58a6ff')
        ax.plot(x_sma, sma, color='#58a6ff', linewidth=2, label=f'SMA-{sma_w}')
        ema = exponential_ma(rewards, sma_w)
        ax.plot(ep_nums, ema, color='#f0883e', linewidth=1.5, alpha=0.8, label=f'EMA-{sma_w}')
    ax.axhline(0, color='#484f58', linestyle=':', linewidth=0.8)
    ax.set_title('Reward (Bollinger Bands)', fontweight='bold')
    ax.set_xlabel('Episode (global)')
    ax.set_ylabel('Reward')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True)

    # 1b. Steps with percentile bands
    ax = fig.add_subplot(gs[0, 1])
    _add_stage_bands(ax, ep_nums, stages_arr, branch_points)
    ax.plot(ep_nums, steps, alpha=0.12, color='#484f58', linewidth=0.5)
    if N > sma_w:
        p25 = rolling_percentile(steps, sma_w, 25)
        p50 = rolling_percentile(steps, sma_w, 50)
        p75 = rolling_percentile(steps, sma_w, 75)
        p95 = rolling_percentile(steps, sma_w, 95)
        mask = ~np.isnan(p50)
        ax.fill_between(ep_nums[mask], p25[mask], p75[mask], alpha=0.2, color='#3fb950')
        ax.plot(ep_nums[mask], p50[mask], color='#3fb950', linewidth=2, label='Median')
        ax.plot(ep_nums[mask], p95[mask], color='#3fb950', linewidth=1, linestyle='--', alpha=0.6, label='P95')
    ax.set_title('Survival Duration (Percentile Bands)', fontweight='bold')
    ax.set_xlabel('Episode (global)')
    ax.set_ylabel('Steps')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True)

    # 1c. Food collection
    ax = fig.add_subplot(gs[1, 0])
    _add_stage_bands(ax, ep_nums, stages_arr, branch_points)
    ax.plot(ep_nums, food, alpha=0.12, color='#484f58', linewidth=0.5)
    if N > sma_w:
        sma_f = moving_average(food, sma_w)
        ax.plot(ep_nums[sma_w - 1:], sma_f, color='#d29922', linewidth=2, label=f'SMA-{sma_w}')
    ax.set_title('Food Collected', fontweight='bold')
    ax.set_xlabel('Episode (global)')
    ax.set_ylabel('Food')
    ax.legend(fontsize=8)
    ax.grid(True)

    # 1d. Loss (log scale)
    ax = fig.add_subplot(gs[1, 1])
    valid_loss = losses > 0
    if np.any(valid_loss):
        ax.plot(ep_nums[valid_loss], losses[valid_loss], alpha=0.15, color='#484f58', linewidth=0.5)
        if np.sum(valid_loss) > sma_w:
            sma_l = moving_average(losses[valid_loss], sma_w)
            ax.plot(ep_nums[valid_loss][sma_w - 1:], sma_l, color='#f85149', linewidth=2, label=f'SMA-{sma_w}')
        ax.set_yscale('log')
    ax.set_title('Training Loss (log scale)', fontweight='bold')
    ax.set_xlabel('Episode (global)')
    ax.set_ylabel('Loss')
    ax.legend(fontsize=8)
    ax.grid(True)

    # 1e. Epsilon + LR dual axis
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(ep_nums, epsilons, color='#f0883e', linewidth=1.5, label='Epsilon')
    ax.set_ylabel('Epsilon', color='#f0883e')
    ax.set_xlabel('Episode (global)')
    ax.set_title('Epsilon & Learning Rate', fontweight='bold')
    if lr_arr is not None and np.any(lr_arr > 0):
        ax2 = ax.twinx()
        ax2.plot(ep_nums, lr_arr, color='#a5d6ff', linewidth=1, alpha=0.7, label='LR')
        ax2.set_ylabel('LR', color='#a5d6ff')
        ax2.set_yscale('log')
        ax2.legend(fontsize=8, loc='center right')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True)

    # 1f. Death cause stacked area
    ax = fig.add_subplot(gs[2, 1])
    w_size = max(50, N // 30)
    if N > w_size:
        x_pts = []
        series = {cn: [] for cn in CAUSE_NAMES}
        for i in range(0, N - w_size, max(1, w_size // 5)):
            window = causes_arr[i:i + w_size]
            x_pts.append(ep_nums[i + w_size // 2])
            total = len(window)
            for cn in CAUSE_NAMES:
                series[cn].append(np.sum(window == cn) / total * 100)

        bottom = np.zeros(len(x_pts))
        for cn in CAUSE_NAMES:
            vals = np.array(series[cn])
            if np.any(vals > 0):
                ax.fill_between(x_pts, bottom, bottom + vals, alpha=0.7,
                               color=CAUSE_COLORS.get(cn, '#888'), label=cn)
                bottom += vals
        ax.set_ylim(0, 100)
        ax.set_ylabel('% of Deaths')
    ax.set_title('Death Causes (Stacked Area)', fontweight='bold')
    ax.set_xlabel('Episode (global)')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True)

    _save(fig, 'chart_01_dashboard.png')

    # ──────────────────────────────────────────────────
    # CHART 2: STAGE PROGRESSION TIMELINE
    # ──────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(20, 10), sharex=True)
    fig.suptitle('CURRICULUM STAGE PROGRESSION', fontsize=16, fontweight='bold')

    for ax_i, (data, label, color) in enumerate([
        (steps, 'Steps', '#3fb950'), (food, 'Food', '#d29922'), (rewards, 'Reward', '#58a6ff')
    ]):
        ax = axes[ax_i]
        _add_stage_bands(ax, ep_nums, stages_arr, branch_points)
        ax.plot(ep_nums, data, alpha=0.12, color='#484f58', linewidth=0.5)
        if N > 20:
            sma_d = moving_average(data, min(20, N // 3))
            ax.plot(ep_nums[len(ep_nums) - len(sma_d):], sma_d, color=color, linewidth=2)
        ax.set_ylabel(label)
        ax.grid(True)

        # Stage labels
        prev_s = stages_arr[0]
        start_i = 0
        for i in range(1, len(stages_arr)):
            if stages_arr[i] != prev_s or i == len(stages_arr) - 1:
                mid = (ep_nums[start_i] + ep_nums[i]) / 2
                col = STAGE_COLORS_HEX.get(prev_s, '#888')
                ax.text(mid, ax.get_ylim()[1] * 0.9, f'S{prev_s}',
                       ha='center', fontsize=9, color=col, fontweight='bold')
                start_i = i
                prev_s = stages_arr[i]

    axes[-1].set_xlabel('Episode')
    _save(fig, 'chart_02_stage_progression.png')

    # ──────────────────────────────────────────────────
    # CHART 3: PER-STAGE BOX + VIOLIN PLOTS
    # ──────────────────────────────────────────────────
    unique_stages = sorted(set(stages_arr))
    if len(unique_stages) >= 1:
        fig, axes = plt.subplots(1, 4, figsize=(22, 6))
        fig.suptitle('PER-STAGE DISTRIBUTIONS', fontsize=16, fontweight='bold')

        for ax_i, (metric_name, get_fn) in enumerate([
            ('Reward', lambda e: e.reward),
            ('Steps', lambda e: e.steps),
            ('Food', lambda e: e.food),
            ('Food/Step', lambda e: e.food_per_step),
        ]):
            ax = axes[ax_i]
            data_per_stage = []
            labels = []
            colors = []
            for s in unique_stages:
                vals = [get_fn(e) for e in episodes if e.stage == s]
                data_per_stage.append(vals)
                labels.append(f'S{s}\n{STAGE_NAMES.get(s, "?")[:8]}')
                colors.append(STAGE_COLORS_HEX.get(s, '#888'))

            parts = ax.violinplot(data_per_stage, positions=range(len(unique_stages)),
                                 showmeans=True, showmedians=True)
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.4)
            for key in ['cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes']:
                if key in parts:
                    parts[key].set_color('#c9d1d9')

            # Overlay box plot
            bp = ax.boxplot(data_per_stage, positions=range(len(unique_stages)),
                           widths=0.15, patch_artist=True, showfliers=False)
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(colors[i])
                patch.set_alpha(0.6)
            for element in ['whiskers', 'caps', 'medians']:
                for item in bp[element]:
                    item.set_color('#c9d1d9')

            ax.set_xticks(range(len(unique_stages)))
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_title(metric_name, fontweight='bold')
            ax.grid(True, axis='y')

        plt.tight_layout()
        _save(fig, 'chart_03_stage_distributions.png')

    # ──────────────────────────────────────────────────
    # CHART 4: HYPERPARAMETER TRACKING
    # ──────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    fig.suptitle('HYPERPARAMETER & TRAINING DYNAMICS', fontsize=16, fontweight='bold')

    # Epsilon decay curve
    ax = axes[0, 0]
    ax.plot(ep_nums, epsilons, color='#f0883e', linewidth=1.5)
    ax.axhline(0.08, color='#3fb950', linestyle='--', alpha=0.5, label='eps_end=0.08')
    ax.set_title('Epsilon Decay')
    ax.set_ylabel('Epsilon')
    ax.set_xlabel('Episode (global)')
    ax.legend(fontsize=8)
    ax.grid(True)

    # Learning rate
    ax = axes[0, 1]
    if lr_arr is not None:
        ax.plot(ep_nums, lr_arr, color='#a5d6ff', linewidth=1.5)
        ax.set_yscale('log')
    ax.set_title('Learning Rate Schedule')
    ax.set_ylabel('LR (log)')
    ax.set_xlabel('Episode (global)')
    ax.grid(True)

    # Beta (PER importance sampling)
    ax = axes[1, 0]
    if beta_arr is not None:
        ax.plot(ep_nums, beta_arr, color='#d2a8ff', linewidth=1.5)
    ax.set_title('PER Beta (Importance Sampling)')
    ax.set_ylabel('Beta')
    ax.set_xlabel('Episode (global)')
    ax.grid(True)

    # Loss with EMA
    ax = axes[1, 1]
    if np.any(losses > 0):
        valid = losses > 0
        ax.plot(ep_nums[valid], losses[valid], alpha=0.1, color='#484f58', linewidth=0.5)
        if np.sum(valid) > 20:
            ema_l = exponential_ma(losses[valid], 50)
            ax.plot(ep_nums[valid], ema_l, color='#f85149', linewidth=2, label='EMA-50')
        ax.set_yscale('log')
    ax.set_title('Loss (EMA + Raw)')
    ax.set_ylabel('Loss (log)')
    ax.set_xlabel('Episode (global)')
    ax.legend(fontsize=8)
    ax.grid(True)

    plt.tight_layout()
    _save(fig, 'chart_04_hyperparameters.png')

    # ──────────────────────────────────────────────────
    # CHART 5a: CORRELATION SCATTER MATRIX (expanded)
    # ──────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 3, figsize=(22, 24))
    fig.suptitle('METRIC CORRELATIONS (Scatter Matrix)', fontsize=16, fontweight='bold')

    # Prepare LR/Beta arrays for correlations (replace None with zeros)
    lr_corr = lr_arr if lr_arr is not None else np.zeros(N)
    beta_corr = beta_arr if beta_arr is not None else np.zeros(N)

    pairs = [
        (steps, rewards, 'Steps', 'Reward'),
        (food, rewards, 'Food', 'Reward'),
        (steps, food, 'Steps', 'Food'),
        (food_per_step, rewards, 'Food/Step', 'Reward'),
        (losses, rewards, 'Loss', 'Reward'),
        (epsilons, rewards, 'Epsilon', 'Reward'),
        (losses, steps, 'Loss', 'Steps'),
        (epsilons, steps, 'Epsilon', 'Steps'),
        (lr_corr, rewards, 'LR', 'Reward'),
        (lr_corr, losses, 'LR', 'Loss'),
        (food_per_step, steps, 'Food/Step', 'Steps'),
        (epsilons, food, 'Epsilon', 'Food'),
    ]

    for idx, (xd, yd, xl, yl) in enumerate(pairs):
        ax = axes[idx // 3, idx % 3]
        # Color by stage
        for s in unique_stages:
            mask = stages_arr == s
            ax.scatter(xd[mask], yd[mask], alpha=0.15, s=8,
                      color=STAGE_COLORS_HEX.get(s, '#888'), label=f'S{s}')
        # Trend line + R²
        valid = np.isfinite(xd) & np.isfinite(yd) & (xd != 0) if xl in ('Loss', 'LR') else np.isfinite(xd) & np.isfinite(yd)
        if np.sum(valid) > 10:
            z = np.polyfit(xd[valid], yd[valid], 1)
            p = np.poly1d(z)
            x_line = np.linspace(np.min(xd[valid]), np.max(xd[valid]), 50)
            ax.plot(x_line, p(x_line), color='#f0883e', linewidth=1.5, alpha=0.8, linestyle='--')
            ss_res = np.sum((yd[valid] - p(xd[valid])) ** 2)
            ss_tot = np.sum((yd[valid] - np.mean(yd[valid])) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            corr = np.corrcoef(xd[valid], yd[valid])[0, 1]
            ax.text(0.02, 0.98, f'r={corr:.3f}  R²={r2:.3f}',
                   transform=ax.transAxes, fontsize=8, va='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#0d1117', edgecolor='#30363d', alpha=0.9),
                   color='#f0883e')
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f'{xl} vs {yl}', fontweight='bold')
        ax.grid(True)
        if idx == 0:
            ax.legend(fontsize=7, markerscale=3)

    plt.tight_layout()
    _save(fig, 'chart_05_correlations.png')

    # ──────────────────────────────────────────────────
    # CHART 5b: CORRELATION HEATMAP MATRIX
    # ──────────────────────────────────────────────────
    q_arr_corr = np.array([e.q_mean for e in episodes])
    gn_arr_corr = np.array([e.grad_norm for e in episodes])
    metric_names = ['Reward', 'Steps', 'Food', 'Food/Step', 'Loss', 'Epsilon', 'LR', 'Beta', 'Q Mean', 'Grad Norm']
    metric_data = [rewards, steps.astype(float), food.astype(float), food_per_step,
                   losses, epsilons, lr_corr, beta_corr, q_arr_corr, gn_arr_corr]

    n_metrics = len(metric_names)
    corr_matrix = np.zeros((n_metrics, n_metrics))
    for i in range(n_metrics):
        for j in range(n_metrics):
            valid = np.isfinite(metric_data[i]) & np.isfinite(metric_data[j])
            if np.sum(valid) > 5 and np.std(metric_data[i][valid]) > 0 and np.std(metric_data[j][valid]) > 0:
                corr_matrix[i, j] = np.corrcoef(metric_data[i][valid], metric_data[j][valid])[0, 1]
            else:
                corr_matrix[i, j] = 0.0

    fig, axes = plt.subplots(1, 2, figsize=(22, 9), gridspec_kw={'width_ratios': [1.2, 1]})
    fig.suptitle('CORRELATION ANALYSIS', fontsize=16, fontweight='bold')

    # Left: Heatmap
    ax = axes[0]
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(n_metrics))
    ax.set_yticks(range(n_metrics))
    ax.set_xticklabels(metric_names, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(metric_names, fontsize=10)
    # Annotate cells
    for i in range(n_metrics):
        for j in range(n_metrics):
            val = corr_matrix[i, j]
            color = '#0d1117' if abs(val) > 0.5 else '#c9d1d9'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=9,
                   fontweight='bold' if abs(val) > 0.3 else 'normal', color=color)
    fig.colorbar(im, ax=ax, shrink=0.8, label='Pearson r')
    ax.set_title('Pearson Correlation Matrix', fontweight='bold', fontsize=12)

    # Right: Top correlations ranked bar chart
    ax = axes[1]
    corr_pairs = []
    for i in range(n_metrics):
        for j in range(i + 1, n_metrics):
            corr_pairs.append((metric_names[i], metric_names[j], corr_matrix[i, j]))
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    top_n = min(12, len(corr_pairs))
    top_pairs = corr_pairs[:top_n]
    labels = [f'{a} × {b}' for a, b, _ in top_pairs]
    values = [v for _, _, v in top_pairs]
    colors = ['#3fb950' if v > 0 else '#f85149' for v in values]
    y_pos = np.arange(top_n)
    ax.barh(y_pos, values, color=colors, alpha=0.8, edgecolor='#30363d')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Pearson r', fontsize=10)
    ax.set_title('Strongest Correlations (ranked)', fontweight='bold', fontsize=12)
    ax.axvline(0, color='#484f58', linewidth=1)
    ax.set_xlim(-1.05, 1.05)
    ax.grid(True, axis='x')
    ax.invert_yaxis()
    # Add value labels on bars
    for i, v in enumerate(values):
        ax.text(v + (0.03 if v >= 0 else -0.03), i, f'{v:.3f}',
               va='center', ha='left' if v >= 0 else 'right', fontsize=8, color='#c9d1d9')

    plt.tight_layout()
    _save(fig, 'chart_05b_correlation_heatmap.png')

    # ──────────────────────────────────────────────────
    # CHART 6: ROLLING PERFORMANCE BANDS
    # ──────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(20, 14), sharex=True)
    fig.suptitle('ROLLING PERFORMANCE PERCENTILE BANDS', fontsize=16, fontweight='bold')

    band_w = min(100, N // 4) if N > 40 else max(5, N // 3)

    for ax_i, (data, label, base_color) in enumerate([
        (rewards, 'Reward', '#58a6ff'),
        (steps, 'Steps', '#3fb950'),
        (food, 'Food', '#d29922'),
    ]):
        ax = axes[ax_i]
        _add_stage_bands(ax, ep_nums, stages_arr, branch_points)
        ax.plot(ep_nums, data, alpha=0.08, color='#484f58', linewidth=0.5)

        if N > band_w:
            p10 = rolling_percentile(data, band_w, 10)
            p25 = rolling_percentile(data, band_w, 25)
            p50 = rolling_percentile(data, band_w, 50)
            p75 = rolling_percentile(data, band_w, 75)
            p90 = rolling_percentile(data, band_w, 90)
            mask = ~np.isnan(p50)

            ax.fill_between(ep_nums[mask], p10[mask], p90[mask], alpha=0.08, color=base_color, label='P10-P90')
            ax.fill_between(ep_nums[mask], p25[mask], p75[mask], alpha=0.2, color=base_color, label='P25-P75')
            ax.plot(ep_nums[mask], p50[mask], color=base_color, linewidth=2.5, label='Median')

        ax.set_ylabel(label)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True)

    axes[-1].set_xlabel('Episode')
    _save(fig, 'chart_06_performance_bands.png')

    # ──────────────────────────────────────────────────
    # CHART 7: DEATH ANALYSIS DEEP DIVE
    # ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle('DEATH ANALYSIS DEEP DIVE', fontsize=16, fontweight='bold')

    # 7a. Overall pie
    ax = fig.add_subplot(gs[0, 0])
    cause_counts = defaultdict(int)
    for ca in causes_arr:
        cause_counts[ca] += 1
    labels_pie = []
    sizes_pie = []
    colors_pie = []
    for cn in CAUSE_NAMES:
        if cause_counts.get(cn, 0) > 0:
            labels_pie.append(cn)
            sizes_pie.append(cause_counts[cn])
            colors_pie.append(CAUSE_COLORS.get(cn, '#888'))
    # Add any unknown causes
    for cn, cnt in cause_counts.items():
        if cn not in CAUSE_NAMES and cnt > 0:
            labels_pie.append(cn)
            sizes_pie.append(cnt)
            colors_pie.append('#666')
    if sizes_pie:
        wedges, texts, autotexts = ax.pie(sizes_pie, labels=labels_pie, autopct='%1.1f%%',
                                           colors=colors_pie, textprops={'fontsize': 9})
        for at in autotexts:
            at.set_fontsize(8)
    ax.set_title('Overall Death Distribution')

    # 7b. Death cause per stage (grouped bar)
    ax = fig.add_subplot(gs[0, 1:])
    stage_cause_data = {}
    for s in unique_stages:
        s_causes = causes_arr[stages_arr == s]
        total_s = len(s_causes)
        stage_cause_data[s] = {}
        for cn in CAUSE_NAMES:
            stage_cause_data[s][cn] = np.sum(s_causes == cn) / max(total_s, 1) * 100

    x = np.arange(len(unique_stages))
    width = 0.15
    for i, cn in enumerate(CAUSE_NAMES):
        vals = [stage_cause_data.get(s, {}).get(cn, 0) for s in unique_stages]
        if any(v > 0 for v in vals):
            ax.bar(x + i * width, vals, width, label=cn, color=CAUSE_COLORS.get(cn, '#888'), alpha=0.8)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([f'S{s}: {STAGE_NAMES.get(s, "?")}' for s in unique_stages], fontsize=9)
    ax.set_ylabel('% of Deaths')
    ax.set_title('Death Causes per Stage')
    ax.legend(fontsize=8)
    ax.grid(True, axis='y')

    # 7c. Steps at death (histogram per cause)
    ax = fig.add_subplot(gs[1, 0])
    for cn in ['Wall', 'SnakeCollision', 'MaxSteps']:
        mask = causes_arr == cn
        if np.any(mask):
            ax.hist(steps[mask], bins=50, alpha=0.5, color=CAUSE_COLORS.get(cn, '#888'),
                   label=f'{cn} (n={np.sum(mask)})', density=True)
    ax.set_title('Steps at Death (by cause)')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.grid(True)

    # 7d. Reward at death (by cause)
    ax = fig.add_subplot(gs[1, 1])
    data_box = []
    labels_box = []
    colors_box = []
    for cn in CAUSE_NAMES:
        mask = causes_arr == cn
        if np.any(mask):
            data_box.append(rewards[mask])
            labels_box.append(cn)
            colors_box.append(CAUSE_COLORS.get(cn, '#888'))
    if data_box:
        bp = ax.boxplot(data_box, labels=labels_box, patch_artist=True, showfliers=False)
        for patch, col in zip(bp['boxes'], colors_box):
            patch.set_facecolor(col)
            patch.set_alpha(0.6)
        for el in ['whiskers', 'caps', 'medians']:
            for item in bp[el]:
                item.set_color('#c9d1d9')
    ax.set_title('Reward by Death Cause')
    ax.set_ylabel('Reward')
    ax.grid(True, axis='y')
    plt.setp(ax.get_xticklabels(), rotation=15, fontsize=8)

    # 7e. Death cause rolling heatmap
    ax = fig.add_subplot(gs[1, 2])
    heatmap_w = max(100, N // 20)
    if N > heatmap_w:
        active_causes = [cn for cn in CAUSE_NAMES if np.any(causes_arr == cn)]
        heat_data = []
        heat_x = []
        for i in range(0, N - heatmap_w, max(1, heatmap_w // 10)):
            window = causes_arr[i:i + heatmap_w]
            total_w = len(window)
            heat_x.append(ep_nums[i + heatmap_w // 2])
            row = [np.sum(window == cn) / total_w * 100 for cn in active_causes]
            heat_data.append(row)

        if heat_data:
            heat_arr = np.array(heat_data).T
            im = ax.imshow(heat_arr, aspect='auto', cmap='YlOrRd',
                          extent=[heat_x[0], heat_x[-1], len(active_causes) - 0.5, -0.5])
            ax.set_yticks(range(len(active_causes)))
            ax.set_yticklabels(active_causes, fontsize=8)
            plt.colorbar(im, ax=ax, label='%', shrink=0.8)
    ax.set_title('Death Cause Heatmap (time)')
    ax.set_xlabel('Episode (global)')

    _save(fig, 'chart_07_death_analysis.png')

    # ──────────────────────────────────────────────────
    # CHART 8: FOOD EFFICIENCY DASHBOARD
    # ──────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('FOOD COLLECTION EFFICIENCY', fontsize=16, fontweight='bold')

    # 8a. Food/step ratio over time
    ax = axes[0, 0]
    _add_stage_bands(ax, ep_nums, stages_arr, branch_points)
    ax.plot(ep_nums, food_per_step, alpha=0.12, color='#484f58', linewidth=0.5)
    if N > sma_w:
        sma_fps = moving_average(food_per_step, sma_w)
        ax.plot(ep_nums[sma_w - 1:], sma_fps, color='#d29922', linewidth=2, label=f'SMA-{sma_w}')
    ax.set_title('Food per Step (Efficiency)')
    ax.set_xlabel('Episode (global)')
    ax.set_ylabel('Food/Step')
    ax.legend(fontsize=8)
    ax.grid(True)

    # 8b. Cumulative food
    ax = axes[0, 1]
    cum_food = np.cumsum(food)
    ax.plot(ep_nums, cum_food, color='#d29922', linewidth=2)
    ax.fill_between(ep_nums, cum_food, alpha=0.1, color='#d29922')
    ax.set_title('Cumulative Food Collected')
    ax.set_xlabel('Episode (global)')
    ax.set_ylabel('Total Food')
    ax.grid(True)

    # 8c. Food distribution per stage
    ax = axes[1, 0]
    data_food_stage = []
    labels_fs = []
    colors_fs = []
    for s in unique_stages:
        vals = food[stages_arr == s]
        data_food_stage.append(vals)
        labels_fs.append(f'S{s}')
        colors_fs.append(STAGE_COLORS_HEX.get(s, '#888'))
    if data_food_stage:
        bp = ax.boxplot(data_food_stage, labels=labels_fs, patch_artist=True, showfliers=False)
        for patch, col in zip(bp['boxes'], colors_fs):
            patch.set_facecolor(col)
            patch.set_alpha(0.6)
        for el in ['whiskers', 'caps', 'medians']:
            for item in bp[el]:
                item.set_color('#c9d1d9')
    ax.set_title('Food Distribution per Stage')
    ax.set_ylabel('Food')
    ax.grid(True, axis='y')

    # 8d. Reward vs Food scatter
    ax = axes[1, 1]
    for s in unique_stages:
        mask = stages_arr == s
        ax.scatter(food[mask], rewards[mask], alpha=0.2, s=10,
                  color=STAGE_COLORS_HEX.get(s, '#888'), label=f'S{s}')
    ax.set_title('Food vs Reward')
    ax.set_xlabel('Food')
    ax.set_ylabel('Reward')
    ax.legend(fontsize=8, markerscale=3)
    ax.grid(True)

    plt.tight_layout()
    _save(fig, 'chart_08_food_efficiency.png')

    # ──────────────────────────────────────────────────
    # CHART 9: REWARD DISTRIBUTIONS (HISTOGRAM + KDE)
    # ──────────────────────────────────────────────────
    n_stages = len(unique_stages)
    fig, axes = plt.subplots(1, max(n_stages, 1), figsize=(6 * max(n_stages, 1), 5))
    fig.suptitle('REWARD DISTRIBUTION PER STAGE', fontsize=16, fontweight='bold')
    if n_stages == 1:
        axes = [axes]

    for i, s in enumerate(unique_stages):
        ax = axes[i]
        s_rewards = rewards[stages_arr == s]
        col = STAGE_COLORS_HEX.get(s, '#888')

        ax.hist(s_rewards, bins=min(60, max(10, len(s_rewards) // 20)),
               color=col, alpha=0.5, density=True, edgecolor='none')

        # Smoothed density curve
        if len(s_rewards) > 20:
            hist_vals, bin_edges = np.histogram(s_rewards, bins=100, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            kernel = np.ones(5) / 5
            smooth = np.convolve(hist_vals.astype(float), kernel, mode='same')
            ax.plot(bin_centers, smooth, color=col, linewidth=2)

        ax.axvline(np.mean(s_rewards), color='white', linestyle='--', linewidth=1.5,
                  label=f'Mean: {np.mean(s_rewards):.1f}')
        ax.axvline(np.median(s_rewards), color='#8b949e', linestyle=':', linewidth=1.5,
                  label=f'Median: {np.median(s_rewards):.1f}')

        ax.set_title(f'S{s}: {STAGE_NAMES.get(s, "?")} (n={len(s_rewards)})', fontweight='bold')
        ax.set_xlabel('Reward')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(True)

    plt.tight_layout()
    _save(fig, 'chart_09_reward_distributions.png')

    # ──────────────────────────────────────────────────
    # CHART 10: LEARNING DETECTION
    # ──────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('LEARNING DETECTION ANALYSIS', fontsize=16, fontweight='bold')

    # 10a. Reward CDF: early vs late
    ax = axes[0, 0]
    half = N // 2
    if half > 20:
        for data, label, color in [
            (sorted(rewards[:half]), f'First half (n={half})', '#f85149'),
            (sorted(rewards[half:]), f'Second half (n={N - half})', '#3fb950'),
        ]:
            y = np.linspace(0, 1, len(data))
            ax.plot(data, y, color=color, linewidth=2, label=label)
    ax.set_title('Reward CDF: Early vs Late')
    ax.set_xlabel('Reward')
    ax.set_ylabel('Cumulative %')
    ax.legend(fontsize=8)
    ax.grid(True)

    # 10b. Cumulative reward
    ax = axes[0, 1]
    cum = np.cumsum(rewards)
    ax.plot(ep_nums, cum, color='#58a6ff', linewidth=2)
    gradient_color = 'green' if cum[-1] > cum[N // 2] else 'red'
    ax.fill_between(ep_nums, cum, alpha=0.08, color=gradient_color)
    ax.set_title('Cumulative Reward')
    ax.set_xlabel('Episode (global)')
    ax.grid(True)

    # 10c. Rolling reward variance
    ax = axes[1, 0]
    if N > sma_w:
        r_std = rolling_std(rewards, sma_w)
        ax.plot(ep_nums, r_std, color='#bc8cff', linewidth=1.5, label=f'Rolling Std ({sma_w})')
        ax.fill_between(ep_nums, r_std, alpha=0.1, color='#bc8cff')
    ax.set_title('Reward Volatility (Rolling Std)')
    ax.set_xlabel('Episode (global)')
    ax.set_ylabel('Std Dev')
    ax.legend(fontsize=8)
    ax.grid(True)

    # 10d. Segmented trends
    ax = axes[1, 1]
    segs = segment_trends(rewards, min(8, N // 50)) if N > 80 else []
    if segs:
        for seg in segs:
            x = np.arange(seg['start_idx'], seg['end_idx'])
            y = rewards[seg['start_idx']:seg['end_idx']]
            sma_seg = moving_average(y, max(3, len(y) // 5))
            ep_seg = ep_nums[seg['start_idx']:seg['start_idx'] + len(sma_seg)]
            col = '#3fb950' if seg['slope'] > 0.1 else '#f85149' if seg['slope'] < -0.1 else '#8b949e'
            ax.plot(ep_seg, sma_seg, color=col, linewidth=2.5)
            mid_ep = ep_nums[(seg['start_idx'] + seg['end_idx']) // 2]
            ax.text(mid_ep, ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else seg['mean'],
                   f'{seg["slope"]:+.2f}', ha='center', fontsize=8, color=col, fontweight='bold')
    ax.set_title('Segmented Reward Trends')
    ax.set_xlabel('Episode (global)')
    ax.set_ylabel('Reward (SMA)')
    ax.grid(True)

    plt.tight_layout()
    _save(fig, 'chart_10_learning_detection.png')

    # ──────────────────────────────────────────────────
    # CHART 11: GOAL PROGRESS GAUGES
    # ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('GOAL PROGRESS', fontsize=16, fontweight='bold')

    theta = np.linspace(0, np.pi, 100)
    gauges = [
        ('Points (Best)', int(np.max(food)), 6000, 'pts'),
        ('Survival (Best)', int(np.max(steps)), 1800, 'steps'),
        ('Avg Food/Ep (Last 100)', float(np.mean(food[-min(100, N):])), 50, 'food'),
    ]

    for i, (title, val, goal, unit) in enumerate(gauges):
        ax = axes[i]
        pct = min(val / goal * 100, 100) if goal > 0 else 0
        ax.plot(np.cos(theta), np.sin(theta), color='#21262d', linewidth=18)
        fill_t = np.linspace(0, np.pi * pct / 100, 100)
        col = '#f85149' if pct < 25 else '#d29922' if pct < 60 else '#3fb950'
        ax.plot(np.cos(fill_t), np.sin(fill_t), color=col, linewidth=18)
        display_val = f'{val}' if isinstance(val, int) else f'{val:.1f}'
        ax.text(0, 0.35, display_val, ha='center', fontsize=28, fontweight='bold', color='white')
        ax.text(0, 0.08, f'/ {goal} {unit}', ha='center', fontsize=11, color='#8b949e')
        ax.text(0, -0.18, f'{pct:.1f}%', ha='center', fontsize=18, color=col, fontweight='bold')
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-0.35, 1.3)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=11)

    _save(fig, 'chart_11_goal_gauges.png')

    # ──────────────────────────────────────────────────
    # CHART 11b: GOAL PROGRESS OVER TIME (time-based X)
    # ──────────────────────────────────────────────────
    timestamps = [e.timestamp for e in episodes]
    has_timestamps = all(t is not None for t in timestamps)
    if has_timestamps:
        from matplotlib.dates import DateFormatter, AutoDateLocator
        ts_arr = np.array(timestamps)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), sharex=True)
        fig.suptitle('GOAL PROGRESS OVER TIME', fontsize=16, fontweight='bold')

        # --- Top panel: Points (food) ---
        ax1.scatter(ts_arr, food, alpha=0.15, s=4, color='#484f58', label='Per episode')
        if N > 10:
            food_sma = moving_average(food, min(50, N // 3))
            ts_sma = ts_arr[len(ts_arr) - len(food_sma):]
            ax1.plot(ts_sma, food_sma, color='#3fb950', linewidth=2, label=f'SMA-{min(50, N // 3)}')
        ax1.axhline(y=6000, color='#f0883e', linestyle='--', alpha=0.6, label='Goal: 6000')
        ax1.set_ylabel('Points (food)')
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        _add_stage_bands(ax1, ts_arr, stages_arr)

        # --- Bottom panel: Survival (steps) ---
        ax2.scatter(ts_arr, steps, alpha=0.15, s=4, color='#484f58', label='Per episode')
        if N > 10:
            steps_sma = moving_average(steps, min(50, N // 3))
            ts_sma2 = ts_arr[len(ts_arr) - len(steps_sma):]
            ax2.plot(ts_sma2, steps_sma, color='#58a6ff', linewidth=2, label=f'SMA-{min(50, N // 3)}')
        ax2.axhline(y=1800, color='#f0883e', linestyle='--', alpha=0.6, label='Goal: 1800')
        ax2.set_ylabel('Survival (steps)')
        ax2.set_xlabel('Time')
        ax2.legend(loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3)
        _add_stage_bands(ax2, ts_arr, stages_arr)

        # Format x-axis as readable timestamps
        locator = AutoDateLocator()
        formatter = DateFormatter('%m-%d %H:%M')
        ax2.xaxis.set_major_locator(locator)
        ax2.xaxis.set_major_formatter(formatter)
        fig.autofmt_xdate(rotation=30)

        plt.tight_layout()
        _save(fig, 'chart_11b_goal_over_time.png')
    else:
        print(c("  chart_11b skipped: no timestamps available", C.YEL))

    # ──────────────────────────────────────────────────
    # CHART 12: HYPERPARAMETER ANALYSIS
    # ──────────────────────────────────────────────────
    # Stage → gamma mapping (from styles.py curriculum)
    STAGE_GAMMA = {1: 0.8, 2: 0.9, 3: 0.95, 4: 0.99}
    gamma_arr = np.array([STAGE_GAMMA.get(s, 0.99) for s in stages_arr])

    fig = plt.figure(figsize=(24, 20))
    fig.suptitle('HYPERPARAMETER PERFORMANCE ANALYSIS', fontsize=16, fontweight='bold')

    # ── Panel 1: Parallel Coordinates ──
    ax = fig.add_subplot(2, 2, (1, 2))  # top row, full width

    # Normalize each HP/metric to [0,1] for parallel axes
    pc_labels = ['Epsilon', 'LR', 'Gamma', 'Beta', 'Loss', 'Reward', 'Steps']
    lr_pc = lr_arr if lr_arr is not None else np.zeros(N)
    beta_pc = beta_arr if beta_arr is not None else np.zeros(N)
    pc_raw = np.column_stack([epsilons, lr_pc, gamma_arr, beta_pc, losses, rewards, steps.astype(float)])

    # Normalize columns to [0,1]
    pc_norm = np.zeros_like(pc_raw)
    for col_i in range(pc_raw.shape[1]):
        col_min = np.nanmin(pc_raw[:, col_i])
        col_max = np.nanmax(pc_raw[:, col_i])
        if col_max - col_min > 1e-10:
            pc_norm[:, col_i] = (pc_raw[:, col_i] - col_min) / (col_max - col_min)
        else:
            pc_norm[:, col_i] = 0.5

    # Color by reward percentile
    reward_pct = np.zeros(N)
    sorted_idx = np.argsort(rewards)
    for rank, idx in enumerate(sorted_idx):
        reward_pct[idx] = rank / max(N - 1, 1)

    # Sample episodes for readability (max 500 lines)
    sample_n = min(500, N)
    sample_idx = np.random.choice(N, sample_n, replace=False) if N > sample_n else np.arange(N)
    sample_idx = sample_idx[np.argsort(reward_pct[sample_idx])]  # draw worst first, best on top

    x_coords = np.arange(len(pc_labels))
    for si in sample_idx:
        color_val = reward_pct[si]
        # Red (bad) → Yellow (mid) → Green (good)
        if color_val < 0.5:
            r, g, b = 0.97, 0.32 + color_val * 1.0, 0.29
        else:
            r, g, b = 0.97 - (color_val - 0.5) * 1.5, 0.72 + (color_val - 0.5) * 0.4, 0.29
        r, g, b = max(0, min(1, r)), max(0, min(1, g)), max(0, min(1, b))
        ax.plot(x_coords, pc_norm[si], color=(r, g, b), alpha=0.08, linewidth=0.8)

    # Draw top 5% bold
    top5_mask = reward_pct >= 0.95
    for si in np.where(top5_mask)[0]:
        ax.plot(x_coords, pc_norm[si], color='#3fb950', alpha=0.6, linewidth=1.5)

    # Axis labels with value ranges
    for i, label in enumerate(pc_labels):
        ax.axvline(i, color='#484f58', linewidth=1, zorder=0)
        vmin, vmax = np.nanmin(pc_raw[:, i]), np.nanmax(pc_raw[:, i])
        ax.text(i, -0.08, f'{vmin:.4g}', ha='center', fontsize=7, color='#8b949e')
        ax.text(i, 1.05, f'{vmax:.4g}', ha='center', fontsize=7, color='#8b949e')

    ax.set_xticks(x_coords)
    ax.set_xticklabels(pc_labels, fontsize=10, fontweight='bold')
    ax.set_ylim(-0.15, 1.12)
    ax.set_yticks([])
    ax.set_title('Parallel Coordinates: HP → Performance  (green=top 5%, red=bottom)',
                fontweight='bold', fontsize=12)
    ax.grid(False)

    # ── Panel 2: HP Sweet Spot Heatmap (Epsilon × LR → Reward) ──
    ax = fig.add_subplot(2, 2, 3)

    # Bin epsilon and LR into grid cells
    n_bins = 15
    valid_lr = lr_pc > 0
    if np.sum(valid_lr) > 20:
        eps_edges = np.linspace(np.min(epsilons[valid_lr]), np.max(epsilons[valid_lr]), n_bins + 1)
        lr_edges = np.logspace(np.log10(max(np.min(lr_pc[valid_lr]), 1e-7)),
                               np.log10(np.max(lr_pc[valid_lr])), n_bins + 1)

        heat_grid = np.full((n_bins, n_bins), np.nan)
        count_grid = np.zeros((n_bins, n_bins))
        for i in range(N):
            if not valid_lr[i]:
                continue
            ei = np.searchsorted(eps_edges, epsilons[i], side='right') - 1
            li = np.searchsorted(lr_edges, lr_pc[i], side='right') - 1
            ei = np.clip(ei, 0, n_bins - 1)
            li = np.clip(li, 0, n_bins - 1)
            if np.isnan(heat_grid[li, ei]):
                heat_grid[li, ei] = 0
            heat_grid[li, ei] += rewards[i]
            count_grid[li, ei] += 1

        # Average reward per cell
        with np.errstate(divide='ignore', invalid='ignore'):
            heat_avg = np.where(count_grid > 0, heat_grid / count_grid, np.nan)

        im = ax.imshow(heat_avg, cmap='RdYlGn', aspect='auto', origin='lower',
                       interpolation='nearest')
        # Tick labels
        eps_ticks = np.linspace(0, n_bins - 1, 5).astype(int)
        lr_ticks = np.linspace(0, n_bins - 1, 5).astype(int)
        ax.set_xticks(eps_ticks)
        ax.set_xticklabels([f'{eps_edges[t]:.3f}' for t in eps_ticks], fontsize=8)
        ax.set_yticks(lr_ticks)
        ax.set_yticklabels([f'{lr_edges[t]:.1e}' for t in lr_ticks], fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.8, label='Avg Reward')

        # Mark best cell
        best_flat = np.nanargmax(heat_avg)
        best_y, best_x = np.unravel_index(best_flat, heat_avg.shape)
        ax.plot(best_x, best_y, marker='*', color='white', markersize=15, markeredgecolor='black')
    else:
        ax.text(0.5, 0.5, 'Not enough LR data', ha='center', va='center',
               transform=ax.transAxes, fontsize=12, color='#8b949e')

    ax.set_xlabel('Epsilon', fontsize=10)
    ax.set_ylabel('Learning Rate', fontsize=10)
    ax.set_title('Sweet Spot: Epsilon × LR → Avg Reward', fontweight='bold', fontsize=12)

    # ── Panel 3: Top vs Bottom HP Comparison ──
    ax = fig.add_subplot(2, 2, 4)

    pct_5 = max(int(N * 0.05), 5)
    top_idx = np.argsort(rewards)[-pct_5:]
    bot_idx = np.argsort(rewards)[:pct_5]

    compare_metrics = {
        'Epsilon': epsilons,
        'LR (×1e4)': lr_pc * 1e4,
        'Gamma': gamma_arr,
        'Beta': beta_pc,
        'Loss': losses,
        'Food/Step': food_per_step,
    }

    metric_names_cmp = list(compare_metrics.keys())
    n_m = len(metric_names_cmp)
    x_pos = np.arange(n_m)
    bar_w = 0.35

    # Normalize each metric to [0,1] for comparison
    top_vals_norm = []
    bot_vals_norm = []
    top_vals_raw = []
    bot_vals_raw = []
    for name, data in compare_metrics.items():
        top_mean = np.mean(data[top_idx])
        bot_mean = np.mean(data[bot_idx])
        top_vals_raw.append(top_mean)
        bot_vals_raw.append(bot_mean)
        vmax = max(abs(top_mean), abs(bot_mean), 1e-10)
        top_vals_norm.append(top_mean / vmax)
        bot_vals_norm.append(bot_mean / vmax)

    bars_top = ax.barh(x_pos - bar_w / 2, top_vals_norm, bar_w,
                       color='#3fb950', alpha=0.8, label=f'Top 5% (n={pct_5})', edgecolor='#30363d')
    bars_bot = ax.barh(x_pos + bar_w / 2, bot_vals_norm, bar_w,
                       color='#f85149', alpha=0.8, label=f'Bottom 5% (n={pct_5})', edgecolor='#30363d')

    # Add raw value labels
    for i, (tv, bv) in enumerate(zip(top_vals_raw, bot_vals_raw)):
        fmt = '.4f' if abs(tv) < 0.1 else '.2f' if abs(tv) < 100 else '.0f'
        ax.text(top_vals_norm[i] + 0.02, i - bar_w / 2, f'{tv:{fmt}}',
               va='center', fontsize=8, color='#3fb950')
        ax.text(bot_vals_norm[i] + 0.02, i + bar_w / 2, f'{bv:{fmt}}',
               va='center', fontsize=8, color='#f85149')

    ax.set_yticks(x_pos)
    ax.set_yticklabels(metric_names_cmp, fontsize=10)
    ax.set_xlabel('Normalized Value', fontsize=10)
    ax.set_title('Top 5% vs Bottom 5% Episodes: HP Snapshot', fontweight='bold', fontsize=12)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, axis='x')
    ax.invert_yaxis()

    # Add summary text box
    top_avg_r = np.mean(rewards[top_idx])
    bot_avg_r = np.mean(rewards[bot_idx])
    top_avg_s = np.mean(steps[top_idx])
    bot_avg_s = np.mean(steps[bot_idx])
    summary = (f'Top 5%: avg_reward={top_avg_r:.1f}, avg_steps={top_avg_s:.0f}\n'
               f'Bot 5%: avg_reward={bot_avg_r:.1f}, avg_steps={bot_avg_s:.0f}')
    ax.text(0.98, 0.02, summary, transform=ax.transAxes, fontsize=8,
           va='bottom', ha='right', color='#c9d1d9',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='#0d1117', edgecolor='#30363d', alpha=0.9))

    plt.tight_layout()
    _save(fig, 'chart_12_hyperparameter_analysis.png')

    # ──────────────────────────────────────────────────
    # CHART 13: Q-VALUE, GRADIENT & TD-ERROR ANALYSIS
    # ──────────────────────────────────────────────────
    # Prefer csv_episodes for Q/gradient data (log parser doesn't extract these fields)
    q_src = csv_episodes if csv_episodes and any(e.q_mean != 0 for e in csv_episodes) else episodes
    q_means_arr = np.array([e.q_mean for e in q_src])
    q_maxes_arr = np.array([e.q_max for e in q_src])
    td_errors_arr = np.array([e.td_error for e in q_src])
    grad_norms_arr = np.array([e.grad_norm for e in q_src])
    # Override ep_nums/rewards/stages for chart 13 scope
    q_ep_nums = np.array([e.global_idx for e in q_src])
    q_branch_points = get_branch_points(q_src)
    q_rewards = np.array([e.reward for e in q_src])
    q_stages_arr = np.array([e.stage for e in q_src])
    q_N = len(q_src)
    q_sma_w = min(50, q_N // 3) if q_N > 30 else max(3, q_N // 3)

    has_q_data = np.any(q_means_arr != 0) or np.any(q_maxes_arr != 0)
    if has_q_data:
        q_unique_stages = sorted(set(q_stages_arr))
        q_losses = np.array([e.loss for e in q_src])
        fig = plt.figure(figsize=(24, 16))
        fig.suptitle('Q-VALUE, TD-ERROR & GRADIENT ANALYSIS', fontsize=20, fontweight='bold', y=0.98)
        gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)

        # 13a. Q-value mean & max over time
        ax = fig.add_subplot(gs[0, 0])
        _add_stage_bands(ax, q_ep_nums, q_stages_arr, q_branch_points)
        ax.plot(q_ep_nums, q_means_arr, alpha=0.12, color='#484f58', linewidth=0.5)
        ax.plot(q_ep_nums, q_maxes_arr, alpha=0.08, color='#484f58', linewidth=0.5)
        if q_N > q_sma_w:
            sma_qm = moving_average(q_means_arr, q_sma_w)
            sma_qx = moving_average(q_maxes_arr, q_sma_w)
            ax.plot(q_ep_nums[q_sma_w-1:], sma_qm, color='#58a6ff', linewidth=2, label=f'Q mean (SMA-{q_sma_w})')
            ax.plot(q_ep_nums[q_sma_w-1:], sma_qx, color='#3fb950', linewidth=2, label=f'Q max (SMA-{q_sma_w})')
        ax.axhline(0, color='#484f58', linestyle=':', linewidth=0.8)
        ax.set_title('Q-Values Over Time', fontweight='bold')
        ax.set_xlabel('Episode (global)')
        ax.set_ylabel('Q-Value')
        ax.legend(fontsize=8)
        ax.grid(True)

        # 13b. Q-value distribution (early vs late)
        ax = fig.add_subplot(gs[0, 1])
        half = q_N // 2
        if half > 20:
            early_q = q_means_arr[:half]
            late_q = q_means_arr[half:]
            bins = np.linspace(min(np.min(early_q), np.min(late_q)),
                              max(np.max(early_q), np.max(late_q)), 50)
            ax.hist(early_q, bins=bins, alpha=0.5, color='#f85149', density=True, label=f'Early (n={half})')
            ax.hist(late_q, bins=bins, alpha=0.5, color='#3fb950', density=True, label=f'Late (n={q_N-half})')
            ax.axvline(np.mean(early_q), color='#f85149', linestyle='--', linewidth=1.5)
            ax.axvline(np.mean(late_q), color='#3fb950', linestyle='--', linewidth=1.5)
        ax.set_title('Q-Value Distribution: Early vs Late', fontweight='bold')
        ax.set_xlabel('Q Mean')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(True)

        # 13c. TD-Error over time
        ax = fig.add_subplot(gs[1, 0])
        _add_stage_bands(ax, q_ep_nums, q_stages_arr, q_branch_points)
        valid_td = td_errors_arr > 0
        if np.any(valid_td):
            ax.plot(q_ep_nums[valid_td], td_errors_arr[valid_td], alpha=0.12, color='#484f58', linewidth=0.5)
            if np.sum(valid_td) > q_sma_w:
                sma_td = moving_average(td_errors_arr[valid_td], q_sma_w)
                ax.plot(q_ep_nums[valid_td][q_sma_w-1:], sma_td, color='#d29922', linewidth=2, label=f'SMA-{q_sma_w}')
            ax.set_yscale('log')
        ax.set_title('TD-Error (log scale)', fontweight='bold')
        ax.set_xlabel('Episode (global)')
        ax.set_ylabel('TD Error')
        ax.legend(fontsize=8)
        ax.grid(True)

        # 13d. Gradient Norm over time
        ax = fig.add_subplot(gs[1, 1])
        _add_stage_bands(ax, q_ep_nums, q_stages_arr, q_branch_points)
        valid_gn = grad_norms_arr > 0
        if np.any(valid_gn):
            ax.plot(q_ep_nums[valid_gn], grad_norms_arr[valid_gn], alpha=0.12, color='#484f58', linewidth=0.5)
            if np.sum(valid_gn) > q_sma_w:
                sma_gn = moving_average(grad_norms_arr[valid_gn], q_sma_w)
                ax.plot(q_ep_nums[valid_gn][q_sma_w-1:], sma_gn, color='#f85149', linewidth=2, label=f'SMA-{q_sma_w}')
            # Clip line
            ax.axhline(1.0, color='#ffaa00', linestyle='--', alpha=0.6, label='grad_clip=1.0')
        ax.set_title('Gradient Norm (pre-clip)', fontweight='bold')
        ax.set_xlabel('Episode (global)')
        ax.set_ylabel('Grad Norm')
        ax.legend(fontsize=8)
        ax.grid(True)

        # 13e. Q-value vs Reward scatter
        ax = fig.add_subplot(gs[2, 0])
        for s in q_unique_stages:
            mask = q_stages_arr == s
            ax.scatter(q_means_arr[mask], q_rewards[mask], alpha=0.15, s=8,
                      color=STAGE_COLORS_HEX.get(s, '#888'), label=f'S{s}')
        if np.any(q_means_arr != 0):
            valid_qr = (q_means_arr != 0) & np.isfinite(q_rewards)
            if np.sum(valid_qr) > 10:
                z = np.polyfit(q_means_arr[valid_qr], q_rewards[valid_qr], 1)
                p = np.poly1d(z)
                x_line = np.linspace(np.min(q_means_arr[valid_qr]), np.max(q_means_arr[valid_qr]), 50)
                ax.plot(x_line, p(x_line), color='#f0883e', linewidth=1.5, linestyle='--')
                corr = np.corrcoef(q_means_arr[valid_qr], q_rewards[valid_qr])[0, 1]
                ax.text(0.02, 0.98, f'r={corr:.3f}', transform=ax.transAxes, fontsize=9, va='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='#0d1117', edgecolor='#30363d'),
                       color='#f0883e')
        ax.set_title('Q-Value vs Reward', fontweight='bold')
        ax.set_xlabel('Q Mean')
        ax.set_ylabel('Reward')
        ax.legend(fontsize=7, markerscale=3)
        ax.grid(True)

        # 13f. Grad norm vs Loss scatter
        ax = fig.add_subplot(gs[2, 1])
        valid_gl = (grad_norms_arr > 0) & (q_losses > 0)
        if np.any(valid_gl):
            for s in q_unique_stages:
                mask = (q_stages_arr == s) & valid_gl
                ax.scatter(grad_norms_arr[mask], q_losses[mask], alpha=0.15, s=8,
                          color=STAGE_COLORS_HEX.get(s, '#888'), label=f'S{s}')
            ax.set_yscale('log')
        ax.set_title('Gradient Norm vs Loss', fontweight='bold')
        ax.set_xlabel('Grad Norm')
        ax.set_ylabel('Loss (log)')
        ax.legend(fontsize=7, markerscale=3)
        ax.grid(True)

        _save(fig, 'chart_13_qvalue_gradients.png')

    # ──────────────────────────────────────────────────
    # CHART 14: ACTION DISTRIBUTION ANALYSIS
    # ──────────────────────────────────────────────────
    # Prefer csv_episodes for action data (log parser doesn't extract these fields)
    act_src = csv_episodes if csv_episodes and any(e.act_straight != 0 for e in csv_episodes) else episodes
    act_ep_nums = np.array([e.global_idx for e in act_src])
    act_branch_points = get_branch_points(act_src)
    act_rewards = np.array([e.reward for e in act_src])
    act_stages_arr = np.array([e.stage for e in act_src])
    act_N = len(act_src)
    act_unique_stages = sorted(set(act_stages_arr))
    act_names = ['Straight', 'Gentle', 'Medium', 'Sharp', 'U-turn', 'Boost']
    act_colors = ['#8b949e', '#58a6ff', '#3fb950', '#d29922', '#f0883e', '#f85149']
    act_arrs = [
        np.array([e.act_straight for e in act_src]),
        np.array([e.act_gentle for e in act_src]),
        np.array([e.act_medium for e in act_src]),
        np.array([e.act_sharp for e in act_src]),
        np.array([e.act_uturn for e in act_src]),
        np.array([e.act_boost for e in act_src]),
    ]

    has_act_data = any(np.any(a != 0) for a in act_arrs)
    act_sma_w = min(50, act_N // 3) if act_N > 30 else max(3, act_N // 3)
    if has_act_data:
        fig = plt.figure(figsize=(24, 14))
        fig.suptitle('ACTION DISTRIBUTION ANALYSIS', fontsize=20, fontweight='bold', y=0.98)
        gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

        # 14a. Stacked area: action distribution over time
        ax = fig.add_subplot(gs[0, :2])
        _add_stage_bands(ax, act_ep_nums, act_stages_arr, act_branch_points)
        roll_w = max(20, act_N // 30)
        if act_N > roll_w:
            x_pts = []
            smoothed = {name: [] for name in act_names}
            for i in range(0, act_N - roll_w, max(1, roll_w // 5)):
                x_pts.append(act_ep_nums[i + roll_w // 2])
                for j, name in enumerate(act_names):
                    smoothed[name].append(np.mean(act_arrs[j][i:i+roll_w]) * 100)
            bottom = np.zeros(len(x_pts))
            for j, name in enumerate(act_names):
                vals = np.array(smoothed[name])
                ax.fill_between(x_pts, bottom, bottom + vals, alpha=0.7,
                               color=act_colors[j], label=name)
                bottom += vals
            ax.set_ylim(0, 100)
        ax.set_title('Action Distribution Over Time (Stacked)', fontweight='bold')
        ax.set_xlabel('Episode (global)')
        ax.set_ylabel('% of Actions')
        ax.legend(fontsize=8, loc='upper right', ncol=3)
        ax.grid(True)

        # 14b. Overall pie chart
        ax = fig.add_subplot(gs[0, 2])
        overall_pcts = [np.mean(a) * 100 for a in act_arrs]
        nonzero = [(name, pct, col) for name, pct, col in zip(act_names, overall_pcts, act_colors) if pct > 0.1]
        if nonzero:
            labels_p = [n for n, _, _ in nonzero]
            sizes_p = [p for _, p, _ in nonzero]
            colors_p = [c_ for _, _, c_ in nonzero]
            wedges, texts, autotexts = ax.pie(sizes_p, labels=labels_p, autopct='%1.1f%%',
                                               colors=colors_p, textprops={'fontsize': 9})
            for at in autotexts:
                at.set_fontsize(8)
        ax.set_title('Overall Action Mix')

        # 14c. Per-stage action bars
        ax = fig.add_subplot(gs[1, 0])
        x_s = np.arange(len(act_unique_stages))
        bar_w = 0.13
        for j, name in enumerate(act_names):
            vals = [np.mean(act_arrs[j][act_stages_arr == s]) * 100 for s in act_unique_stages]
            ax.bar(x_s + j * bar_w, vals, bar_w, label=name, color=act_colors[j], alpha=0.8)
        ax.set_xticks(x_s + bar_w * 2.5)
        ax.set_xticklabels([f'S{s}' for s in act_unique_stages])
        ax.set_title('Action Mix per Stage')
        ax.set_ylabel('% of Actions')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, axis='y')

        # 14d. Action entropy over time (diversity measure)
        ax = fig.add_subplot(gs[1, 1])
        entropy_arr = np.zeros(act_N)
        for i in range(act_N):
            probs = np.array([a[i] for a in act_arrs])
            probs = probs[probs > 0]
            if len(probs) > 0:
                entropy_arr[i] = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(6)  # 6 action categories
        _add_stage_bands(ax, act_ep_nums, act_stages_arr, act_branch_points)
        ax.plot(act_ep_nums, entropy_arr, alpha=0.12, color='#484f58', linewidth=0.5)
        if act_N > act_sma_w:
            sma_ent = moving_average(entropy_arr, act_sma_w)
            ax.plot(act_ep_nums[act_sma_w-1:], sma_ent, color='#bc8cff', linewidth=2, label=f'SMA-{act_sma_w}')
        ax.axhline(max_entropy, color='#3fb950', linestyle='--', alpha=0.5, label=f'Max entropy ({max_entropy:.2f})')
        ax.axhline(max_entropy * 0.5, color='#f85149', linestyle=':', alpha=0.5, label='50% diversity')
        ax.set_title('Action Entropy (Policy Diversity)', fontweight='bold')
        ax.set_xlabel('Episode (global)')
        ax.set_ylabel('Entropy (bits)')
        ax.legend(fontsize=8)
        ax.grid(True)

        # 14e. Action vs Reward relationship
        ax = fig.add_subplot(gs[1, 2])
        for j, name in enumerate(act_names):
            valid_a = act_arrs[j] > 0
            if np.sum(valid_a) > 10:
                corr = np.corrcoef(act_arrs[j][valid_a], act_rewards[valid_a])[0, 1]
                ax.barh(j, corr, color=act_colors[j], alpha=0.8, edgecolor='#30363d')
                ax.text(corr + (0.02 if corr >= 0 else -0.02), j, f'{corr:.3f}',
                       va='center', ha='left' if corr >= 0 else 'right', fontsize=9, color='#c9d1d9')
        ax.set_yticks(range(len(act_names)))
        ax.set_yticklabels(act_names, fontsize=10)
        ax.axvline(0, color='#484f58', linewidth=1)
        ax.set_xlim(-1, 1)
        ax.set_title('Action → Reward Correlation', fontweight='bold')
        ax.set_xlabel('Pearson r')
        ax.grid(True, axis='x')

        _save(fig, 'chart_14_action_distribution.png')

    # ──────────────────────────────────────────────────
    # CHART 15: ACTIVE AGENTS OVER TIME (auto-scaling)
    # ──────────────────────────────────────────────────
    # num_agents is only in CSV data, not in log-parsed episodes
    csv_src = csv_episodes if csv_episodes else episodes
    csv_agents_arr = np.array([e.num_agents for e in csv_src])
    has_agents_data = np.any(csv_agents_arr > 0)
    if has_agents_data:
        csv_ep_nums = np.array([e.global_idx for e in csv_src])
        csv_branch_points = get_branch_points(csv_src)
        csv_rewards = np.array([e.reward for e in csv_src])
        csv_stages = np.array([e.stage for e in csv_src])
        csv_N = len(csv_src)
        csv_sma_w = min(50, csv_N // 3) if csv_N > 30 else max(3, csv_N // 3)

        fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True,
                                  gridspec_kw={'height_ratios': [2, 1]})
        fig.suptitle('AUTO-SCALING: ACTIVE AGENTS', fontsize=16, fontweight='bold')

        # 15a. Number of agents over time (step plot)
        ax = axes[0]
        _add_stage_bands(ax, csv_ep_nums, csv_stages, csv_branch_points)
        ax.step(csv_ep_nums, csv_agents_arr, where='post', color='#58a6ff', linewidth=2, label='Active Agents')
        ax.fill_between(csv_ep_nums, 0, csv_agents_arr, step='post', alpha=0.15, color='#58a6ff')
        ax.set_ylabel('Active Agents')
        ax.set_ylim(0, max(csv_agents_arr.max() + 1, 2))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(True)

        # 15b. Reward per agent (efficiency)
        ax = axes[1]
        _add_stage_bands(ax, csv_ep_nums, csv_stages, csv_branch_points)
        safe_agents = np.maximum(csv_agents_arr, 1)
        rw_per_agent = csv_rewards / safe_agents
        ax.plot(csv_ep_nums, rw_per_agent, alpha=0.12, color='#484f58', linewidth=0.5)
        if csv_N > csv_sma_w:
            sma_rpa = moving_average(rw_per_agent, csv_sma_w)
            ax.plot(csv_ep_nums[csv_sma_w-1:], sma_rpa, color='#3fb950', linewidth=2, label=f'Reward/Agent SMA-{csv_sma_w}')
        ax.set_title('Reward per Agent (Efficiency)', fontweight='bold')
        ax.set_xlabel('Episode (global)')
        ax.set_ylabel('Reward / Agent')
        ax.axhline(0, color='#484f58', linestyle=':', linewidth=0.8)
        ax.legend(fontsize=8)
        ax.grid(True)

        _save(fig, 'chart_15_auto_scaling.png')

    # ──────────────────────────────────────────────────
    # CHART 16: MAX STEPS ANALYSIS
    # ──────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle('MAX STEPS ANALYSIS', fontsize=16, fontweight='bold', y=0.98)
    fig.subplots_adjust(hspace=0.35)

    # Max steps limits per stage (from styles.py)
    stage_max_steps = {1: 300, 2: 500, 3: 1000, 4: 2000}

    # 16a. Steps per episode with max_steps ceiling lines
    ax = axes[0]
    _add_stage_bands(ax, ep_nums, stages_arr, branch_points)

    # Plot all steps
    ax.plot(ep_nums, steps, alpha=0.15, color='#484f58', linewidth=0.5)
    if N > sma_w:
        sma_steps = moving_average(steps, sma_w)
        ax.plot(ep_nums[sma_w-1:], sma_steps, color='#58a6ff', linewidth=2, label=f'Steps SMA-{sma_w}')

    # Draw max_steps ceiling per stage segment
    unique_stages_sorted = sorted(set(stages_arr))
    for s in unique_stages_sorted:
        s_mask = stages_arr == s
        if not np.any(s_mask):
            continue
        s_ep_nums = ep_nums[s_mask]
        limit = stage_max_steps.get(s, 99999)
        if limit < 10000:
            ax.hlines(limit, s_ep_nums[0], s_ep_nums[-1],
                      colors=STAGE_COLORS_HEX.get(s, '#888'), linestyles='--',
                      linewidth=2, alpha=0.8, label=f'S{s} limit={limit}')

    # Mark MaxSteps episodes
    maxstep_mask = causes_arr == 'MaxSteps'
    if np.any(maxstep_mask):
        ax.scatter(ep_nums[maxstep_mask], steps[maxstep_mask],
                   color='#f0e68c', s=40, zorder=5, marker='D',
                   edgecolors='#b8860b', linewidths=0.8, label='MaxSteps hit')

    ax.set_title('Steps per Episode vs Max Steps Limit', fontweight='bold')
    _setup_ep_axis(ax, ep_nums, orig_ep_nums)
    ax.set_ylabel('Steps')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True)

    # 16b. MaxSteps hit rate per stage (bar chart)
    ax = axes[1]
    bar_stages = []
    bar_rates = []
    bar_counts = []
    bar_totals = []
    bar_colors = []
    for s in unique_stages_sorted:
        s_mask = stages_arr == s
        n_total = np.sum(s_mask)
        n_maxsteps = np.sum((causes_arr == 'MaxSteps') & s_mask)
        bar_stages.append(f'S{s}: {STAGE_NAMES.get(s, "?")}')
        bar_rates.append(n_maxsteps / n_total * 100 if n_total > 0 else 0)
        bar_counts.append(n_maxsteps)
        bar_totals.append(n_total)
        bar_colors.append(STAGE_COLORS_HEX.get(s, '#888'))

    bars = ax.bar(bar_stages, bar_rates, color=bar_colors, alpha=0.8, edgecolor='#30363d')
    for bar, rate, cnt, total in zip(bars, bar_rates, bar_counts, bar_totals):
        if cnt > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{cnt}/{total}\n({rate:.1f}%)', ha='center', va='bottom', fontsize=9,
                    color='#c9d1d9', fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, 0.3,
                    f'0/{total}', ha='center', va='bottom', fontsize=9,
                    color='#8b949e')

    ax.set_title('MaxSteps Hit Rate by Stage', fontweight='bold')
    ax.set_ylabel('MaxSteps %')
    ax.set_ylim(0, max(max(bar_rates) * 1.3, 5) if bar_rates else 5)
    ax.grid(True, axis='y')

    _save(fig, 'chart_16_maxsteps_analysis.png')

    # ──────────────────────────────────────────────────
    # CHART 17: SURVIVAL PERCENTILES (per 100-episode windows)
    # ──────────────────────────────────────────────────
    SURV_WINDOW = 100
    if N >= SURV_WINDOW:
        n_windows = N // SURV_WINDOW
        win_x = []       # window center episode
        win_avg = []
        win_p25 = []
        win_p50 = []
        win_p75 = []
        win_p90 = []
        win_best = []

        for w_i in range(n_windows):
            start_i = w_i * SURV_WINDOW
            end_i = start_i + SURV_WINDOW
            w_steps = np.sort(steps[start_i:end_i])
            w_ep = ep_nums[start_i + SURV_WINDOW // 2]  # center of window
            win_x.append(w_ep)
            win_avg.append(np.mean(w_steps))
            win_p25.append(np.percentile(w_steps, 25))
            win_p50.append(np.percentile(w_steps, 50))
            win_p75.append(np.percentile(w_steps, 75))
            win_p90.append(np.percentile(w_steps, 90))
            win_best.append(np.max(w_steps))

        fig, axes = plt.subplots(2, 1, figsize=(18, 12), gridspec_kw={'height_ratios': [2, 1]})
        fig.suptitle(f'SURVIVAL PERCENTILES (per {SURV_WINDOW}-episode windows)', fontsize=16, fontweight='bold', y=0.98)
        fig.subplots_adjust(hspace=0.3)

        # 17a. Main view — clipped Y axis for detail
        ax = axes[0]
        _add_stage_bands(ax, ep_nums, stages_arr, branch_points)

        # Fill between p25-p75
        ax.fill_between(win_x, win_p25, win_p75, alpha=0.15, color='#58a6ff', label='_nolegend_')
        # Fill between p75-p90
        ax.fill_between(win_x, win_p75, win_p90, alpha=0.08, color='#bc8cff', label='_nolegend_')

        # Lines
        ax.plot(win_x, win_p25, color='#8b949e', linewidth=1.2, marker='v', markersize=3, label='p25')
        ax.plot(win_x, win_p50, color='#58a6ff', linewidth=2, marker='s', markersize=4, label='Median (p50)')
        ax.plot(win_x, win_avg, color='#f0883e', linewidth=2.5, marker='o', markersize=5, label='Average', zorder=5)
        ax.plot(win_x, win_p75, color='#3fb950', linewidth=1.5, marker='^', markersize=4, label='p75')
        ax.plot(win_x, win_p90, color='#bc8cff', linewidth=1.5, marker='D', markersize=4, label='p90')
        ax.plot(win_x, win_best, color='#f85149', linewidth=1.5, linestyle='--', marker='*', markersize=6, label='Best (ep)', alpha=0.7)

        # Clip Y to 2x the max p90 for readability
        y_clip = max(win_p90) * 2.2 if win_p90 else 500
        y_clip = max(y_clip, 200)  # at least 200
        ax.set_ylim(0, y_clip)
        _setup_ep_axis(ax, ep_nums, orig_ep_nums)
        ax.set_ylabel('Steps Survived')
        ax.set_title('Detailed View (Y clipped for readability)', fontweight='bold', fontsize=11)
        ax.legend(fontsize=9, loc='upper left', ncol=3)
        ax.grid(True, alpha=0.3)

        # Annotate latest values
        if len(win_x) > 0:
            last_x = win_x[-1]
            for val, label, color, va in [
                (win_avg[-1], f'avg={win_avg[-1]:.0f}', '#f0883e', 'bottom'),
                (win_p50[-1], f'p50={win_p50[-1]:.0f}', '#58a6ff', 'top'),
                (win_p90[-1], f'p90={win_p90[-1]:.0f}', '#bc8cff', 'bottom'),
            ]:
                ax.annotate(label, xy=(last_x, val), fontsize=8, color=color,
                            fontweight='bold', ha='left',
                            xytext=(8, 4 if va == 'bottom' else -4), textcoords='offset points')

        # 17b. Average + Median trend with linear regression
        ax = axes[1]
        _add_stage_bands(ax, ep_nums, stages_arr, branch_points)

        ax.plot(win_x, win_avg, color='#f0883e', linewidth=2.5, marker='o', markersize=5, label='Average')
        ax.plot(win_x, win_p50, color='#58a6ff', linewidth=2, marker='s', markersize=4, label='Median (p50)')

        # Add trend lines per stage
        for s in sorted(set(stages_arr)):
            s_mask_win = []
            s_avg_win = []
            s_x_win = []
            for wi in range(len(win_x)):
                start_i = wi * SURV_WINDOW
                end_i = start_i + SURV_WINDOW
                mid_stage = stages_arr[start_i + SURV_WINDOW // 2] if start_i + SURV_WINDOW // 2 < N else stages_arr[-1]
                if mid_stage == s:
                    s_x_win.append(win_x[wi])
                    s_avg_win.append(win_avg[wi])
            if len(s_x_win) >= 3:
                z = np.polyfit(range(len(s_x_win)), s_avg_win, 1)
                trend_y = np.polyval(z, range(len(s_x_win)))
                col = STAGE_COLORS_HEX.get(s, '#888')
                ax.plot(s_x_win, trend_y, color=col, linewidth=2, linestyle=':', alpha=0.8,
                        label=f'S{s} trend ({z[0]:+.2f}/win)')

        _setup_ep_axis(ax, ep_nums, orig_ep_nums)
        ax.set_ylabel('Steps Survived')
        ax.set_title('Average & Median with Trend Lines', fontweight='bold', fontsize=11)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9, loc='upper left', ncol=2)
        ax.grid(True, alpha=0.3)

        _save(fig, 'chart_17_survival_percentiles.png')

    print(c(f'\n  Total: up to 17 chart files generated.', C.GRN, C.B))


# ═══════════════════════════════════════════════════════
#  MARKDOWN REPORT
# ═══════════════════════════════════════════════════════

def generate_markdown(episodes, csv_episodes, sessions, verdict, output_path):
    rewards = np.array([e.reward for e in episodes])
    steps = np.array([e.steps for e in episodes])
    food = np.array([e.food for e in episodes])
    losses = np.array([e.loss for e in episodes])
    stages_arr = np.array([e.stage for e in episodes])
    causes_arr = np.array([e.cause for e in episodes])
    N = len(episodes)

    with open(output_path, 'w') as f:
        f.write("# Slither.io Bot - Training Progress Report v3\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Total Episodes:** {N}  \n")
        f.write(f"**Training Sessions:** {len(sessions)}\n\n")

        status = "LEARNING" if verdict.is_learning else "NOT LEARNING"
        f.write(f"## Verdict: {status} (Confidence: {verdict.confidence * 100:.0f}%)\n\n")
        f.write(f"**Goal Feasibility:** {verdict.goal_feasibility}\n\n")

        for label, items in [('Critical Issues', verdict.issues),
                              ('Warnings', verdict.warnings),
                              ('Positive Signals', verdict.positives)]:
            if items:
                f.write(f"### {label}\n")
                for item in items:
                    f.write(f"- {item}\n")
                f.write("\n")

        # Stage breakdown
        unique_stages = sorted(set(stages_arr))
        f.write("## Curriculum Stage Breakdown\n\n")
        f.write("| Stage | Name | Episodes | Avg Reward | Avg Steps | Avg Food | Food/Step | Wall% | Snake% | MaxSteps% |\n")
        f.write("|-------|------|----------|------------|-----------|----------|-----------|-------|--------|----------|\n")
        for s in unique_stages:
            mask = stages_arr == s
            s_eps = [e for e in episodes if e.stage == s]
            sc = causes_arr[mask]
            n = len(s_eps)
            wp = np.sum(sc == 'Wall') / n * 100
            sp = np.sum(sc == 'SnakeCollision') / n * 100
            mp = np.sum(sc == 'MaxSteps') / n * 100
            avg_fps = np.mean([e.food_per_step for e in s_eps])
            f.write(f"| S{s} | {STAGE_NAMES.get(s, '?')} | {n} | "
                    f"{np.mean(rewards[mask]):.1f} | {np.mean(steps[mask]):.1f} | "
                    f"{np.mean(food[mask]):.1f} | {avg_fps:.4f} | "
                    f"{wp:.1f}% | {sp:.1f}% | {mp:.1f}% |\n")
        f.write("\n")

        # Key statistics
        f.write("## Key Statistics\n\n")
        f.write("| Metric | Mean | Std | Min | P25 | Median | P75 | P95 | Max |\n")
        f.write("|--------|------|-----|-----|-----|--------|-----|-----|-----|\n")
        for name, data in [('Reward', rewards), ('Steps', steps.astype(float)),
                           ('Food', food.astype(float)), ('Loss', losses),
                           ('Food/Step', np.array([e.food_per_step for e in episodes]))]:
            p = compute_percentiles(data)
            f.write(f"| {name} | {p['mean']:.2f} | {p['std']:.2f} | {p['min']:.2f} | "
                    f"{p['p25']:.2f} | {p['p50']:.2f} | {p['p75']:.2f} | {p['p95']:.2f} | {p['max']:.2f} |\n")
        f.write("\n")

        # Q-Value & Gradient Analysis
        q_means_md = np.array([e.q_mean for e in episodes])
        q_maxes_md = np.array([e.q_max for e in episodes])
        td_errors_md = np.array([e.td_error for e in episodes])
        grad_norms_md = np.array([e.grad_norm for e in episodes])
        has_q_md = np.any(q_means_md != 0) or np.any(q_maxes_md != 0)

        if has_q_md:
            f.write("## Q-Value & Gradient Health\n\n")
            f.write("| Metric | Last | Avg (50) | Min | Max | Trend |\n")
            f.write("|--------|------|----------|-----|-----|-------|\n")
            n_r = min(50, N)
            for name, arr in [('Q Mean', q_means_md), ('Q Max', q_maxes_md),
                               ('TD Error', td_errors_md), ('Grad Norm', grad_norms_md)]:
                sl, r2 = linear_trend(arr)
                trend = "UP" if sl > 0.01 else ("DOWN" if sl < -0.01 else "FLAT")
                f.write(f"| {name} | {arr[-1]:.4f} | {np.mean(arr[-n_r:]):.4f} | "
                        f"{np.min(arr):.4f} | {np.max(arr):.4f} | {trend} (slope={sl:.4f}) |\n")
            f.write("\n")

        # Action Distribution
        act_md = {
            'Straight': np.array([e.act_straight for e in episodes]),
            'Gentle': np.array([e.act_gentle for e in episodes]),
            'Medium': np.array([e.act_medium for e in episodes]),
            'Sharp': np.array([e.act_sharp for e in episodes]),
            'U-turn': np.array([e.act_uturn for e in episodes]),
            'Boost': np.array([e.act_boost for e in episodes]),
        }
        has_act_md = any(np.any(v != 0) for v in act_md.values())

        if has_act_md:
            f.write("## Action Distribution\n\n")
            f.write("| Action | Overall % | Last 100 % | First 100 % | Change |\n")
            f.write("|--------|----------|-----------|------------|--------|\n")
            n_100 = min(100, N)
            for name, arr in act_md.items():
                overall = np.mean(arr) * 100
                recent = np.mean(arr[-n_100:]) * 100
                early = np.mean(arr[:n_100]) * 100
                change = recent - early
                arrow = "+" if change > 0 else ""
                f.write(f"| {name} | {overall:.1f}% | {recent:.1f}% | {early:.1f}% | {arrow}{change:.1f}% |\n")

            # Entropy
            entropy_vals = []
            for i in range(N):
                probs = np.array([arr[i] for arr in act_md.values()])
                probs = probs[probs > 0]
                if len(probs) > 0:
                    entropy_vals.append(-np.sum(probs * np.log2(probs + 1e-10)))
                else:
                    entropy_vals.append(0)
            max_ent = np.log2(6)
            avg_ent = np.mean(entropy_vals[-n_100:])
            f.write(f"\n**Action Entropy (last 100):** {avg_ent:.2f} / {max_ent:.2f} bits "
                    f"({avg_ent/max_ent*100:.0f}% diversity)\n\n")

        # Windowed trends
        f.write("## Windowed Trend Analysis\n\n")
        f.write("| Window | Mean Reward | Std | Slope | R\u00b2 |\n")
        f.write("|--------|-----------|-----|-------|----|\n")
        for w in [50, 100, 200, 500, 1000]:
            if N >= w:
                wr = rewards[-w:]
                sl, r2 = linear_trend(wr)
                f.write(f"| Last {w} | {np.mean(wr):.2f} | {np.std(wr):.2f} | {sl:+.4f} | {r2:.4f} |\n")
        f.write("\n")

        # Death analysis
        f.write("## Death Cause Analysis\n\n")
        f.write("| Cause | Count | % | Avg Steps | Avg Reward |\n")
        f.write("|-------|-------|---|-----------|------------|\n")
        for cn in CAUSE_NAMES:
            mask = causes_arr == cn
            n_c = np.sum(mask)
            if n_c > 0:
                f.write(f"| {cn} | {n_c} | {n_c / N * 100:.1f}% | "
                        f"{np.mean(steps[mask]):.1f} | {np.mean(rewards[mask]):.1f} |\n")
        f.write("\n")

        # Goal progress
        f.write("## Goal Progress\n\n")
        best_food = int(np.max(food))
        best_steps = int(np.max(steps))
        f.write(f"| Target | Best | Goal | Progress |\n")
        f.write(f"|--------|------|------|----------|\n")
        f.write(f"| Points | {best_food} | 6,000 | {best_food / 6000 * 100:.1f}% |\n")
        f.write(f"| Survival | {best_steps} steps | 1,800 steps | {best_steps / 1800 * 100:.1f}% |\n\n")

        # Recommendations
        f.write("## Recommendations\n\n")
        f.write(f"{verdict.recommendation}\n\n")
        recs = generate_specific_recommendations(episodes, csv_episodes, sessions, verdict)
        for i, rec in enumerate(recs, 1):
            f.write(f"{i}. {rec}\n\n")

        # Charts (only include if file exists)
        f.write("## Charts\n\n")
        chart_dir = os.path.dirname(output_path)
        chart_files = [
            ('chart_01_dashboard.png', 'Main Dashboard'),
            ('chart_02_stage_progression.png', 'Stage Progression'),
            ('chart_03_stage_distributions.png', 'Per-Stage Distributions'),
            ('chart_04_hyperparameters.png', 'Hyperparameter Tracking'),
            ('chart_05_correlations.png', 'Metric Correlations (Scatter)'),
            ('chart_05b_correlation_heatmap.png', 'Correlation Heatmap & Rankings'),
            ('chart_06_performance_bands.png', 'Performance Percentile Bands'),
            ('chart_07_death_analysis.png', 'Death Analysis'),
            ('chart_08_food_efficiency.png', 'Food Efficiency'),
            ('chart_09_reward_distributions.png', 'Reward Distributions'),
            ('chart_10_learning_detection.png', 'Learning Detection'),
            ('chart_11_goal_gauges.png', 'Goal Progress'),
            ('chart_11b_goal_over_time.png', 'Goal Progress Over Time'),
            ('chart_12_hyperparameter_analysis.png', 'Hyperparameter Analysis'),
            ('chart_13_qvalue_gradients.png', 'Q-Value & Gradient Analysis'),
            ('chart_14_action_distribution.png', 'Action Distribution Analysis'),
            ('chart_15_auto_scaling.png', 'Active Agents Over Time'),
            ('chart_16_maxsteps_analysis.png', 'MaxSteps Analysis'),
            ('chart_17_survival_percentiles.png', 'Survival Percentiles'),
        ]
        for fname, title in chart_files:
            chart_path = os.path.join(chart_dir, fname)
            if os.path.exists(chart_path):
                f.write(f"### {title}\n![{title}]({fname})\n\n")

    print(c(f'  Report saved: {output_path}', C.GRN))


# ═══════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Slither.io Bot Training Progress Analyzer v3')
    parser.add_argument('--log', default=None, help='Path to train.log')
    parser.add_argument('--csv', default=None, help='Path to training_stats.csv')
    parser.add_argument('--no-charts', action='store_true', help='Skip chart generation')
    parser.add_argument('--no-report', action='store_true', help='Skip markdown report')
    parser.add_argument('--uid', type=str, default=None, help='Filter by run UID')
    parser.add_argument('--latest', action='store_true', help='Analyze only latest UID')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = args.log or os.path.join(script_dir, 'logs', 'train.log')
    csv_path = args.csv or os.path.join(script_dir, 'training_stats.csv')

    print(c('\n  Loading training data...', C.CYN))

    episodes = []
    sessions = []
    if os.path.exists(log_path):
        print(f'  Parsing log: {log_path}')
        episodes, sessions = parse_log(log_path)
        print(c(f'  Found {len(episodes)} episodes in {len(sessions)} sessions', C.GRN))
    else:
        print(c(f'  Log not found: {log_path}', C.YEL))

    csv_episodes = []
    if os.path.exists(csv_path):
        print(f'  Parsing CSV: {csv_path}')
        csv_episodes = parse_csv(csv_path)
        print(c(f'  Found {len(csv_episodes)} episodes in CSV', C.GRN))

    if not episodes and csv_episodes:
        episodes = csv_episodes

    if not episodes:
        print(c('\n  No training data found!', C.RED, C.B))
        return

    # UID discovery
    uids = discover_uids(csv_episodes) if csv_episodes else []
    if uids and any(u[0] != 'unknown' for u in uids):
        print()
        _print_uid_tree(uids)

        if args.uid:
            matched = [u for u in uids if args.uid in u[0]]
            if matched:
                target_uid = matched[0][0]
                # Include full ancestor chain
                chain_uids = _get_ancestor_chain(target_uid, csv_episodes)
                csv_episodes = [e for e in csv_episodes if e.uid in chain_uids]
                if not episodes or episodes is csv_episodes:
                    episodes = csv_episodes
                print(c(f'  Filtering: {target_uid} + {len(chain_uids)-1} ancestors ({len(csv_episodes)} eps)', C.GRN))
            else:
                print(c(f'  UID "{args.uid}" not found!', C.RED))
                return
        elif args.latest:
            latest_uid = uids[-1][0]
            # Include full ancestor chain for latest UID
            chain_uids = _get_ancestor_chain(latest_uid, csv_episodes)
            csv_episodes = [e for e in csv_episodes if e.uid in chain_uids]
            if not episodes or episodes[0].uid:
                episodes = csv_episodes
            print(c(f'  Latest UID: {latest_uid} + {len(chain_uids)-1} ancestors ({len(csv_episodes)} eps)', C.GRN))
        else:
            # No filter: use longest chain as primary, keep all for overlay
            longest_chain = _find_longest_chain(uids)
            if len(longest_chain) < len(set(u[0] for u in uids)):
                print(c(f'  Using longest chain ({len(longest_chain)} UIDs) as primary view', C.CYN))
            csv_episodes = [e for e in csv_episodes if e.uid in longest_chain]
            if not episodes or episodes[0].uid:
                episodes = csv_episodes

    # Build lineage chain — assigns global_idx to all episodes
    if csv_episodes:
        csv_episodes = build_lineage_chain(csv_episodes)
    if episodes and episodes is not csv_episodes:
        episodes = build_lineage_chain(episodes)
    elif episodes is csv_episodes:
        pass  # already built
    else:
        for i, ep in enumerate(episodes):
            ep.global_idx = i

    print(c('  Analyzing...', C.CYN))
    verdict = assess_learning(episodes, csv_episodes, sessions)

    print_full_report(episodes, csv_episodes, sessions, verdict)

    if not args.no_charts:
        section('GENERATING CHARTS')
        generate_charts(episodes, csv_episodes, sessions, verdict, script_dir)

    if not args.no_report:
        section('GENERATING REPORT')
        md_path = os.path.join(script_dir, 'progress_report.md')
        generate_markdown(episodes, csv_episodes, sessions, verdict, md_path)

    print()
    print(c('  Analysis complete.', C.GRN, C.B))
    print()


if __name__ == '__main__':
    main()
