#!/usr/bin/env python3
"""
=============================================================================
  SLITHER.IO BOT - DEEP TRAINING PROGRESS ANALYZER
=============================================================================
  Full analysis tool for AI model learning progress.
  Parses train.log (full history) + training_stats.csv (current session).

  Provides:
    - Learning detection (is the model actually learning?)
    - Trend analysis with statistical significance
    - Goal feasibility estimation (6000 pts / 1 hour survival)
    - Training health diagnostics (LR, epsilon, loss, death patterns)
    - Multi-session comparison
    - Detailed charts and terminal report

  Usage:
    python training_progress_analyzer.py
    python training_progress_analyzer.py --log logs/train.log --csv training_stats.csv
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
    """Terminal color codes."""
    R = '\033[0m'     # Reset
    B = '\033[1m'     # Bold
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

def bar(value, max_val, width=30, fill_char='█', empty_char='░', color=C.CYN):
    if max_val <= 0:
        return empty_char * width
    ratio = min(max(value / max_val, 0), 1.0)
    filled = int(ratio * width)
    return c(fill_char * filled, color) + c(empty_char * (width - filled), C.DIM)


# ═══════════════════════════════════════════════════════
#  DATA STRUCTURES
# ═══════════════════════════════════════════════════════

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

@dataclass
class TrainingSession:
    """A continuous training run (between restarts)."""
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
    r'(\w+)'
)

WALL_ENEMY_PATTERN = re.compile(r'Wall:(\d+)\s*(?:Enemy:(\d+))?')

CONFIG_STYLE_PATTERN = re.compile(r'Style: (.+)')
CONFIG_AGENTS_PATTERN = re.compile(r'Agents: (\d+)')
CONFIG_LR_PATTERN = re.compile(r'LR: ([\d.e\-]+)')
RESUMED_PATTERN = re.compile(r'Resumed from episode (\d+)')


def parse_log(log_path: str) -> Tuple[List[Episode], List[TrainingSession]]:
    """Parse the full train.log and extract all episodes + sessions."""
    episodes = []
    sessions = []

    current_style = "Unknown"
    current_agents = 1
    current_lr = 0.0
    session_start_ep = None
    current_session_episodes = []
    session_start_time = None

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Detect configuration blocks
            m_style = CONFIG_STYLE_PATTERN.search(line)
            if m_style:
                # Save previous session if exists
                if current_session_episodes:
                    sess = TrainingSession(
                        style=prev_style if 'prev_style' in dir() else current_style,
                        start_episode=current_session_episodes[0].number,
                        end_episode=current_session_episodes[-1].number,
                        episodes=current_session_episodes,
                        agents=prev_agents if 'prev_agents' in dir() else current_agents,
                        lr_config=prev_lr if 'prev_lr' in dir() else current_lr,
                        start_time=session_start_time,
                        end_time=current_session_episodes[-1].timestamp,
                    )
                    sessions.append(sess)

                prev_style = current_style
                prev_agents = current_agents
                prev_lr = current_lr
                current_style = m_style.group(1).strip()
                current_session_episodes = []
                session_start_time = None
                continue

            m_agents = CONFIG_AGENTS_PATTERN.search(line)
            if m_agents:
                current_agents = int(m_agents.group(1))
                continue

            m_lr = CONFIG_LR_PATTERN.search(line)
            if m_lr:
                current_lr = float(m_lr.group(1))
                continue

            # Parse episode line
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
                    number=int(m.group(2)),
                    stage=int(m.group(3)),
                    stage_name=m.group(4),
                    reward=float(m.group(5)),
                    steps=int(m.group(6)),
                    food=int(m.group(7)),
                    food_per_step=float(m.group(8)),
                    epsilon=float(m.group(9)),
                    loss=float(m.group(10)),
                    cause=m.group(11),
                    wall_dist=wall_dist,
                    enemy_dist=enemy_dist,
                    timestamp=ts,
                    style=current_style,
                )

                episodes.append(ep)
                current_session_episodes.append(ep)

                if session_start_time is None:
                    session_start_time = ts

    # Save last session
    if current_session_episodes:
        sess = TrainingSession(
            style=current_style,
            start_episode=current_session_episodes[0].number,
            end_episode=current_session_episodes[-1].number,
            episodes=current_session_episodes,
            agents=current_agents,
            lr_config=current_lr,
            start_time=session_start_time,
            end_time=current_session_episodes[-1].timestamp if current_session_episodes else None,
        )
        sessions.append(sess)

    return episodes, sessions


def parse_csv(csv_path: str) -> List[Episode]:
    """Parse training_stats.csv for current session detail."""
    episodes = []
    try:
        with open(csv_path, 'r') as f:
            header = f.readline().strip().split(',')
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 10:
                    continue
                ep = Episode(
                    number=int(parts[0]),
                    steps=int(parts[1]),
                    reward=float(parts[2]),
                    epsilon=float(parts[3]),
                    loss=float(parts[4]),
                    beta=float(parts[5]),
                    lr=float(parts[6]),
                    cause=parts[7],
                    stage=int(parts[8]),
                    food=int(parts[9]),
                    stage_name='',
                    food_per_step=int(parts[9]) / max(int(parts[1]), 1),
                )
                episodes.append(ep)
    except Exception as e:
        print(c(f"  Warning: Could not parse CSV: {e}", C.YEL))
    return episodes


# ═══════════════════════════════════════════════════════
#  STATISTICAL ANALYSIS
# ═══════════════════════════════════════════════════════

def moving_average(data, window):
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')

def linear_trend(data):
    """Returns slope and r-squared of linear fit."""
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
    """Split data into segments and compute trend for each."""
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
            'segment': i + 1,
            'start_idx': start,
            'end_idx': end,
            'mean': np.mean(seg),
            'std': np.std(seg),
            'slope': slope,
            'r2': r2,
        })
    return results

def detect_plateau(data, window=100, threshold=0.01):
    """Detect if metric is in plateau (no significant change)."""
    if len(data) < window * 2:
        return False, 0
    recent = data[-window:]
    earlier = data[-window*2:-window]
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
        'p5': np.percentile(data, 5),
        'p25': np.percentile(data, 25),
        'p50': np.percentile(data, 50),
        'p75': np.percentile(data, 75),
        'p95': np.percentile(data, 95),
        'mean': np.mean(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
    }


# ═══════════════════════════════════════════════════════
#  LEARNING DETECTION ENGINE
# ═══════════════════════════════════════════════════════

@dataclass
class LearningVerdict:
    is_learning: bool
    confidence: float  # 0-1
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    positives: List[str] = field(default_factory=list)
    recommendation: str = ''
    goal_feasibility: str = ''
    estimated_episodes_to_goal: Optional[int] = None


def assess_learning(episodes: List[Episode], csv_episodes: List[Episode],
                    sessions: List[TrainingSession]) -> LearningVerdict:
    """Main learning assessment engine."""
    verdict = LearningVerdict(is_learning=False, confidence=0.0)
    score = 50  # Start neutral

    if len(episodes) < 20:
        verdict.issues.append("Insufficient data (< 20 episodes)")
        verdict.confidence = 0.1
        return verdict

    rewards = np.array([e.reward for e in episodes])
    steps = np.array([e.steps for e in episodes])
    food = np.array([e.food for e in episodes])
    losses = np.array([e.loss for e in episodes])
    epsilons = np.array([e.epsilon for e in episodes])

    # --- CHECK 1: Learning Rate ---
    if csv_episodes:
        current_lr = csv_episodes[-1].lr
        if current_lr == 0.0:
            verdict.issues.append(
                "CRITICAL: Learning Rate = 0.0 - Model CANNOT learn! "
                "The LR scheduler has reduced LR to zero. "
                "Weights are frozen, no gradient updates are happening."
            )
            score -= 40
        elif current_lr < 1e-7:
            verdict.warnings.append(f"Learning rate extremely low: {current_lr:.2e}")
            score -= 15

    # --- CHECK 2: Reward Trend ---
    n_recent = min(len(rewards), 100)
    recent_rewards = rewards[-n_recent:]

    if len(rewards) > 200:
        first_half = rewards[:len(rewards)//2]
        second_half = rewards[len(rewards)//2:]
        reward_improvement = np.mean(second_half) - np.mean(first_half)

        if reward_improvement > 50:
            verdict.positives.append(f"Rewards improving: +{reward_improvement:.1f} (2nd half vs 1st half)")
            score += 15
        elif reward_improvement < -50:
            verdict.issues.append(f"Rewards DECLINING: {reward_improvement:.1f} (getting worse)")
            score -= 15
        else:
            verdict.warnings.append(f"Rewards flat: change = {reward_improvement:.1f} between halves")
            score -= 5

    reward_slope, reward_r2 = linear_trend(rewards)
    if reward_slope > 0.1 and reward_r2 > 0.05:
        verdict.positives.append(f"Positive reward trend (slope={reward_slope:.4f}, R²={reward_r2:.3f})")
        score += 10
    elif reward_slope < -0.1 and reward_r2 > 0.05:
        verdict.issues.append(f"Negative reward trend (slope={reward_slope:.4f}, R²={reward_r2:.3f})")
        score -= 10

    # --- CHECK 3: Survival Duration ---
    steps_slope, steps_r2 = linear_trend(steps.astype(float))
    if steps_slope > 0.05 and steps_r2 > 0.03:
        verdict.positives.append(f"Episodes getting longer (slope={steps_slope:.3f}/ep)")
        score += 10

    avg_steps = np.mean(steps)
    max_steps = np.max(steps)
    if avg_steps < 50:
        verdict.issues.append(f"Very short episodes: avg={avg_steps:.0f} steps (dying too fast)")
        score -= 10
    elif avg_steps > 500:
        verdict.positives.append(f"Good episode length: avg={avg_steps:.0f} steps")
        score += 5

    # --- CHECK 4: Food Collection Trend ---
    food_slope, food_r2 = linear_trend(food.astype(float))
    if food_slope > 0.01:
        verdict.positives.append(f"Food collection improving (slope={food_slope:.4f}/ep)")
        score += 5

    # --- CHECK 5: Loss Behavior ---
    if len(losses) > 50:
        loss_slope, loss_r2 = linear_trend(losses)
        recent_loss_mean = np.mean(losses[-50:])

        if recent_loss_mean < 0.01:
            verdict.warnings.append(f"Loss near zero ({recent_loss_mean:.6f}) - model may have collapsed")
            score -= 10
        elif recent_loss_mean > 10:
            verdict.warnings.append(f"Loss very high ({recent_loss_mean:.2f}) - training unstable")
            score -= 5

        if loss_slope < -0.001 and loss_r2 > 0.05:
            verdict.positives.append("Loss decreasing (model converging)")
            score += 5

    # --- CHECK 6: Epsilon Progress ---
    current_eps = epsilons[-1]
    if current_eps > 0.8:
        verdict.warnings.append(f"Epsilon very high ({current_eps:.3f}) - still mostly random")
        score -= 5
    elif current_eps > 0.3:
        verdict.warnings.append(f"Epsilon moderate ({current_eps:.3f}) - still significant random exploration")
    elif current_eps < 0.15:
        verdict.positives.append(f"Epsilon low ({current_eps:.3f}) - model exploiting learned policy")
        score += 5

    # --- CHECK 7: Death Pattern ---
    causes = [e.cause for e in episodes[-min(200, len(episodes)):]]
    cause_counts = defaultdict(int)
    for ca in causes:
        cause_counts[ca] += 1
    total_deaths = len(causes)

    wall_pct = cause_counts.get('Wall', 0) / total_deaths * 100 if total_deaths > 0 else 0
    if wall_pct > 60:
        verdict.issues.append(f"Wall deaths dominant ({wall_pct:.0f}%) - bot not learning wall avoidance")
        score -= 10

    # --- CHECK 8: Plateau Detection ---
    is_plateau, change = detect_plateau(rewards)
    if is_plateau:
        verdict.warnings.append(f"Reward in plateau (change < 1% over last 200 episodes)")
        score -= 5

    # --- CHECK 9: Multiple Restarts ---
    if len(sessions) > 3:
        verdict.warnings.append(f"Multiple training restarts detected ({len(sessions)} sessions) - fragmented learning")
        score -= 5

    # --- CHECK 10: Best Performance vs Goal ---
    best_reward = np.max(rewards)
    best_steps = int(np.max(steps))
    best_food = int(np.max(food))

    # Goal: 6000 points, 1 hour (~3600 steps at ~1 step/sec, or more depending on speed)
    GOAL_REWARD = 6000
    GOAL_STEPS = 3600  # ~1 hour at 1 step/sec

    reward_pct = (best_reward / GOAL_REWARD) * 100 if GOAL_REWARD > 0 else 0
    steps_pct = (best_steps / GOAL_STEPS) * 100 if GOAL_STEPS > 0 else 0

    # --- FINAL VERDICT ---
    score = max(0, min(100, score))
    verdict.confidence = score / 100.0
    verdict.is_learning = score > 55

    # Goal feasibility
    if score < 25:
        verdict.goal_feasibility = "IMPOSSIBLE with current setup"
        verdict.recommendation = (
            "Training is fundamentally broken. Fix critical issues first:\n"
            "  1. Restore learning rate > 0 (check LR scheduler)\n"
            "  2. Consider resetting epsilon to allow fresh exploration\n"
            "  3. Review reward shaping - excessive penalties prevent learning"
        )
    elif score < 45:
        verdict.goal_feasibility = "VERY UNLIKELY (<5% chance)"
        verdict.recommendation = (
            "Significant issues detected. Major changes needed:\n"
            "  1. Fix learning rate and optimizer state\n"
            "  2. Simplify reward structure\n"
            "  3. Ensure episodes can last long enough to learn from"
        )
    elif score < 65:
        verdict.goal_feasibility = "UNLIKELY (5-25% chance) without tuning"
        verdict.recommendation = (
            "Some learning signals present but not strong enough.\n"
            "  1. Fine-tune hyperparameters\n"
            "  2. Increase training duration significantly\n"
            "  3. Consider curriculum adjustments"
        )
    elif score < 80:
        verdict.goal_feasibility = "POSSIBLE (25-60% chance) with continued training"
        verdict.recommendation = "Keep training. Monitor for sustained improvement."
    else:
        verdict.goal_feasibility = "LIKELY (>60% chance)"
        verdict.recommendation = "Training looks healthy. Continue and monitor."

    # Estimate episodes to goal
    if reward_slope > 0.1:
        remaining_reward = GOAL_REWARD - np.mean(recent_rewards)
        if remaining_reward > 0:
            verdict.estimated_episodes_to_goal = int(remaining_reward / reward_slope)

    return verdict


# ═══════════════════════════════════════════════════════
#  REPORT PRINTER
# ═══════════════════════════════════════════════════════

W = 78  # Terminal width

def sep(char='─', color=C.DIM):
    print(c(char * W, color))

def header(text, color=C.CYN):
    print()
    sep('═', color)
    print(c(f'  {text}', color, C.B))
    sep('═', color)

def section(text, color=C.BLU):
    print()
    print(c(f'  [{text}]', color, C.B))
    sep('─', C.DIM)


def print_full_report(episodes, csv_episodes, sessions, verdict):
    """Print the complete analysis report to terminal."""

    rewards = np.array([e.reward for e in episodes])
    steps = np.array([e.steps for e in episodes])
    food = np.array([e.food for e in episodes])
    losses = np.array([e.loss for e in episodes])
    epsilons = np.array([e.epsilon for e in episodes])

    # ══════════════════════════════════════
    #  HEADER
    # ══════════════════════════════════════
    print()
    print(c('=' * W, C.CYN, C.B))
    print(c('  SLITHER.IO BOT - DEEP TRAINING PROGRESS ANALYSIS', C.CYN, C.B))
    print(c('  ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'), C.DIM))
    print(c('=' * W, C.CYN, C.B))

    # ══════════════════════════════════════
    #  VERDICT (TOP)
    # ══════════════════════════════════════
    section('LEARNING VERDICT', C.MAG)

    if verdict.is_learning:
        status = c(' LEARNING ', C.B, C.BG_GRN)
    else:
        status = c(' NOT LEARNING ', C.B, C.BG_RED)

    confidence_bar = bar(verdict.confidence, 1.0, 30, color=C.GRN if verdict.confidence > 0.5 else C.RED)

    print(f'  Status:     {status}')
    print(f'  Confidence: {confidence_bar} {verdict.confidence*100:.0f}%')
    print(f'  Goal:       {c(verdict.goal_feasibility, C.YEL, C.B)}')

    if verdict.estimated_episodes_to_goal:
        print(f'  ETA:        ~{verdict.estimated_episodes_to_goal:,} episodes to goal')

    # Issues
    if verdict.issues:
        print()
        print(c('  CRITICAL ISSUES:', C.RED, C.B))
        for issue in verdict.issues:
            print(c(f'    ✗ {issue}', C.RED))

    if verdict.warnings:
        print()
        print(c('  WARNINGS:', C.YEL, C.B))
        for w in verdict.warnings:
            print(c(f'    ⚠ {w}', C.YEL))

    if verdict.positives:
        print()
        print(c('  POSITIVE SIGNALS:', C.GRN, C.B))
        for p in verdict.positives:
            print(c(f'    ✓ {p}', C.GRN))

    # ══════════════════════════════════════
    #  TRAINING OVERVIEW
    # ══════════════════════════════════════
    section('TRAINING OVERVIEW')

    total_episodes = len(episodes)
    first_ep = episodes[0].number if episodes else 0
    last_ep = episodes[-1].number if episodes else 0

    if episodes[0].timestamp and episodes[-1].timestamp:
        total_time = episodes[-1].timestamp - episodes[0].timestamp
        hours = total_time.total_seconds() / 3600
    else:
        hours = 0

    print(f'  Total Episodes:    {c(total_episodes, C.WHT, C.B)}')
    print(f'  Episode Range:     {first_ep} -> {last_ep}')
    print(f'  Training Time:     {hours:.1f} hours')
    print(f'  Sessions:          {len(sessions)}')
    print(f'  Current Style:     {c(sessions[-1].style if sessions else "?", C.CYN)}')

    # ══════════════════════════════════════
    #  SESSION HISTORY
    # ══════════════════════════════════════
    section('SESSION HISTORY')

    print(f'  {"#":<3} {"Style":<25} {"Episodes":<18} {"Avg Reward":<12} {"Avg Steps":<10} {"Eps Range":<14}')
    sep('─', C.DIM)

    for i, sess in enumerate(sessions):
        if not sess.episodes:
            continue
        s_rewards = [e.reward for e in sess.episodes]
        s_steps = [e.steps for e in sess.episodes]
        s_eps = [e.epsilon for e in sess.episodes]

        ep_range = f"{sess.start_episode}-{sess.end_episode}"
        eps_range = f"{s_eps[0]:.3f}-{s_eps[-1]:.3f}"

        style_color = C.CYN if i == len(sessions) - 1 else C.DIM
        print(f'  {c(i+1, style_color):<12} {c(sess.style[:24], style_color):<34} '
              f'{ep_range:<18} {np.mean(s_rewards):>9.1f}   {np.mean(s_steps):>7.0f}   {eps_range}')

    # ══════════════════════════════════════
    #  REWARD ANALYSIS
    # ══════════════════════════════════════
    section('REWARD ANALYSIS')

    rp = compute_percentiles(rewards)
    print(f'  Mean:    {rp["mean"]:>10.2f}    Std:  {rp["std"]:>10.2f}')
    print(f'  Median:  {rp["p50"]:>10.2f}    Min:  {rp["min"]:>10.2f}')
    print(f'  P95:     {rp["p95"]:>10.2f}    Max:  {rp["max"]:>10.2f}')

    # Reward by time windows
    windows = [50, 100, 500, 1000]
    print()
    print(f'  {"Window":<10} {"Mean Rw":<12} {"Std Rw":<12} {"Best":<12} {"Trend":<12}')
    sep('─', C.DIM)

    for w in windows:
        if len(rewards) >= w:
            wr = rewards[-w:]
            sl, r2 = linear_trend(wr)
            trend = c("↗", C.GRN) if sl > 0.1 else (c("↘", C.RED) if sl < -0.1 else c("→", C.DIM))
            print(f'  Last {w:<4} {np.mean(wr):>10.2f}  {np.std(wr):>10.2f}  {np.max(wr):>10.2f}  {trend} ({sl:+.3f})')

    # ══════════════════════════════════════
    #  SURVIVAL ANALYSIS
    # ══════════════════════════════════════
    section('SURVIVAL ANALYSIS')

    sp = compute_percentiles(steps.astype(float))
    print(f'  Mean Steps:  {sp["mean"]:>8.1f}    Max: {sp["max"]:>8.0f}')
    print(f'  P75 Steps:   {sp["p75"]:>8.1f}    P95: {sp["p95"]:>8.0f}')

    # Duration estimation (assuming ~1 step = ~2 seconds of game time)
    STEP_DURATION = 2.0  # seconds per step (approximate)
    avg_duration_sec = sp["mean"] * STEP_DURATION
    max_duration_sec = sp["max"] * STEP_DURATION

    print()
    print(f'  Estimated avg survival: {c(f"{avg_duration_sec:.0f}s ({avg_duration_sec/60:.1f} min)", C.YEL)}')
    print(f'  Estimated max survival: {c(f"{max_duration_sec:.0f}s ({max_duration_sec/60:.1f} min)", C.CYN)}')
    print(f'  Goal (1 hour):          {c("3600s (60 min)", C.GRN, C.B)}')

    goal_pct = (max_duration_sec / 3600) * 100
    print(f'  Progress to goal:       {bar(goal_pct, 100, 30)} {goal_pct:.1f}%')

    # ══════════════════════════════════════
    #  FOOD COLLECTION
    # ══════════════════════════════════════
    section('FOOD COLLECTION')

    fp = compute_percentiles(food.astype(float))
    food_per_step = np.array([e.food_per_step for e in episodes])
    fps_p = compute_percentiles(food_per_step)

    print(f'  Mean Food/Ep:    {fp["mean"]:>8.1f}    Max: {int(fp["max"])}')
    print(f'  Mean Food/Step:  {fps_p["mean"]:>8.4f}    Max: {fps_p["max"]:.3f}')

    # Food = score proxy. Goal: 6000 points
    # In slither.io, each food gives ~1 point
    GOAL_FOOD = 6000
    best_food = int(fp["max"])
    food_goal_pct = (best_food / GOAL_FOOD) * 100

    print(f'  Best food collected:  {c(best_food, C.CYN, C.B)}')
    print(f'  Goal (6000 pts):      {c(GOAL_FOOD, C.GRN, C.B)}')
    print(f'  Progress to goal:     {bar(food_goal_pct, 100, 30)} {food_goal_pct:.1f}%')

    # ══════════════════════════════════════
    #  DEATH ANALYSIS
    # ══════════════════════════════════════
    section('DEATH CAUSE ANALYSIS')

    # Full history
    cause_counts = defaultdict(int)
    for e in episodes:
        cause_counts[e.cause] += 1
    total = len(episodes)

    sorted_causes = sorted(cause_counts.items(), key=lambda x: x[1], reverse=True)

    for cause, count in sorted_causes:
        pct = count / total * 100
        cc = C.RED if cause == 'Wall' else C.YEL if cause == 'SnakeCollision' else C.DIM
        print(f'  {cause:<18} {count:>5} ({pct:>5.1f}%)  {bar(pct, 100, 25, color=cc)}')

    # Trend in death causes (early vs recent)
    if len(episodes) > 200:
        early = episodes[:len(episodes)//2]
        late = episodes[len(episodes)//2:]

        early_wall = sum(1 for e in early if e.cause == 'Wall') / len(early) * 100
        late_wall = sum(1 for e in late if e.cause == 'Wall') / len(late) * 100

        early_snake = sum(1 for e in early if e.cause == 'SnakeCollision') / len(early) * 100
        late_snake = sum(1 for e in late if e.cause == 'SnakeCollision') / len(late) * 100

        print()
        print(f'  Death cause evolution:')
        wall_arrow = c("↘", C.GRN) if late_wall < early_wall else c("↗", C.RED)
        snake_arrow = c("↘", C.GRN) if late_snake < early_snake else c("↗", C.RED)
        print(f'    Wall:           {early_wall:>5.1f}% -> {late_wall:>5.1f}% {wall_arrow}')
        print(f'    SnakeCollision: {early_snake:>5.1f}% -> {late_snake:>5.1f}% {snake_arrow}')

    # ══════════════════════════════════════
    #  TRAINING HEALTH
    # ══════════════════════════════════════
    section('TRAINING HEALTH')

    current_eps = epsilons[-1]
    current_loss = losses[-1] if len(losses) > 0 else 0

    # Epsilon
    eps_color = C.RED if current_eps > 0.5 else C.YEL if current_eps > 0.2 else C.GRN
    print(f'  Epsilon:    {c(f"{current_eps:.4f}", eps_color)}  {bar(1-current_eps, 1.0, 20, color=C.GRN)} exploitation')

    # LR from CSV
    if csv_episodes:
        current_lr = csv_episodes[-1].lr
        lr_color = C.RED if current_lr == 0 else C.YEL if current_lr < 1e-6 else C.GRN
        print(f'  Learn Rate: {c(f"{current_lr:.8f}", lr_color)}', end='')
        if current_lr == 0:
            print(c('  *** ZERO - NO LEARNING POSSIBLE ***', C.RED, C.B))
        else:
            print()

        current_beta = csv_episodes[-1].beta
        print(f'  Beta (PER): {current_beta:.4f}')

    # Loss
    print(f'  Last Loss:  {current_loss:.4f}')
    if len(losses) > 50:
        print(f'  Avg Loss (last 50): {np.mean(losses[-50:]):.4f}')

    # ══════════════════════════════════════
    #  TREND ANALYSIS (SEGMENTED)
    # ══════════════════════════════════════
    section('SEGMENTED TREND ANALYSIS')

    segments = segment_trends(rewards, 4)
    if segments:
        print(f'  {"Quarter":<10} {"Mean Rw":<12} {"Std":<10} {"Trend":<14} {"R²":<8}')
        sep('─', C.DIM)
        for seg in segments:
            trend_str = c("↗ UP", C.GRN) if seg['slope'] > 0.5 else (c("↘ DOWN", C.RED) if seg['slope'] < -0.5 else c("→ FLAT", C.YEL))
            print(f'  Q{seg["segment"]}        {seg["mean"]:>10.1f}  {seg["std"]:>8.1f}  {trend_str:<22} {seg["r2"]:.4f}')

    # ══════════════════════════════════════
    #  GOAL DISTANCE CALCULATOR
    # ══════════════════════════════════════
    section('GOAL DISTANCE ANALYSIS', C.MAG)

    GOAL_POINTS = 6000
    GOAL_SURVIVAL_SECS = 3600  # 1 hour
    GOAL_SURVIVAL_STEPS = GOAL_SURVIVAL_SECS / STEP_DURATION

    best_food_val = int(np.max(food))
    avg_food_recent = np.mean(food[-min(50, len(food)):])
    best_steps_val = int(np.max(steps))
    avg_steps_recent = np.mean(steps[-min(50, len(steps)):])

    print(f'  {"Metric":<22} {"Current Best":<16} {"Recent Avg":<16} {"Goal":<12} {"% Done":<10}')
    sep('─', C.DIM)

    pts_pct = min(best_food_val / GOAL_POINTS * 100, 100)
    surv_pct = min(best_steps_val / GOAL_SURVIVAL_STEPS * 100, 100)

    pts_color = C.GRN if pts_pct > 50 else C.YEL if pts_pct > 20 else C.RED
    surv_color = C.GRN if surv_pct > 50 else C.YEL if surv_pct > 20 else C.RED

    print(f'  {"Points (food)":<22} {best_food_val:<16} {avg_food_recent:<16.1f} {GOAL_POINTS:<12} {c(f"{pts_pct:.1f}%", pts_color)}')
    print(f'  {"Survival (steps)":<22} {best_steps_val:<16} {avg_steps_recent:<16.1f} {int(GOAL_SURVIVAL_STEPS):<12} {c(f"{surv_pct:.1f}%", surv_color)}')

    print()
    print(f'  Points progress:   {bar(pts_pct, 100, 40, color=pts_color)} {pts_pct:.1f}%')
    print(f'  Survival progress: {bar(surv_pct, 100, 40, color=surv_color)} {surv_pct:.1f}%')

    # ══════════════════════════════════════
    #  RECOMMENDATIONS
    # ══════════════════════════════════════
    section('RECOMMENDATIONS', C.MAG)

    print(c(f'  {verdict.recommendation}', C.WHT))

    # Additional specific recommendations
    print()
    recs = generate_specific_recommendations(episodes, csv_episodes, sessions, verdict)
    for i, rec in enumerate(recs, 1):
        print(c(f'  {i}. {rec}', C.CYN))

    sep('═', C.CYN)
    print()


def generate_specific_recommendations(episodes, csv_episodes, sessions, verdict):
    """Generate specific actionable recommendations."""
    recs = []

    if csv_episodes and csv_episodes[-1].lr == 0:
        recs.append(
            "URGENT: Reset learning rate. In config.py, the LR scheduler has reduced LR to 0. "
            "Either restart with a fresh optimizer state or set a minimum LR floor (e.g., 1e-6)."
        )

    epsilons = [e.epsilon for e in episodes]
    if epsilons[-1] > 0.3:
        recs.append(
            f"Epsilon is {epsilons[-1]:.3f} after {len(episodes)} episodes. "
            f"Consider faster decay (eps_decay={int(50000)}) or lower eps_start if resuming."
        )

    steps_arr = np.array([e.steps for e in episodes[-100:]])
    if np.mean(steps_arr) < 100:
        recs.append(
            "Average episode too short. Consider:\n"
            "     - Reducing death penalties to avoid discouraging exploration\n"
            "     - Adding survival bonus to incentivize staying alive"
        )

    causes = [e.cause for e in episodes[-200:]]
    wall_pct = sum(1 for c in causes if c == 'Wall') / len(causes)
    if wall_pct > 0.5:
        recs.append(
            f"Wall deaths are {wall_pct*100:.0f}% of recent deaths. The bot hits map boundaries. "
            f"Increase wall_proximity_penalty or wall_alert_dist in reward config."
        )

    if len(sessions) > 2:
        recs.append(
            "Multiple session restarts detected. Each restart disrupts learning continuity. "
            "Try to maintain consistent training runs of 10,000+ episodes."
        )

    rewards = [e.reward for e in episodes]
    if np.mean(rewards) < 0:
        recs.append(
            f"Average reward is negative ({np.mean(rewards):.1f}). The bot is being penalized "
            f"more than rewarded. Reduce penalty magnitudes or increase food_reward."
        )

    if not recs:
        recs.append("No critical issues detected. Continue training and monitor progress.")

    return recs


# ═══════════════════════════════════════════════════════
#  CHART GENERATION
# ═══════════════════════════════════════════════════════

def generate_charts(episodes, csv_episodes, sessions, verdict, output_dir):
    """Generate comprehensive analysis charts."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print(c("  matplotlib not installed, skipping charts.", C.YEL))
        return

    plt.style.use('dark_background')

    rewards = np.array([e.reward for e in episodes])
    steps = np.array([e.steps for e in episodes])
    food = np.array([e.food for e in episodes])
    losses = np.array([e.loss for e in episodes])
    epsilons = np.array([e.epsilon for e in episodes])
    ep_nums = np.array([e.number for e in episodes])

    # ─── CHART 1: COMPREHENSIVE OVERVIEW (6 panels) ───
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('Slither.io Bot - Training Progress Analysis',
                 fontsize=18, fontweight='bold', color='white', y=0.98)

    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    # 1. Reward History
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.fill_between(ep_nums, rewards, alpha=0.1, color='gray')
    ax1.plot(ep_nums, rewards, alpha=0.15, color='gray', linewidth=0.5)

    if len(rewards) > 50:
        sma50 = moving_average(rewards, 50)
        ax1.plot(ep_nums[49:], sma50, color='#00ff88', linewidth=2, label='SMA-50')
    if len(rewards) > 200:
        sma200 = moving_average(rewards, 200)
        ax1.plot(ep_nums[199:], sma200, color='#ff6600', linewidth=2, label='SMA-200')

    ax1.axhline(y=0, color='white', linestyle=':', alpha=0.3)
    ax1.set_title('Reward History', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.15)

    # 2. Survival Duration
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(ep_nums, steps, alpha=0.15, color='gray', linewidth=0.5)
    if len(steps) > 50:
        sma = moving_average(steps.astype(float), 50)
        ax2.plot(ep_nums[49:], sma, color='#00ccff', linewidth=2, label='Steps (SMA-50)')
    ax2.axhline(y=1800, color='lime', linestyle='--', alpha=0.5, label='Goal: 1hr (1800 steps)')
    ax2.set_title('Survival Duration', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.15)

    # 3. Food Collection
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(ep_nums, food, alpha=0.15, color='gray', linewidth=0.5)
    if len(food) > 50:
        sma = moving_average(food.astype(float), 50)
        ax3.plot(ep_nums[49:], sma, color='#ffaa00', linewidth=2, label='Food (SMA-50)')
    ax3.axhline(y=6000, color='lime', linestyle='--', alpha=0.5, label='Goal: 6000')
    ax3.set_title('Food Collected per Episode', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Food')
    ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.15)

    # 4. Loss
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(ep_nums, losses, alpha=0.15, color='gray', linewidth=0.5)
    if len(losses) > 50:
        sma = moving_average(losses, 50)
        ax4.plot(ep_nums[49:], sma, color='#ff4444', linewidth=2, label='Loss (SMA-50)')
    ax4.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Loss')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.15)

    # 5. Epsilon + LR
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(ep_nums, epsilons, color='#ff6600', linewidth=1.5, label='Epsilon')
    ax5.set_ylabel('Epsilon', color='#ff6600')
    ax5.set_title('Exploration Rate (Epsilon)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Episode')
    ax5.legend(loc='upper right', fontsize=8)
    ax5.grid(True, alpha=0.15)

    # 6. Death Cause Distribution
    ax6 = fig.add_subplot(gs[2, 1])
    cause_counts = defaultdict(int)
    for e in episodes:
        cause_counts[e.cause] += 1

    # Death cause over time (rolling window)
    window_size = max(50, len(episodes) // 20)
    cause_names = ['Wall', 'SnakeCollision', 'Unknown']
    cause_colors = ['#ff4444', '#ffaa00', '#888888']

    if len(episodes) > window_size:
        x_points = []
        cause_series = {cn: [] for cn in cause_names}

        for i in range(0, len(episodes) - window_size, max(1, window_size // 5)):
            window = episodes[i:i + window_size]
            x_points.append(window[len(window)//2].number)
            total = len(window)
            for cn in cause_names:
                cnt = sum(1 for e in window if e.cause == cn)
                cause_series[cn].append(cnt / total * 100)

        for cn, cc in zip(cause_names, cause_colors):
            if cause_series[cn]:
                ax6.plot(x_points, cause_series[cn], color=cc, linewidth=1.5, label=cn)
        ax6.set_ylabel('% of Deaths')
    else:
        labels = list(cause_counts.keys())
        sizes = list(cause_counts.values())
        ax6.pie(sizes, labels=labels, autopct='%1.1f%%', colors=cause_colors[:len(labels)])

    ax6.set_title('Death Causes Over Time', fontsize=12, fontweight='bold')
    ax6.legend(loc='upper right', fontsize=8)
    ax6.grid(True, alpha=0.15)

    # Save
    overview_path = os.path.join(output_dir, 'training_progress_overview.png')
    plt.savefig(overview_path, dpi=120, facecolor='#1a1a2e', bbox_inches='tight')
    plt.close()
    print(c(f'  Chart saved: {overview_path}', C.GRN))

    # ─── CHART 2: LEARNING DETECTION ───
    fig2, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig2.suptitle('Learning Detection Analysis', fontsize=16, fontweight='bold', color='white')

    # Reward distribution: early vs late
    ax = axes[0, 0]
    half = len(rewards) // 2
    if half > 20:
        ax.hist(rewards[:half], bins=40, alpha=0.5, color='#ff6666', label=f'First half (n={half})', density=True)
        ax.hist(rewards[half:], bins=40, alpha=0.5, color='#66ff66', label=f'Second half (n={len(rewards)-half})', density=True)
        ax.axvline(np.mean(rewards[:half]), color='#ff6666', linestyle='--', linewidth=2)
        ax.axvline(np.mean(rewards[half:]), color='#66ff66', linestyle='--', linewidth=2)
    ax.set_title('Reward Distribution: Early vs Late')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.15)

    # Cumulative reward
    ax = axes[0, 1]
    cum_reward = np.cumsum(rewards)
    ax.plot(ep_nums, cum_reward, color='#00ff88', linewidth=1.5)
    ax.set_title('Cumulative Reward')
    ax.set_xlabel('Episode')
    ax.grid(True, alpha=0.15)
    # If declining = bad
    if cum_reward[-1] < cum_reward[len(cum_reward)//2]:
        ax.fill_between(ep_nums, cum_reward, alpha=0.1, color='red')
    else:
        ax.fill_between(ep_nums, cum_reward, alpha=0.1, color='green')

    # Steps distribution: early vs late
    ax = axes[1, 0]
    if half > 20:
        ax.hist(steps[:half], bins=40, alpha=0.5, color='#ff6666', label='First half', density=True)
        ax.hist(steps[half:], bins=40, alpha=0.5, color='#66ff66', label='Second half', density=True)
    ax.set_title('Steps Distribution: Early vs Late')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.15)

    # Session comparison
    ax = axes[1, 1]
    sess_means = []
    sess_labels = []
    for i, sess in enumerate(sessions):
        if sess.episodes:
            mean_r = np.mean([e.reward for e in sess.episodes])
            sess_means.append(mean_r)
            label = f"S{i+1}\n{sess.style[:10]}"
            sess_labels.append(label)

    if sess_means:
        colors_list = ['#ff6666' if m < 0 else '#66ff66' for m in sess_means]
        ax.bar(range(len(sess_means)), sess_means, color=colors_list)
        ax.set_xticks(range(len(sess_means)))
        ax.set_xticklabels(sess_labels, fontsize=7)
        ax.axhline(y=0, color='white', linestyle=':', alpha=0.3)
    ax.set_title('Average Reward by Session')
    ax.grid(True, alpha=0.15)

    learning_path = os.path.join(output_dir, 'training_learning_detection.png')
    plt.tight_layout()
    plt.savefig(learning_path, dpi=120, facecolor='#1a1a2e', bbox_inches='tight')
    plt.close()
    print(c(f'  Chart saved: {learning_path}', C.GRN))

    # ─── CHART 3: GOAL PROGRESS GAUGE ───
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
    fig3.suptitle('Goal Progress', fontsize=14, fontweight='bold', color='white')

    # Points gauge
    ax = axes3[0]
    best_food_val = int(np.max(food))
    pct = min(best_food_val / 6000 * 100, 100)
    theta = np.linspace(0, np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), color='#333', linewidth=15)
    fill_theta = np.linspace(0, np.pi * pct / 100, 100)
    color = '#ff4444' if pct < 20 else '#ffaa00' if pct < 50 else '#00ff88'
    ax.plot(np.cos(fill_theta), np.sin(fill_theta), color=color, linewidth=15)
    ax.text(0, 0.3, f'{best_food_val}', ha='center', fontsize=28, fontweight='bold', color='white')
    ax.text(0, 0.05, f'/ 6000 pts', ha='center', fontsize=12, color='gray')
    ax.text(0, -0.2, f'{pct:.1f}%', ha='center', fontsize=16, color=color)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.4, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Points (Best)')

    # Survival gauge
    ax = axes3[1]
    best_steps_val = int(np.max(steps))
    STEP_DURATION = 2.0
    best_min = best_steps_val * STEP_DURATION / 60
    goal_min = 60
    pct2 = min(best_min / goal_min * 100, 100)
    ax.plot(np.cos(theta), np.sin(theta), color='#333', linewidth=15)
    fill_theta2 = np.linspace(0, np.pi * pct2 / 100, 100)
    color2 = '#ff4444' if pct2 < 20 else '#ffaa00' if pct2 < 50 else '#00ff88'
    ax.plot(np.cos(fill_theta2), np.sin(fill_theta2), color=color2, linewidth=15)
    ax.text(0, 0.3, f'{best_min:.1f} min', ha='center', fontsize=28, fontweight='bold', color='white')
    ax.text(0, 0.05, f'/ 60 min', ha='center', fontsize=12, color='gray')
    ax.text(0, -0.2, f'{pct2:.1f}%', ha='center', fontsize=16, color=color2)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.4, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Survival (Best)')

    gauge_path = os.path.join(output_dir, 'training_goal_progress.png')
    plt.savefig(gauge_path, dpi=120, facecolor='#1a1a2e', bbox_inches='tight')
    plt.close()
    print(c(f'  Chart saved: {gauge_path}', C.GRN))


# ═══════════════════════════════════════════════════════
#  MARKDOWN REPORT
# ═══════════════════════════════════════════════════════

def generate_markdown(episodes, csv_episodes, sessions, verdict, output_path):
    """Generate comprehensive markdown report."""

    rewards = np.array([e.reward for e in episodes])
    steps = np.array([e.steps for e in episodes])
    food = np.array([e.food for e in episodes])
    losses = np.array([e.loss for e in episodes])

    with open(output_path, 'w') as f:
        f.write("# Slither.io Bot - Training Progress Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Total Episodes:** {len(episodes)}  \n")
        f.write(f"**Training Sessions:** {len(sessions)}\n\n")

        # Verdict
        status = "LEARNING" if verdict.is_learning else "NOT LEARNING"
        f.write(f"## Verdict: {status} (Confidence: {verdict.confidence*100:.0f}%)\n\n")
        f.write(f"**Goal Feasibility:** {verdict.goal_feasibility}\n\n")

        if verdict.issues:
            f.write("### Critical Issues\n")
            for issue in verdict.issues:
                f.write(f"- {issue}\n")
            f.write("\n")

        if verdict.warnings:
            f.write("### Warnings\n")
            for w in verdict.warnings:
                f.write(f"- {w}\n")
            f.write("\n")

        if verdict.positives:
            f.write("### Positive Signals\n")
            for p in verdict.positives:
                f.write(f"- {p}\n")
            f.write("\n")

        # Statistics
        f.write("## Key Statistics\n\n")
        f.write("| Metric | Mean | Std | Min | Max | P50 | P95 |\n")
        f.write("|--------|------|-----|-----|-----|-----|-----|\n")

        for name, data in [('Reward', rewards), ('Steps', steps.astype(float)),
                           ('Food', food.astype(float)), ('Loss', losses)]:
            p = compute_percentiles(data)
            f.write(f"| {name} | {p['mean']:.2f} | {p['std']:.2f} | {p['min']:.2f} | "
                    f"{p['max']:.2f} | {p['p50']:.2f} | {p['p95']:.2f} |\n")
        f.write("\n")

        # Goal Progress
        f.write("## Goal Progress\n\n")
        best_food_val = int(np.max(food))
        best_steps_val = int(np.max(steps))
        STEP_DURATION = 2.0

        f.write(f"| Target | Current Best | Goal | Progress |\n")
        f.write(f"|--------|-------------|------|----------|\n")
        f.write(f"| Points | {best_food_val} | 6,000 | {best_food_val/6000*100:.1f}% |\n")
        f.write(f"| Survival | {best_steps_val*STEP_DURATION/60:.1f} min | 60 min | {best_steps_val*STEP_DURATION/3600*100:.1f}% |\n\n")

        # Session History
        f.write("## Session History\n\n")
        f.write("| # | Style | Episodes | Avg Reward | Avg Steps |\n")
        f.write("|---|-------|----------|------------|----------|\n")
        for i, sess in enumerate(sessions):
            if not sess.episodes:
                continue
            s_rewards = [e.reward for e in sess.episodes]
            s_steps = [e.steps for e in sess.episodes]
            f.write(f"| {i+1} | {sess.style} | {sess.start_episode}-{sess.end_episode} | "
                    f"{np.mean(s_rewards):.1f} | {np.mean(s_steps):.0f} |\n")
        f.write("\n")

        # Recommendations
        f.write("## Recommendations\n\n")
        f.write(f"{verdict.recommendation}\n\n")

        recs = generate_specific_recommendations(episodes, csv_episodes, sessions, verdict)
        for i, rec in enumerate(recs, 1):
            f.write(f"{i}. {rec}\n\n")

        # Charts
        f.write("## Charts\n\n")
        f.write("![Overview](training_progress_overview.png)\n\n")
        f.write("![Learning Detection](training_learning_detection.png)\n\n")
        f.write("![Goal Progress](training_goal_progress.png)\n\n")

    print(c(f'  Report saved: {output_path}', C.GRN))


# ═══════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Slither.io Bot Training Progress Analyzer')
    parser.add_argument('--log', default=None, help='Path to train.log')
    parser.add_argument('--csv', default=None, help='Path to training_stats.csv')
    parser.add_argument('--no-charts', action='store_true', help='Skip chart generation')
    parser.add_argument('--no-report', action='store_true', help='Skip markdown report')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Find files
    log_path = args.log or os.path.join(script_dir, 'logs', 'train.log')
    csv_path = args.csv or os.path.join(script_dir, 'training_stats.csv')

    print(c('\n  Loading training data...', C.CYN))

    # Parse log
    episodes = []
    sessions = []
    if os.path.exists(log_path):
        print(f'  Parsing log: {log_path}')
        episodes, sessions = parse_log(log_path)
        print(c(f'  Found {len(episodes)} episodes in {len(sessions)} sessions', C.GRN))
    else:
        print(c(f'  Log not found: {log_path}', C.YEL))

    # Parse CSV
    csv_episodes = []
    if os.path.exists(csv_path):
        print(f'  Parsing CSV: {csv_path}')
        csv_episodes = parse_csv(csv_path)
        print(c(f'  Found {len(csv_episodes)} episodes in CSV', C.GRN))

    # Use log episodes if available, otherwise CSV
    if not episodes and csv_episodes:
        episodes = csv_episodes

    if not episodes:
        print(c('\n  No training data found!', C.RED, C.B))
        return

    # Assess learning
    print(c('  Analyzing learning progress...', C.CYN))
    verdict = assess_learning(episodes, csv_episodes, sessions)

    # Print full report
    print_full_report(episodes, csv_episodes, sessions, verdict)

    # Generate charts
    if not args.no_charts:
        section('GENERATING CHARTS')
        generate_charts(episodes, csv_episodes, sessions, verdict, script_dir)

    # Generate markdown
    if not args.no_report:
        section('GENERATING REPORT')
        md_path = os.path.join(script_dir, 'progress_report.md')
        generate_markdown(episodes, csv_episodes, sessions, verdict, md_path)

    print()
    print(c('  Analysis complete.', C.GRN, C.B))
    print()


if __name__ == '__main__':
    main()
