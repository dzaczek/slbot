#!/usr/bin/env python3
"""
=============================================================================
  KARPATHY MOD EXPERIMENT ANALYZER v1
=============================================================================
  Comprehensive analysis and charting tool for Karpathy Mod autonomous
  reward-parameter mutation experiments.

  Parses karpathy_mod_results.tsv and generates professional charts,
  terminal reports, and markdown summaries.

  Usage:
    python karpathy_mod_analyzer.py                    # full analysis
    python karpathy_mod_analyzer.py --no-charts        # text report only
    python karpathy_mod_analyzer.py --no-report        # skip markdown
    python karpathy_mod_analyzer.py --tsv PATH         # custom TSV path
    python karpathy_mod_analyzer.py --last N           # only last N rounds
=============================================================================
"""

import os
import re
import sys
import csv
import argparse
import warnings
from datetime import datetime
from collections import defaultdict, Counter
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


W = 80


def sep(char='\u2500', color=C.DIM):
    print(c(char * W, color))


def header(text, color=C.CYN):
    sep('\u2550', color)
    print(c(f'  {text}', color, C.B))
    sep('\u2550', color)


def section(text, color=C.BLU):
    print()
    print(c(f'  [{text}]', color, C.B))
    sep('\u2500', C.DIM)


# ═══════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════

STAGE_NAMES = {
    1: 'FOOD_VECTOR', 2: 'WALL_AVOID', 3: 'ENEMY_AVOID',
    4: 'MASS_MANAGEMENT', 5: 'MASTERY_SURVIVAL', 6: 'APEX_PREDATOR',
}

STAGE_COLORS_HEX = {
    1: '#ff6666', 2: '#ffaa00', 3: '#00ccff',
    4: '#00ff88', 5: '#ff66ff', 6: '#ff3333',
}

STRATEGY_COLORS = {
    'tweak': '#58a6ff',
    'explore': '#f0883e',
    'radical': '#f85149',
    'targeted': '#3fb950',
    'crossover': '#bc8cff',
}

DECISION_COLORS = {
    'keep': '#3fb950',
    'discard': '#f85149',
    'inconclusive': '#f0883e',
}

# GitHub-style palette
PAL_BLUE = '#58a6ff'
PAL_ORANGE = '#f0883e'
PAL_GREEN = '#3fb950'
PAL_RED = '#f85149'
PAL_PURPLE = '#bc8cff'
PAL_CORAL = '#ff7b72'
BG_COLOR = '#0d1117'
TEXT_COLOR = '#c9d1d9'
GRID_COLOR = '#21262d'

DEFAULT_TSV = 'karpathy_mod_results.tsv'
CHARTS_DIR = 'charts'

TSV_COLUMNS = [
    'timestamp', 'round', 'experiment_id', 'strategy', 'stage',
    'decision', 'improvement_pct', 'baseline_score', 'experiment_score',
    'avg_steps', 'avg_food', 'peak_length', 'snake_death_rate', 'description',
]


# ═══════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════

def _safe_float(val, default=0.0):
    try:
        return float(val) if val else default
    except (ValueError, TypeError):
        return default


def _safe_int(val, default=0):
    try:
        return int(val) if val else default
    except (ValueError, TypeError):
        return default


def load_tsv(tsv_path: str, last_n: Optional[int] = None) -> List[dict]:
    """Load experiment results from TSV file."""
    if not os.path.exists(tsv_path):
        print(c(f'  ERROR: TSV file not found: {tsv_path}', C.RED, C.B))
        return []

    experiments = []
    with open(tsv_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            exp = {
                'timestamp': row.get('timestamp', ''),
                'round': _safe_int(row.get('round')),
                'experiment_id': row.get('experiment_id', ''),
                'strategy': row.get('strategy', 'unknown'),
                'stage': _safe_int(row.get('stage'), 1),
                'decision': row.get('decision', 'inconclusive'),
                'improvement_pct': _safe_float(row.get('improvement_pct')),
                'baseline_score': _safe_float(row.get('baseline_score')),
                'experiment_score': _safe_float(row.get('experiment_score')),
                'avg_steps': _safe_float(row.get('avg_steps')),
                'avg_food': _safe_float(row.get('avg_food')),
                'peak_length': _safe_float(row.get('peak_length')),
                'snake_death_rate': _safe_float(row.get('snake_death_rate')),
                'description': row.get('description', ''),
            }
            experiments.append(exp)

    if not experiments:
        print(c('  WARNING: TSV file is empty or has no valid rows', C.YEL))
        return []

    if last_n is not None and last_n > 0:
        experiments = experiments[-last_n:]

    return experiments


# ═══════════════════════════════════════════════════════
#  ANALYSIS HELPERS
# ═══════════════════════════════════════════════════════

def compute_stats(experiments: List[dict]) -> dict:
    """Compute summary statistics from experiments."""
    total = len(experiments)
    if total == 0:
        return {}

    decisions = Counter(e['decision'] for e in experiments)
    strategies = Counter(e['strategy'] for e in experiments)
    stages = Counter(e['stage'] for e in experiments)
    improvements = [e['improvement_pct'] for e in experiments]

    kept = [e for e in experiments if e['decision'] == 'keep']
    discarded = [e for e in experiments if e['decision'] == 'discard']
    inconclusive = [e for e in experiments if e['decision'] == 'inconclusive']

    keep_rate = len(kept) / total * 100 if total > 0 else 0.0

    # Strategy breakdown
    strategy_stats = {}
    for strat in sorted(strategies.keys()):
        strat_exps = [e for e in experiments if e['strategy'] == strat]
        strat_kept = [e for e in strat_exps if e['decision'] == 'keep']
        strat_imps = [e['improvement_pct'] for e in strat_exps]
        strategy_stats[strat] = {
            'count': len(strat_exps),
            'kept': len(strat_kept),
            'keep_rate': len(strat_kept) / len(strat_exps) * 100 if strat_exps else 0,
            'avg_improvement': np.mean(strat_imps) if strat_imps else 0,
            'max_improvement': max(strat_imps) if strat_imps else 0,
        }

    # Stage breakdown
    stage_stats = {}
    for stg in sorted(stages.keys()):
        stg_exps = [e for e in experiments if e['stage'] == stg]
        stg_kept = [e for e in stg_exps if e['decision'] == 'keep']
        stg_imps = [e['improvement_pct'] for e in stg_exps]
        stage_stats[stg] = {
            'count': len(stg_exps),
            'kept': len(stg_kept),
            'keep_rate': len(stg_kept) / len(stg_exps) * 100 if stg_exps else 0,
            'avg_improvement': np.mean(stg_imps) if stg_imps else 0,
        }

    # Top experiments
    best_kept = sorted(kept, key=lambda e: e['improvement_pct'], reverse=True)[:5]
    worst_discarded = sorted(discarded, key=lambda e: e['improvement_pct'])[:5]

    # Trend: compare first half vs second half keep rates
    mid = total // 2
    if mid > 0:
        first_half = experiments[:mid]
        second_half = experiments[mid:]
        first_keep = sum(1 for e in first_half if e['decision'] == 'keep') / len(first_half) * 100
        second_keep = sum(1 for e in second_half if e['decision'] == 'keep') / len(second_half) * 100
        if second_keep > first_keep + 5:
            trend = 'IMPROVING'
        elif second_keep < first_keep - 5:
            trend = 'REGRESSING'
        else:
            trend = 'PLATEAUING'
    else:
        trend = 'INSUFFICIENT_DATA'

    return {
        'total': total,
        'decisions': decisions,
        'strategies': strategies,
        'stages': stages,
        'keep_rate': keep_rate,
        'kept': kept,
        'discarded': discarded,
        'inconclusive': inconclusive,
        'improvements': improvements,
        'best_improvement': max(improvements) if improvements else 0,
        'worst_regression': min(improvements) if improvements else 0,
        'avg_improvement': np.mean(improvements) if improvements else 0,
        'strategy_stats': strategy_stats,
        'stage_stats': stage_stats,
        'best_kept': best_kept,
        'worst_discarded': worst_discarded,
        'trend': trend,
    }


def extract_parameters(description: str) -> List[str]:
    """Extract parameter names from experiment description text."""
    params = []
    # Match patterns like "param_name: value -> value" or "param_name=value"
    patterns = [
        re.compile(r'(\w+(?:\.\w+)*)\s*[:=]\s*[\d.\-]+\s*(?:->|=>|to)\s*[\d.\-]+'),
        re.compile(r'(\w+(?:\.\w+)*)\s*(?:increased|decreased|changed|set|adjusted)'),
        re.compile(r'(?:increased|decreased|changed|set|adjusted)\s+(\w+(?:\.\w+)*)'),
        re.compile(r'(\w+(?:_\w+)+)\s*[:=]'),
    ]
    for pat in patterns:
        for m in pat.finditer(description):
            param = m.group(1).strip()
            if len(param) > 2 and param not in params:
                params.append(param)
    # Fallback: extract any snake_case or dot.notation tokens
    if not params:
        tokens = re.findall(r'[a-z]\w*(?:\.\w+)+|[a-z]\w*_\w+', description, re.IGNORECASE)
        params = list(dict.fromkeys(tokens))[:5]
    return params


# ═══════════════════════════════════════════════════════
#  TERMINAL REPORT
# ═══════════════════════════════════════════════════════

def print_report(experiments: List[dict], stats: dict):
    """Print colored terminal report."""
    header('KARPATHY MOD EXPERIMENT ANALYSIS')

    # --- Summary ---
    section('SUMMARY STATISTICS')
    total = stats['total']
    decisions = stats['decisions']
    kr = stats['keep_rate']
    best_imp = stats['best_improvement']
    worst_reg = stats['worst_regression']
    avg_imp = stats['avg_improvement']
    print(f"  Total experiments:    {c(total, C.WHT, C.B)}")
    print(f"  Total rounds:         {c(max(e['round'] for e in experiments) if experiments else 0, C.WHT)}")
    print(f"  Keep rate:            {c('%.1f%%' % kr, C.GRN if kr > 20 else C.YEL)}")
    print(f"  Kept / Discarded:     {c(decisions.get('keep', 0), C.GRN)} / {c(decisions.get('discard', 0), C.RED)}"
          f"  ({c(decisions.get('inconclusive', 0), C.YEL)} inconclusive)")
    print(f"  Best improvement:     {c('+%.2f%%' % best_imp, C.GRN, C.B)}")
    print(f"  Worst regression:     {c('%.2f%%' % worst_reg, C.RED)}")
    print(f"  Avg improvement:      {c('%+.2f%%' % avg_imp, C.CYN)}")

    # --- Strategy Leaderboard ---
    section('STRATEGY LEADERBOARD')
    print(f"  {'Strategy':<14} {'Count':>6} {'Kept':>6} {'Rate':>8} {'Avg Impr':>10} {'Best':>10}")
    sep('\u2500', C.DIM)
    for strat in sorted(stats['strategy_stats'], key=lambda s: stats['strategy_stats'][s]['keep_rate'], reverse=True):
        ss = stats['strategy_stats'][strat]
        strat_color = C.GRN if ss['keep_rate'] > 30 else (C.YEL if ss['keep_rate'] > 10 else C.RED)
        kr_str = '%5.1f%%' % ss['keep_rate']
        ai_str = '%+7.2f%%' % ss['avg_improvement']
        mi_str = '%+7.2f%%' % ss['max_improvement']
        print(f"  {c(strat, C.CYN):<25} {ss['count']:>6} {ss['kept']:>6}"
              f"  {c(kr_str, strat_color)}"
              f"  {c(ai_str, C.GRN if ss['avg_improvement'] > 0 else C.RED)}"
              f"  {c(mi_str, C.WHT)}")

    # --- Stage Performance ---
    section('STAGE PERFORMANCE')
    print(f"  {'Stage':<25} {'Count':>6} {'Kept':>6} {'Rate':>8} {'Avg Impr':>10}")
    sep('\u2500', C.DIM)
    for stg in sorted(stats['stage_stats']):
        ss = stats['stage_stats'][stg]
        name = STAGE_NAMES.get(stg, f'Stage {stg}')
        stg_color = C.GRN if ss['keep_rate'] > 30 else (C.YEL if ss['keep_rate'] > 10 else C.RED)
        kr_str = '%5.1f%%' % ss['keep_rate']
        ai_str = '%+7.2f%%' % ss['avg_improvement']
        print(f"  S{stg}:{c(name, C.MAG):<30} {ss['count']:>6} {ss['kept']:>6}"
              f"  {c(kr_str, stg_color)}"
              f"  {c(ai_str, C.GRN if ss['avg_improvement'] > 0 else C.RED)}")

    # --- Top 5 Best Kept ---
    section('TOP 5 BEST KEPT EXPERIMENTS', C.GRN)
    for i, e in enumerate(stats['best_kept'], 1):
        desc = e['description'][:60] if e['description'] else 'N/A'
        imp_str = '+%.2f%%' % e['improvement_pct']
        print(f"  {i}. R{e['round']:>3} | {c(imp_str, C.GRN, C.B)}"
              f" | {e['strategy']:<10} | S{e['stage']} | {c(desc, C.DIM)}")

    # --- Top 5 Worst Discarded ---
    section('TOP 5 WORST DISCARDED EXPERIMENTS', C.RED)
    for i, e in enumerate(stats['worst_discarded'], 1):
        desc = e['description'][:60] if e['description'] else 'N/A'
        imp_str = '%+.2f%%' % e['improvement_pct']
        print(f"  {i}. R{e['round']:>3} | {c(imp_str, C.RED, C.B)}"
              f" | {e['strategy']:<10} | S{e['stage']} | {c(desc, C.DIM)}")

    # --- Trend ---
    section('TREND ASSESSMENT')
    trend = stats['trend']
    trend_colors = {'IMPROVING': C.GRN, 'REGRESSING': C.RED, 'PLATEAUING': C.YEL, 'INSUFFICIENT_DATA': C.DIM}
    print(f"  Current trend: {c(trend, trend_colors.get(trend, C.WHT), C.B)}")

    if trend == 'IMPROVING':
        print(f"  {c('Keep rate is increasing over time. The mutation system is learning what works.', C.GRN)}")
    elif trend == 'REGRESSING':
        print(f"  {c('Keep rate is decreasing. Consider reducing radical experiments or narrowing search space.', C.RED)}")
    elif trend == 'PLATEAUING':
        print(f"  {c('Keep rate is stable. Consider trying more explore/radical strategies to escape local optimum.', C.YEL)}")
    else:
        print(f"  {c('Not enough data to determine trend.', C.DIM)}")

    # --- Recommendations ---
    section('RECOMMENDATIONS')
    strategy_stats = stats['strategy_stats']
    if strategy_stats:
        best_strat = max(strategy_stats, key=lambda s: strategy_stats[s]['keep_rate'])
        worst_strat = min(strategy_stats, key=lambda s: strategy_stats[s]['keep_rate'])
        print(f"  1. Best-performing strategy: {c(best_strat, C.GRN, C.B)}"
              f" ({strategy_stats[best_strat]['keep_rate']:.0f}% keep rate)")
        if strategy_stats[worst_strat]['count'] >= 3:
            print(f"  2. Consider reducing: {c(worst_strat, C.RED)}"
                  f" ({strategy_stats[worst_strat]['keep_rate']:.0f}% keep rate)")

    stage_stats = stats['stage_stats']
    if stage_stats:
        best_stage = max(stage_stats, key=lambda s: stage_stats[s]['avg_improvement'])
        print(f"  3. Most improvable stage: S{best_stage} {c(STAGE_NAMES.get(best_stage, ''), C.MAG)}"
              f" (avg {stage_stats[best_stage]['avg_improvement']:+.2f}%)")

    if stats['keep_rate'] < 15:
        print(f"  4. {c('Low keep rate. Consider tightening tweak ranges or focusing on targeted strategy.', C.YEL)}")
    elif stats['keep_rate'] > 40:
        print(f"  4. {c('High keep rate. Consider more aggressive mutations to find bigger improvements.', C.GRN)}")

    print()
    sep('\u2550', C.CYN)


# ═══════════════════════════════════════════════════════
#  CHART GENERATION
# ═══════════════════════════════════════════════════════

def generate_charts(experiments: List[dict], stats: dict, charts_dir: str):
    """Generate all matplotlib charts."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        print(c('  WARNING: matplotlib not available, skipping charts', C.YEL))
        return

    os.makedirs(charts_dir, exist_ok=True)

    def _save(fig, name):
        path = os.path.join(charts_dir, name)
        fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close(fig)
        print(c(f'  Chart saved: charts/{name}', C.GRN))

    def _setup_fig(title, nrows=1, ncols=1, figsize=(14, 8)):
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        fig.patch.set_facecolor(BG_COLOR)
        fig.suptitle(title, fontsize=16, fontweight='bold', color=TEXT_COLOR, y=0.98)
        return fig, axes

    def _style_ax(ax, xlabel='', ylabel='', title=''):
        ax.set_facecolor(BG_COLOR)
        ax.tick_params(colors=TEXT_COLOR, which='both')
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
        ax.grid(True, alpha=0.15, color=GRID_COLOR)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=10)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=10)
        if title:
            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

    rounds = np.array([e['round'] for e in experiments])
    improvements = np.array([e['improvement_pct'] for e in experiments])
    baselines = np.array([e['baseline_score'] for e in experiments])
    exp_scores = np.array([e['experiment_score'] for e in experiments])
    decisions = [e['decision'] for e in experiments]
    strategies = [e['strategy'] for e in experiments]
    stages_arr = np.array([e['stage'] for e in experiments])

    # ── Chart 1: EXPERIMENT OVERVIEW DASHBOARD ──
    try:
        fig, axes = _setup_fig('KARPATHY MOD - EXPERIMENT OVERVIEW', 2, 2, figsize=(16, 12))

        # 1a: Score timeline
        ax = axes[0, 0]
        _style_ax(ax, 'Round', 'Score', 'Baseline vs Experiment Scores')
        ax.plot(rounds, baselines, color=PAL_BLUE, alpha=0.7, linewidth=1, label='Baseline')
        ax.plot(rounds, exp_scores, color=PAL_ORANGE, alpha=0.7, linewidth=1, label='Experiment')
        for i, e in enumerate(experiments):
            if e['decision'] == 'keep':
                ax.scatter(e['round'], e['experiment_score'], color=PAL_GREEN, s=40, zorder=5, marker='o')
            elif e['decision'] == 'discard':
                ax.scatter(e['round'], e['experiment_score'], color=PAL_RED, s=30, zorder=5, marker='x')
        ax.legend(facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=9)

        # 1b: Improvement % over time
        ax = axes[0, 1]
        _style_ax(ax, 'Round', 'Improvement %', 'Improvement % per Round')
        colors_bar = [PAL_GREEN if v >= 0 else PAL_RED for v in improvements]
        ax.bar(rounds, improvements, color=colors_bar, alpha=0.8, width=0.8)
        ax.axhline(y=2.0, color=PAL_ORANGE, linestyle='--', alpha=0.6, linewidth=1, label='2% threshold')
        ax.axhline(y=0.0, color=TEXT_COLOR, linestyle='-', alpha=0.3, linewidth=0.5)
        ax.legend(facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=9)

        # 1c: Decision pie
        ax = axes[1, 0]
        ax.set_facecolor(BG_COLOR)
        ax.set_title('Decision Distribution', fontsize=12, fontweight='bold', color=TEXT_COLOR, pad=10)
        dec_counts = stats['decisions']
        labels = []
        sizes = []
        colors_pie = []
        for dec in ['keep', 'discard', 'inconclusive']:
            cnt = dec_counts.get(dec, 0)
            if cnt > 0:
                labels.append(f"{dec} ({cnt})")
                sizes.append(cnt)
                colors_pie.append(DECISION_COLORS[dec])
        if sizes:
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_pie,
                                              autopct='%1.1f%%', startangle=90,
                                              textprops={'color': TEXT_COLOR, 'fontsize': 10})
            for at in autotexts:
                at.set_fontsize(9)
                at.set_color('#ffffff')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    color=TEXT_COLOR, transform=ax.transAxes)

        # 1d: Cumulative keep rate
        ax = axes[1, 1]
        _style_ax(ax, 'Round', 'Cumulative Keep Rate %', 'Keep Rate Over Time')
        cum_keeps = np.cumsum([1 if d == 'keep' else 0 for d in decisions])
        cum_total = np.arange(1, len(decisions) + 1)
        cum_rate = cum_keeps / cum_total * 100
        ax.plot(rounds, cum_rate, color=PAL_GREEN, linewidth=2)
        ax.fill_between(rounds, 0, cum_rate, alpha=0.1, color=PAL_GREEN)
        ax.set_ylim(0, max(100, cum_rate.max() + 5) if len(cum_rate) > 0 else 100)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        _save(fig, 'karpathy_overview.png')
    except Exception as ex:
        print(c(f'  WARNING: Chart 1 failed: {ex}', C.YEL))

    # ── Chart 2: STRATEGY EFFECTIVENESS ──
    try:
        fig, ax = _setup_fig('KARPATHY MOD - STRATEGY EFFECTIVENESS', figsize=(14, 7))
        if isinstance(ax, np.ndarray):
            ax = ax[0]
        _style_ax(ax, '', '', 'Strategy Effectiveness')

        strat_names = sorted(stats['strategy_stats'].keys())
        n_strats = len(strat_names)
        if n_strats > 0:
            x = np.arange(n_strats)
            width = 0.25

            keep_rates = [stats['strategy_stats'][s]['keep_rate'] for s in strat_names]
            avg_imps = [stats['strategy_stats'][s]['avg_improvement'] for s in strat_names]
            counts = [stats['strategy_stats'][s]['count'] for s in strat_names]
            strat_colors = [STRATEGY_COLORS.get(s, PAL_BLUE) for s in strat_names]

            bars1 = ax.bar(x - width, keep_rates, width, label='Keep Rate %',
                          color=strat_colors, alpha=0.8)
            bars2 = ax.bar(x, avg_imps, width, label='Avg Improvement %',
                          color=[PAL_ORANGE] * n_strats, alpha=0.8)
            bars3 = ax.bar(x + width, counts, width, label='Experiment Count',
                          color=[PAL_PURPLE] * n_strats, alpha=0.8)

            ax.set_xticks(x)
            ax.set_xticklabels(strat_names, fontsize=11, color=TEXT_COLOR)
            ax.legend(facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=10)
            ax.axhline(y=0, color=TEXT_COLOR, linewidth=0.5, alpha=0.3)

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        _save(fig, 'karpathy_strategy.png')
    except Exception as ex:
        print(c(f'  WARNING: Chart 2 failed: {ex}', C.YEL))

    # ── Chart 3: STAGE ANALYSIS ──
    try:
        fig, axes = _setup_fig('KARPATHY MOD - STAGE ANALYSIS', 1, 3, figsize=(16, 6))

        active_stages = sorted(stats['stage_stats'].keys())
        n_stages = len(active_stages)

        if n_stages > 0:
            x = np.arange(n_stages)
            stage_labels = [f"S{s}\n{STAGE_NAMES.get(s, '')[:8]}" for s in active_stages]
            stage_colors = [STAGE_COLORS_HEX.get(s, PAL_BLUE) for s in active_stages]

            # Experiments run
            ax = axes[0]
            _style_ax(ax, '', 'Count', 'Experiments per Stage')
            counts = [stats['stage_stats'][s]['count'] for s in active_stages]
            ax.bar(x, counts, color=stage_colors, alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(stage_labels, fontsize=9, color=TEXT_COLOR)

            # Keep rate
            ax = axes[1]
            _style_ax(ax, '', 'Keep Rate %', 'Keep Rate per Stage')
            rates = [stats['stage_stats'][s]['keep_rate'] for s in active_stages]
            ax.bar(x, rates, color=stage_colors, alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(stage_labels, fontsize=9, color=TEXT_COLOR)
            ax.axhline(y=20, color=PAL_ORANGE, linestyle='--', alpha=0.5)

            # Avg improvement
            ax = axes[2]
            _style_ax(ax, '', 'Avg Improvement %', 'Avg Improvement per Stage')
            imps = [stats['stage_stats'][s]['avg_improvement'] for s in active_stages]
            bar_colors = [PAL_GREEN if v >= 0 else PAL_RED for v in imps]
            ax.bar(x, imps, color=bar_colors, alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(stage_labels, fontsize=9, color=TEXT_COLOR)
            ax.axhline(y=0, color=TEXT_COLOR, linewidth=0.5, alpha=0.3)

        fig.tight_layout(rect=[0, 0, 1, 0.92])
        _save(fig, 'karpathy_stages.png')
    except Exception as ex:
        print(c(f'  WARNING: Chart 3 failed: {ex}', C.YEL))

    # ── Chart 4: METRIC EVOLUTION ──
    try:
        fig, axes = _setup_fig('KARPATHY MOD - METRIC EVOLUTION (kept experiments)', 2, 2, figsize=(16, 12))

        kept_exps = sorted([e for e in experiments if e['decision'] == 'keep'], key=lambda e: e['round'])
        if len(kept_exps) >= 2:
            kr = [e['round'] for e in kept_exps]

            metrics = [
                ('avg_steps', 'Avg Steps', axes[0, 0]),
                ('avg_food', 'Avg Food', axes[0, 1]),
                ('peak_length', 'Peak Length', axes[1, 0]),
                ('snake_death_rate', 'Snake Death Rate', axes[1, 1]),
            ]

            for key, label, ax in metrics:
                _style_ax(ax, 'Round', label, f'{label} Evolution')
                baseline_vals = [e['baseline_score'] for e in kept_exps]
                exp_vals = [e[key] for e in kept_exps]
                ax.plot(kr, exp_vals, color=PAL_GREEN, linewidth=1.5, marker='o', markersize=4, label='Experiment')
                # Show trend
                if len(kr) >= 3:
                    z = np.polyfit(kr, exp_vals, 1)
                    trend_line = np.poly1d(z)(kr)
                    ax.plot(kr, trend_line, color=PAL_ORANGE, linestyle='--', alpha=0.6, label='Trend')
                ax.legend(facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=9)
        else:
            for ax_item in axes.flat:
                _style_ax(ax_item)
                ax_item.text(0.5, 0.5, 'Not enough kept experiments', ha='center', va='center',
                           color=TEXT_COLOR, transform=ax_item.transAxes)

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        _save(fig, 'karpathy_metrics.png')
    except Exception as ex:
        print(c(f'  WARNING: Chart 4 failed: {ex}', C.YEL))

    # ── Chart 5: PARAMETER IMPACT ANALYSIS ──
    try:
        fig, ax = _setup_fig('KARPATHY MOD - PARAMETER IMPACT ANALYSIS', figsize=(14, 8))
        if isinstance(ax, np.ndarray):
            ax = ax[0]
        _style_ax(ax, 'Count', '', 'Parameters in Keep vs Discard Decisions')

        keep_params = Counter()
        discard_params = Counter()
        for e in experiments:
            params = extract_parameters(e['description'])
            if e['decision'] == 'keep':
                keep_params.update(params)
            elif e['decision'] == 'discard':
                discard_params.update(params)

        all_params = set(keep_params.keys()) | set(discard_params.keys())
        if all_params:
            # Sort by total appearances
            sorted_params = sorted(all_params,
                                   key=lambda p: keep_params.get(p, 0) + discard_params.get(p, 0))
            sorted_params = sorted_params[-20:]  # Top 20

            y = np.arange(len(sorted_params))
            keep_vals = [keep_params.get(p, 0) for p in sorted_params]
            discard_vals = [discard_params.get(p, 0) for p in sorted_params]

            ax.barh(y - 0.2, keep_vals, 0.4, color=PAL_GREEN, alpha=0.85, label='Keep')
            ax.barh(y + 0.2, discard_vals, 0.4, color=PAL_RED, alpha=0.85, label='Discard')
            ax.set_yticks(y)
            ax.set_yticklabels(sorted_params, fontsize=9, color=TEXT_COLOR)
            ax.legend(facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=10)
        else:
            ax.text(0.5, 0.5, 'No parameter data extracted', ha='center', va='center',
                   color=TEXT_COLOR, transform=ax.transAxes)

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        _save(fig, 'karpathy_parameters.png')
    except Exception as ex:
        print(c(f'  WARNING: Chart 5 failed: {ex}', C.YEL))

    # ── Chart 6: IMPROVEMENT DISTRIBUTION ──
    try:
        fig, ax = _setup_fig('KARPATHY MOD - IMPROVEMENT DISTRIBUTION', figsize=(14, 7))
        if isinstance(ax, np.ndarray):
            ax = ax[0]
        _style_ax(ax, 'Improvement %', 'Count', 'Distribution of Improvement %')

        bins = np.linspace(min(improvements) - 1, max(improvements) + 1,
                          min(40, max(10, len(improvements) // 3)))

        for dec, color, label in [('keep', PAL_GREEN, 'Keep'),
                                   ('discard', PAL_RED, 'Discard'),
                                   ('inconclusive', PAL_ORANGE, 'Inconclusive')]:
            vals = [e['improvement_pct'] for e in experiments if e['decision'] == dec]
            if vals:
                ax.hist(vals, bins=bins, alpha=0.6, color=color, label=label, edgecolor='none')

        ax.axvline(x=0, color=TEXT_COLOR, linestyle='-', alpha=0.5, linewidth=1, label='0%')
        ax.axvline(x=2, color=PAL_ORANGE, linestyle='--', alpha=0.7, linewidth=1.5, label='+2% threshold')
        ax.legend(facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=10)

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        _save(fig, 'karpathy_distribution.png')
    except Exception as ex:
        print(c(f'  WARNING: Chart 6 failed: {ex}', C.YEL))

    # ── Chart 7: STRATEGY x STAGE HEATMAP ──
    try:
        fig, ax = _setup_fig('KARPATHY MOD - STRATEGY x STAGE HEATMAP', figsize=(14, 8))
        if isinstance(ax, np.ndarray):
            ax = ax[0]

        unique_strategies = sorted(set(strategies))
        unique_stages = sorted(set(stages_arr))

        if len(unique_strategies) > 0 and len(unique_stages) > 0:
            heatmap_data = np.zeros((len(unique_strategies), len(unique_stages)))
            count_data = np.zeros((len(unique_strategies), len(unique_stages)), dtype=int)

            for e in experiments:
                si = unique_strategies.index(e['strategy'])
                sj = unique_stages.index(e['stage']) if e['stage'] in unique_stages else -1
                if sj >= 0:
                    count_data[si, sj] += 1
                    heatmap_data[si, sj] += e['improvement_pct']

            # Average
            with np.errstate(invalid='ignore', divide='ignore'):
                avg_data = np.where(count_data > 0, heatmap_data / count_data, np.nan)

            # Custom diverging colormap: red-dark-green
            from matplotlib.colors import LinearSegmentedColormap
            cmap = LinearSegmentedColormap.from_list('rdgn', [PAL_RED, BG_COLOR, PAL_GREEN])

            vmax = np.nanmax(np.abs(avg_data)) if not np.all(np.isnan(avg_data)) else 10
            im = ax.imshow(avg_data, cmap=cmap, aspect='auto', vmin=-vmax, vmax=vmax)

            ax.set_xticks(range(len(unique_stages)))
            ax.set_xticklabels([f"S{s}" for s in unique_stages], fontsize=11, color=TEXT_COLOR)
            ax.set_yticks(range(len(unique_strategies)))
            ax.set_yticklabels(unique_strategies, fontsize=11, color=TEXT_COLOR)
            ax.set_xlabel('Stage', fontsize=12, color=TEXT_COLOR)
            ax.set_ylabel('Strategy', fontsize=12, color=TEXT_COLOR)

            # Annotate cells
            for i in range(len(unique_strategies)):
                for j in range(len(unique_stages)):
                    cnt = count_data[i, j]
                    val = avg_data[i, j]
                    if cnt > 0:
                        txt = f"{val:+.1f}%\n(n={cnt})"
                        ax.text(j, i, txt, ha='center', va='center',
                               fontsize=9, fontweight='bold', color='white')

            cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label('Avg Improvement %', color=TEXT_COLOR, fontsize=10)
            cbar.ax.tick_params(colors=TEXT_COLOR)
            ax.set_title('Average Improvement % (cell count in parentheses)',
                        fontsize=12, fontweight='bold', color=TEXT_COLOR, pad=10)

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        _save(fig, 'karpathy_heatmap.png')
    except Exception as ex:
        print(c(f'  WARNING: Chart 7 failed: {ex}', C.YEL))

    print(c(f'\n  All charts saved to {charts_dir}/', C.GRN, C.B))


# ═══════════════════════════════════════════════════════
#  INTERACTIVE PLOTLY HTML
# ═══════════════════════════════════════════════════════

def generate_interactive_html(experiments: List[dict], charts_dir: str):
    """Generate interactive Plotly HTML chart."""
    try:
        # We build raw HTML with inline Plotly.js — no plotly python import needed
        pass
    except Exception:
        pass

    os.makedirs(charts_dir, exist_ok=True)

    rounds = [e['round'] for e in experiments]
    improvements = [e['improvement_pct'] for e in experiments]
    decisions_list = [e['decision'] for e in experiments]
    strategies_list = [e['strategy'] for e in experiments]
    stages_list = [e['stage'] for e in experiments]
    baselines = [e['baseline_score'] for e in experiments]
    exp_scores = [e['experiment_score'] for e in experiments]
    descriptions = [e['description'].replace("'", "\\'").replace('"', '\\"').replace('\n', ' ')
                    for e in experiments]
    avg_steps = [e['avg_steps'] for e in experiments]
    avg_food = [e['avg_food'] for e in experiments]
    peak_lengths = [e['peak_length'] for e in experiments]
    death_rates = [e['snake_death_rate'] for e in experiments]

    # Build traces as JSON-like strings
    traces = []
    for dec in ['keep', 'discard', 'inconclusive']:
        idxs = [i for i, d in enumerate(decisions_list) if d == dec]
        if not idxs:
            continue
        r = [rounds[i] for i in idxs]
        imp = [improvements[i] for i in idxs]
        hover = [
            f"R{rounds[i]} | {strategies_list[i]} | S{stages_list[i]}<br>"
            f"Improvement: {improvements[i]:+.2f}%<br>"
            f"Baseline: {baselines[i]:.1f} | Exp: {exp_scores[i]:.1f}<br>"
            f"Steps: {avg_steps[i]:.0f} | Food: {avg_food[i]:.1f}<br>"
            f"Peak: {peak_lengths[i]:.0f} | Death: {death_rates[i]:.1f}%<br>"
            f"Desc: {descriptions[i][:80]}"
            for i in idxs
        ]
        color = DECISION_COLORS[dec]
        marker_sym = 'circle' if dec == 'keep' else ('x' if dec == 'discard' else 'diamond')
        trace = {
            'x': r, 'y': imp, 'mode': 'markers', 'type': 'scatter',
            'name': dec.capitalize(),
            'text': hover, 'hoverinfo': 'text',
            'marker': {'color': color, 'size': 10, 'symbol': marker_sym,
                      'line': {'width': 1, 'color': '#ffffff'}},
        }
        traces.append(trace)

    import json
    traces_json = json.dumps(traces)

    layout = json.dumps({
        'title': {'text': 'Karpathy Mod - Experiment Explorer', 'font': {'color': TEXT_COLOR, 'size': 20}},
        'paper_bgcolor': BG_COLOR,
        'plot_bgcolor': BG_COLOR,
        'font': {'color': TEXT_COLOR},
        'xaxis': {'title': 'Round', 'gridcolor': GRID_COLOR, 'zerolinecolor': GRID_COLOR},
        'yaxis': {'title': 'Improvement %', 'gridcolor': GRID_COLOR, 'zerolinecolor': GRID_COLOR},
        'hovermode': 'closest',
        'legend': {'bgcolor': BG_COLOR, 'bordercolor': GRID_COLOR},
        'shapes': [
            {'type': 'line', 'x0': min(rounds), 'x1': max(rounds),
             'y0': 0, 'y1': 0, 'line': {'color': TEXT_COLOR, 'width': 1, 'dash': 'dot'}},
            {'type': 'line', 'x0': min(rounds), 'x1': max(rounds),
             'y0': 2, 'y1': 2, 'line': {'color': PAL_ORANGE, 'width': 1.5, 'dash': 'dash'}},
        ],
    })

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Karpathy Mod - Experiment Explorer</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ margin: 0; padding: 0; background: {BG_COLOR}; color: {TEXT_COLOR};
         font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; }}
  #chart {{ width: 100vw; height: 100vh; }}
  .header {{ padding: 10px 20px; font-size: 14px; opacity: 0.6; }}
</style>
</head>
<body>
<div class="header">Karpathy Mod Experiment Explorer | {len(experiments)} experiments | Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
<div id="chart"></div>
<script>
var traces = {traces_json};
var layout = {layout};
layout.margin = {{t: 60, r: 30, b: 60, l: 60}};
Plotly.newPlot('chart', traces, layout, {{responsive: true}});
</script>
</body>
</html>"""

    path = os.path.join(charts_dir, 'karpathy_experiment_explorer.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(c(f'  Interactive chart saved: charts/karpathy_experiment_explorer.html', C.GRN))


# ═══════════════════════════════════════════════════════
#  MARKDOWN REPORT
# ═══════════════════════════════════════════════════════

def generate_markdown_report(experiments: List[dict], stats: dict, output_path: str):
    """Generate a comprehensive markdown report."""
    lines = []

    def add(text=''):
        lines.append(text)

    add('# Karpathy Mod Experiment Report')
    add(f'*Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*')
    add()

    # Summary
    add('## Summary')
    add()
    add(f'| Metric | Value |')
    add(f'|--------|-------|')
    add(f'| Total experiments | {stats["total"]} |')
    add(f'| Keep rate | {stats["keep_rate"]:.1f}% |')
    add(f'| Kept | {stats["decisions"].get("keep", 0)} |')
    add(f'| Discarded | {stats["decisions"].get("discard", 0)} |')
    add(f'| Inconclusive | {stats["decisions"].get("inconclusive", 0)} |')
    add(f'| Best improvement | +{stats["best_improvement"]:.2f}% |')
    add(f'| Worst regression | {stats["worst_regression"]:.2f}% |')
    add(f'| Avg improvement | {stats["avg_improvement"]:+.2f}% |')
    add(f'| Trend | **{stats["trend"]}** |')
    add()

    # Strategy table
    add('## Strategy Effectiveness')
    add()
    add('| Strategy | Count | Kept | Keep Rate | Avg Improvement | Best |')
    add('|----------|------:|-----:|----------:|----------------:|-----:|')
    for strat in sorted(stats['strategy_stats'], key=lambda s: stats['strategy_stats'][s]['keep_rate'], reverse=True):
        ss = stats['strategy_stats'][strat]
        add(f"| {strat} | {ss['count']} | {ss['kept']} | {ss['keep_rate']:.1f}% | "
            f"{ss['avg_improvement']:+.2f}% | {ss['max_improvement']:+.2f}% |")
    add()

    # Stage table
    add('## Stage Performance')
    add()
    add('| Stage | Name | Count | Kept | Keep Rate | Avg Improvement |')
    add('|------:|------|------:|-----:|----------:|----------------:|')
    for stg in sorted(stats['stage_stats']):
        ss = stats['stage_stats'][stg]
        name = STAGE_NAMES.get(stg, f'Stage {stg}')
        add(f"| S{stg} | {name} | {ss['count']} | {ss['kept']} | {ss['keep_rate']:.1f}% | "
            f"{ss['avg_improvement']:+.2f}% |")
    add()

    # Top kept
    add('## Top 5 Best Kept Experiments')
    add()
    add('| Round | Improvement | Strategy | Stage | Description |')
    add('|------:|------------:|----------|------:|-------------|')
    for e in stats['best_kept']:
        desc = e['description'][:80] if e['description'] else 'N/A'
        add(f"| R{e['round']} | +{e['improvement_pct']:.2f}% | {e['strategy']} | S{e['stage']} | {desc} |")
    add()

    # Worst discarded
    add('## Top 5 Worst Discarded Experiments')
    add()
    add('| Round | Improvement | Strategy | Stage | Description |')
    add('|------:|------------:|----------|------:|-------------|')
    for e in stats['worst_discarded']:
        desc = e['description'][:80] if e['description'] else 'N/A'
        add(f"| R{e['round']} | {e['improvement_pct']:+.2f}% | {e['strategy']} | S{e['stage']} | {desc} |")
    add()

    # Charts
    add('## Charts')
    add()
    chart_files = [
        ('karpathy_overview.png', 'Experiment Overview Dashboard'),
        ('karpathy_strategy.png', 'Strategy Effectiveness'),
        ('karpathy_stages.png', 'Stage Analysis'),
        ('karpathy_metrics.png', 'Metric Evolution'),
        ('karpathy_parameters.png', 'Parameter Impact Analysis'),
        ('karpathy_distribution.png', 'Improvement Distribution'),
        ('karpathy_heatmap.png', 'Strategy x Stage Heatmap'),
    ]
    for filename, title in chart_files:
        add(f'### {title}')
        add(f'![{title}](charts/{filename})')
        add()

    add('### Interactive Explorer')
    add('[Open Experiment Explorer](charts/karpathy_experiment_explorer.html)')
    add()

    # Trend
    add('## Trend Assessment')
    add()
    add(f'**{stats["trend"]}**')
    add()
    if stats['trend'] == 'IMPROVING':
        add('Keep rate is increasing over time. The mutation system is learning what works.')
    elif stats['trend'] == 'REGRESSING':
        add('Keep rate is decreasing. Consider reducing radical experiments or narrowing search space.')
    elif stats['trend'] == 'PLATEAUING':
        add('Keep rate is stable. Consider trying more explore/radical strategies to escape local optimum.')
    else:
        add('Not enough data to determine trend.')
    add()

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(c(f'  Report saved: {output_path}', C.GRN))


# ═══════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Karpathy Mod Experiment Analyzer')
    parser.add_argument('--tsv', default=DEFAULT_TSV, help='Path to TSV results file')
    parser.add_argument('--last', type=int, default=None, help='Only analyze last N rounds')
    parser.add_argument('--no-charts', action='store_true', help='Skip chart generation')
    parser.add_argument('--no-report', action='store_true', help='Skip markdown report')
    args = parser.parse_args()

    # Resolve paths relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tsv_path = args.tsv if os.path.isabs(args.tsv) else os.path.join(script_dir, args.tsv)
    charts_dir = os.path.join(script_dir, CHARTS_DIR)
    report_path = os.path.join(script_dir, 'karpathy_mod_report.md')

    print()
    header('KARPATHY MOD EXPERIMENT ANALYZER')
    print(f'  TSV: {c(tsv_path, C.CYN)}')

    # Load data
    experiments = load_tsv(tsv_path, last_n=args.last)
    if not experiments:
        print(c('\n  No experiment data found. Nothing to analyze.', C.RED, C.B))
        print(c(f'  Expected TSV at: {tsv_path}', C.DIM))
        print(c(f'  Columns: {", ".join(TSV_COLUMNS)}', C.DIM))
        print()
        return

    print(f'  Loaded: {c(len(experiments), C.WHT, C.B)} experiments')
    if args.last:
        print(f'  Filter: last {c(args.last, C.YEL)} rounds')
    print()

    # Compute stats
    stats = compute_stats(experiments)
    if not stats:
        print(c('  Failed to compute statistics.', C.RED))
        return

    # Terminal report (always)
    print_report(experiments, stats)

    # Charts
    if not args.no_charts:
        section('GENERATING CHARTS', C.MAG)
        generate_charts(experiments, stats, charts_dir)
        generate_interactive_html(experiments, charts_dir)

    # Markdown report
    if not args.no_report:
        section('GENERATING MARKDOWN REPORT', C.MAG)
        generate_markdown_report(experiments, stats, report_path)

    print()
    sep('\u2550', C.CYN)
    print(c('  Analysis complete.', C.GRN, C.B))
    print()


if __name__ == '__main__':
    main()
