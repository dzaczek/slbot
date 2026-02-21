import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import os
import sys
import time
import hashlib
import json
import multiprocessing as mp
import argparse
from collections import deque
import logging
import psutil
import threading
import signal

# Rich TUI Dashboard (optional)
try:
    from rich.live import Live
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.console import Console
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Global graceful shutdown flag (set by Ctrl+E listener)
_shutdown_requested = False

# Add gen2 to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from slither_env import SlitherEnv
from config import Config
from agent import DDQNAgent
from styles import STYLES

# Setup logging
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers (file only — console output handled by Rich dashboard)
f_handler_app = logging.FileHandler('logs/app.log')
f_handler_train = logging.FileHandler('logs/train.log')

f_handler_app.setLevel(logging.INFO)
f_handler_train.setLevel(logging.INFO)

# Create formatters and add it to handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
f_handler_app.setFormatter(formatter)
f_handler_train.setFormatter(formatter)

# Add handlers to the logger
if not logger.hasHandlers():
    logger.addHandler(f_handler_app)
    logger.addHandler(f_handler_train)


class TrainingDashboard:
    """Rich TUI dashboard for real-time training visualization."""

    SPARKLINE_CHARS = " ▁▂▃▄▅▆▇█"

    def __init__(self):
        self.console = Console()
        self.live = None
        self.start_time = time.time()
        self._key_thread = None
        # Rolling data
        self.reward_history = deque(maxlen=100)
        self.steps_history = deque(maxlen=100)
        self.food_history = deque(maxlen=100)
        self.death_causes = deque(maxlen=100)
        self.action_history = deque(maxlen=100)
        self.loss_history = deque(maxlen=100)
        self.epsilon_history = deque(maxlen=100)
        self.q_mean_history = deque(maxlen=100)
        self.td_error_history = deque(maxlen=100)
        self.food_ratio_history = deque(maxlen=100)
        # Long-term history for learning progress (500 episodes)
        self.long_steps = deque(maxlen=500)
        self.long_reward = deque(maxlen=500)
        self.long_food = deque(maxlen=500)
        self.long_survival_sma = deque(maxlen=500)  # SMA20 of steps
        self.long_reward_sma = deque(maxlen=500)    # SMA20 of reward
        self.events = deque(maxlen=12)
        # Current episode data
        self.episode = 0
        self.stage = 1
        self.stage_name = ""
        self.epsilon = 1.0
        self.lr = 0.0
        self.loss = 0.0
        self.q_mean = 0.0
        self.q_max = 0.0
        self.td_error = 0.0
        self.grad_norm = 0.0
        self.num_agents = 1
        self.uid = ""
        self.style_name = ""
        self.best_avg_reward = -float('inf')
        self.episodes_per_min = 0.0
        self._ep_timestamps = deque(maxlen=60)
        # Agent board: list of dicts per agent
        self.agents_board = []
        self._last_board_refresh = 0.0

    def start(self):
        if not RICH_AVAILABLE:
            return
        self.start_time = time.time()
        self.live = Live(console=self.console,
                         refresh_per_second=2, screen=True,
                         get_renderable=self._build_layout)
        self.live.start()
        self._start_key_listener()

    def stop(self):
        if self.live:
            self.live.stop()
            self.live = None

    def _start_key_listener(self):
        """Background thread watching for graceful shutdown signal.
        Two methods: SIGUSR1 signal, or touch file 'STOP' in script dir.
        Usage: kill -USR1 <pid>  OR  touch STOP
        """
        global _shutdown_requested
        _stop_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'STOP')

        # Register SIGUSR1 handler
        def _sigusr1_handler(signum, frame):
            global _shutdown_requested
            _shutdown_requested = True
            self.log_event("SIGUSR1: Graceful shutdown requested — finishing current episodes...")
        try:
            signal.signal(signal.SIGUSR1, _sigusr1_handler)
        except (OSError, AttributeError):
            pass  # SIGUSR1 not available on Windows

        # File watcher thread: check for STOP file every 0.5s
        def _watch():
            global _shutdown_requested
            while self.live and not _shutdown_requested:
                if os.path.exists(_stop_file):
                    _shutdown_requested = True
                    try:
                        os.remove(_stop_file)
                    except OSError:
                        pass
                    self.log_event("STOP file detected: Graceful shutdown — finishing current episodes...")
                    break
                time.sleep(0.5)

        self._key_thread = threading.Thread(target=_watch, daemon=True)
        self._key_thread.start()

    def log_event(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.events.append(f"{ts} {msg}")
        self._refresh()

    def update(self, episode, stage, stage_name, epsilon, lr, loss, q_mean, q_max,
               td_error, grad_norm, reward, steps, food, cause, action_pcts, num_agents):
        self.episode = episode
        self.stage = stage
        self.stage_name = stage_name
        self.epsilon = epsilon
        self.lr = lr
        self.loss = loss
        self.q_mean = q_mean
        self.q_max = q_max
        self.td_error = td_error
        self.grad_norm = grad_norm
        self.num_agents = num_agents
        self.reward_history.append(reward)
        self.steps_history.append(steps)
        self.food_history.append(food)
        self.death_causes.append(cause)
        self.action_history.append(action_pcts)
        self.loss_history.append(loss)
        self.epsilon_history.append(epsilon)
        self.q_mean_history.append(q_mean)
        self.td_error_history.append(td_error)
        fr = food / max(steps, 1)
        self.food_ratio_history.append(fr)
        # Long-term tracking
        self.long_steps.append(steps)
        self.long_reward.append(reward)
        self.long_food.append(food)
        # Compute SMA20
        sma_window = min(20, len(self.long_steps))
        sma_steps = sum(list(self.long_steps)[-sma_window:]) / sma_window
        sma_reward = sum(list(self.long_reward)[-sma_window:]) / sma_window
        self.long_survival_sma.append(sma_steps)
        self.long_reward_sma.append(sma_reward)
        # Track best
        if len(self.reward_history) >= 20:
            avg = sum(self.reward_history) / len(self.reward_history)
            if avg > self.best_avg_reward:
                self.best_avg_reward = avg
        # Episodes per minute
        now = time.time()
        self._ep_timestamps.append(now)
        if len(self._ep_timestamps) >= 2:
            span = self._ep_timestamps[-1] - self._ep_timestamps[0]
            if span > 0:
                self.episodes_per_min = (len(self._ep_timestamps) - 1) / span * 60
        self._refresh()

    def update_agent_board(self, agents_data):
        """Update live agent board. agents_data: list of dicts per agent.
        Each dict: {name, reward, food, steps, ep_time, total_eps, last_cause}
        """
        self.agents_board = agents_data
        now = time.time()
        if now - self._last_board_refresh >= 0.5:
            self._last_board_refresh = now
            self._refresh()

    def _refresh(self):
        if self.live:
            try:
                self.live.refresh()
            except Exception as e:
                logger.debug(f"[TUI] _refresh error: {e}")

    def _sparkline(self, data, width=40):
        if not data:
            return ""
        values = list(data)[-width:]
        if len(values) < 2:
            return self.SPARKLINE_CHARS[4] * len(values)
        lo, hi = min(values), max(values)
        rng = hi - lo if hi != lo else 1
        chars = []
        for v in values:
            idx = int((v - lo) / rng * 8)
            idx = max(0, min(8, idx))
            chars.append(self.SPARKLINE_CHARS[idx])
        return "".join(chars)

    def _bar(self, pct, width=10):
        filled = int(pct / 100 * width)
        return "█" * filled + "░" * (width - filled)

    def _elapsed(self):
        secs = int(time.time() - self.start_time)
        h, m, s = secs // 3600, (secs % 3600) // 60, secs % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _build_layout(self):
        global _shutdown_requested
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body_upper"),
            Layout(name="body_lower", size=12),
            Layout(name="footer", size=3),
            Layout(name="bottom_bar"),
        )
        layout["body_upper"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="middle", ratio=1),
            Layout(name="right", ratio=1),
        )

        # Header
        try:
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory()
            ram_free = ram.available / (1024 * 1024)
            sys_info = f"CPU:{cpu:.0f}%  RAM:{ram_free:.0f}MB"
        except Exception:
            sys_info = ""

        header_text = Text()
        header_text.append("  SlitherBot DQN", style="bold cyan")
        header_text.append(f"  │  {self.style_name}", style="bold white")
        header_text.append(f"  │  UID: {self.uid}", style="dim")
        header_text.append(f"  │  {self._elapsed()}", style="green")
        header_text.append(f"  │  {self.num_agents} agents", style="yellow")
        header_text.append(f"  │  {self.episodes_per_min:.1f} ep/min", style="magenta")
        header_text.append(f"  │  {sys_info}", style="dim")
        if _shutdown_requested:
            header_text.append("  │  STOPPING...", style="bold red")
        layout["header"].update(Panel(header_text, style="bold blue"))

        # ── LEFT COLUMN: Training Status + Performance ──
        left_layout = Layout()
        left_layout.split_column(
            Layout(name="status", ratio=2),
            Layout(name="perf", ratio=1),
        )

        status_table = Table(show_header=False, box=None, padding=(0, 1))
        status_table.add_column("key", style="bold", width=12)
        status_table.add_column("value")
        status_table.add_row("Episode", f"{self.episode:,}")
        status_table.add_row("Stage", f"S{self.stage}: {self.stage_name}")
        status_table.add_row("Epsilon", f"{self.epsilon:.4f}")
        status_table.add_row("LR", f"{self.lr:.6f}")
        status_table.add_row("Loss", f"{self.loss:.4f}")
        status_table.add_row("Q Mean/Max", f"{self.q_mean:.2f} / {self.q_max:.2f}")
        status_table.add_row("TD Error", f"{self.td_error:.4f}")
        status_table.add_row("Grad Norm", f"{self.grad_norm:.4f}")
        left_layout["status"].update(Panel(status_table, title="[bold]Training Status", border_style="cyan"))

        # Performance summary
        perf_table = Table(show_header=False, box=None, padding=(0, 1))
        perf_table.add_column("key", style="bold", width=12)
        perf_table.add_column("value")
        best_str = f"{self.best_avg_reward:.2f}" if self.best_avg_reward > -1e9 else "—"
        perf_table.add_row("Best Avg", best_str)
        if self.reward_history:
            last10 = list(self.reward_history)[-10:]
            perf_table.add_row("Last 10 Avg", f"{sum(last10)/len(last10):.2f}")
        if self.food_ratio_history:
            avg_fr = sum(self.food_ratio_history) / len(self.food_ratio_history)
            perf_table.add_row("Food/Step", f"{avg_fr:.4f}")
        left_layout["perf"].update(Panel(perf_table, title="[bold]Performance", border_style="green"))
        layout["left"].update(left_layout)

        # ── MIDDLE COLUMN: Averages + Deaths + Epsilon/Loss sparklines ──
        mid_layout = Layout()
        mid_layout.split_column(
            Layout(name="averages", ratio=1),
            Layout(name="deaths", ratio=1),
            Layout(name="model_trends", ratio=1),
        )

        avg_table = Table(show_header=False, box=None, padding=(0, 1))
        avg_table.add_column("key", style="bold", width=12)
        avg_table.add_column("value")
        if self.reward_history:
            avg_r = sum(self.reward_history) / len(self.reward_history)
            avg_s = sum(self.steps_history) / len(self.steps_history)
            avg_f = sum(self.food_history) / len(self.food_history)
            avg_table.add_row("Avg Reward", f"{avg_r:.2f}")
            avg_table.add_row("Avg Steps", f"{avg_s:.1f}")
            avg_table.add_row("Avg Food", f"{avg_f:.2f}")
            avg_table.add_row("Window", f"{len(self.reward_history)}/100")
        else:
            avg_table.add_row("", "Waiting for data...")
        mid_layout["averages"].update(Panel(avg_table, title="[bold]Averages (100ep)", border_style="green"))

        # Death causes
        death_table = Table(show_header=False, box=None, padding=(0, 1))
        death_table.add_column("cause", style="bold", width=12)
        death_table.add_column("bar")
        if self.death_causes:
            total = len(self.death_causes)
            causes_count = {}
            for c in self.death_causes:
                causes_count[c] = causes_count.get(c, 0) + 1
            for cause_name in ["Wall", "SnakeCollision", "MaxSteps", "InvalidFrame", "BrowserError"]:
                cnt = causes_count.get(cause_name, 0)
                pct = cnt / total * 100
                if cnt > 0:
                    color = "red" if cause_name == "Wall" else "yellow" if cause_name == "SnakeCollision" else "dim"
                    death_table.add_row(cause_name[:10], f"[{color}]{pct:5.1f}% {self._bar(pct, 8)}[/]")
        mid_layout["deaths"].update(Panel(death_table, title="[bold]Deaths (100ep)", border_style="red"))

        # Model trends (epsilon, loss, Q-mean, TD error)
        mt = Text()
        mt.append("Epsilon: ", style="bold")
        mt.append(self._sparkline(self.epsilon_history, 30), style="yellow")
        mt.append("\n")
        mt.append("Loss:    ", style="bold")
        mt.append(self._sparkline(self.loss_history, 30), style="red")
        mt.append("\n")
        mt.append("Q-Mean:  ", style="bold")
        mt.append(self._sparkline(self.q_mean_history, 30), style="cyan")
        mt.append("\n")
        mt.append("TD Err:  ", style="bold")
        mt.append(self._sparkline(self.td_error_history, 30), style="magenta")
        mid_layout["model_trends"].update(Panel(mt, title="[bold]Model Trends", border_style="blue"))
        layout["middle"].update(mid_layout)

        # ── RIGHT COLUMN: Reward/Steps/Food sparklines + Food/Step + Actions ──
        right_layout = Layout()
        right_layout.split_column(
            Layout(name="trends", ratio=2),
            Layout(name="actions", ratio=1),
        )

        trend_text = Text()
        trend_text.append("Reward:    ", style="bold")
        trend_text.append(self._sparkline(self.reward_history), style="green")
        if self.reward_history:
            trend_text.append(f"  {list(self.reward_history)[-1]:.1f}", style="dim")
        trend_text.append("\n\n")
        trend_text.append("Steps:     ", style="bold")
        trend_text.append(self._sparkline(self.steps_history), style="cyan")
        if self.steps_history:
            trend_text.append(f"  {list(self.steps_history)[-1]:.0f}", style="dim")
        trend_text.append("\n\n")
        trend_text.append("Food:      ", style="bold")
        trend_text.append(self._sparkline(self.food_history), style="yellow")
        if self.food_history:
            trend_text.append(f"  {list(self.food_history)[-1]:.0f}", style="dim")
        trend_text.append("\n\n")
        trend_text.append("Food/Step: ", style="bold")
        trend_text.append(self._sparkline(self.food_ratio_history), style="magenta")
        if self.food_ratio_history:
            trend_text.append(f"  {list(self.food_ratio_history)[-1]:.4f}", style="dim")
        right_layout["trends"].update(Panel(trend_text, title="[bold]Episode Trends (100ep)", border_style="magenta"))

        # Action distribution
        act_table = Table(show_header=False, box=None, padding=(0, 1))
        act_table.add_column("action", style="bold", width=10)
        act_table.add_column("bar")
        action_names = ["Straight", "Gentle", "Medium", "Sharp", "UTurn", "Boost"]
        if self.action_history:
            avg_acts = [0.0] * 6
            for ap in self.action_history:
                for j in range(6):
                    avg_acts[j] += ap[j]
            n = len(self.action_history)
            avg_acts = [a / n * 100 for a in avg_acts]
            for name, pct in zip(action_names, avg_acts):
                act_table.add_row(name, f"{pct:5.1f}% {self._bar(pct, 12)}")
        right_layout["actions"].update(Panel(act_table, title="[bold]Actions (100ep)", border_style="blue"))
        layout["right"].update(right_layout)

        # ── BOTTOM ROW: Learning Progress ──
        progress_layout = Layout()
        progress_layout.split_row(
            Layout(name="survival_chart", ratio=2),
            Layout(name="survival_stats", ratio=1),
        )

        # Survival sparkline (long-term SMA20, up to 500 episodes)
        surv_text = Text()
        surv_text.append("Survival SMA20:  ", style="bold")
        surv_text.append(self._sparkline(self.long_survival_sma, 60), style="green")
        if self.long_survival_sma:
            surv_text.append(f"  {list(self.long_survival_sma)[-1]:.0f}", style="dim green")
        surv_text.append("\n\n")
        surv_text.append("Reward SMA20:    ", style="bold")
        surv_text.append(self._sparkline(self.long_reward_sma, 60), style="cyan")
        if self.long_reward_sma:
            surv_text.append(f"  {list(self.long_reward_sma)[-1]:.1f}", style="dim cyan")
        surv_text.append("\n\n")
        surv_text.append("Food (raw):      ", style="bold")
        surv_text.append(self._sparkline(self.long_food, 60), style="yellow")
        if self.long_food:
            surv_text.append(f"  {list(self.long_food)[-1]:.0f}", style="dim yellow")
        progress_layout["survival_chart"].update(
            Panel(surv_text, title=f"[bold]Learning Progress ({len(self.long_steps)}/500 ep)", border_style="green"))

        # Survival stats: min/avg/max + trend arrows
        stat_text = Text()
        if self.steps_history:
            s_list = list(self.steps_history)
            r_list = list(self.reward_history)
            f_list = list(self.food_history)
            s_min, s_avg, s_max = min(s_list), sum(s_list)/len(s_list), max(s_list)
            r_min, r_avg, r_max = min(r_list), sum(r_list)/len(r_list), max(r_list)
            f_min, f_avg, f_max = min(f_list), sum(f_list)/len(f_list), max(f_list)

            # Trend: compare last 20 vs previous 20
            def _trend(data):
                if len(data) < 40:
                    return "—", "dim"
                recent = list(data)[-20:]
                prev = list(data)[-40:-20]
                r_avg = sum(recent) / len(recent)
                p_avg = sum(prev) / len(prev)
                diff = (r_avg - p_avg) / max(abs(p_avg), 0.01) * 100
                if diff > 10:
                    return f"▲ +{diff:.0f}%", "bold green"
                elif diff > 2:
                    return f"↗ +{diff:.0f}%", "green"
                elif diff < -10:
                    return f"▼ {diff:.0f}%", "bold red"
                elif diff < -2:
                    return f"↘ {diff:.0f}%", "red"
                else:
                    return f"→ {diff:+.0f}%", "yellow"

            s_trend, s_style = _trend(self.long_steps)
            r_trend, r_style = _trend(self.long_reward)
            f_trend, f_style = _trend(self.long_food)

            stat_text.append("         Min   Avg   Max  Trend\n", style="bold dim")
            stat_text.append(f"Steps  {s_min:5.0f} {s_avg:5.0f} {s_max:5.0f}  ")
            stat_text.append(f"{s_trend}\n", style=s_style)
            stat_text.append(f"Reward {r_min:5.1f} {r_avg:5.1f} {r_max:5.1f}  ")
            stat_text.append(f"{r_trend}\n", style=r_style)
            stat_text.append(f"Food   {f_min:5.0f} {f_avg:5.0f} {f_max:5.0f}  ")
            stat_text.append(f"{f_trend}\n", style=f_style)

            # All-time bests
            if len(self.long_steps) > 0:
                stat_text.append("\n")
                stat_text.append(f"All-time max steps:  ", style="dim")
                stat_text.append(f"{max(self.long_steps):.0f}\n", style="bold green")
                stat_text.append(f"All-time max reward: ", style="dim")
                stat_text.append(f"{max(self.long_reward):.1f}", style="bold green")
        else:
            stat_text.append("Waiting for data...", style="dim")

        progress_layout["survival_stats"].update(
            Panel(stat_text, title="[bold]Survival Stats (100ep)", border_style="cyan"))
        layout["body_lower"].update(progress_layout)

        # Footer
        footer_text = Text()
        footer_text.append("  Ctrl+C", style="bold red")
        footer_text.append(" Force quit  │  ", style="dim")
        footer_text.append("touch STOP", style="bold yellow")
        footer_text.append(" or ", style="dim")
        footer_text.append("kill -USR1 " + str(os.getpid()), style="bold yellow")
        footer_text.append(" Graceful shutdown", style="dim")
        layout["footer"].update(Panel(footer_text, style="dim"))

        # ── BOTTOM BAR: Agents Board + Events ──
        layout["bottom_bar"].split_row(
            Layout(name="agents_board", ratio=1),
            Layout(name="events", ratio=1),
        )

        # Agents Board
        agent_table = Table(box=None, padding=(0, 1), expand=True)
        agent_table.add_column("#", style="dim", width=3)
        agent_table.add_column("Name", style="bold cyan", width=12)
        agent_table.add_column("Reward", justify="right", width=8)
        agent_table.add_column("Food", justify="right", width=5)
        agent_table.add_column("Size", justify="right", width=5)
        agent_table.add_column("Steps", justify="right", width=6)
        agent_table.add_column("Time", justify="right", width=7)
        agent_table.add_column("Eps", justify="right", width=5)
        agent_table.add_column("Last Death", width=11)
        agent_table.add_column("Server", style="dim", width=18)

        if self.agents_board:
            for a in self.agents_board:
                ep_secs = int(a.get('ep_time', 0))
                if ep_secs >= 60:
                    time_str = f"{ep_secs // 60}m{ep_secs % 60:02d}s"
                else:
                    time_str = f"{ep_secs}s"
                reward_val = a.get('reward', 0)
                rw_style = "green" if reward_val > 0 else "red" if reward_val < 0 else "dim"
                cause = a.get('last_cause', '—')
                cause_style = "red" if cause == "Wall" else "yellow" if cause == "Snake" else "dim"
                size_val = a.get('length', 0)
                size_style = "bold green" if size_val >= 100 else "green" if size_val >= 30 else "dim"
                server = a.get('server', '')
                # Show short form: just IP or last segment
                short_srv = server.split('/')[-1] if '/' in server else server
                agent_table.add_row(
                    str(a.get('idx', 0)),
                    a.get('name', '?'),
                    f"[{rw_style}]{reward_val:.2f}[/]",
                    str(a.get('food', 0)),
                    f"[{size_style}]{size_val}[/]",
                    str(a.get('steps', 0)),
                    time_str,
                    str(a.get('total_eps', 0)),
                    f"[{cause_style}]{cause}[/]",
                    short_srv,
                )
        else:
            agent_table.add_row("—", "Waiting...", "", "", "", "", "", "", "", "")

        layout["agents_board"].update(Panel(agent_table, title="[bold]Agents Board", border_style="cyan"))

        # Events
        event_lines = Text()
        if self.events:
            for ev in self.events:
                event_lines.append(ev + "\n")
        else:
            event_lines.append("Waiting for events...\n", style="dim")
        layout["events"].update(Panel(event_lines, title="[bold]Events", border_style="yellow"))

        return layout


def generate_uid():
    """Generate a unique run ID in format YYYYMMDD-8hexchars."""
    timestamp = str(time.time())
    rand_part = str(random.getrandbits(64))
    pid = str(os.getpid())
    raw = f"{timestamp}-{rand_part}-{pid}"
    hash_hex = hashlib.sha256(raw.encode()).hexdigest()[:8]
    date_str = time.strftime('%Y%m%d')
    return f"{date_str}-{hash_hex}"


def select_style_and_model(args):
    """
    Interactive menu or CLI argument handling for selecting Style and Model.
    """
    # Determine Style
    style_name = "Standard (Curriculum)"

    if args.style_name:
        # Check if valid
        found = False
        for s in STYLES:
            if args.style_name.lower() in s.lower():
                style_name = s
                found = True
                break
        if not found:
            print(f"Warning: Style '{args.style_name}' not found. Using default.")
    elif sys.stdin.isatty():
        # Interactive Menu
        print("\n" + "="*40)
        print(" SELECT LEARNING STYLE")
        print("="*40)
        style_keys = list(STYLES.keys())
        for i, s in enumerate(style_keys):
            desc = STYLES[s].get('description', '')
            print(f"{i+1}. {s}")
            print(f"   {desc}")

        try:
            choice = input(f"\nChoice (1-{len(style_keys)}, default 1): ").strip()
            if choice:
                idx = int(choice) - 1
                if 0 <= idx < len(style_keys):
                    style_name = style_keys[idx]
        except:
            pass

    print(f"Selected Style: {style_name}")

    # Determine Model
    model_path = args.model_path

    if not model_path and sys.stdin.isatty():
        print("\n" + "="*60)
        print(" SELECT MODEL CHECKPOINT")
        print("="*60)

        # Scan for .pth files
        base_dir = os.path.dirname(os.path.abspath(__file__))
        backup_dir = os.path.join(base_dir, 'backup_models')

        checkpoints = []
        backups = []

        # 1. Main Checkpoints (cwd, gen2/)
        search_paths = [os.getcwd(), base_dir]
        seen = set()

        for p in search_paths:
            if os.path.exists(p):
                for f in os.listdir(p):
                    if f.endswith('.pth'):
                        full_path = os.path.join(p, f)
                        if full_path not in seen:
                            checkpoints.append(full_path)
                            seen.add(full_path)

        # 2. Backup Models (gen2/backup_models/)
        if os.path.exists(backup_dir):
            for f in os.listdir(backup_dir):
                if f.endswith('.pth'):
                    full_path = os.path.join(backup_dir, f)
                    backups.append(full_path)

        # Sort: Newest first
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        backups.sort(key=os.path.getmtime, reverse=True)

        all_models = checkpoints + backups

        print(f" [0] New Random Agent (Start from scratch)")
        print("-" * 60)

        list_idx = 1
        if checkpoints:
            print(" --- MAIN CHECKPOINTS ---")
            for cp in checkpoints:
                ts = time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(cp)))
                name = os.path.basename(cp)
                print(f" [{list_idx}] {ts} | {name}")
                list_idx += 1
            print("-" * 60)

        if backups:
            print(" --- PROMISING BACKUPS (Auto-Saved) ---")
            for cp in backups:
                ts = time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(cp)))
                name = os.path.basename(cp)
                print(f" [{list_idx}] {ts} | {name}")
                list_idx += 1

        try:
            choice = input(f"\nSelect Model (0-{len(all_models)}, default 0): ").strip()
            if choice and choice != '0':
                sel_idx = int(choice) - 1
                if 0 <= sel_idx < len(all_models):
                    model_path = all_models[sel_idx]
        except:
            pass

    if model_path:
        print(f"Selected Model: {model_path}")
    else:
        print("Selected Model: New Random Agent")

    return style_name, model_path


class CurriculumManager:
    """
    Manages rewards and curriculum progression based on the selected Style.
    Supports both 'curriculum' (multi-stage) and 'static' (single-config) modes.
    """

    def __init__(self, style_name="Standard (Curriculum)", start_stage=1):
        self.style_name = style_name
        self.style_config = STYLES[style_name]
        self.mode = self.style_config["type"] # "curriculum" or "static"

        self.current_stage = start_stage
        # maxlen must cover the largest promote_window across all stages
        stages = self.style_config.get("stages", {})
        max_window = max((s.get("promote_window", 100) for s in stages.values()), default=500)
        self.episode_food_history = deque(maxlen=max_window)
        self.episode_steps_history = deque(maxlen=max_window)
        self.episode_food_ratio_history = deque(maxlen=max_window)
        self.episode_cause_history = deque(maxlen=max_window)

    def get_config(self):
        """Return current stage config dict."""
        if self.mode == "static":
            return self.style_config["config"]
        else:
            return self.style_config["stages"][self.current_stage]

    def get_max_steps(self):
        if self.mode == "static":
            return self.style_config["config"]["max_steps"]
        else:
            return self.style_config["stages"][self.current_stage]["max_steps"]

    def record_episode(self, food_eaten, steps, cause="SnakeCollision"):
        """Record episode metrics for promotion check."""
        self.episode_food_history.append(food_eaten)
        self.episode_steps_history.append(steps)
        ratio = food_eaten / max(steps, 1)
        self.episode_food_ratio_history.append(ratio)
        self.episode_cause_history.append(cause)

    def check_promotion(self):
        """Check if we should advance to the next stage. Returns True if promoted."""
        if self.mode == "static":
            return False

        cfg = self.style_config["stages"][self.current_stage]
        metric = cfg["promote_metric"]
        window = cfg["promote_window"]

        if metric is None:
            return False  # Final stage

        if metric == "compound":
            # All conditions in promote_conditions must be met simultaneously
            conditions = cfg.get("promote_conditions", {})
            if not conditions:
                return False
            min_len = max(len(self.episode_food_history), len(self.episode_steps_history))
            if min_len < window:
                return False

            recent_food = list(self.episode_food_history)[-window:]
            recent_steps = list(self.episode_steps_history)[-window:]
            avg_food = sum(recent_food) / len(recent_food)
            avg_steps = sum(recent_steps) / len(recent_steps)

            blocked = []
            if "avg_food" in conditions and avg_food < conditions["avg_food"]:
                blocked.append(f"avg_food={avg_food:.1f}<{conditions['avg_food']}")
            if "avg_steps" in conditions and avg_steps < conditions["avg_steps"]:
                blocked.append(f"avg_steps={avg_steps:.0f}<{conditions['avg_steps']}")

            if blocked:
                if len(self.episode_steps_history) % 50 == 0:
                    logger.info(f"  [Curriculum] Compound blocked: {', '.join(blocked)}")
                return False

            self._promote()
            return True

        elif metric == "food_per_step":
            threshold = cfg["promote_threshold"]
            if len(self.episode_food_ratio_history) < window:
                return False
            recent = list(self.episode_food_ratio_history)[-window:]
            avg = sum(recent) / len(recent)
            if avg >= threshold:
                self._promote()
                return True

        elif metric == "avg_food":
            threshold = cfg["promote_threshold"]
            if len(self.episode_food_history) < window:
                return False
            recent = list(self.episode_food_history)[-window:]
            avg = sum(recent) / len(recent)
            if avg >= threshold:
                self._promote()
                return True

        elif metric == "avg_steps":
            threshold = cfg["promote_threshold"]
            if len(self.episode_steps_history) < window:
                return False
            recent = list(self.episode_steps_history)[-window:]
            avg = sum(recent) / len(recent)
            if avg < threshold:
                return False

            # Compound check: wall death ratio must be below max
            wall_death_max = cfg.get("promote_wall_death_max")
            if wall_death_max is not None and len(self.episode_cause_history) >= window:
                recent_causes = list(self.episode_cause_history)[-window:]
                wall_deaths = sum(1 for c in recent_causes if c == "Wall")
                wall_ratio = wall_deaths / len(recent_causes)
                if wall_ratio > wall_death_max:
                    if len(self.episode_steps_history) % 50 == 0:
                        logger.info(f"  [Curriculum] avg_steps={avg:.0f} OK but wall_death={wall_ratio:.0%} > {wall_death_max:.0%} — blocked")
                    return False

            self._promote()
            return True

        return False

    def _promote(self):
        stages = self.style_config["stages"]
        old_name = stages[self.current_stage]["name"]
        self.current_stage = min(self.current_stage + 1, max(stages.keys()))
        new_name = stages[self.current_stage]["name"]
        logger.info(f"STAGE UP! {old_name} -> {new_name} (Stage {self.current_stage})")
        # Clear history so new stage must earn its own promotion
        self.episode_food_history.clear()
        self.episode_steps_history.clear()
        self.episode_food_ratio_history.clear()
        self.episode_cause_history.clear()

    def get_state(self):
        """Serialize for checkpoint."""
        return {
            "stage": self.current_stage,
            "food_history": list(self.episode_food_history),
            "steps_history": list(self.episode_steps_history),
            "food_ratio_history": list(self.episode_food_ratio_history),
            "cause_history": list(self.episode_cause_history),
        }

    def load_state(self, state):
        """Restore from checkpoint."""
        if state:
            self.current_stage = state.get("stage", 1)
            stages = self.style_config.get("stages", {})
            max_window = max((s.get("promote_window", 100) for s in stages.values()), default=500)
            self.episode_food_history = deque(state.get("food_history", []), maxlen=max_window)
            self.episode_steps_history = deque(state.get("steps_history", []), maxlen=max_window)
            self.episode_food_ratio_history = deque(state.get("food_ratio_history", []), maxlen=max_window)
            self.episode_cause_history = deque(state.get("cause_history", []), maxlen=max_window)


class SuperPatternOptimizer:
    def __init__(self, base_stage_cfg, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.base_stage_cfg = dict(base_stage_cfg)
        self.current_stage_cfg = dict(base_stage_cfg)
        self.causes = deque(maxlen=cfg.opt.super_pattern_window)
        self.food_ratios = deque(maxlen=cfg.opt.super_pattern_window)
        self.steps = deque(maxlen=cfg.opt.super_pattern_window)
        self.rewards = deque(maxlen=cfg.opt.super_pattern_window)

    def reset_stage(self, base_stage_cfg):
        self.base_stage_cfg = dict(base_stage_cfg)
        self.current_stage_cfg = dict(base_stage_cfg)
        self.causes.clear()
        self.food_ratios.clear()
        self.steps.clear()
        self.rewards.clear()

    def record_episode(self, cause, food_eaten, steps, reward):
        self.causes.append(cause)
        self.food_ratios.append(food_eaten / max(steps, 1))
        self.steps.append(steps)
        self.rewards.append(reward)

    def maybe_update(self):
        if not self.cfg.opt.super_pattern_enabled:
            return None
        if len(self.causes) < self.cfg.opt.super_pattern_window:
            return None

        wall_ratio = self.causes.count("Wall") / len(self.causes)
        snake_ratio = self.causes.count("SnakeCollision") / len(self.causes)
        avg_food_ratio = sum(self.food_ratios) / len(self.food_ratios)

        updated = dict(self.current_stage_cfg)
        changed = False

        wall_base = self.base_stage_cfg.get("wall_proximity_penalty", 0.0)
        wall_current = updated.get("wall_proximity_penalty", 0.0)
        wall_cap = self.cfg.opt.super_pattern_wall_penalty_cap
        wall_step = self.cfg.opt.super_pattern_penalty_step
        if wall_ratio > self.cfg.opt.super_pattern_wall_ratio:
            wall_target = min(wall_current + wall_step, wall_cap)
        elif wall_ratio < self.cfg.opt.super_pattern_wall_ratio * 0.7:
            wall_target = max(wall_current - wall_step, wall_base)
        else:
            wall_target = wall_current
        if wall_target != wall_current:
            updated["wall_proximity_penalty"] = wall_target
            changed = True

        enemy_base = self.base_stage_cfg.get("enemy_proximity_penalty", 0.0)
        enemy_current = updated.get("enemy_proximity_penalty", 0.0)
        enemy_cap = self.cfg.opt.super_pattern_enemy_penalty_cap
        if snake_ratio > self.cfg.opt.super_pattern_snake_ratio:
            enemy_target = min(enemy_current + wall_step, enemy_cap)
        elif snake_ratio < self.cfg.opt.super_pattern_snake_ratio * 0.7:
            enemy_target = max(enemy_current - wall_step, enemy_base)
        else:
            enemy_target = enemy_current
        if enemy_target != enemy_current:
            updated["enemy_proximity_penalty"] = enemy_target
            changed = True

        straight_base = self.base_stage_cfg.get("straight_penalty", 0.0)
        straight_current = updated.get("straight_penalty", 0.0)
        straight_cap = self.cfg.opt.super_pattern_straight_penalty_cap
        if wall_ratio > self.cfg.opt.super_pattern_wall_ratio:
            straight_target = min(straight_current + wall_step * 0.5, straight_cap)
        elif wall_ratio < self.cfg.opt.super_pattern_wall_ratio * 0.7:
            straight_target = max(straight_current - wall_step * 0.5, straight_base)
        else:
            straight_target = straight_current
        if straight_target != straight_current:
            updated["straight_penalty"] = straight_target
            changed = True

        food_base = self.base_stage_cfg.get("food_reward", 0.0)
        food_current = updated.get("food_reward", 0.0)
        food_cap = self.cfg.opt.super_pattern_food_reward_cap
        if avg_food_ratio < self.cfg.opt.super_pattern_food_ratio_low:
            food_target = min(food_current + self.cfg.opt.super_pattern_reward_step, food_cap)
        elif avg_food_ratio > self.cfg.opt.super_pattern_food_ratio_high:
            food_target = max(food_current - self.cfg.opt.super_pattern_reward_step, food_base)
        else:
            food_target = food_current
        if food_target != food_current:
            updated["food_reward"] = food_target
            changed = True

        if not changed:
            return None

        self.current_stage_cfg = updated
        self.logger.info(
            "[SuperPattern] Adjusted rewards: "
            f"wall_pen={wall_current:.2f}->{updated.get('wall_proximity_penalty', wall_current):.2f}, "
            f"enemy_pen={enemy_current:.2f}->{updated.get('enemy_proximity_penalty', enemy_current):.2f}, "
            f"straight_pen={straight_current:.2f}->{updated.get('straight_penalty', straight_current):.2f}, "
            f"food_reward={food_current:.2f}->{updated.get('food_reward', food_current):.2f} "
            f"(wall_ratio={wall_ratio:.2f}, snake_ratio={snake_ratio:.2f}, food_ratio={avg_food_ratio:.3f})"
        )
        return updated

class ResourceMonitor:
    """Monitors system resources and recommends agent scaling decisions."""

    def __init__(self, check_interval=15, cooldown_up=20, cooldown_down=15):
        self.step_times = deque(maxlen=50)
        self.check_interval = check_interval
        self.cooldown_up = cooldown_up
        self.cooldown_down = cooldown_down
        self.last_check = time.time()
        self.last_scale_time = 0  # no cooldown on first check

    def record_step(self, duration):
        self.step_times.append(duration)

    def should_check(self):
        return time.time() - self.last_check > self.check_interval

    def get_metrics(self):
        cpu = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory()
        avg_step = sum(self.step_times) / len(self.step_times) if self.step_times else 0
        self.last_check = time.time()
        return {
            'cpu_percent': cpu,
            'ram_free_mb': ram.available / (1024 * 1024),
            'ram_percent': ram.percent,
            'avg_step_ms': avg_step * 1000,
        }

    def recommend(self, current_agents, metrics, backend="selenium"):
        """Returns +1 (scale up), 0 (no change), or -1 (scale down)."""
        now = time.time()
        cpu = metrics['cpu_percent']
        ram_free = metrics['ram_free_mb']
        step_ms = metrics['avg_step_ms']

        # Backend-aware thresholds
        if backend == "websocket":
            step_down_threshold = 100   # ms
            step_up_threshold = 50      # ms
            ram_down_threshold = 200    # MB (WS uses ~5MB/agent vs ~500MB)
            ram_up_threshold = 500
        else:
            step_down_threshold = 500
            step_up_threshold = 200
            ram_down_threshold = 500
            ram_up_threshold = 2000

        # Scale DOWN: any critical threshold breached
        if cpu > 90 or ram_free < ram_down_threshold or step_ms > step_down_threshold:
            if now - self.last_scale_time >= self.cooldown_down:
                self.last_scale_time = now
                return -1
            return 0

        # Scale UP: all conditions must be met
        if cpu < 70 and ram_free > ram_up_threshold and step_ms < step_up_threshold:
            if now - self.last_scale_time >= self.cooldown_up:
                self.last_scale_time = now
                return 1
            return 0

        return 0


class VecFrameStack:
    """
    Wraps SubprocVecEnv to stack frames.
    Observations are dicts: {'matrix': (3,H,W), 'sectors': (75,)}.
    Only matrices get stacked (4 frames -> 12 channels).
    Sectors are passed through from the current frame (no stacking).
    """
    def __init__(self, venv, k):
        self.venv = venv
        self.k = k
        self.num_agents = venv.num_agents
        self.frames = [deque(maxlen=k) for _ in range(self.num_agents)]

    def _stack_obs(self, i, current_sectors):
        """Build stacked observation dict from frame deque + current sectors."""
        stacked_matrix = np.concatenate([f for f in self.frames[i]], axis=0)
        return {'matrix': stacked_matrix, 'sectors': current_sectors}

    def reset(self):
        obs_list = self.venv.reset()
        stacked_obs = []
        for i, o in enumerate(obs_list):
            mat = o['matrix']
            self.frames[i].clear()
            for _ in range(self.k):
                self.frames[i].append(mat)
            stacked_obs.append(self._stack_obs(i, o['sectors']))
        return stacked_obs

    def step(self, actions):
        obs_list, rews, dones, infos = self.venv.step(actions)
        stacked_obs = []

        for i in range(self.num_agents):
            if dones[i]:
                # 1. Handle terminal observation stacking
                term_obs = infos[i]['terminal_observation']
                term_mat = term_obs['matrix']
                term_sectors = term_obs['sectors']
                term_stack_deque = self.frames[i].copy()
                term_stack_deque.append(term_mat)
                term_stacked_matrix = np.concatenate(list(term_stack_deque), axis=0)
                infos[i]['terminal_observation'] = {
                    'matrix': term_stacked_matrix,
                    'sectors': term_sectors,
                }

                # 2. Handle new episode start
                new_mat = obs_list[i]['matrix']
                self.frames[i].clear()
                for _ in range(self.k):
                    self.frames[i].append(new_mat)
            else:
                self.frames[i].append(obs_list[i]['matrix'])

            stacked_obs.append(self._stack_obs(i, obs_list[i]['sectors']))

        return stacked_obs, rews, dones, infos

    def reset_agent(self, i):
        """Force reset specific agent."""
        return self.reset_one(i)

    def reset_one(self, i):
        obs = self.venv.reset_one(i)
        mat = obs['matrix']
        self.frames[i].clear()
        for _ in range(self.k):
            self.frames[i].append(mat)
        return self._stack_obs(i, obs['sectors'])

    def close(self):
        self.venv.close()

    def add_agent(self):
        """Add a new agent dynamically. Returns stacked initial observation."""
        obs = self.venv.add_agent()
        mat = obs['matrix']
        new_deque = deque(maxlen=self.k)
        for _ in range(self.k):
            new_deque.append(mat)
        self.frames.append(new_deque)
        self.num_agents += 1
        return self._stack_obs(self.num_agents - 1, obs['sectors'])

    def remove_agent(self):
        """Remove the last agent. Returns False if only 1 agent left."""
        if not self.venv.remove_agent():
            return False
        self.frames.pop()
        self.num_agents -= 1
        return True

    def set_stage(self, stage_config):
        self.venv.set_stage(stage_config)

def worker(remote, parent_remote, worker_id, headless, nickname_prefix, matrix_size, frame_skip, view_plus=False, base_url="http://slither.io", backend="selenium", ws_server_url="", suppress_stdout=False):
    parent_remote.close()

    # Suppress stdout/stderr in workers to avoid corrupting Rich TUI
    if suppress_stdout:
        devnull = open(os.devnull, 'w')
        sys.stdout = devnull
        sys.stderr = devnull

    agent_names = [
        "Picard", "Riker", "Data", "Worf", "Troi", "LaForge",
        "Crusher", "Q", "Seven", "Raffi", "Rios", "Jurati"
    ]
    chosen_name = agent_names[worker_id % len(agent_names)]

    try:
        env = SlitherEnv(
            headless=headless,
            nickname=chosen_name,
            matrix_size=matrix_size,
            view_plus=view_plus,
            base_url=base_url,
            frame_skip=frame_skip,
            backend=backend,
            ws_server_url=ws_server_url,
        )

        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                action = data
                next_state, reward, done, info = env.step(action)
                if done:
                    info['terminal_observation'] = next_state
                    reset_state = env.reset()
                    next_state = reset_state
                remote.send((next_state, reward, done, info))

            elif cmd == 'reset':
                state = env.reset()
                remote.send(state)
            elif cmd == 'reset_one':
                state = env.reset()
                remote.send(state)
            elif cmd == 'set_stage':
                env.set_curriculum_stage(data)
                remote.send('ok')
            elif cmd == 'close':
                env.close()
                break
    except Exception as e:
        import traceback
        crash_msg = f"Worker {worker_id} crashed: {e}\n{traceback.format_exc()}"
        try:
            with open("logs/worker_crashes.log", "a") as _wf:
                _wf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {crash_msg}\n")
        except Exception:
            pass
        print(crash_msg)
    finally:
        remote.close()

class SubprocVecEnv:
    def __init__(self, num_agents, matrix_size, frame_skip, view_first=False, view_plus=False, nickname="dzaczekAI", base_url="http://slither.io", backend="selenium", ws_server_url="", suppress_stdout=False):
        self.num_agents = num_agents
        # Store constructor args for spawning new workers dynamically
        self._matrix_size = matrix_size
        self._frame_skip = frame_skip
        self._nickname = nickname
        self._base_url = base_url
        self._backend = backend
        self._ws_server_url = ws_server_url
        self._suppress_stdout = suppress_stdout
        self._current_stage_config = None

        pipes = [mp.Pipe() for _ in range(num_agents)]
        self.remotes = [p[0] for p in pipes]
        work_remotes = [p[1] for p in pipes]
        self.ps = []

        for i in range(num_agents):
            is_headless = not (view_first and i == 0)
            # Enable view_plus only for the first agent when view mode is active
            agent_view_plus = view_plus and (i == 0) and not is_headless
            p = mp.Process(
                target=worker,
                args=(work_remotes[i], self.remotes[i], i, is_headless, nickname, matrix_size, frame_skip, agent_view_plus, base_url, backend, ws_server_url, suppress_stdout),
            )
            p.daemon = True
            p.start()
            self.ps.append(p)

        for remote in work_remotes:
            remote.close()

    def _make_dummy_obs(self):
        """Return a zero observation for when respawn fails."""
        return {
            'matrix': np.zeros((3, self._matrix_size, self._matrix_size), dtype=np.float32),
            'sectors': np.zeros(99, dtype=np.float32),
        }

    def _respawn_worker(self, index, max_retries=3):
        """Respawn a crashed worker process with retries."""
        for attempt in range(max_retries):
            logger.warning(f"[SubprocVecEnv] Respawning worker {index} (attempt {attempt+1}/{max_retries})...")
            try:
                self.ps[index].terminate()
                self.ps[index].join(timeout=3)
            except Exception:
                pass
            try:
                self.remotes[index].close()
            except Exception:
                pass

            remote, work_remote = mp.Pipe()
            p = mp.Process(
                target=worker,
                args=(work_remote, remote, index, True, self._nickname,
                      self._matrix_size, self._frame_skip, False, self._base_url,
                      self._backend, self._ws_server_url, self._suppress_stdout),
            )
            p.daemon = True
            p.start()
            work_remote.close()
            self.remotes[index] = remote
            self.ps[index] = p

            try:
                if self._current_stage_config is not None:
                    remote.send(('set_stage', self._current_stage_config))
                    remote.recv()

                remote.send(('reset', None))
                obs = remote.recv()
                logger.info(f"[SubprocVecEnv] Worker {index} respawned successfully.")
                return obs
            except (EOFError, BrokenPipeError, ConnectionResetError) as e:
                logger.warning(f"[SubprocVecEnv] Respawn attempt {attempt+1} failed: {e}")
                time.sleep(2)

        logger.error(f"[SubprocVecEnv] Worker {index} failed after {max_retries} retries. Using dummy obs.")
        return self._make_dummy_obs()

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return [remote.recv() for remote in self.remotes]

    def reset_one(self, index):
        try:
            self.remotes[index].send(('reset_one', None))
            return self.remotes[index].recv()
        except (EOFError, BrokenPipeError, ConnectionResetError):
            return self._respawn_worker(index)

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            try:
                remote.send(('step', action))
            except (EOFError, BrokenPipeError, ConnectionResetError):
                pass  # will be caught on recv

        results = []
        for i, remote in enumerate(self.remotes):
            try:
                results.append(remote.recv())
            except (EOFError, BrokenPipeError, ConnectionResetError):
                logger.warning(f"[SubprocVecEnv] Worker {i} crashed (EOFError). Respawning...")
                obs = self._respawn_worker(i)
                # Return a "death" result for this agent
                zero_obs = obs
                results.append((zero_obs, 0.0, True, {
                    'terminal_observation': obs,
                    'cause': 'BrowserError',
                    'food_eaten': 0,
                    'pos': (0, 0),
                    'wall_dist': -1,
                    'enemy_dist': -1,
                }))
        states, rewards, dones, infos = zip(*results)
        return states, rewards, dones, infos

    def close(self):
        for i, remote in enumerate(self.remotes):
            try:
                remote.send(('close', None))
            except (EOFError, BrokenPipeError, ConnectionResetError):
                pass
        for p in self.ps:
            try:
                p.join(timeout=5)
            except Exception:
                p.terminate()

    def set_stage(self, stage_config):
        """Send curriculum stage config to all workers."""
        self._current_stage_config = stage_config
        for remote in self.remotes:
            remote.send(('set_stage', stage_config))
        for remote in self.remotes:
            remote.recv()  # Wait for ack

    def add_agent(self):
        """Spawn a new worker process dynamically. Returns initial observation."""
        i = self.num_agents
        remote, work_remote = mp.Pipe()
        p = mp.Process(
            target=worker,
            args=(work_remote, remote, i, True, self._nickname,
                  self._matrix_size, self._frame_skip, False, self._base_url,
                  self._backend, self._ws_server_url, self._suppress_stdout),
        )
        p.daemon = True
        p.start()
        work_remote.close()
        self.remotes.append(remote)
        self.ps.append(p)
        self.num_agents += 1

        # Apply current stage config to new worker
        try:
            if self._current_stage_config is not None:
                remote.send(('set_stage', self._current_stage_config))
                remote.recv()

            # Reset and get initial observation
            remote.send(('reset', None))
            return remote.recv()
        except (EOFError, BrokenPipeError) as e:
            # Worker crashed during init — clean up and re-raise with info
            logger.error(f"Worker {i} died during add_agent: {e} — check logs/worker_crashes.log")
            try:
                p.terminate()
            except Exception:
                pass
            self.remotes.pop()
            self.ps.pop()
            self.num_agents -= 1
            raise RuntimeError(f"Failed to add agent #{i}: worker crashed during init") from e

    def remove_agent(self):
        """Shut down the last worker process. Returns False if only 1 agent left."""
        if self.num_agents <= 1:
            return False
        idx = self.num_agents - 1
        try:
            self.remotes[idx].send(('close', None))
            self.ps[idx].join(timeout=5)
        except Exception:
            self.ps[idx].terminate()
        self.remotes.pop()
        self.ps.pop()
        self.num_agents -= 1
        return True


def train(args):
    # Select Style and Model
    style_name, model_path = select_style_and_model(args)

    # Load Config
    cfg = Config()
    
    # Override config with args if necessary
    if args.auto_num_agents:
        cfg.env.num_agents = 1  # Auto-scale starts with 1 agent
    elif args.num_agents > 0:
        cfg.env.num_agents = args.num_agents
    
    if args.vision_size:
        cfg.env.resolution = (args.vision_size, args.vision_size)

    # Backend selection
    cfg.browser_backend = args.backend
    cfg.ws_server_url = args.ws_server_url

    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(base_dir, 'checkpoint.pth')
    backup_dir = os.path.join(base_dir, 'backup_models')
    os.makedirs(backup_dir, exist_ok=True)
    stats_file = os.path.join(base_dir, 'training_stats.csv')

    # Initialize Curriculum/Style Manager
    curriculum = CurriculumManager(style_name=style_name, start_stage=args.stage if args.stage > 0 else 1)

    # Auto-scaling setup
    auto_scale = args.auto_num_agents
    max_agents = args.max_agents
    monitor = ResourceMonitor() if auto_scale else None

    logger.info(f"Configuration:")
    logger.info(f"  Style: {style_name}")
    logger.info(f"  Mode: {curriculum.mode}")
    if auto_scale:
        logger.info(f"  Agents: AUTO (start=1, max={max_agents})")
    else:
        logger.info(f"  Agents: {cfg.env.num_agents}")
    logger.info(f"  Frame Stack: {cfg.env.frame_stack}")
    logger.info(f"  Resolution: {cfg.env.resolution}")
    logger.info(f"  Model: {cfg.model.architecture} ({cfg.model.activation})")
    logger.info(f"  LR: {cfg.opt.lr}")
    logger.info(f"  PER: {cfg.buffer.prioritized}")
    logger.info(f"  Backend: {cfg.browser_backend}")
    if cfg.ws_server_url:
        logger.info(f"  WS Server: {cfg.ws_server_url}")

    # Generate unique Run UID
    run_uid = generate_uid()
    parent_uid = None
    logger.info(f"Training Run UID: {run_uid}")

    # Initialize Env
    raw_env = SubprocVecEnv(
        num_agents=cfg.env.num_agents,
        matrix_size=cfg.env.resolution[0],
        frame_skip=cfg.env.frame_skip,
        view_first=args.view or args.view_plus,
        view_plus=args.view_plus,
        nickname="AI_Opt",
        base_url=args.url,
        backend=cfg.browser_backend,
        ws_server_url=cfg.ws_server_url,
        suppress_stdout=RICH_AVAILABLE,
    )
    env = VecFrameStack(raw_env, k=cfg.env.frame_stack)

    # Initialize Agent
    agent = DDQNAgent(cfg)

    # Resume / Load Model
    start_episode = 0
    load_path = model_path if model_path else (checkpoint_path if args.resume else None)

    if load_path and os.path.exists(load_path):
        start_episode, _, supervisor_state, checkpoint_uid = agent.load_checkpoint(load_path)

        # Set parent_uid from the checkpoint's run_uid (lineage tracking)
        if checkpoint_uid:
            parent_uid = checkpoint_uid
            logger.info(f"  Parent UID: {parent_uid}")

        # Fix stale LR from old checkpoints: reset if below 10% of target
        current_lr = agent.optimizer.param_groups[0]['lr']
        if current_lr < cfg.opt.lr * 0.1:
            for pg in agent.optimizer.param_groups:
                pg['lr'] = cfg.opt.lr
            logger.info(f"  LR was {current_lr:.8f} (stale). Reset to {cfg.opt.lr}")

        # Restore curriculum state only if we are in curriculum mode
        if curriculum.mode == 'curriculum' and supervisor_state:
            curriculum.load_state(supervisor_state)

        # --stage flag overrides checkpoint stage
        if args.stage > 0 and curriculum.current_stage != args.stage:
            logger.info(f"  --stage override: {curriculum.current_stage} -> {args.stage}")
            curriculum.current_stage = args.stage
            curriculum.episode_food_history.clear()
            curriculum.episode_steps_history.clear()
            curriculum.episode_food_ratio_history.clear()
            curriculum.episode_cause_history.clear()

        logger.info(f"Resumed from episode {start_episode}")
        if curriculum.mode == 'curriculum':
            logger.info(f"  Stage: {curriculum.current_stage} ({curriculum.get_config()['name']})")
    elif load_path:
        logger.warning(f"Model path {load_path} not found. Starting from scratch.")

    # Apply current curriculum stage/style to environments
    stage_cfg = curriculum.get_config()
    env.set_stage(stage_cfg)
    max_steps_per_episode = curriculum.get_max_steps()
    agent.set_gamma(stage_cfg.get('gamma', cfg.opt.gamma))
    logger.info(f"  Curriculum Stage: {curriculum.current_stage} ({stage_cfg['name']})")
    logger.info(f"  Max Steps: {max_steps_per_episode}")
    super_pattern = SuperPatternOptimizer(stage_cfg, cfg, logger)

    # Initialize stats file — update header if columns were added
    CSV_HEADER = "UID,ParentUID,Episode,Steps,Reward,Epsilon,Loss,Beta,LR,Cause,Stage,Food,QMean,QMax,TDError,GradNorm,ActStraight,ActGentle,ActMedium,ActSharp,ActUturn,ActBoost,NumAgents\n"
    if not os.path.exists(stats_file):
        with open(stats_file, 'w') as f:
            f.write(CSV_HEADER)
    else:
        with open(stats_file, 'r') as f:
            old_header = f.readline()
        if old_header.strip() != CSV_HEADER.strip():
            # Re-write file with updated header, preserving all data rows
            with open(stats_file, 'r') as f:
                _ = f.readline()  # skip old header
                data_lines = f.readlines()
            with open(stats_file, 'w') as f:
                f.write(CSV_HEADER)
                f.writelines(data_lines)
            logger.info(f"[CSV] Updated header ({old_header.strip().count(',') + 1} -> {CSV_HEADER.strip().count(',') + 1} columns)")

    # LR Scheduler (Linear Warmup) — based on batch_count, decoupled from num_agents
    warmup_batches = 2000
    target_lr = cfg.opt.lr
    batch_count = 0

    # Autonomy / Stabilization Vars
    reward_window = deque(maxlen=100)
    food_window = deque(maxlen=100)
    steps_window = deque(maxlen=100)
    best_avg_reward = -float('inf')
    best_fitness = -float('inf')
    episodes_since_improvement = 0

    # Metrics tracking
    total_steps = agent.steps_done
    episode_rewards = [0] * cfg.env.num_agents
    episode_steps = [0] * cfg.env.num_agents
    episode_food = [0] * cfg.env.num_agents
    agent_length = [0] * cfg.env.num_agents
    agent_server = [''] * cfg.env.num_agents
    # Per-episode action distribution: [straight, gentle, medium, sharp, uturn, boost]
    episode_actions = [[0]*6 for _ in range(cfg.env.num_agents)]
    # Running training metrics from optimize_model
    last_metrics = {'loss': 0, 'q_mean': 0, 'q_max': 0, 'td_error_mean': 0, 'grad_norm': 0}

    # Death Counters
    death_stats = {"Wall": 0, "SnakeCollision": 0, "InvalidFrame": 0, "BrowserError": 0, "MaxSteps": 0}

    # Per-agent board tracking
    AGENT_NAMES = ["Picard", "Riker", "Data", "Worf", "Troi", "LaForge",
                   "Crusher", "Q", "Seven", "Raffi", "Rios", "Jurati"]
    agent_ep_start = [time.time()] * cfg.env.num_agents
    agent_total_eps = [0] * cfg.env.num_agents
    agent_last_cause = ["—"] * cfg.env.num_agents

    # Initial Reset
    states = env.reset()

    # Initialize Rich TUI Dashboard
    dashboard = None
    if RICH_AVAILABLE:
        dashboard = TrainingDashboard()
        dashboard.uid = run_uid
        dashboard.style_name = style_name
        dashboard.num_agents = cfg.env.num_agents
        dashboard.stage = curriculum.current_stage
        dashboard.stage_name = curriculum.get_config()['name']
        dashboard.start()

    # Initialize AI Supervisor (if enabled)
    ai_supervisor = None
    ai_config_path = os.path.join(base_dir, 'config_ai.json')
    ai_config_mtime = 0  # track file modification time

    if args.ai_supervisor:
        from ai_supervisor import AISupervisor
        ai_supervisor = AISupervisor(
            stats_file=stats_file,
            config_ai_path=ai_config_path,
            provider=args.ai_supervisor,
            api_key=args.ai_key,
            model_name=args.ai_model,
            interval_episodes=args.ai_interval,
            lookback_episodes=args.ai_lookback,
        )
        ai_supervisor.start()
        logger.info(f"AI Supervisor enabled: provider={args.ai_supervisor}, interval={args.ai_interval}")
        if dashboard:
            dashboard.log_event(f"AI Supervisor: {args.ai_supervisor}")

    def _apply_ai_config():
        """Check for new config_ai.json and apply parameter changes."""
        nonlocal ai_config_mtime
        if not os.path.exists(ai_config_path):
            return
        try:
            mtime = os.path.getmtime(ai_config_path)
        except OSError:
            return
        if mtime <= ai_config_mtime:
            return
        ai_config_mtime = mtime

        try:
            with open(ai_config_path, 'r') as f:
                ai_cfg = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"[AI] Failed to read config_ai.json: {e}")
            return

        params = ai_cfg.get('parameters', {})
        if not params:
            return

        reasoning = ai_cfg.get('reasoning', '')
        logger.info(f"[AI] Applying config_ai.json: {params}")
        logger.info(f"[AI] Reasoning: {reasoning}")

        # Separate reward params from agent/training params
        from ai_supervisor import TUNABLE_PARAMS
        reward_params = {}
        for key, value in params.items():
            meta = TUNABLE_PARAMS.get(key)
            if not meta:
                continue
            _, _, _, group = meta

            if group == "reward":
                reward_params[key] = value
            elif key == "gamma":
                agent.set_gamma(value)
                logger.info(f"[AI] gamma -> {value}")
            elif key == "lr":
                for pg in agent.optimizer.param_groups:
                    pg['lr'] = value
                logger.info(f"[AI] lr -> {value}")
            elif key == "epsilon_target":
                agent.boost_exploration(target_eps=value)
                logger.info(f"[AI] epsilon_target -> {value}")
            elif key == "target_update_freq":
                cfg.opt.target_update_freq = int(value)
                logger.info(f"[AI] target_update_freq -> {int(value)}")

        # Apply reward params to environment via stage config
        if reward_params:
            merged = dict(super_pattern.current_stage_cfg)
            merged.update(reward_params)
            env.set_stage(merged)
            super_pattern.reset_stage(merged)
            logger.info(f"[AI] Reward params applied: {reward_params}")

        if dashboard:
            short = reasoning[:80] + "..." if len(reasoning) > 80 else reasoning
            dashboard.log_event(f"AI: {short}")

    def finalize_episode(agent_index, terminal_state, cause, force_done_flag):
        nonlocal start_episode, max_steps_per_episode, best_avg_reward, best_fitness, episodes_since_improvement
        total_steps_local = episode_steps[agent_index]
        total_reward = episode_rewards[agent_index]
        food_eaten = episode_food[agent_index]

        # Store transition (n-step)
        agent.remember_nstep(states[agent_index], actions[agent_index], rewards[agent_index], terminal_state, True, agent_id=agent_index)

        # Log
        start_episode += 1
        eps = agent.get_epsilon()
        loss_val = last_metrics.get('loss', 0)
        q_mean = last_metrics.get('q_mean', 0)
        q_max = last_metrics.get('q_max', 0)
        td_err = last_metrics.get('td_error_mean', 0)
        g_norm = last_metrics.get('grad_norm', 0)

        current_beta = min(1.0, agent.memory.beta_start + agent.memory.frame * (1.0 - agent.memory.beta_start) / agent.memory.beta_frames)
        lr = agent.optimizer.param_groups[0]['lr']

        if force_done_flag:
            cause_label = "MaxSteps"
        else:
            cause_label = cause if cause else "SnakeCollision"

        # Update death stats
        if cause_label in death_stats:
            death_stats[cause_label] += 1
        else:
            death_stats["SnakeCollision"] += 1

        pos = infos[agent_index].get('pos', (0, 0))
        wall_dist = infos[agent_index].get('wall_dist', -1)
        # MODIFIED: Use enemy_dist from HEAD logic
        enemy_dist = infos[agent_index].get('enemy_dist', -1)

        pos_str = f"Pos:({pos[0]:.0f},{pos[1]:.0f})"
        wall_str = f" Wall:{wall_dist:.0f}" if wall_dist >= 0 else ""
        enemy_str = f" Enemy:{enemy_dist:.0f}" if enemy_dist >= 0 else ""
        stage_name = curriculum.get_config()['name']

        food_ratio = food_eaten / max(total_steps_local, 1)

        log_msg = (f"Ep {start_episode} | S{curriculum.current_stage}:{stage_name} | "
                   f"Rw: {total_reward:.2f} | St: {total_steps_local} | "
                   f"Fd: {food_eaten} ({food_ratio:.3f}/st) | "
                   f"Eps: {eps:.3f} | L: {loss_val:.4f} | Q: {q_mean:.2f}/{q_max:.2f} | "
                   f"{cause_label} | {pos_str}{wall_str}{enemy_str}")

        logger.info(log_msg)

        # Print stats occasionally
        if start_episode % 10 == 0:
            logger.info(f"Death Stats: {death_stats}")

        # Action distribution for this episode
        act = episode_actions[agent_index]
        act_total = max(sum(act), 1)
        act_pcts = [a / act_total for a in act]  # [straight, gentle, medium, sharp, uturn, boost]

        with open(stats_file, 'a') as f:
            f.write(f"{run_uid},{parent_uid or ''},{start_episode},{total_steps_local},{total_reward:.2f},"
                    f"{eps:.4f},{loss_val:.4f},{current_beta:.2f},{lr:.6f},{cause_label},"
                    f"{curriculum.current_stage},{food_eaten},"
                    f"{q_mean:.4f},{q_max:.4f},{td_err:.4f},{g_norm:.4f},"
                    f"{act_pcts[0]:.3f},{act_pcts[1]:.3f},{act_pcts[2]:.3f},{act_pcts[3]:.3f},{act_pcts[4]:.3f},{act_pcts[5]:.3f},{env.num_agents}\n")

        # Update dashboard
        if dashboard:
            dashboard.update(
                episode=start_episode, stage=curriculum.current_stage,
                stage_name=stage_name, epsilon=eps, lr=lr, loss=loss_val,
                q_mean=q_mean, q_max=q_max, td_error=td_err, grad_norm=g_norm,
                reward=total_reward, steps=total_steps_local, food=food_eaten,
                cause=cause_label, action_pcts=act_pcts, num_agents=env.num_agents,
            )

        # Autonomy Logic (Scheduler & Watchdog)
        reward_window.append(total_reward)
        food_window.append(food_eaten)
        steps_window.append(total_steps_local)
        if len(reward_window) >= 20:
            avg_reward = sum(reward_window) / len(reward_window)
            avg_food = sum(food_window) / len(food_window)
            avg_steps = sum(steps_window) / len(steps_window)

            # Fitness = survival time + food eaten (weighted)
            # Steps dominate (survival skill), food multiplied to reward eating
            fitness = avg_steps + avg_food * 10

            # Scheduler step
            agent.step_scheduler(avg_reward)

            # Watchdog for stagnation (still tracks reward for exploration boost)
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                episodes_since_improvement = 0
            else:
                episodes_since_improvement += 1

            # Save BEST model based on fitness (steps + food), not raw reward
            if fitness > best_fitness:
                best_fitness = fitness
                backup_name = f"best_model_{run_uid}_ep{start_episode}_s{int(avg_steps)}_f{int(avg_food)}.pth"
                backup_path = os.path.join(backup_dir, backup_name)
                agent.save_checkpoint(backup_path, start_episode, max_steps_per_episode, curriculum.get_state(), run_uid=run_uid, parent_uid=parent_uid)
                logger.info(f"  >> New Best Fitness: {fitness:.1f} (steps={avg_steps:.1f}, food={avg_food:.1f}). Saved: {backup_name}")
                if dashboard:
                    dashboard.log_event(f"New best fitness: {fitness:.1f} (s={avg_steps:.0f} f={avg_food:.0f}) — saved {backup_name}")

            if episodes_since_improvement > cfg.opt.adaptive_eps_patience * env.num_agents:
                current_eps = agent.get_epsilon()
                new_eps = max(current_eps + 0.1, 0.3)
                logger.info(f"[Autonomy] Stagnation detected ({episodes_since_improvement} eps). Gentle eps boost: {current_eps:.3f} -> {new_eps:.3f}. LR unchanged.")
                if dashboard:
                    dashboard.log_event(f"Stagnation ({episodes_since_improvement} eps) — eps boost {current_eps:.3f} -> {new_eps:.3f}")
                agent.boost_exploration(target_eps=new_eps)
                # Do NOT reset LR — let the network keep learning at current rate
                episodes_since_improvement = 0

        # SuperPattern only active from stage 3+
        if curriculum.current_stage >= 3:
            super_pattern.record_episode(cause_label, food_eaten, total_steps_local, total_reward)
            updated_stage_cfg = super_pattern.maybe_update()
            if updated_stage_cfg:
                env.set_stage(updated_stage_cfg)
                if dashboard:
                    dashboard.log_event("SuperPattern adjusted rewards")

        # Track metrics for curriculum promotion
        curriculum.record_episode(food_eaten, total_steps_local, cause_label)

        # Check for stage promotion
        if curriculum.check_promotion():
            stage_cfg = curriculum.get_config()
            max_steps_per_episode = curriculum.get_max_steps()
            env.set_stage(stage_cfg)
            agent.set_gamma(stage_cfg.get('gamma', cfg.opt.gamma))
            super_pattern.reset_stage(stage_cfg)
            # Save checkpoint on promotion
            agent.save_checkpoint(checkpoint_path, start_episode, max_steps_per_episode, curriculum.get_state(), run_uid=run_uid, parent_uid=parent_uid)
            if dashboard:
                dashboard.log_event(f"STAGE UP -> S{curriculum.current_stage}: {stage_cfg['name']}")

        if start_episode % cfg.opt.checkpoint_every == 0:
            agent.save_checkpoint(checkpoint_path, start_episode, max_steps_per_episode, curriculum.get_state(), run_uid=run_uid, parent_uid=parent_uid)

        # AI Supervisor: notify episode and check for new config
        if ai_supervisor:
            ai_supervisor.notify_episode(start_episode)
            _apply_ai_config()

        # Per-agent board tracking
        agent_total_eps[agent_index] += 1
        agent_last_cause[agent_index] = cause_label[:10]
        agent_ep_start[agent_index] = time.time()

        episode_rewards[agent_index] = 0
        episode_steps[agent_index] = 0
        episode_food[agent_index] = 0
        agent_length[agent_index] = 0
        episode_actions[agent_index] = [0]*6

    try:
        _agents_draining = set()  # agents finishing their last episode during shutdown
        while start_episode < cfg.opt.max_episodes:
            batch_count += 1

            # LR Warmup (based on batch_count, independent of num_agents)
            if batch_count < warmup_batches:
                lr_scale = batch_count / warmup_batches
                for param_group in agent.optimizer.param_groups:
                    param_group['lr'] = target_lr * lr_scale

            # Select actions — steps_done increments once per batch (decoupled from num_agents)
            actions = [agent.select_action(s) for s in states]
            agent.steps_done += 1

            # Track action distribution per agent
            for i, a in enumerate(actions):
                if a == 0: episode_actions[i][0] += 1        # straight
                elif a in (1, 2): episode_actions[i][1] += 1  # gentle
                elif a in (3, 4): episode_actions[i][2] += 1  # medium
                elif a in (5, 6): episode_actions[i][3] += 1  # sharp
                elif a in (7, 8): episode_actions[i][4] += 1  # uturn
                elif a == 9: episode_actions[i][5] += 1        # boost

            # Step (with latency tracking for auto-scale)
            step_start = time.time()
            next_states, rewards, dones, infos = env.step(actions)
            if monitor:
                monitor.record_step(time.time() - step_start)

            loss = None

            # Increment total_steps once per batch step (not per agent)
            # to keep target_update_freq independent of num_agents
            total_steps += 1

            for i in range(env.num_agents):
                # Skip agents that already finished during graceful shutdown
                if i in _agents_draining:
                    continue

                episode_rewards[i] += rewards[i]
                episode_steps[i] += 1
                episode_food[i] += infos[i].get('food_eaten', 0)
                agent_length[i] = infos[i].get('length', agent_length[i])
                srv = infos[i].get('server_id', '')
                if srv:
                    agent_server[i] = srv

                force_done = episode_steps[i] >= max_steps_per_episode
                episode_done = dones[i] or force_done

                if episode_done:
                    terminal_state = infos[i]['terminal_observation'] if dones[i] else next_states[i]

                    if force_done and not dones[i]:
                        reset_state = env.reset_one(i)
                        next_states[i] = reset_state

                    finalize_episode(
                        agent_index=i,
                        terminal_state=terminal_state,
                        cause=infos[i].get('cause', 'SnakeCollision'),
                        force_done_flag=force_done and not dones[i]
                    )

                    # Graceful shutdown: don't start new episodes, mark agent as drained
                    if _shutdown_requested:
                        _agents_draining.add(i)
                else:
                    agent.remember_nstep(states[i], actions[i], rewards[i], next_states[i], False, agent_id=i)

            # Graceful shutdown: all agents finished their episodes — exit loop
            if _shutdown_requested and len(_agents_draining) >= env.num_agents:
                break

            # Update agent board (live per-step)
            if dashboard:
                now = time.time()
                board = []
                for i in range(env.num_agents):
                    board.append({
                        'idx': i,
                        'name': AGENT_NAMES[i % len(AGENT_NAMES)],
                        'reward': episode_rewards[i],
                        'food': episode_food[i],
                        'length': agent_length[i],
                        'steps': episode_steps[i],
                        'ep_time': now - agent_ep_start[i],
                        'total_eps': agent_total_eps[i],
                        'last_cause': agent_last_cause[i],
                        'server': agent_server[i],
                    })
                dashboard.update_agent_board(board)

            # Update states
            states = list(next_states) if not isinstance(next_states, list) else next_states

            # Auto-scale agents based on system resources
            if monitor and monitor.should_check():
                metrics = monitor.get_metrics()
                rec = monitor.recommend(env.num_agents, metrics, backend=cfg.browser_backend)
                if rec > 0 and env.num_agents < max_agents:
                    try:
                        new_state = env.add_agent()
                        episode_rewards.append(0)
                        episode_steps.append(0)
                        episode_food.append(0)
                        agent_length.append(0)
                        agent_server.append('')
                        episode_actions.append([0] * 6)
                        agent_ep_start.append(time.time())
                        agent_total_eps.append(0)
                        agent_last_cause.append("—")
                        states.append(new_state)
                        logger.info(f"[AUTO-SCALE] Added agent #{env.num_agents} "
                                    f"(CPU:{metrics['cpu_percent']:.0f}% "
                                    f"RAM:{metrics['ram_free_mb']:.0f}MB "
                                    f"Step:{metrics['avg_step_ms']:.0f}ms)")
                        if dashboard:
                            dashboard.log_event(f"Scale UP -> {env.num_agents} agents")
                    except RuntimeError as e:
                        logger.warning(f"[AUTO-SCALE] Failed to add agent: {e}")
                        if dashboard:
                            dashboard.log_event(f"Scale UP FAILED: {e}")
                elif rec < 0 and env.num_agents > 1:
                    env.remove_agent()
                    episode_rewards.pop()
                    episode_steps.pop()
                    episode_food.pop()
                    agent_length.pop()
                    agent_server.pop()
                    episode_actions.pop()
                    agent_ep_start.pop()
                    agent_total_eps.pop()
                    agent_last_cause.pop()
                    states.pop()
                    logger.info(f"[AUTO-SCALE] Removed agent -> {env.num_agents} "
                                f"(CPU:{metrics['cpu_percent']:.0f}% "
                                f"RAM:{metrics['ram_free_mb']:.0f}MB "
                                f"Step:{metrics['avg_step_ms']:.0f}ms)")
                    if dashboard:
                        dashboard.log_event(f"Scale DOWN -> {env.num_agents} agents")

            # Train
            metrics = agent.optimize_model()
            if metrics is not None:
                last_metrics = metrics
                train._last_loss = metrics['loss']

            # Target Update
            if total_steps % cfg.opt.target_update_freq == 0:
                agent.update_target()
                logger.info(">> Target Network Updated")
                if dashboard:
                    dashboard.log_event("Target network updated")

        # Graceful shutdown: loop ended because _shutdown_requested
        if _shutdown_requested:
            if dashboard:
                dashboard.log_event("Saving checkpoint and shutting down...")
                dashboard._refresh()
            logger.info("[Shutdown] Graceful shutdown via Ctrl+E. Saving checkpoint.")
            agent.save_checkpoint(checkpoint_path, start_episode, max_steps_per_episode, curriculum.get_state(), run_uid=run_uid, parent_uid=parent_uid)

    except KeyboardInterrupt:
        if dashboard:
            dashboard.stop()
        logger.info("[Shutdown] KeyboardInterrupt. Saving checkpoint.")
        print("Interrupted. Saving...")
        agent.save_checkpoint(checkpoint_path, start_episode, max_steps_per_episode, curriculum.get_state(), run_uid=run_uid, parent_uid=parent_uid)
    finally:
        if ai_supervisor:
            ai_supervisor.stop()
        if dashboard:
            dashboard.stop()
        env.close()
        logger.info(f"[Shutdown] Training ended at episode {start_episode}. UID: {run_uid}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, default=0, help="Override config num_agents")
    parser.add_argument("--view", action="store_true", help="View first agent")
    parser.add_argument("--view-plus", action="store_true", help="View first agent with bot vision overlay grid")
    parser.add_argument("--resume", action="store_true", help="Resume")
    parser.add_argument("--stage", type=int, default=0, help="Force start at specific stage (1-6)")
    parser.add_argument("--style-name", type=str, help="Learning style name (e.g. 'Aggressive')")
    parser.add_argument("--model-path", type=str, help="Path to model checkpoint to load")
    parser.add_argument("--url", type=str, default="http://slither.io", help="Game URL (e.g. http://eslither.io)")
    parser.add_argument("--vision-size", type=int, default=0, help="Vision input size override (default: from config)")
    parser.add_argument("--auto-num-agents", action="store_true", help="Auto-scale agents based on system resources")
    parser.add_argument("--max-agents", type=int, default=5, help="Max agents for auto-scaling (default: 5)")
    parser.add_argument("--backend", choices=["selenium", "websocket"], default="selenium", help="Browser backend: selenium (default) or websocket")
    parser.add_argument("--ws-server-url", type=str, default="", help="WebSocket server URL override (e.g. ws://1.2.3.4:444/slither)")
    parser.add_argument("--reset", action="store_true", help="Total reset: delete logs, CSV, checkpoints, events")
    parser.add_argument("--ai-supervisor", choices=["claude", "openai", "gemini", "ollama"], default=None, help="Enable AI Supervisor with chosen LLM provider")
    parser.add_argument("--ai-interval", type=int, default=200, help="AI Supervisor: consult every N episodes (default: 200)")
    parser.add_argument("--ai-lookback", type=int, default=500, help="AI Supervisor: analyze last N episodes (default: 500)")
    parser.add_argument("--ai-model", type=str, default=None, help="AI Supervisor: override LLM model name")
    parser.add_argument("--ai-key", type=str, default=None, help="AI Supervisor: API key (default: from env var)")
    args = parser.parse_args()

    if args.reset:
        import glob as glob_mod
        import shutil
        base = os.path.dirname(os.path.abspath(__file__))
        targets = []
        # Checkpoints
        for p in glob_mod.glob(os.path.join(base, '*.pth')):
            targets.append(p)
        for p in glob_mod.glob(os.path.join(base, '..', '*.pth')):
            targets.append(p)
        # Backup models dir
        backup_dir = os.path.join(base, 'backup_models')
        if os.path.isdir(backup_dir):
            targets.append(backup_dir)
        backup_dir2 = os.path.join(base, 'backup')
        if os.path.isdir(backup_dir2):
            targets.append(backup_dir2)
        # CSV
        for p in glob_mod.glob(os.path.join(base, '*.csv')):
            targets.append(p)
        # Logs
        for p in glob_mod.glob(os.path.join(base, 'logs', '*.log')):
            targets.append(p)
        # Charts
        for p in glob_mod.glob(os.path.join(base, '*.png')):
            targets.append(p)
        # Events
        events_dir = os.path.join(base, 'events')
        if os.path.isdir(events_dir):
            targets.append(events_dir)

        if not targets:
            print("Nothing to clean.")
            exit(0)

        print("\n=== TOTAL RESET ===")
        print("The following will be DELETED:\n")
        dirs = [t for t in targets if os.path.isdir(t)]
        files = [t for t in targets if not os.path.isdir(t)]
        for d in dirs:
            count = sum(len(fs) for _, _, fs in os.walk(d))
            print(f"  [DIR]  {os.path.relpath(d, base)}/  ({count} files)")
        for f in files:
            sz = os.path.getsize(f)
            unit = 'B'
            if sz > 1024*1024: sz /= 1024*1024; unit = 'MB'
            elif sz > 1024: sz /= 1024; unit = 'KB'
            print(f"  [FILE] {os.path.relpath(f, base)}  ({sz:.1f} {unit})")

        print(f"\nTotal: {len(files)} files + {len(dirs)} directories")
        confirm = input("\nType 'yes' to confirm deletion: ").strip()
        if confirm.lower() != 'yes':
            print("Aborted.")
            exit(0)

        for t in targets:
            if os.path.isdir(t):
                shutil.rmtree(t)
                print(f"  Removed dir:  {os.path.relpath(t, base)}/")
            elif os.path.isfile(t):
                os.remove(t)
                print(f"  Removed file: {os.path.relpath(t, base)}")

        print("\nReset complete. Ready for fresh training.")
        exit(0)

    mp.set_start_method('spawn', force=True)
    train(args)
