#!/usr/bin/env python3
"""
AI Supervisor Log Reader — pretty-prints consultations from ai_supervisor.log.

Usage:
    python3 ai_reader.py                  # all consultations
    python3 ai_reader.py --last 3         # last 3 consultations
    python3 ai_reader.py --episode 3400   # specific episode
    python3 ai_reader.py --live           # follow mode (like tail -f)
"""

import re
import json
import argparse
import time
import os
import sys

# ── ANSI colors ──────────────────────────────────────────────────────────────

RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
CYAN    = "\033[36m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
RED     = "\033[31m"
MAGENTA = "\033[35m"
WHITE   = "\033[97m"
BLUE    = "\033[34m"
BG_DARK = "\033[48;5;236m"

# Box-drawing
H_LINE = "─"
V_LINE = "│"
TL = "┌"; TR = "┐"; BL = "└"; BR = "┘"
T_DOWN = "┬"; T_UP = "┴"; T_RIGHT = "├"; T_LEFT = "┤"; CROSS = "┼"


def parse_log(log_path):
    """Parse ai_supervisor.log into a list of consultation dicts."""
    if not os.path.exists(log_path):
        return []

    with open(log_path, "r") as f:
        lines = f.readlines()

    consultations = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        # Look for consultation start
        m = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - INFO - === AI Consultation at episode (\d+) ===", line)
        if not m:
            # Check for errors
            if "ERROR" in line and consultations:
                err_m = re.search(r"ERROR - (.+)", line)
                if err_m:
                    consultations[-1].setdefault("errors", []).append(err_m.group(1))
            i += 1
            continue

        consultation = {
            "timestamp": m.group(1),
            "episode": int(m.group(2)),
            "stats": None,
            "prompt_system": "",
            "prompt_user": "",
            "response": None,
            "config_written": None,
            "elapsed": None,
            "changes": None,
            "errors": [],
        }

        # Read ahead to collect all info for this consultation
        i += 1
        while i < len(lines):
            line = lines[i].rstrip()

            # Stats block
            if "INFO - Aggregated stats over" in line:
                json_lines = []
                i += 1
                while i < len(lines) and not re.match(r"^\d{4}-\d{2}-\d{2}", lines[i]):
                    json_lines.append(lines[i].rstrip())
                    i += 1
                stats_str = line.split("episodes: ", 1)[-1] + "\n" + "\n".join(json_lines)
                try:
                    consultation["stats"] = json.loads(stats_str)
                except json.JSONDecodeError:
                    pass
                continue

            # Prompt (DEBUG level)
            if "DEBUG - Prompt:" in line:
                prompt_lines = []
                i += 1
                while i < len(lines) and not re.match(r"^\d{4}-\d{2}-\d{2}", lines[i]):
                    prompt_lines.append(lines[i].rstrip())
                    i += 1
                try:
                    prompt_data = eval("\n".join(prompt_lines))  # it's a Python dict repr
                    consultation["prompt_system"] = prompt_data.get("system", "")
                    consultation["prompt_user"] = prompt_data.get("user", "")
                except Exception:
                    pass
                continue

            # LLM Response
            if "INFO - LLM response:" in line:
                resp_lines = []
                i += 1
                while i < len(lines) and not re.match(r"^\d{4}-\d{2}-\d{2}", lines[i]):
                    resp_lines.append(lines[i].rstrip())
                    i += 1
                resp_str = "\n".join(resp_lines).strip()
                try:
                    consultation["response"] = json.loads(resp_str)
                except json.JSONDecodeError:
                    consultation["response"] = {"raw": resp_str}
                continue

            # Config written
            if "INFO - Wrote" in line:
                m2 = re.search(r"Wrote .+?: (.+)", line)
                if m2:
                    try:
                        consultation["config_written"] = json.loads(m2.group(1))
                    except json.JSONDecodeError:
                        pass
                i += 1
                continue

            # Consultation complete
            if "INFO - Consultation complete" in line:
                m2 = re.search(r"in (\d+\.\d+)s\. Changes: (.+)", line)
                if m2:
                    consultation["elapsed"] = float(m2.group(1))
                    try:
                        consultation["changes"] = eval(m2.group(2))
                    except Exception:
                        pass
                i += 1
                break

            # Error
            if "ERROR" in line:
                err_m = re.search(r"ERROR - (.+)", line)
                if err_m:
                    consultation["errors"].append(err_m.group(1))
                i += 1
                # If error, this consultation is done
                if "Empty LLM response" in line or "Failed to parse" in line:
                    break
                continue

            # Next consultation or unrelated line — done with this one
            if re.match(r"^\d{4}-\d{2}-\d{2}.*=== AI Consultation", line):
                break

            i += 1

        consultations.append(consultation)

    return consultations


def box(text, width, color=WHITE):
    """Wrap text in a box."""
    lines = text.split("\n")
    result = []
    result.append(f"{DIM}{TL}{H_LINE * (width - 2)}{TR}{RESET}")
    for l in lines:
        visible_len = len(re.sub(r'\033\[[0-9;]*m', '', l))
        pad = max(0, width - 4 - visible_len)
        result.append(f"{DIM}{V_LINE}{RESET} {l}{' ' * pad} {DIM}{V_LINE}{RESET}")
    result.append(f"{DIM}{BL}{H_LINE * (width - 2)}{BR}{RESET}")
    return "\n".join(result)


def format_stats_panel(stats):
    """Format the stats/query side."""
    if not stats:
        return f"{DIM}(no stats available){RESET}"

    lines = []
    lines.append(f"{BOLD}{CYAN}  TRAINING STATS{RESET}")
    lines.append(f"{DIM}  {'─' * 38}{RESET}")

    metrics = [
        ("Avg Reward",   stats.get("avg_reward"),    None),
        ("Avg Steps",    stats.get("avg_steps"),      None),
        ("Avg Food",     stats.get("avg_food"),       None),
        ("Avg Loss",     stats.get("avg_loss"),       None),
        ("Avg Q-Mean",   stats.get("avg_q_mean"),     None),
        ("Epsilon",      stats.get("current_epsilon"), None),
    ]
    for label, val, _ in metrics:
        if val is not None:
            color = GREEN if "Food" in label else WHITE
            if "Reward" in label:
                color = RED if val < 0 else GREEN
            if "Loss" in label:
                color = YELLOW if val > 5 else GREEN
            lines.append(f"  {DIM}{label:<16}{RESET} {color}{val:>10}{RESET}")

    lines.append("")
    lines.append(f"{BOLD}{CYAN}  TRENDS (2nd vs 1st half){RESET}")
    lines.append(f"{DIM}  {'─' * 38}{RESET}")

    trends = [
        ("Reward", stats.get("reward_trend")),
        ("Steps",  stats.get("steps_trend")),
        ("Loss",   stats.get("loss_trend")),
        ("Q-Value", stats.get("q_trend")),
    ]
    for label, val in trends:
        if val is not None:
            arrow = "▲" if val > 0 else "▼" if val < 0 else "─"
            color = GREEN if val > 0 else RED if val < 0 else DIM
            if "Loss" in label:
                color = GREEN if val < 0 else RED  # lower loss = better
            lines.append(f"  {DIM}{label:<16}{RESET} {color}{arrow} {val:>+10.2f}{RESET}")

    # Death distribution
    deaths = stats.get("death_distribution_pct", {})
    if deaths:
        lines.append("")
        lines.append(f"{BOLD}{CYAN}  DEATHS{RESET}")
        lines.append(f"{DIM}  {'─' * 38}{RESET}")
        for cause, pct in sorted(deaths.items(), key=lambda x: -x[1]):
            bar_len = int(pct / 100 * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            color = RED if pct > 50 else YELLOW if pct > 25 else GREEN
            lines.append(f"  {DIM}{cause:<18}{RESET} {color}{bar} {pct:>5.1f}%{RESET}")

    # Action distribution
    actions = stats.get("action_distribution", {})
    if actions:
        lines.append("")
        lines.append(f"{BOLD}{CYAN}  ACTIONS{RESET}")
        lines.append(f"{DIM}  {'─' * 38}{RESET}")
        for act, pct in actions.items():
            short = act.replace("Act", "")
            bar_len = int(pct * 30)
            bar = "▓" * bar_len + "░" * (30 - bar_len)
            color = MAGENTA if pct > 0.3 else WHITE
            lines.append(f"  {DIM}{short:<12}{RESET} {color}{bar} {pct:>5.1%}{RESET}")

    return "\n".join(lines)


def format_response_panel(consultation):
    """Format the AI response side."""
    lines = []

    resp = consultation.get("response")
    errors = consultation.get("errors", [])

    if errors:
        lines.append(f"{BOLD}{RED}  ERRORS{RESET}")
        lines.append(f"{DIM}  {'─' * 38}{RESET}")
        for err in errors:
            lines.append(f"  {RED}{err}{RESET}")
        return "\n".join(lines)

    if not resp:
        lines.append(f"{DIM}  (no response){RESET}")
        return "\n".join(lines)

    # Reasoning
    reasoning = resp.get("reasoning", resp.get("raw", ""))
    if reasoning:
        lines.append(f"{BOLD}{GREEN}  AI REASONING{RESET}")
        lines.append(f"{DIM}  {'─' * 38}{RESET}")
        # Word-wrap reasoning
        words = reasoning.split()
        current_line = "  "
        for word in words:
            if len(current_line) + len(word) + 1 > 44:
                lines.append(f"{WHITE}{current_line}{RESET}")
                current_line = "  " + word
            else:
                current_line += " " + word if current_line.strip() else "  " + word
        if current_line.strip():
            lines.append(f"{WHITE}{current_line}{RESET}")

    # Parameter changes
    params = resp.get("parameters", {})
    changes = consultation.get("changes", params)
    if changes:
        lines.append("")
        lines.append(f"{BOLD}{YELLOW}  PARAMETER CHANGES{RESET}")
        lines.append(f"{DIM}  {'─' * 38}{RESET}")
        lines.append(f"  {DIM}{'Parameter':<24} {'New Value':>14}{RESET}")
        lines.append(f"  {DIM}{'─' * 24} {'─' * 14}{RESET}")
        for param, val in changes.items():
            if isinstance(val, float):
                val_str = f"{val:.6f}" if abs(val) < 0.01 and val != 0 else f"{val:.4f}"
            else:
                val_str = str(val)
            lines.append(f"  {CYAN}{param:<24}{RESET} {BOLD}{YELLOW}{val_str:>14}{RESET}")
    elif not errors:
        lines.append("")
        lines.append(f"  {GREEN}No changes recommended.{RESET}")

    # Timing
    elapsed = consultation.get("elapsed")
    if elapsed:
        lines.append("")
        lines.append(f"  {DIM}Completed in {elapsed:.1f}s{RESET}")

    return "\n".join(lines)


def render_consultation(c, idx=None, total=None):
    """Render a single consultation as a side-by-side display."""
    try:
        term_width = os.get_terminal_size().columns
    except OSError:
        term_width = 120
    panel_width = min((term_width - 3) // 2, 48)

    # Header
    ep = c["episode"]
    ts = c["timestamp"]
    stage = c["stats"].get("current_stage", "?") if c["stats"] else "?"
    counter = f" [{idx}/{total}]" if idx else ""

    header = (
        f"\n{BOLD}{BG_DARK}"
        f"  {'═' * (term_width - 4)}  {RESET}\n"
        f"{BOLD}{BG_DARK}"
        f"  Episode {CYAN}{ep}{WHITE}  {DIM}{V_LINE}  "
        f"{WHITE}Stage {MAGENTA}{stage}{WHITE}  {DIM}{V_LINE}  "
        f"{DIM}{ts}{WHITE}{counter}"
        f"{' ' * max(0, term_width - 50 - len(counter))}"
        f"  {RESET}\n"
        f"{BOLD}{BG_DARK}"
        f"  {'═' * (term_width - 4)}  {RESET}"
    )
    print(header)

    # Build panels
    left = format_stats_panel(c["stats"])
    right = format_response_panel(c)

    left_lines = left.split("\n")
    right_lines = right.split("\n")

    # Pad to same height
    max_lines = max(len(left_lines), len(right_lines))
    left_lines += [""] * (max_lines - len(left_lines))
    right_lines += [""] * (max_lines - len(right_lines))

    # Column headers
    lh = f"{BOLD}{'QUERY (Stats sent to AI)':<{panel_width}}{RESET}"
    rh = f"{BOLD}{'RESPONSE (AI Recommendation)':<{panel_width}}{RESET}"
    print(f"  {lh} {DIM}{V_LINE}{RESET} {rh}")
    print(f"  {DIM}{'─' * panel_width} {V_LINE} {'─' * panel_width}{RESET}")

    # Side by side
    for l, r in zip(left_lines, right_lines):
        l_visible = len(re.sub(r'\033\[[0-9;]*m', '', l))
        r_visible = len(re.sub(r'\033\[[0-9;]*m', '', r))
        l_pad = max(0, panel_width - l_visible)
        r_pad = max(0, panel_width - r_visible)
        print(f"  {l}{' ' * l_pad} {DIM}{V_LINE}{RESET} {r}{' ' * r_pad}")

    print(f"  {DIM}{'─' * panel_width} {T_UP} {'─' * panel_width}{RESET}")


def main():
    parser = argparse.ArgumentParser(
        description="AI Supervisor Log Reader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python3 ai_reader.py              # show all consultations\n"
               "  python3 ai_reader.py --last 3      # last 3\n"
               "  python3 ai_reader.py --episode 3400 # specific episode\n"
               "  python3 ai_reader.py --live         # follow mode\n"
    )
    parser.add_argument("--log", default=None, help="Path to ai_supervisor.log")
    parser.add_argument("--last", type=int, default=0, help="Show last N consultations")
    parser.add_argument("--episode", type=int, default=0, help="Show specific episode")
    parser.add_argument("--live", action="store_true", help="Follow mode (tail -f)")
    parser.add_argument("--summary", action="store_true", help="Compact summary table only")
    args = parser.parse_args()

    log_path = args.log or os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "ai_supervisor.log")

    if not os.path.exists(log_path):
        print(f"{RED}Log not found: {log_path}{RESET}")
        sys.exit(1)

    if args.live:
        print(f"{BOLD}{CYAN}AI Supervisor Live Monitor{RESET}")
        print(f"{DIM}Watching: {log_path}{RESET}")
        print(f"{DIM}Press Ctrl+C to stop{RESET}\n")
        last_size = 0
        last_count = 0
        try:
            while True:
                cur_size = os.path.getsize(log_path)
                if cur_size != last_size:
                    last_size = cur_size
                    consultations = parse_log(log_path)
                    if len(consultations) > last_count:
                        for c in consultations[last_count:]:
                            render_consultation(c, len(consultations), len(consultations))
                        last_count = len(consultations)
                time.sleep(2)
        except KeyboardInterrupt:
            print(f"\n{DIM}Stopped.{RESET}")
            return

    consultations = parse_log(log_path)

    if not consultations:
        print(f"{YELLOW}No consultations found in {log_path}{RESET}")
        sys.exit(0)

    # Filter
    if args.episode:
        consultations = [c for c in consultations if c["episode"] == args.episode]
        if not consultations:
            print(f"{YELLOW}No consultation found for episode {args.episode}{RESET}")
            sys.exit(0)
    elif args.last:
        consultations = consultations[-args.last:]

    if args.summary:
        # Compact table
        print(f"\n{BOLD}{'Ep':>6}  {'Time':<19}  {'Stage':>5}  {'Reward':>9}  {'Food':>6}  "
              f"{'Deaths':>20}  {'Changes'}{RESET}")
        print(f"{DIM}{'─' * 100}{RESET}")
        for c in consultations:
            s = c["stats"] or {}
            deaths = s.get("death_distribution_pct", {})
            top_death = max(deaths, key=deaths.get) if deaths else "-"
            top_pct = deaths.get(top_death, 0) if deaths else 0
            changes = c.get("changes", {})
            ch_str = ", ".join(f"{k}={v}" for k, v in changes.items()) if changes else "-"
            err = " ERROR" if c["errors"] else ""
            print(
                f"{c['episode']:>6}  {c['timestamp']:<19}  "
                f"{s.get('current_stage', '?'):>5}  "
                f"{s.get('avg_reward', 0):>+9.1f}  "
                f"{s.get('avg_food', 0):>6.1f}  "
                f"{top_death}({top_pct:.0f}%){'':>8}  "
                f"{YELLOW}{ch_str}{RESET}{RED}{err}{RESET}"
            )
        print()
        return

    # Full render
    total = len(consultations)
    for i, c in enumerate(consultations, 1):
        render_consultation(c, i, total)

    # Summary footer
    print(f"\n{BOLD}{CYAN}Summary:{RESET} {total} consultation(s)")
    all_changes = {}
    for c in consultations:
        if c.get("changes"):
            for k, v in c["changes"].items():
                all_changes.setdefault(k, []).append((c["episode"], v))
    if all_changes:
        print(f"{BOLD}Parameter history:{RESET}")
        for param, history in all_changes.items():
            trail = " -> ".join(f"ep{ep}:{YELLOW}{v}{RESET}" for ep, v in history)
            print(f"  {CYAN}{param}{RESET}: {trail}")
    print()


if __name__ == "__main__":
    main()
