"""
AI Supervisor for SlitherBot DDQN Trainer.

An LLM-based hyperparameter tuner that periodically analyzes training statistics
and adjusts reward/agent parameters. Works alongside SuperPatternOptimizer:
- SuperPattern: fast, rule-based, small adjustments every 50 episodes (4 params)
- AI Supervisor: slow, LLM-based, broader adjustments every 200+ episodes (14+ params)
"""

import os
import sys
import json
import time
import threading
import logging
from datetime import datetime
from collections import deque

# Load .env file if present (no external dependency)
def _load_dotenv():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if not os.environ.get(key):  # don't override existing env vars
                os.environ[key] = value

_load_dotenv()

logger = logging.getLogger("ai_supervisor")
logger.setLevel(logging.DEBUG)

# File handler for dedicated log
os.makedirs("logs", exist_ok=True)
_fh = logging.FileHandler("logs/ai_supervisor.log")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
if not logger.hasHandlers():
    logger.addHandler(_fh)


# ---------------------------------------------------------------------------
# Tunable parameters: name -> (min, max, type, group)
# ---------------------------------------------------------------------------
TUNABLE_PARAMS = {
    # Reward shaping
    "food_reward":             (1.0,   30.0,  float, "reward"),
    "food_shaping":            (0.0,   1.0,   float, "reward"),
    "survival":                (0.0,   1.0,   float, "reward"),
    "death_wall":              (-100,  -5,    float, "reward"),
    "death_snake":             (-100,  -5,    float, "reward"),
    "wall_proximity_penalty":  (0.0,   3.0,   float, "reward"),
    "enemy_proximity_penalty": (0.0,   3.0,   float, "reward"),
    "enemy_approach_penalty":  (0.0,   2.0,   float, "reward"),
    "starvation_penalty":      (0.0,   0.05,  float, "reward"),
    "starvation_grace_steps":  (20,    200,   int,   "reward"),
    # Agent
    "gamma":                   (0.8,   0.999, float, "agent"),
    "lr":                      (1e-6,  1e-3,  float, "agent"),
    "epsilon_target":          (0.05,  0.5,   float, "agent"),
    # Training
    "target_update_freq":      (200,   5000,  int,   "training"),
}


def _clamp(value, param_name):
    """Clamp value to safe range defined in TUNABLE_PARAMS."""
    if param_name not in TUNABLE_PARAMS:
        return None
    lo, hi, typ, _ = TUNABLE_PARAMS[param_name]
    try:
        value = typ(value)
    except (ValueError, TypeError):
        return None
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# LLM System Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an AI supervisor for a reinforcement learning agent learning to play slither.io.

The agent uses Double DQN with Prioritized Experience Replay, n-step returns, and a curriculum
of stages (1: Food Vector, 2: Wall Avoid, 3: Enemy Avoid, 4: Mass Management).

Your job: analyze training statistics and recommend hyperparameter adjustments to improve learning.

## Tunable Parameters

{param_table}

## Guidelines

- Focus on the BIGGEST problem first (e.g., if wall deaths are 60%+, prioritize wall penalties).
- Make conservative changes — the agent is learning online; drastic shifts destabilize training.
- Only change parameters you have a clear reason for. Do NOT change everything at once.
- If training looks healthy (improving reward, reasonable death distribution), recommend NO changes.
- Consider the current stage: early stages focus on food, later stages on survival and enemies.
- Lower lr if loss is unstable; raise if learning is too slow.
- gamma should increase as the agent matures (0.85 early → 0.97+ late).
- epsilon_target controls minimum exploration; raise if stuck in local optima, lower if nearly converged.

## Response Format

Respond with ONLY a JSON object (no markdown fences, no extra text):

{{
  "reasoning": "Brief explanation of your analysis and changes (2-3 sentences)",
  "parameters": {{
    "param_name": new_value,
    ...
  }}
}}

If no changes needed, return:
{{
  "reasoning": "Training looks healthy, no changes needed.",
  "parameters": {{}}
}}
"""


def _build_param_table(current_values):
    """Build a markdown table of tunable params with current values and ranges."""
    lines = ["| Parameter | Current | Min | Max | Group |",
             "|---|---|---|---|---|"]
    for name, (lo, hi, typ, group) in TUNABLE_PARAMS.items():
        cur = current_values.get(name, "?")
        if isinstance(cur, float):
            if abs(cur) < 0.001 and cur != 0:
                cur_str = f"{cur:.6f}"
            else:
                cur_str = f"{cur:.4f}"
        else:
            cur_str = str(cur)
        lines.append(f"| {name} | {cur_str} | {lo} | {hi} | {group} |")
    return "\n".join(lines)


class AISupervisor:
    """LLM-based hyperparameter supervisor for DDQN training."""

    def __init__(
        self,
        stats_file,
        config_ai_path,
        provider="claude",
        api_key=None,
        model_name=None,
        interval_episodes=200,
        lookback_episodes=500,
    ):
        self.stats_file = stats_file
        self.config_ai_path = config_ai_path
        self.provider = provider.lower()
        self.api_key = api_key or self._default_api_key()
        self.model_name = model_name or self._default_model()
        self.interval = interval_episodes
        self.lookback = lookback_episodes

        self._lock = threading.Lock()
        self._thread = None
        self._stop_event = threading.Event()
        self._pending_episode = None  # episode number queued for consultation

        # Track last consultation to avoid duplicates
        self._last_consulted_ep = 0

    def _default_api_key(self):
        if self.provider == "ollama":
            return ""  # Ollama needs no API key
        env_map = {
            "claude": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "gemini": "GOOGLE_API_KEY",
        }
        var = env_map.get(self.provider, "ANTHROPIC_API_KEY")
        return os.environ.get(var, "")

    def _default_model(self):
        defaults = {
            "claude": "claude-sonnet-4-20250514",
            "openai": "gpt-4o",
            "gemini": "gemini-2.0-flash",
            "ollama": os.environ.get("OLLAMA_MODEL", "llama3.1"),
        }
        return defaults.get(self.provider, "claude-sonnet-4-20250514")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self):
        """Start the background consultation thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="ai-supervisor")
        self._thread.start()
        logger.info(f"AI Supervisor started (provider={self.provider}, model={self.model_name}, "
                    f"interval={self.interval}, lookback={self.lookback})")

    def stop(self):
        """Signal the background thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        logger.info("AI Supervisor stopped.")

    def notify_episode(self, episode_num):
        """Called from trainer after each episode. Queues consultation if interval reached."""
        if episode_num > 0 and episode_num % self.interval == 0 and episode_num > self._last_consulted_ep:
            with self._lock:
                self._pending_episode = episode_num
            logger.debug(f"Consultation queued for episode {episode_num}")

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------
    def _run_loop(self):
        """Background thread: checks for pending consultation requests."""
        while not self._stop_event.is_set():
            ep = None
            with self._lock:
                if self._pending_episode is not None:
                    ep = self._pending_episode
                    self._pending_episode = None

            if ep is not None:
                try:
                    self._consult(ep)
                except Exception as e:
                    logger.error(f"Consultation failed for episode {ep}: {e}", exc_info=True)

            self._stop_event.wait(timeout=2.0)

    # ------------------------------------------------------------------
    # Core consultation pipeline
    # ------------------------------------------------------------------
    def _consult(self, episode_num):
        """Full consultation: collect stats → build prompt → call LLM → apply."""
        logger.info(f"=== AI Consultation at episode {episode_num} ===")
        t0 = time.time()

        rows = self._collect_stats()
        if not rows:
            logger.warning("No stats rows found, skipping consultation.")
            return

        stats = self._aggregate_stats(rows)
        logger.info(f"Aggregated stats over {len(rows)} episodes: {json.dumps(stats, indent=2)}")

        current_values = self._get_current_values(rows)
        prompt = self._build_prompt(stats, current_values, episode_num)
        logger.debug(f"Prompt:\n{prompt}")

        response_text = self._call_llm(prompt, current_values)
        if not response_text:
            logger.error("Empty LLM response.")
            return
        logger.info(f"LLM response:\n{response_text}")

        result = self._parse_response(response_text)
        if result is None:
            logger.error("Failed to parse LLM response.")
            return

        if not result.get("parameters"):
            logger.info("LLM recommended no changes.")
            self._last_consulted_ep = episode_num
            return

        self._write_config(result, episode_num)
        self._last_consulted_ep = episode_num
        elapsed = time.time() - t0
        logger.info(f"Consultation complete in {elapsed:.1f}s. Changes: {result['parameters']}")

    # ------------------------------------------------------------------
    # Stats collection
    # ------------------------------------------------------------------
    def _collect_stats(self):
        """Read last N rows from training_stats.csv using tail-read (efficient for large files)."""
        if not os.path.exists(self.stats_file):
            return []

        # Read header first
        with open(self.stats_file, "r") as f:
            header_line = f.readline().strip()
        if not header_line:
            return []
        headers = header_line.split(",")

        # Tail-read: read last chunk of file
        try:
            file_size = os.path.getsize(self.stats_file)
            # Estimate ~200 bytes per line, read extra to be safe
            read_size = min(file_size, self.lookback * 300)
            with open(self.stats_file, "rb") as f:
                f.seek(max(0, file_size - read_size))
                chunk = f.read().decode("utf-8", errors="replace")

            lines = chunk.strip().split("\n")
            # Skip partial first line (unless we read from start)
            if file_size > read_size:
                lines = lines[1:]  # first line is likely partial

            # Filter out header if present
            data_lines = [l for l in lines if l and not l.startswith("UID,")]
            # Take last N
            data_lines = data_lines[-self.lookback:]

            rows = []
            for line in data_lines:
                parts = line.split(",")
                if len(parts) < len(headers):
                    continue
                row = {}
                for i, h in enumerate(headers):
                    row[h] = parts[i] if i < len(parts) else ""
                rows.append(row)
            return rows
        except Exception as e:
            logger.error(f"Failed to read stats: {e}")
            return []

    def _aggregate_stats(self, rows):
        """Compute aggregated statistics from raw CSV rows."""
        def safe_float(v, default=0.0):
            try:
                return float(v)
            except (ValueError, TypeError):
                return default

        n = len(rows)
        rewards = [safe_float(r.get("Reward")) for r in rows]
        steps = [safe_float(r.get("Steps")) for r in rows]
        food = [safe_float(r.get("Food")) for r in rows]
        losses = [safe_float(r.get("Loss")) for r in rows]
        q_means = [safe_float(r.get("QMean")) for r in rows]
        epsilons = [safe_float(r.get("Epsilon")) for r in rows]
        causes = [r.get("Cause", "Unknown") for r in rows]

        # Death distribution
        cause_counts = {}
        for c in causes:
            cause_counts[c] = cause_counts.get(c, 0) + 1
        cause_pcts = {k: round(v / n * 100, 1) for k, v in cause_counts.items()}

        # Trend: compare first half vs second half
        mid = n // 2
        def trend(values):
            if mid == 0:
                return 0.0
            first = sum(values[:mid]) / max(mid, 1)
            second = sum(values[mid:]) / max(n - mid, 1)
            return round(second - first, 4)

        # Action entropy (from last row's action distribution)
        import math
        act_keys = ["ActStraight", "ActGentle", "ActMedium", "ActSharp", "ActUturn", "ActBoost"]
        # Average action distribution over all rows
        act_avgs = []
        for key in act_keys:
            vals = [safe_float(r.get(key)) for r in rows]
            act_avgs.append(sum(vals) / max(len(vals), 1))
        total_act = sum(act_avgs) or 1.0
        act_probs = [a / total_act for a in act_avgs]
        entropy = -sum(p * math.log(p + 1e-10) for p in act_probs)
        max_entropy = math.log(len(act_keys))

        return {
            "num_episodes": n,
            "avg_reward": round(sum(rewards) / n, 2),
            "avg_steps": round(sum(steps) / n, 1),
            "avg_food": round(sum(food) / n, 2),
            "avg_loss": round(sum(losses) / n, 6),
            "avg_q_mean": round(sum(q_means) / n, 4),
            "current_epsilon": round(epsilons[-1], 4) if epsilons else 0,
            "reward_trend": trend(rewards),
            "steps_trend": trend(steps),
            "loss_trend": trend(losses),
            "q_trend": trend(q_means),
            "death_distribution_pct": cause_pcts,
            "action_entropy": round(entropy, 3),
            "action_entropy_normalized": round(entropy / max_entropy, 3),
            "action_distribution": {k: round(v, 3) for k, v in zip(act_keys, act_probs)},
            "current_stage": rows[-1].get("Stage", "?") if rows else "?",
        }

    def _get_current_values(self, rows):
        """Extract current param values from latest stats + any existing config_ai.json."""
        values = {}

        # Start with defaults from the latest CSV row
        if rows:
            last = rows[-1]
            values["lr"] = float(last.get("LR", 1e-4))
            values["gamma"] = 0.95  # can't read from CSV, use reasonable default

        # Override with existing config_ai.json if present
        if os.path.exists(self.config_ai_path):
            try:
                with open(self.config_ai_path, "r") as f:
                    existing = json.load(f)
                for k, v in existing.get("parameters", {}).items():
                    values[k] = v
            except (json.JSONDecodeError, IOError):
                pass

        # Fill remaining with midpoint defaults
        for name, (lo, hi, typ, _) in TUNABLE_PARAMS.items():
            if name not in values:
                values[name] = typ((lo + hi) / 2)

        return values

    def _build_prompt(self, stats, current_values, episode_num):
        """Build the user prompt with stats and current values."""
        param_table = _build_param_table(current_values)
        system = SYSTEM_PROMPT.format(param_table=param_table)

        user = (
            f"## Current State\n\n"
            f"Episode: {episode_num}\n"
            f"Stage: {stats['current_stage']}\n\n"
            f"## Training Statistics (last {stats['num_episodes']} episodes)\n\n"
            f"| Metric | Value |\n|---|---|\n"
            f"| Avg Reward | {stats['avg_reward']} |\n"
            f"| Avg Steps | {stats['avg_steps']} |\n"
            f"| Avg Food | {stats['avg_food']} |\n"
            f"| Avg Loss | {stats['avg_loss']} |\n"
            f"| Avg Q-Mean | {stats['avg_q_mean']} |\n"
            f"| Current Epsilon | {stats['current_epsilon']} |\n"
            f"| Reward Trend (2nd half - 1st half) | {stats['reward_trend']} |\n"
            f"| Steps Trend | {stats['steps_trend']} |\n"
            f"| Loss Trend | {stats['loss_trend']} |\n"
            f"| Q-Value Trend | {stats['q_trend']} |\n"
            f"| Action Entropy (normalized) | {stats['action_entropy_normalized']} |\n\n"
            f"## Death Distribution\n\n"
        )
        for cause, pct in stats["death_distribution_pct"].items():
            user += f"- {cause}: {pct}%\n"

        user += (
            f"\n## Action Distribution\n\n"
        )
        for act, pct in stats["action_distribution"].items():
            user += f"- {act}: {pct:.1%}\n"

        user += "\nAnalyze the training progress and recommend parameter adjustments."

        return {"system": system, "user": user}

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------
    def _call_llm(self, prompt, current_values):
        """Dispatch to the appropriate LLM provider."""
        dispatch = {
            "claude": self._call_claude,
            "openai": self._call_openai,
            "gemini": self._call_gemini,
            "ollama": self._call_ollama,
        }
        fn = dispatch.get(self.provider)
        if not fn:
            logger.error(f"Unknown provider: {self.provider}")
            return None
        return fn(prompt)

    def _call_claude(self, prompt):
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                system=prompt["system"],
                messages=[{"role": "user", "content": prompt["user"]}],
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return None

    def _call_openai(self, prompt):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model_name,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]},
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return None

    def _call_gemini(self, prompt):
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(
                self.model_name,
                system_instruction=prompt["system"],
            )
            response = model.generate_content(prompt["user"])
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return None

    def _call_ollama(self, prompt):
        try:
            import urllib.request
            host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
            url = f"{host}/api/chat"
            payload = json.dumps({
                "model": self.model_name,
                "stream": False,
                "messages": [
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]},
                ],
            }).encode("utf-8")
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data.get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            return None

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------
    def _parse_response(self, text):
        """Extract JSON from LLM response, validate and clamp parameters."""
        # Try to find JSON in the response
        text = text.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        # Find JSON object boundaries
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            logger.error(f"No JSON found in response: {text[:200]}")
            return None

        json_str = text[start:end]
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}\nRaw: {json_str[:500]}")
            return None

        if "parameters" not in data:
            logger.error(f"Missing 'parameters' key in response: {data}")
            return None

        # Validate and clamp each parameter
        validated = {}
        for key, value in data["parameters"].items():
            if key not in TUNABLE_PARAMS:
                logger.warning(f"Unknown parameter '{key}' — skipping.")
                continue
            clamped = _clamp(value, key)
            if clamped is None:
                logger.warning(f"Invalid value for '{key}': {value} — skipping.")
                continue
            validated[key] = clamped

        return {
            "reasoning": data.get("reasoning", ""),
            "parameters": validated,
        }

    # ------------------------------------------------------------------
    # Config writing
    # ------------------------------------------------------------------
    def _write_config(self, result, episode_num):
        """Atomically write config_ai.json (write .tmp → os.replace)."""
        config = {
            "version": 2,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "episode": episode_num,
            "source": self.provider,
            "model": self.model_name,
            "reasoning": result["reasoning"],
            "parameters": result["parameters"],
        }

        tmp_path = self.config_ai_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(config, f, indent=2)
        os.replace(tmp_path, self.config_ai_path)
        logger.info(f"Wrote {self.config_ai_path}: {json.dumps(result['parameters'])}")


# ---------------------------------------------------------------------------
# CLI test mode
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="AI Supervisor — test mode")
    parser.add_argument("--test", action="store_true", help="Run one consultation without writing config")
    parser.add_argument("--provider", choices=["claude", "openai", "gemini", "ollama"], default="claude")
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    parser.add_argument("--key", type=str, default=None, help="API key (default: from env)")
    parser.add_argument("--stats-file", type=str, default=None, help="Path to training_stats.csv")
    parser.add_argument("--lookback", type=int, default=500, help="Episodes to look back")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    stats_file = args.stats_file or os.path.join(base_dir, "training_stats.csv")
    config_path = os.path.join(base_dir, "config_ai.json")

    # Also log to console in test mode
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(console_handler)

    supervisor = AISupervisor(
        stats_file=stats_file,
        config_ai_path=config_path,
        provider=args.provider,
        api_key=args.key,
        model_name=args.model,
        lookback_episodes=args.lookback,
    )

    rows = supervisor._collect_stats()
    if not rows:
        print(f"No data in {stats_file}")
        sys.exit(1)

    stats = supervisor._aggregate_stats(rows)
    current_values = supervisor._get_current_values(rows)
    prompt = supervisor._build_prompt(stats, current_values, episode_num=int(rows[-1].get("Episode", 0)))

    print("\n=== SYSTEM PROMPT ===")
    print(prompt["system"])
    print("\n=== USER PROMPT ===")
    print(prompt["user"])

    if args.test:
        print("\n=== CALLING LLM ===")
        response = supervisor._call_llm(prompt, current_values)
        if response:
            print("\n=== RAW RESPONSE ===")
            print(response)
            result = supervisor._parse_response(response)
            if result:
                print("\n=== PARSED RESULT ===")
                print(json.dumps(result, indent=2))
            else:
                print("\nFailed to parse response.")
        else:
            print("\nNo response from LLM.")
    else:
        print("\n(Use --test to actually call the LLM)")


if __name__ == "__main__":
    main()
