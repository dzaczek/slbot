#!/usr/bin/env python3
"""
Karpathy-style self-improvement loop for SlitherBot.

Autonomous experiment runner:
  1. Generate mutation(s) to reward config
  2. Apply to styles.py (via git worktree for isolation)
  3. Run training for N episodes
  4. Evaluate against baseline
  5. Keep best, revert rest
  6. Log everything, repeat forever

Supports parallel agents: each runs in its own git worktree
with an independent mutation.

Usage:
    # Sequential (1 experiment at a time)
    python karpathy_mod_runner.py --budget 500

    # Parallel (4 independent mutations, best wins)
    python karpathy_mod_runner.py --budget 500 --parallel 4

    # Target specific stage
    python karpathy_mod_runner.py --budget 500 --parallel 4 --stage 5

    # Dry run (show mutations without executing)
    python karpathy_mod_runner.py --dry-run --parallel 4
"""

import argparse
import copy
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from karpathy_mod_mutator import Mutator, apply_mutation_to_styles, styles_dict_to_python
from karpathy_mod_evaluator import (
    read_csv_tail, compute_metrics, compare_experiments, count_csv_rows
)

# === Constants ===
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
STYLES_FILE = os.path.join(PROJECT_ROOT, 'styles.py')
CSV_FILE = os.path.join(PROJECT_ROOT, 'training_stats.csv')
LOG_FILE = os.path.join(PROJECT_ROOT, 'karpathy_mod_results.tsv')
STATE_FILE = os.path.join(PROJECT_ROOT, 'karpathy_mod_state.json')

# Trainer command template
TRAINER_CMD = [
    sys.executable, 'trainer.py',
    '--resume',
    '--num_agents', '1',
]


class ExperimentWorker:
    """Manages a single experiment in a git worktree."""

    def __init__(self, experiment_id: str, mutation: Dict,
                 worktree_base: str, budget: int, trainer_args: List[str]):
        self.experiment_id = experiment_id
        self.mutation = mutation
        self.budget = budget
        self.trainer_args = trainer_args
        self.worktree_dir = os.path.join(worktree_base, f'exp_{experiment_id}')
        self.branch_name = f'karpathy/exp-{experiment_id}'
        self.process: Optional[subprocess.Popen] = None
        self.csv_path = ''
        self.start_episode = 0
        self.status = 'pending'  # pending, running, done, failed

    def setup_worktree(self) -> bool:
        """Create a git worktree with the mutated styles.py."""
        try:
            # Create branch from current HEAD
            subprocess.run(
                ['git', 'branch', self.branch_name, 'HEAD'],
                cwd=PROJECT_ROOT, capture_output=True, check=True
            )

            # Create worktree
            subprocess.run(
                ['git', 'worktree', 'add', self.worktree_dir, self.branch_name],
                cwd=PROJECT_ROOT, capture_output=True, check=True
            )

            # Apply mutation to worktree's styles.py
            from styles import STYLES
            mutated_styles = apply_mutation_to_styles(STYLES, self.mutation)
            styles_code = styles_dict_to_python(mutated_styles)

            styles_path = os.path.join(self.worktree_dir, 'styles.py')
            with open(styles_path, 'w') as f:
                f.write(styles_code)

            # Set up CSV path in worktree
            self.csv_path = os.path.join(self.worktree_dir, 'training_stats.csv')

            # Copy existing CSV so trainer can --resume properly
            if os.path.exists(CSV_FILE):
                shutil.copy2(CSV_FILE, self.csv_path)

            # Copy model checkpoint if exists
            # Trainer uses checkpoint.pth (--resume) or best model from backup_models
            checkpoint_src = os.path.join(PROJECT_ROOT, 'checkpoint.pth')
            checkpoint_dst = os.path.join(self.worktree_dir, 'checkpoint.pth')
            if os.path.exists(checkpoint_src):
                shutil.copy2(checkpoint_src, checkpoint_dst)
                print(f"  [+] Copied checkpoint.pth to worktree")
            else:
                # Fallback: find best model in backup_models/
                backup_dir = os.path.join(PROJECT_ROOT, 'backup_models')
                if os.path.isdir(backup_dir):
                    backups = sorted(
                        [f for f in os.listdir(backup_dir) if f.endswith('.pth')],
                        key=lambda f: os.path.getmtime(os.path.join(backup_dir, f)),
                        reverse=True
                    )
                    if backups:
                        best_src = os.path.join(backup_dir, backups[0])
                        shutil.copy2(best_src, checkpoint_dst)
                        print(f"  [+] Copied best backup model {backups[0]} to worktree")

            # Record starting episode count
            self.start_episode = count_csv_rows(self.csv_path)

            return True
        except subprocess.CalledProcessError as e:
            print(f"  [!] Worktree setup failed for {self.experiment_id}: {e.stderr}")
            self.status = 'failed'
            return False

    def start_training(self):
        """Start trainer subprocess in the worktree."""
        cmd = [sys.executable, 'trainer.py'] + self.trainer_args
        env = os.environ.copy()
        env['KARPATHY_EXPERIMENT_ID'] = self.experiment_id

        # Ensure logs dir exists in worktree
        logs_dir = os.path.join(self.worktree_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)

        log_file = open(os.path.join(logs_dir, f'karpathy_{self.experiment_id}.log'), 'w')

        self.process = subprocess.Popen(
            cmd,
            cwd=self.worktree_dir,
            stdin=subprocess.DEVNULL,  # No interactive menus
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid,  # New process group for clean kill
        )
        self.status = 'running'
        print(f"  [>] Started experiment {self.experiment_id} (PID {self.process.pid})")

    def check_progress(self) -> int:
        """Return number of new episodes completed."""
        if not os.path.exists(self.csv_path):
            return 0
        current = count_csv_rows(self.csv_path)
        return max(0, current - self.start_episode)

    def stop_training(self):
        """Gracefully stop the trainer process."""
        if self.process and self.process.poll() is None:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=30)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    self.process.wait(timeout=5)
                except Exception:
                    pass
        self.status = 'done'

    def get_metrics(self, warmup: int = 50):
        """Read experiment results from CSV."""
        rows = read_csv_tail(self.csv_path, self.budget)
        # Only use rows from THIS experiment (after start_episode)
        if len(rows) > self.budget:
            rows = rows[-self.budget:]
        return compute_metrics(rows, warmup=warmup)

    def cleanup(self, keep: bool = False):
        """Remove worktree and branch."""
        self.stop_training()

        try:
            subprocess.run(
                ['git', 'worktree', 'remove', '--force', self.worktree_dir],
                cwd=PROJECT_ROOT, capture_output=True
            )
        except Exception:
            # Force remove if git worktree remove fails
            if os.path.exists(self.worktree_dir):
                shutil.rmtree(self.worktree_dir, ignore_errors=True)

        if not keep:
            try:
                subprocess.run(
                    ['git', 'branch', '-D', self.branch_name],
                    cwd=PROJECT_ROOT, capture_output=True
                )
            except Exception:
                pass


class KarpathyRunner:
    """Main self-improvement loop orchestrator."""

    def __init__(self, args):
        self.budget = args.budget
        self.parallel = args.parallel
        self.target_stage = args.stage
        self.warmup = args.warmup
        self.max_time = args.max_time * 3600 if args.max_time else 0  # hours → seconds
        self.global_start = time.time()
        self.trainer_args = self._build_trainer_args(args)
        self.mutator = Mutator()
        self.dry_run = args.dry_run
        self.max_rounds = args.max_rounds
        self.worktree_base = os.path.join(PROJECT_ROOT, '.karpathy_worktrees')
        self.round_num = 0
        self.total_kept = 0
        self.total_discarded = 0

        # Load state if exists
        self._load_state()

    def _build_trainer_args(self, args) -> List[str]:
        """Build trainer command line args."""
        trainer_args = [
            '--resume',
            '--num_agents', '1',
            '--style-name', 'Standard',  # Skip interactive style menu
            '--model-path', 'checkpoint.pth',  # Skip interactive model menu
        ]
        if args.url:
            trainer_args += ['--url', args.url]
        if args.backend:
            trainer_args += ['--backend', args.backend]
        return trainer_args

    def _load_state(self):
        """Load persistent state (round counter, history)."""
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
                self.round_num = state.get('round_num', 0)
                self.total_kept = state.get('total_kept', 0)
                self.total_discarded = state.get('total_discarded', 0)

    def _save_state(self):
        """Save persistent state."""
        state = {
            'round_num': self.round_num,
            'total_kept': self.total_kept,
            'total_discarded': self.total_discarded,
            'last_update': datetime.now().isoformat(),
        }
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)

    def run(self):
        """Main loop — runs forever until interrupted."""
        print("=" * 70)
        print("  KARPATHY MOD — Self-Improvement Loop for SlitherBot")
        print(f"  Budget: {self.budget} episodes/experiment")
        print(f"  Parallel: {self.parallel} agents")
        print(f"  Target stage: {self.target_stage or 'auto'}")
        print(f"  Warmup: {self.warmup} episodes")
        print(f"  Max rounds: {self.max_rounds or 'unlimited'}")
        print(f"  Max time: {self.max_time/3600:.1f}h" if self.max_time else "  Max time: unlimited")
        print("=" * 70)

        os.makedirs(self.worktree_base, exist_ok=True)

        try:
            while True:
                self.round_num += 1
                if self.max_rounds and self.round_num > self.max_rounds:
                    print(f"\n[*] Reached max rounds ({self.max_rounds}). Stopping.")
                    break

                if self.max_time and (time.time() - self.global_start) > self.max_time:
                    elapsed_h = (time.time() - self.global_start) / 3600
                    print(f"\n[*] Reached max time ({elapsed_h:.1f}h). Stopping.")
                    break

                print(f"\n{'='*60}")
                print(f"  ROUND {self.round_num}  (kept={self.total_kept}, "
                      f"discarded={self.total_discarded})")
                print(f"{'='*60}")

                result = self._run_round()
                self._save_state()

                if result == 'error':
                    print("  [!] Round failed. Sleeping 60s before retry...")
                    time.sleep(60)

        except KeyboardInterrupt:
            print("\n\n[*] Interrupted by user. Cleaning up...")
            self._cleanup_all_worktrees()
            self._save_state()
            print("[*] Done. State saved.")

    def _run_round(self) -> str:
        """Execute one round of experiments. Returns 'keep'/'discard'/'error'."""
        # 1. Load current styles as baseline
        from importlib import reload
        import styles as styles_module
        reload(styles_module)
        from styles import STYLES
        curriculum = STYLES["Standard (Curriculum)"]

        # 2. Collect baseline metrics from recent training
        print("\n  [1/5] Collecting baseline metrics...")
        baseline_rows = read_csv_tail(CSV_FILE, self.budget)
        if len(baseline_rows) < self.warmup + 50:
            print(f"  [!] Not enough baseline data ({len(baseline_rows)} rows). "
                  f"Need at least {self.warmup + 50}. Train more first.")
            return 'error'

        baseline = compute_metrics(baseline_rows, warmup=self.warmup)
        print(f"  Baseline: avg_steps={baseline.avg_steps:.0f}, "
              f"score={baseline.score():.1f}, "
              f"snake_death={baseline.snake_death_rate:.1%}")

        # 3. Generate mutations
        print(f"\n  [2/5] Generating {self.parallel} mutation(s)...")
        mutations = self.mutator.generate_batch(
            curriculum, count=self.parallel, target_stage=self.target_stage
        )
        for i, m in enumerate(mutations):
            print(f"    [{i}] {m['description']}")

        if self.dry_run:
            print("\n  [DRY RUN] Would execute above mutations. Exiting.")
            raise KeyboardInterrupt("dry-run")

        # 4. Setup workers
        print(f"\n  [3/5] Setting up {self.parallel} worktree(s)...")
        workers = []
        for mutation in mutations:
            w = ExperimentWorker(
                experiment_id=mutation['experiment_id'],
                mutation=mutation,
                worktree_base=self.worktree_base,
                budget=self.budget,
                trainer_args=self.trainer_args,
            )
            if w.setup_worktree():
                workers.append(w)
            else:
                w.cleanup()

        if not workers:
            print("  [!] All worktree setups failed.")
            return 'error'

        # 5. Start training in all worktrees
        print(f"\n  [4/5] Starting {len(workers)} trainer(s)...")
        for w in workers:
            w.start_training()
            time.sleep(2)  # Stagger launches to avoid port conflicts

        # 6. Monitor progress
        print(f"\n  [5/5] Training {self.budget} episodes per agent...")
        try:
            self._monitor_progress(workers)
        except KeyboardInterrupt:
            print("\n  [!] Interrupted during training. Stopping workers...")
            for w in workers:
                w.stop_training()
            for w in workers:
                w.cleanup()
            raise

        # 7. Evaluate all experiments
        print("\n  Evaluating experiments...")
        results = []
        for w in workers:
            w.stop_training()
            metrics = w.get_metrics(warmup=self.warmup)
            comparison = compare_experiments(baseline, metrics)
            mutation = next(m for m in mutations if m['experiment_id'] == w.experiment_id)

            results.append({
                'worker': w,
                'mutation': mutation,
                'metrics': metrics,
                'comparison': comparison,
            })

            status_icon = {'keep': '+', 'discard': 'x', 'inconclusive': '?'}[comparison['decision']]
            print(f"    [{status_icon}] {w.experiment_id}: {comparison['details']}")

        # 8. Pick best
        best = self._pick_best(results, baseline)

        if best:
            # Apply best mutation to main repo
            print(f"\n  >>> KEEPING experiment {best['worker'].experiment_id}")
            print(f"      {best['mutation']['description']}")
            self._apply_to_main(best['mutation'], STYLES)
            self.total_kept += 1
            self._log_result(best, 'keep')
        else:
            print("\n  >>> No improvement found. Discarding all.")
            self.total_discarded += len(workers)
            for r in results:
                self._log_result(r, 'discard')

        # 9. Cleanup all worktrees
        for w in workers:
            w.cleanup()

        return 'keep' if best else 'discard'

    def _monitor_progress(self, workers: List[ExperimentWorker]):
        """Poll workers until all reach budget or crash.

        Safety features:
        - Per-round timeout: budget * 30s (reasonable upper bound)
        - Stall detection: if no new episodes for 5 min, kill stalled workers
        - Dead process detection: mark crashed workers immediately
        """
        start_time = time.time()
        timeout = self.budget * 30  # ~30s per episode max (was 120 — way too generous)
        stall_timeout = 300  # 5 min without progress = stalled

        # Track last known progress per worker for stall detection
        last_progress = {w.experiment_id: 0 for w in workers}
        last_progress_time = {w.experiment_id: time.time() for w in workers}

        while True:
            all_done = True
            status_parts = []

            for w in workers:
                if w.status == 'done':
                    progress = w.check_progress()
                    status_parts.append(f"{w.experiment_id[:6]}:DONE({progress})")
                    continue

                if w.process and w.process.poll() is not None:
                    # Process exited (crashed or finished)
                    progress = w.check_progress()
                    exit_code = w.process.returncode
                    status_parts.append(f"{w.experiment_id[:6]}:EXIT({exit_code},ep={progress})")
                    w.status = 'done'
                    continue

                progress = w.check_progress()

                # Stall detection: no new episodes for stall_timeout
                if progress > last_progress[w.experiment_id]:
                    last_progress[w.experiment_id] = progress
                    last_progress_time[w.experiment_id] = time.time()
                elif time.time() - last_progress_time[w.experiment_id] > stall_timeout:
                    stall_mins = (time.time() - last_progress_time[w.experiment_id]) / 60
                    print(f"\n  [!] Worker {w.experiment_id[:6]} stalled for {stall_mins:.0f}min "
                          f"at {progress}/{self.budget} episodes. Killing.")
                    w.stop_training()
                    status_parts.append(f"{w.experiment_id[:6]}:STALLED({progress})")
                    continue

                if progress < self.budget:
                    all_done = False
                status_parts.append(f"{w.experiment_id[:6]}:{progress}/{self.budget}")

            elapsed = time.time() - start_time
            elapsed_min = elapsed / 60
            print(f"\r  [{elapsed_min:.0f}m] {' | '.join(status_parts)}", end='', flush=True)

            if all_done:
                print()
                break

            if elapsed > timeout:
                print(f"\n  [!] Round timeout after {elapsed_min:.0f}min. Killing all workers.")
                for w in workers:
                    if w.status != 'done':
                        w.stop_training()
                break

            time.sleep(15)

    def _pick_best(self, results: List[Dict], baseline) -> Optional[Dict]:
        """Pick the best experiment from results."""
        candidates = [r for r in results if r['comparison']['decision'] == 'keep']

        if not candidates:
            # Also consider 'inconclusive' if improvement is positive
            candidates = [r for r in results
                          if r['comparison']['improvement_pct'] > 0.01]

        if not candidates:
            return None

        # Sort by improvement
        candidates.sort(key=lambda r: r['comparison']['improvement_pct'], reverse=True)
        return candidates[0]

    def _apply_to_main(self, mutation: Dict, current_styles: Dict):
        """Apply winning mutation to the main styles.py and git commit."""
        mutated = apply_mutation_to_styles(current_styles, mutation)
        code = styles_dict_to_python(mutated)

        with open(STYLES_FILE, 'w') as f:
            f.write(code)

        # Git commit
        desc = mutation['description'][:200]
        commit_msg = (
            f"karpathy: {desc}\n\n"
            f"Round {self.round_num}, experiment {mutation['experiment_id']}\n"
            f"Strategy: {mutation['strategy']}\n"
        )
        try:
            subprocess.run(['git', 'add', 'styles.py'],
                           cwd=PROJECT_ROOT, capture_output=True, check=True)
            subprocess.run(['git', 'commit', '-m', commit_msg],
                           cwd=PROJECT_ROOT, capture_output=True, check=True)
            print(f"  Committed to git: {desc[:80]}")
        except subprocess.CalledProcessError as e:
            print(f"  [!] Git commit failed: {e.stderr}")

    def _log_result(self, result: Dict, decision: str):
        """Append result to TSV log."""
        write_header = not os.path.exists(LOG_FILE)

        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            if write_header:
                writer.writerow([
                    'timestamp', 'round', 'experiment_id', 'strategy',
                    'stage', 'decision', 'improvement_pct',
                    'baseline_score', 'experiment_score',
                    'avg_steps', 'avg_food', 'peak_length',
                    'snake_death_rate', 'description'
                ])

            m = result['metrics']
            c = result['comparison']
            mut = result['mutation']

            writer.writerow([
                datetime.now().isoformat(),
                self.round_num,
                mut['experiment_id'],
                mut['strategy'],
                mut['target_stage'],
                decision,
                f"{c['improvement_pct']:.4f}",
                f"{c['baseline_score']:.1f}",
                f"{c['experiment_score']:.1f}",
                f"{m.avg_steps:.0f}",
                f"{m.avg_food:.0f}",
                f"{m.avg_peak_length:.0f}",
                f"{m.snake_death_rate:.3f}",
                mut['description'][:300],
            ])

    def _cleanup_all_worktrees(self):
        """Emergency cleanup of all worktrees."""
        if os.path.exists(self.worktree_base):
            # Kill any running trainers
            try:
                subprocess.run(['pkill', '-f', 'KARPATHY_EXPERIMENT_ID'],
                               capture_output=True)
            except Exception:
                pass

            # Remove worktrees
            try:
                result = subprocess.run(
                    ['git', 'worktree', 'list', '--porcelain'],
                    cwd=PROJECT_ROOT, capture_output=True, text=True
                )
                for line in result.stdout.split('\n'):
                    if line.startswith('worktree ') and '.karpathy_worktrees' in line:
                        wt_path = line.split(' ', 1)[1]
                        subprocess.run(
                            ['git', 'worktree', 'remove', '--force', wt_path],
                            cwd=PROJECT_ROOT, capture_output=True
                        )
            except Exception:
                pass

            shutil.rmtree(self.worktree_base, ignore_errors=True)

        # Clean up karpathy branches
        try:
            result = subprocess.run(
                ['git', 'branch', '--list', 'karpathy/*'],
                cwd=PROJECT_ROOT, capture_output=True, text=True
            )
            for line in result.stdout.strip().split('\n'):
                branch = line.strip()
                if branch:
                    subprocess.run(
                        ['git', 'branch', '-D', branch],
                        cwd=PROJECT_ROOT, capture_output=True
                    )
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description='Karpathy-style self-improvement loop for SlitherBot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --budget 500                    # Sequential, 500 eps/experiment
  %(prog)s --budget 500 --parallel 4       # 4 parallel agents
  %(prog)s --budget 300 --stage 5          # Focus on stage 5
  %(prog)s --dry-run --parallel 4          # Show mutations without running
  %(prog)s --budget 500 --max-rounds 10    # Stop after 10 rounds
        """
    )
    parser.add_argument('--budget', type=int, default=500,
                        help='Episodes per experiment (default: 500)')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Number of parallel experiments (default: 1)')
    parser.add_argument('--stage', type=int, default=0,
                        help='Target stage to mutate (0=auto, default: 0)')
    parser.add_argument('--warmup', type=int, default=50,
                        help='Warmup episodes to skip in evaluation (default: 50)')
    parser.add_argument('--max-rounds', type=int, default=0,
                        help='Max rounds before stopping (0=unlimited)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show mutations without executing')
    parser.add_argument('--url', type=str, default='http://slither.io',
                        help='Game URL')
    parser.add_argument('--backend', type=str, default='selenium',
                        help='Browser backend')
    parser.add_argument('--max-time', type=float, default=0,
                        help='Max total runtime in hours (0=unlimited, default: 0)')
    parser.add_argument('--cleanup', action='store_true',
                        help='Clean up leftover worktrees and exit')

    args = parser.parse_args()

    if args.cleanup:
        print("Cleaning up worktrees...")
        runner = KarpathyRunner(args)
        runner._cleanup_all_worktrees()
        print("Done.")
        return

    runner = KarpathyRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
