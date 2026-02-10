import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import os
import sys
import time
import multiprocessing as mp
import argparse
from collections import deque
import logging

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

# Create handlers
c_handler = logging.StreamHandler()
f_handler_app = logging.FileHandler('logs/app.log')
f_handler_train = logging.FileHandler('logs/train.log')

c_handler.setLevel(logging.INFO)
f_handler_app.setLevel(logging.INFO)
f_handler_train.setLevel(logging.INFO)

# Create formatters and add it to handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
c_handler.setFormatter(formatter)
f_handler_app.setFormatter(formatter)
f_handler_train.setFormatter(formatter)

# Add handlers to the logger
if not logger.hasHandlers():
    logger.addHandler(c_handler)
    logger.addHandler(f_handler_app)
    logger.addHandler(f_handler_train)


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
        self.episode_food_history = deque(maxlen=100)
        self.episode_steps_history = deque(maxlen=100)
        self.episode_food_ratio_history = deque(maxlen=100)

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

    def record_episode(self, food_eaten, steps):
        """Record episode metrics for promotion check."""
        self.episode_food_history.append(food_eaten)
        self.episode_steps_history.append(steps)
        ratio = food_eaten / max(steps, 1)
        self.episode_food_ratio_history.append(ratio)

    def check_promotion(self):
        """Check if we should advance to the next stage. Returns True if promoted."""
        if self.mode == "static":
            return False

        cfg = self.style_config["stages"][self.current_stage]
        metric = cfg["promote_metric"]
        threshold = cfg["promote_threshold"]
        window = cfg["promote_window"]

        if metric is None:
            return False  # Final stage

        if metric == "food_per_step":
            if len(self.episode_food_ratio_history) < window:
                return False
            recent = list(self.episode_food_ratio_history)[-window:]
            avg = sum(recent) / len(recent)
            if avg >= threshold:
                self._promote()
                return True

        elif metric == "avg_food":
            if len(self.episode_food_history) < window:
                return False
            recent = list(self.episode_food_history)[-window:]
            avg = sum(recent) / len(recent)
            if avg >= threshold:
                self._promote()
                return True

        elif metric == "avg_steps":
            if len(self.episode_steps_history) < window:
                return False
            recent = list(self.episode_steps_history)[-window:]
            avg = sum(recent) / len(recent)
            if avg >= threshold:
                self._promote()
                return True

        return False

    def _promote(self):
        stages = self.style_config["stages"]
        old_name = stages[self.current_stage]["name"]
        self.current_stage = min(self.current_stage + 1, max(stages.keys()))
        new_name = stages[self.current_stage]["name"]
        print(f"\n{'='*60}")
        print(f"  STAGE UP! {old_name} -> {new_name} (Stage {self.current_stage})")
        print(f"{'='*60}\n")

    def get_state(self):
        """Serialize for checkpoint."""
        return {
            "stage": self.current_stage,
            "food_history": list(self.episode_food_history),
            "steps_history": list(self.episode_steps_history),
            "food_ratio_history": list(self.episode_food_ratio_history),
        }

    def load_state(self, state):
        """Restore from checkpoint."""
        if state:
            self.current_stage = state.get("stage", 1)
            self.episode_food_history = deque(state.get("food_history", []), maxlen=100)
            self.episode_steps_history = deque(state.get("steps_history", []), maxlen=100)
            self.episode_food_ratio_history = deque(state.get("food_ratio_history", []), maxlen=100)


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

class VecFrameStack:
    """
    Wraps SubprocVecEnv to stack frames.
    """
    def __init__(self, venv, k):
        self.venv = venv
        self.k = k
        self.num_agents = venv.num_agents
        self.frames = [deque(maxlen=k) for _ in range(self.num_agents)]

        # Initialize frames with initial reset
        # We don't call reset here to avoid double reset, or we do?
        # Standard wrappers often wait for explicit reset.
        pass

    def reset(self):
        obs = self.venv.reset()
        stacked_obs = []
        for i, o in enumerate(obs):
            self.frames[i].clear()
            for _ in range(self.k):
                self.frames[i].append(o)
            stacked_obs.append(np.concatenate(list(self.frames[i]), axis=0))
        return stacked_obs

    def step(self, actions):
        obs, rews, dones, infos = self.venv.step(actions)
        stacked_obs = []

        for i in range(self.num_agents):
            # If done, the 'obs' is the first frame of the new episode.
            # The terminal observation (last frame of old episode) is in info.
            if dones[i]:
                # 1. Handle terminal observation stacking
                term_frame = infos[i]['terminal_observation']
                # Copy current deque to create terminal stack
                term_stack_deque = self.frames[i].copy()
                term_stack_deque.append(term_frame) # Now has k frames ending with terminal
                infos[i]['terminal_observation'] = np.concatenate(list(term_stack_deque), axis=0)

                # 2. Handle new episode start
                # Clear deque and fill with new frame
                self.frames[i].clear()
                for _ in range(self.k):
                    self.frames[i].append(obs[i])
            else:
                self.frames[i].append(obs[i])

            # Create stacked observation
            stacked_obs.append(np.concatenate(list(self.frames[i]), axis=0))

        return stacked_obs, rews, dones, infos

    def reset_agent(self, i):
        """Force reset specific agent."""
        return self.reset_one(i)

    def reset_one(self, i):
        obs = self.venv.reset_one(i)
        self.frames[i].clear()
        for _ in range(self.k):
            self.frames[i].append(obs)
        return np.concatenate(list(self.frames[i]), axis=0)

    def close(self):
        self.venv.close()

    def set_stage(self, stage_config):
        self.venv.set_stage(stage_config)

def worker(remote, parent_remote, worker_id, headless, nickname_prefix, matrix_size, frame_skip, view_plus=False, base_url="http://slither.io"):
    parent_remote.close()
    
    picard_names = [
        "Picard", "Riker", "Data", "Worf", "Troi", "LaForge",
        "Crusher", "Q", "Seven", "Raffi", "Rios", "Jurati"
    ]
    chosen_name = f"{random.choice(picard_names)}_{worker_id}"
    
    try:
        env = SlitherEnv(
            headless=headless,
            nickname=chosen_name,
            matrix_size=matrix_size,
            view_plus=view_plus,
            base_url=base_url,
            frame_skip=frame_skip,
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
        print(f"Worker {worker_id} crashed: {e}")
    finally:
        remote.close()

class SubprocVecEnv:
    def __init__(self, num_agents, matrix_size, frame_skip, view_first=False, view_plus=False, nickname="dzaczekAI", base_url="http://slither.io"):
        self.num_agents = num_agents
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_agents)])
        self.ps = []

        for i in range(num_agents):
            is_headless = not (view_first and i == 0)
            # Enable view_plus only for the first agent when view mode is active
            agent_view_plus = view_plus and (i == 0) and not is_headless
            p = mp.Process(
                target=worker,
                args=(self.work_remotes[i], self.remotes[i], i, is_headless, nickname, matrix_size, frame_skip, agent_view_plus, base_url),
            )
            p.daemon = True
            p.start()
            self.ps.append(p)

        for remote in self.work_remotes:
            remote.close()

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return [remote.recv() for remote in self.remotes]

    def reset_one(self, index):
        self.remotes[index].send(('reset_one', None))
        return self.remotes[index].recv()

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        states, rewards, dones, infos = zip(*results)
        return states, rewards, dones, infos

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def set_stage(self, stage_config):
        """Send curriculum stage config to all workers."""
        for remote in self.remotes:
            remote.send(('set_stage', stage_config))
        for remote in self.remotes:
            remote.recv()  # Wait for ack

def train(args):
    # Select Style and Model
    style_name, model_path = select_style_and_model(args)

    # Load Config
    cfg = Config()
    
    # Override config with args if necessary
    if args.num_agents > 0:
        cfg.env.num_agents = args.num_agents
    
    if args.vision_size:
        cfg.env.resolution = (args.vision_size, args.vision_size)

    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(base_dir, 'checkpoint.pth')
    backup_dir = os.path.join(base_dir, 'backup_models')
    os.makedirs(backup_dir, exist_ok=True)
    stats_file = os.path.join(base_dir, 'training_stats.csv')

    # Initialize Curriculum/Style Manager
    curriculum = CurriculumManager(style_name=style_name, start_stage=args.stage if args.stage > 0 else 1)

    logger.info(f"Configuration:")
    logger.info(f"  Style: {style_name}")
    logger.info(f"  Mode: {curriculum.mode}")
    logger.info(f"  Agents: {cfg.env.num_agents}")
    logger.info(f"  Frame Stack: {cfg.env.frame_stack}")
    logger.info(f"  Resolution: {cfg.env.resolution}")
    logger.info(f"  Model: {cfg.model.architecture} ({cfg.model.activation})")
    logger.info(f"  LR: {cfg.opt.lr}")
    logger.info(f"  PER: {cfg.buffer.prioritized}")

    # Initialize Env
    raw_env = SubprocVecEnv(
        num_agents=cfg.env.num_agents,
        matrix_size=cfg.env.resolution[0],
        frame_skip=cfg.env.frame_skip,
        view_first=args.view or args.view_plus,
        view_plus=args.view_plus,
        nickname="AI_Opt",
        base_url=args.url
    )
    env = VecFrameStack(raw_env, k=cfg.env.frame_stack)

    # Initialize Agent
    agent = DDQNAgent(cfg)

    # Resume / Load Model
    start_episode = 0
    load_path = model_path if model_path else (checkpoint_path if args.resume else None)

    if load_path and os.path.exists(load_path):
        start_episode, _, supervisor_state = agent.load_checkpoint(load_path)

        # Fix LR=0 from old checkpoints: restore to min_lr if frozen
        current_lr = agent.optimizer.param_groups[0]['lr']
        if current_lr < cfg.opt.scheduler_min_lr:
            for pg in agent.optimizer.param_groups:
                pg['lr'] = cfg.opt.lr
            logger.info(f"  LR was {current_lr:.8f} (frozen). Reset to {cfg.opt.lr}")

        # Restore curriculum state only if we are in curriculum mode
        if curriculum.mode == 'curriculum' and supervisor_state:
            curriculum.load_state(supervisor_state)

        logger.info(f"Resumed from episode {start_episode}")
        if curriculum.mode == 'curriculum':
            logger.info(f"  Stage: {curriculum.current_stage} ({curriculum.get_config()['name']})")
    elif load_path:
        logger.warning(f"Model path {load_path} not found. Starting from scratch.")

    # Apply current curriculum stage/style to environments
    stage_cfg = curriculum.get_config()
    env.set_stage(stage_cfg)
    max_steps_per_episode = curriculum.get_max_steps()
    logger.info(f"  Curriculum Stage: {curriculum.current_stage} ({stage_cfg['name']})")
    logger.info(f"  Max Steps: {max_steps_per_episode}")
    super_pattern = SuperPatternOptimizer(stage_cfg, cfg, logger)

    # Initialize stats file
    if not os.path.exists(stats_file) or not args.resume:
        with open(stats_file, 'w') as f:
            f.write("Episode,Steps,Reward,Epsilon,Loss,Beta,LR,Cause,Stage,Food\n")

    # LR Scheduler (Linear Warmup)
    warmup_steps = 10000
    target_lr = cfg.opt.lr

    # Autonomy / Stabilization Vars
    reward_window = deque(maxlen=100)
    best_avg_reward = -float('inf')
    episodes_since_improvement = 0

    # Metrics tracking
    total_steps = agent.steps_done
    episode_rewards = [0] * cfg.env.num_agents
    episode_steps = [0] * cfg.env.num_agents
    episode_food = [0] * cfg.env.num_agents

    # Death Counters
    death_stats = {"Wall": 0, "SnakeCollision": 0, "Unknown": 0, "MaxSteps": 0}

    # Initial Reset
    states = env.reset()

    def finalize_episode(agent_index, terminal_state, cause, force_done_flag):
        nonlocal start_episode, max_steps_per_episode, best_avg_reward, episodes_since_improvement
        total_steps_local = episode_steps[agent_index]
        total_reward = episode_rewards[agent_index]
        food_eaten = episode_food[agent_index]

        # Store transition
        agent.remember(states[agent_index], actions[agent_index], rewards[agent_index], terminal_state, True)

        # Log
        start_episode += 1
        eps = agent.get_epsilon()
        loss_val = getattr(train, '_last_loss', 0) or 0

        current_beta = min(1.0, agent.memory.beta_start + agent.memory.frame * (1.0 - agent.memory.beta_start) / agent.memory.beta_frames)
        lr = agent.optimizer.param_groups[0]['lr']

        if force_done_flag:
            cause_label = "MaxSteps"
        else:
            cause_label = cause or "Unknown"

        # Update death stats
        if cause_label in death_stats:
            death_stats[cause_label] += 1
        else:
            death_stats["Unknown"] = death_stats.get("Unknown", 0) + 1

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
                   f"Eps: {eps:.3f} | L: {loss_val:.4f} | {cause_label} | {pos_str}{wall_str}{enemy_str}")

        logger.info(log_msg)

        # Print stats occasionally
        if start_episode % 10 == 0:
            logger.info(f"Death Stats: {death_stats}")

        with open(stats_file, 'a') as f:
            f.write(f"{start_episode},{total_steps_local},{total_reward:.2f},{eps:.4f},{loss_val:.4f},{current_beta:.2f},{lr:.6f},{cause_label},{curriculum.current_stage},{food_eaten}\n")

        # Autonomy Logic (Scheduler & Watchdog)
        reward_window.append(total_reward)
        if len(reward_window) >= 20:
            avg_reward = sum(reward_window) / len(reward_window)

            # Scheduler step
            agent.step_scheduler(avg_reward)

            # Watchdog for stagnation
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                episodes_since_improvement = 0

                # Save PROMISING model backup
                if avg_reward > 0: # Only backup if positive reward
                    backup_name = f"best_model_ep{start_episode}_rw{int(avg_reward)}.pth"
                    backup_path = os.path.join(backup_dir, backup_name)
                    agent.save_checkpoint(backup_path, start_episode, max_steps_per_episode, curriculum.get_state())
                    logger.info(f"  >> New Best Avg Reward: {avg_reward:.2f}. Saved backup: {backup_name}")

            else:
                episodes_since_improvement += 1

            if episodes_since_improvement > cfg.opt.adaptive_eps_patience:
                logger.info(f"\n[Autonomy] Stagnation detected ({episodes_since_improvement} eps). Gentle exploration boost.")
                agent.boost_exploration(target_eps=0.3)
                episodes_since_improvement = 0

        super_pattern.record_episode(cause_label, food_eaten, total_steps_local, total_reward)
        updated_stage_cfg = super_pattern.maybe_update()
        if updated_stage_cfg:
            env.set_stage(updated_stage_cfg)

        # Track metrics for curriculum promotion
        curriculum.record_episode(food_eaten, total_steps_local)

        # Check for stage promotion
        if curriculum.check_promotion():
            stage_cfg = curriculum.get_config()
            max_steps_per_episode = curriculum.get_max_steps()
            env.set_stage(stage_cfg)
            super_pattern.reset_stage(stage_cfg)
            # Save checkpoint on promotion
            agent.save_checkpoint(checkpoint_path, start_episode, max_steps_per_episode, curriculum.get_state())

        if start_episode % cfg.opt.checkpoint_every == 0:
            agent.save_checkpoint(checkpoint_path, start_episode, max_steps_per_episode, curriculum.get_state())

        episode_rewards[agent_index] = 0
        episode_steps[agent_index] = 0
        episode_food[agent_index] = 0

    try:
        while start_episode < cfg.opt.max_episodes:
            # LR Warmup
            if total_steps < warmup_steps:
                lr_scale = total_steps / warmup_steps
                for param_group in agent.optimizer.param_groups:
                    param_group['lr'] = target_lr * lr_scale

            # Select actions (increment steps_done once per batch, not per agent)
            actions = [agent.select_action(s) for s in states]
            agent.steps_done += 1

            # Step
            next_states, rewards, dones, infos = env.step(actions)

            loss = None

            for i in range(cfg.env.num_agents):
                total_steps += 1
                episode_rewards[i] += rewards[i]
                episode_steps[i] += 1
                episode_food[i] += infos[i].get('food_eaten', 0)

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
                        cause=infos[i].get('cause', 'Unknown'),
                        force_done_flag=force_done and not dones[i]
                    )
                else:
                    agent.remember(states[i], actions[i], rewards[i], next_states[i], False)

            # Update states
            states = next_states

            # Train
            loss = agent.optimize_model()
            if loss is not None:
                train._last_loss = loss

            # Target Update
            if total_steps % cfg.opt.target_update_freq == 0:
                agent.update_target()
                logger.info(">> Target Network Updated")

    except KeyboardInterrupt:
        print("Interrupted. Saving...")
        agent.save_checkpoint(checkpoint_path, start_episode, max_steps_per_episode, curriculum.get_state())
    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, default=0, help="Override config num_agents")
    parser.add_argument("--view", action="store_true", help="View first agent")
    parser.add_argument("--view-plus", action="store_true", help="View first agent with bot vision overlay grid")
    parser.add_argument("--resume", action="store_true", help="Resume")
    parser.add_argument("--stage", type=int, default=0, help="Force start at specific stage (1-3)")
    parser.add_argument("--style-name", type=str, help="Learning style name (e.g. 'Aggressive')")
    parser.add_argument("--model-path", type=str, help="Path to model checkpoint to load")
    parser.add_argument("--url", type=str, default="http://slither.io", help="Game URL (e.g. http://eslither.io)")
    parser.add_argument("--vision-size", type=int, default=84, help="Vision input size (64, 84, 128, etc.)")
    args = parser.parse_args()

    mp.set_start_method('spawn', force=True)
    train(args)
