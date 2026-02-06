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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CurriculumManager:
    """
    Multi-stage curriculum learning.
    Stage 1: EAT   - Learn to eat food (high food reward, low death penalty)
    Stage 2: SURVIVE - Learn to avoid walls and enemies (higher death penalty)
    Stage 3: GROW  - Eat a lot and survive long (length bonus)
    """
    STAGES = {
        1: {
            "name": "EAT",
            "food_reward": 10.0,
            "food_shaping": 0.01,
            "survival": 0.05,
            "death_wall": -15,  # Increased penalty to prevent suicide-eating
            "death_snake": -15,
            "straight_penalty": 0.0,
            "length_bonus": 0.0,
            "max_steps": 200,
            "promote_metric": "food_per_step",  # Must eat more than 1.5x steps
            "promote_threshold": 0.05,           # Reduced threshold (10 food in 200 steps)
            "promote_window": 50,
        },
        2: {
            "name": "SURVIVE",
            "food_reward": 5.0,
            "food_shaping": 0.005,
            "survival": 0.2,
            "death_wall": -100,
            "death_snake": -20,
            "straight_penalty": 0.05,
            "length_bonus": 0.0,
            "max_steps": 500,
            "promote_metric": "avg_steps",
            "promote_threshold": 150,
            "promote_window": 50,
        },
        3: {
            "name": "GROW",
            "food_reward": 5.0,
            "food_shaping": 0.005,
            "survival": 0.2,
            "death_wall": -100,
            "death_snake": -20,
            "straight_penalty": 0.05,
            "length_bonus": 0.01,
            "max_steps": 99999,
            "promote_metric": None,
            "promote_threshold": None,
            "promote_window": 50,
        },
    }

    def __init__(self, start_stage=1):
        self.current_stage = start_stage
        self.episode_food_history = deque(maxlen=100)
        self.episode_steps_history = deque(maxlen=100)
        self.episode_food_ratio_history = deque(maxlen=100)

    def get_config(self):
        """Return current stage config dict."""
        return self.STAGES[self.current_stage]

    def get_max_steps(self):
        return self.STAGES[self.current_stage]["max_steps"]

    def record_episode(self, food_eaten, steps):
        """Record episode metrics for promotion check."""
        self.episode_food_history.append(food_eaten)
        self.episode_steps_history.append(steps)
        ratio = food_eaten / max(steps, 1)
        self.episode_food_ratio_history.append(ratio)

    def check_promotion(self):
        """Check if we should advance to the next stage. Returns True if promoted."""
        cfg = self.STAGES[self.current_stage]
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
        old_name = self.STAGES[self.current_stage]["name"]
        self.current_stage = min(self.current_stage + 1, max(self.STAGES.keys()))
        new_name = self.STAGES[self.current_stage]["name"]
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
        # This requires SubprocVecEnv to support reset_one, which we haven't implemented yet.
        # For now, we will ignore partial resets or implement them if needed.
        pass

    def close(self):
        self.venv.close()

    def set_stage(self, stage_config):
        self.venv.set_stage(stage_config)

def worker(remote, parent_remote, worker_id, headless, nickname_prefix, matrix_size, view_plus=False):
    parent_remote.close()
    
    picard_names = [
        "Picard", "Riker", "Data", "Worf", "Troi", "LaForge",
        "Crusher", "Q", "Seven", "Raffi", "Rios", "Jurati"
    ]
    chosen_name = f"{random.choice(picard_names)}_{worker_id}"
    
    try:
        env = SlitherEnv(headless=headless, nickname=chosen_name, matrix_size=matrix_size, view_plus=view_plus)

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
    def __init__(self, num_agents, matrix_size, view_first=False, view_plus=False, nickname="dzaczekAI"):
        self.num_agents = num_agents
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_agents)])
        self.ps = []

        for i in range(num_agents):
            is_headless = not (view_first and i == 0)
            # Enable view_plus only for the first agent when view mode is active
            agent_view_plus = view_plus and (i == 0) and not is_headless
            p = mp.Process(target=worker, args=(self.work_remotes[i], self.remotes[i], i, is_headless, nickname, matrix_size, agent_view_plus))
            p.daemon = True
            p.start()
            self.ps.append(p)

        for remote in self.work_remotes:
            remote.close()

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return [remote.recv() for remote in self.remotes]

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
    # Load Config
    cfg = Config()
    
    # Override config with args if necessary
    if args.num_agents > 0:
        cfg.env.num_agents = args.num_agents
    
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(base_dir, 'checkpoint.pth')
    stats_file = os.path.join(base_dir, 'training_stats.csv')

    # Initialize Curriculum
    curriculum = CurriculumManager(start_stage=args.stage if args.stage > 0 else 1)

    print(f"Configuration:")
    print(f"  Agents: {cfg.env.num_agents}")
    print(f"  Frame Stack: {cfg.env.frame_stack}")
    print(f"  Resolution: {cfg.env.resolution}")
    print(f"  Model: {cfg.model.architecture} ({cfg.model.activation})")
    print(f"  LR: {cfg.opt.lr}")
    print(f"  PER: {cfg.buffer.prioritized}")

    # Initialize Env
    raw_env = SubprocVecEnv(
        num_agents=cfg.env.num_agents,
        matrix_size=cfg.env.resolution[0],
        view_first=args.view or args.view_plus,
        view_plus=args.view_plus,
        nickname="AI_Opt"
    )
    env = VecFrameStack(raw_env, k=cfg.env.frame_stack)

    # Initialize Agent
    agent = DDQNAgent(cfg)

    # Resume
    start_episode = 0
    if args.resume and os.path.exists(checkpoint_path):
        start_episode, _, supervisor_state = agent.load_checkpoint(checkpoint_path)
        curriculum.load_state(supervisor_state)
        print(f"Resumed from episode {start_episode}, Stage {curriculum.current_stage} ({curriculum.get_config()['name']})")

    # Apply current curriculum stage to environments
    stage_cfg = curriculum.get_config()
    env.set_stage(stage_cfg)
    max_steps_per_episode = curriculum.get_max_steps()
    print(f"  Curriculum Stage: {curriculum.current_stage} ({stage_cfg['name']})")
    print(f"  Max Steps: {max_steps_per_episode}")

    # Initialize stats file
    if not os.path.exists(stats_file) or not args.resume:
        with open(stats_file, 'w') as f:
            f.write("Episode,Steps,Reward,Epsilon,Loss,Beta,LR,Cause,Stage,Food\n")

    # LR Scheduler (Linear Warmup)
    warmup_steps = 10000
    target_lr = cfg.opt.lr

    # Metrics tracking
    total_steps = agent.steps_done
    episode_rewards = [0] * cfg.env.num_agents
    episode_steps = [0] * cfg.env.num_agents
    episode_food = [0] * cfg.env.num_agents

    # Initial Reset
    states = env.reset()

    try:
        while start_episode < cfg.opt.max_episodes:
            # LR Warmup
            if total_steps < warmup_steps:
                lr_scale = total_steps / warmup_steps
                for param_group in agent.optimizer.param_groups:
                    param_group['lr'] = target_lr * lr_scale

            # Select actions
            actions = [agent.select_action(s) for s in states]

            # Step
            next_states, rewards, dones, infos = env.step(actions)

            loss = None

            for i in range(cfg.env.num_agents):
                total_steps += 1
                episode_rewards[i] += rewards[i]
                episode_steps[i] += 1
                episode_food[i] += infos[i].get('food_eaten', 0)

                force_done = False
                if episode_steps[i] >= max_steps_per_episode:
                    force_done = True

                if dones[i] or force_done:
                    terminal_state = infos[i]['terminal_observation'] if dones[i] else next_states[i]

                    # Store transition
                    agent.remember(states[i], actions[i], rewards[i], terminal_state, True)

                    # Log
                    start_episode += 1
                    eps = agent.get_epsilon()
                    loss_val = getattr(train, '_last_loss', 0) or 0

                    current_beta = min(1.0, agent.memory.beta_start + agent.memory.frame * (1.0 - agent.memory.beta_start) / agent.memory.beta_frames)
                    lr = agent.optimizer.param_groups[0]['lr']
                    
                    if force_done and not dones[i]:
                        cause = "MaxSteps"
                    else:
                        cause = infos[i].get('cause', 'Unknown')

                    pos = infos[i].get('pos', (0,0))
                    wall_dist = infos[i].get('wall_dist', -1)
                    pos_str = f"Pos:({pos[0]:.0f},{pos[1]:.0f})"
                    wall_str = f" Wall:{wall_dist:.0f}" if wall_dist >= 0 else ""
                    stage_name = curriculum.get_config()['name']

                    food_ratio = episode_food[i] / max(episode_steps[i], 1)
                    print(f"Ep {start_episode} | S{curriculum.current_stage}:{stage_name} | Rw: {episode_rewards[i]:.2f} | St: {episode_steps[i]} | Fd: {episode_food[i]} ({food_ratio:.3f}/st) | Eps: {eps:.3f} | L: {loss_val:.4f} | {cause} | {pos_str}{wall_str}")

                    with open(stats_file, 'a') as f:
                        f.write(f"{start_episode},{episode_steps[i]},{episode_rewards[i]:.2f},{eps:.4f},{loss_val:.4f},{current_beta:.2f},{lr:.6f},{cause},{curriculum.current_stage},{episode_food[i]}\n")

                    # Track metrics for curriculum promotion
                    curriculum.record_episode(episode_food[i], episode_steps[i])

                    # Check for stage promotion
                    if curriculum.check_promotion():
                        stage_cfg = curriculum.get_config()
                        max_steps_per_episode = curriculum.get_max_steps()
                        env.set_stage(stage_cfg)
                        # Save checkpoint on promotion
                        agent.save_checkpoint(checkpoint_path, start_episode, max_steps_per_episode, curriculum.get_state())

                    if start_episode % cfg.opt.checkpoint_every == 0:
                        agent.save_checkpoint(checkpoint_path, start_episode, max_steps_per_episode, curriculum.get_state())

                    episode_rewards[i] = 0
                    episode_steps[i] = 0
                    episode_food[i] = 0
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
                print(">> Target Network Updated")

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
    args = parser.parse_args()

    mp.set_start_method('spawn', force=True)
    train(args)
