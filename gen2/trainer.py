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

def worker(remote, parent_remote, worker_id, headless, nickname_prefix, matrix_size):
    parent_remote.close()
    
    picard_names = [
        "Picard", "Riker", "Data", "Worf", "Troi", "LaForge",
        "Crusher", "Q", "Seven", "Raffi", "Rios", "Jurati"
    ]
    chosen_name = f"{random.choice(picard_names)}_{worker_id}"
    
    try:
        env = SlitherEnv(headless=headless, nickname=chosen_name, matrix_size=matrix_size)

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
            elif cmd == 'close':
                env.close()
                break
    except Exception as e:
        print(f"Worker {worker_id} crashed: {e}")
    finally:
        remote.close()

class SubprocVecEnv:
    def __init__(self, num_agents, matrix_size, view_first=False, nickname="dzaczekAI"):
        self.num_agents = num_agents
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_agents)])
        self.ps = []

        for i in range(num_agents):
            is_headless = not (view_first and i == 0)
            p = mp.Process(target=worker, args=(self.work_remotes[i], self.remotes[i], i, is_headless, nickname, matrix_size))
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
        matrix_size=cfg.env.resolution[0], # Assuming square
        view_first=args.view,
        nickname="AI_Opt"
    )
    env = VecFrameStack(raw_env, k=cfg.env.frame_stack)

    # Initialize Agent
    agent = DDQNAgent(cfg)
    
    # Resume
    start_episode = 0
    if args.resume and os.path.exists(checkpoint_path):
        start_episode, _ = agent.load_checkpoint(checkpoint_path)
        print(f"Resumed from episode {start_episode}")

    # Initialize stats file
    if not os.path.exists(stats_file) or not args.resume:
        with open(stats_file, 'w') as f:
            f.write("Episode,Steps,Reward,Epsilon,Loss,Beta,LR,Cause\n")

    # Curriculum Settings
    max_steps_per_episode = 200 # Progressive Episode Limit (Start small)

    # LR Scheduler (Linear Warmup)
    # 0 to target_lr in 10000 steps
    warmup_steps = 10000
    target_lr = cfg.opt.lr

    # Metrics tracking
    total_steps = agent.steps_done
    episode_rewards = [0] * cfg.env.num_agents
    episode_steps = [0] * cfg.env.num_agents

    # Initial Reset
    states = env.reset()

    try:
        while start_episode < cfg.opt.max_episodes:
            # LR Warmup
            if total_steps < warmup_steps:
                lr_scale = total_steps / warmup_steps
                for param_group in agent.optimizer.param_groups:
                    param_group['lr'] = target_lr * lr_scale
            else:
                # Normal LR (could add decay here)
                pass

            # Select actions
            actions = [agent.select_action(s) for s in states]

            # Step
            next_states, rewards, dones, infos = env.step(actions)

            loss = None

            for i in range(cfg.env.num_agents):
                total_steps += 1
                episode_rewards[i] += rewards[i]
                episode_steps[i] += 1

                force_done = False
                if episode_steps[i] >= max_steps_per_episode:
                    force_done = True

                # Treat as done if environment finished OR we force it (TimeLimit)
                # Note: If force_done is True, we use 'next_states[i]' as the next state,
                # but we mark 'done' as True in memory to reset the return calculation (bootstrapping).
                # However, for correct bootstrapping in PPO/DQN with TimeLimit, we usually use done=False
                # and handle 'truncated' separately. But for this simple implementation,
                # marking done=True is a safe heuristic to prevent infinite loops,
                # effectively treating timeout as a terminal state "you ran out of time".

                if dones[i] or force_done:
                    terminal_state = infos[i]['terminal_observation'] if dones[i] else next_states[i]

                    # Store transition
                    # If force_done, we treat it as done.
                    agent.remember(states[i], actions[i], rewards[i], terminal_state, True)

                    # Log
                    start_episode += 1
                    eps = agent.get_epsilon()
                    loss_val = loss if loss is not None else 0

                    # Calculate current beta
                    current_beta = min(1.0, agent.memory.beta_start + agent.memory.frame * (1.0 - agent.memory.beta_start) / agent.memory.beta_frames)

                    lr = agent.optimizer.param_groups[0]['lr']
                    cause = infos[i].get('cause', 'Unknown')

                    print(f"Ep {start_episode} | Ag {i} | Rw: {episode_rewards[i]:.2f} | St: {episode_steps[i]} | Eps: {eps:.3f} | L: {loss_val:.4f} | {cause}")

                    with open(stats_file, 'a') as f:
                        f.write(f"{start_episode},{episode_steps[i]},{episode_rewards[i]:.2f},{eps:.4f},{loss_val:.4f},{current_beta:.2f},{lr:.6f},{cause}\n")

                    if start_episode % cfg.opt.checkpoint_every == 0:
                        agent.save_checkpoint(checkpoint_path, start_episode)

                        # Curriculum Update
                        if start_episode % 100 == 0:
                            max_steps_per_episode += 50
                            print(f">> Curriculum: Increased max steps to {max_steps_per_episode}")

                    episode_rewards[i] = 0
                    episode_steps[i] = 0
                else:
                    agent.remember(states[i], actions[i], rewards[i], next_states[i], False)

            # Update states
            states = next_states

            # Train
            loss = agent.optimize_model()

            # Target Update
            if total_steps % cfg.opt.target_update_freq == 0:
                agent.update_target()
                print(">> Target Network Updated")

    except KeyboardInterrupt:
        print("Interrupted. Saving...")
        agent.save_checkpoint(checkpoint_path, start_episode)
    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, default=0, help="Override config num_agents")
    parser.add_argument("--view", action="store_true", help="View first agent")
    parser.add_argument("--resume", action="store_true", help="Resume")
    args = parser.parse_args()

    mp.set_start_method('spawn', force=True)
    train(args)
