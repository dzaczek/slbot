import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import os
import sys
from collections import deque
import time
import multiprocessing as mp
import argparse

# Add gen2 to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from slither_env import SlitherEnv
from model import MatrixPolicy

BATCH_SIZE = 64       # Smaller batch for slower Selenium
GAMMA = 0.99          # Discount factor
EPS_START = 1.0       # Start with full exploration
EPS_END = 0.05        # Minimum exploration
EPS_DECAY = 100000    # Much slower decay for longer exploration (was 50000)
TARGET_UPDATE = 2000  # Less frequent target updates for stability (was 1000)
MEMORY_SIZE = 100000  # Larger replay buffer to retain good experiences (was 50000)

class TrainingSupervisor:
    """
    Monitors training progress to detect stagnation or degradation.
    """
    def __init__(self, patience=100, improvement_threshold=0.01, degradation_threshold=0.4):
        self.patience = patience
        self.improvement_threshold = improvement_threshold
        self.degradation_threshold = degradation_threshold

        self.rewards_window = deque(maxlen=100) # Moving average window (increased for stability)
        self.best_avg_reward = -float('inf')
        self.episodes_since_improvement = 0

        self.actions_log = []

    def step(self, episode_reward):
        self.rewards_window.append(episode_reward)

        if len(self.rewards_window) < 10:
            return None # Not enough data

        current_avg = np.mean(self.rewards_window)

        # Check for improvement
        if current_avg > self.best_avg_reward + self.improvement_threshold:
            self.best_avg_reward = current_avg
            self.episodes_since_improvement = 0
            return "SAVE_CHECKPOINT"

        self.episodes_since_improvement += 1

        # Check for stagnation
        if self.episodes_since_improvement >= self.patience:
            self.episodes_since_improvement = 0 # Reset to give time for LR decay to work
            return "DECAY_LR"

        # Check for degradation
        # If current avg drops below 50% of best (and best is positive)
        if self.best_avg_reward > 0 and current_avg < self.best_avg_reward * self.degradation_threshold:
             return "WARNING_DEGRADATION"

        return None
    
    def get_state(self):
        """Get supervisor state for checkpointing."""
        return {
            'rewards_window': list(self.rewards_window),
            'best_avg_reward': self.best_avg_reward,
            'episodes_since_improvement': self.episodes_since_improvement,
        }
    
    def load_state(self, state):
        """Restore supervisor state from checkpoint."""
        if state is None:
            return
        self.rewards_window = deque(state['rewards_window'], maxlen=50)
        self.best_avg_reward = state['best_avg_reward']
        self.episodes_since_improvement = state['episodes_since_improvement']
        print(f"   Supervisor: best_avg={self.best_avg_reward:.2f}, window_size={len(self.rewards_window)}")

def worker(remote, parent_remote, worker_id, headless, nickname_prefix):
    parent_remote.close()
    
    # Picard Character Names
    picard_names = [
        "Picard!!.", "Riker!!.", "Data!!.", "Worf!!.", "Troi!!.", "LaForge!!.",
        "Crusher!!.", "Q!!.", "Seven!!.", "Raffi!!.", "Rios!!.", "Jurati!!.",
        "Soji!!.", "Guinan!!.", "Locutus!!.", "Borg!!."
    ]
    
    # Select random name or consistent based on worker_id if we want stability
    # Using random choice here so re-launches get different names
    chosen_name = random.choice(picard_names)
    
    try:
        # Initialize environment inside the worker process
        # We ignore the prefix/worker_id in the actual game nickname now
        env = SlitherEnv(headless=headless, nickname=chosen_name)

        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                action = data
                next_state, reward, done, info = env.step(action)
                if done:
                    # Automatically reset if done
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
        import traceback
        traceback.print_exc()
    finally:
        remote.close()

class SubprocVecEnv:
    def __init__(self, num_agents, view_first=False, nickname="dzaczekAI"):
        self.num_agents = num_agents
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_agents)])
        self.ps = []

        for i in range(num_agents):
            # First agent is visible if view_first is True
            is_headless = not (view_first and i == 0)
            p = mp.Process(target=worker, args=(self.work_remotes[i], self.remotes[i], i, is_headless, nickname))
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
        # unzip
        states, rewards, dones, infos = zip(*results)
        return states, rewards, dones, infos

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

class DDQNAgent:
    def __init__(self, learning_rate=0.0001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = MatrixPolicy().to(self.device)
        self.target_net = MatrixPolicy().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.steps_done = 0

    def get_epsilon(self):
        return EPS_END + (EPS_START - EPS_END) * \
            np.exp(-1. * self.steps_done / EPS_DECAY)

    def select_actions(self, states):
        # states: list of numpy arrays (3, 64, 64)
        eps_threshold = self.get_epsilon()
        self.steps_done += len(states) # Increment by batch size

        # Convert to tensor batch
        states_np = np.array(states) # (B, 3, 64, 64)
        states_t = torch.tensor(states_np, dtype=torch.float32).to(self.device)

        # Get model predictions for all
        with torch.no_grad():
            q_values = self.policy_net(states_t) # (B, 6)
            best_actions = q_values.max(1)[1].cpu().numpy() # (B,)

        actions = []
        for i in range(len(states)):
            if random.random() > eps_threshold:
                actions.append(best_actions[i])
            else:
                actions.append(random.randrange(6))

        return actions

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return

        transitions = random.sample(self.memory, BATCH_SIZE)
        batch_state, batch_action, batch_reward, batch_next, batch_done = zip(*transitions)

        state_batch = torch.tensor(np.array(batch_state), dtype=torch.float32).to(self.device)
        action_batch = torch.tensor(batch_action, dtype=torch.long).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(batch_reward, dtype=torch.float32).to(self.device)
        next_batch = torch.tensor(np.array(batch_next), dtype=torch.float32).to(self.device)
        done_batch = torch.tensor(batch_done, dtype=torch.float32).to(self.device)

        # Q(s, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # V(s') = max Q(s', a') from target
        # Double DQN: action from policy, value from target
        next_actions = self.policy_net(next_batch).max(1)[1].unsqueeze(1)
        next_state_values = self.target_net(next_batch).gather(1, next_actions).squeeze(1)

        expected_state_action_values = (next_state_values * GAMMA * (1 - done_batch)) + reward_batch

        loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_checkpoint(self, filepath, episode, supervisor_state=None):
        """Save full training state to resume later."""
        checkpoint = {
            'episode': episode,
            'steps_done': self.steps_done,
            'policy_net_state': self.policy_net.state_dict(),
            'target_net_state': self.target_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'memory': list(self.memory),  # Convert deque to list for pickle
            'supervisor_state': supervisor_state,
        }
        torch.save(checkpoint, filepath)
        print(f">> Checkpoint saved: {filepath} (Episode {episode}, Steps {self.steps_done})")
    
    def load_checkpoint(self, filepath):
        """Load training state from checkpoint. Returns (episode, supervisor_state)."""
        if not os.path.exists(filepath):
            print(f"No checkpoint found at {filepath}")
            return 0, None
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state'])
        self.target_net.load_state_dict(checkpoint['target_net_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.steps_done = checkpoint['steps_done']
        
        # Restore memory
        self.memory = deque(checkpoint['memory'], maxlen=MEMORY_SIZE)
        
        episode = checkpoint['episode']
        supervisor_state = checkpoint.get('supervisor_state')
        
        print(f">> Checkpoint loaded: {filepath}")
        print(f"   Episode: {episode}, Steps: {self.steps_done}, Memory: {len(self.memory)}")
        print(f"   Epsilon: {self.get_epsilon():.4f}")
        
        return episode, supervisor_state

def train(args):
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(base_dir, 'checkpoint.pth')
    best_model_path = os.path.join(base_dir, 'best_model.pth')
    stats_file = os.path.join(base_dir, 'matrix_stats.csv')
    
    # Settings
    CHECKPOINT_EVERY = 50  # Save checkpoint every N episodes
    MAX_EPISODES = 5000000   # Total episodes to train
    
    num_agents = args.num_agents
    print(f"=" * 60)
    print(f"Matrix-based Slither.io Training")
    print(f"=" * 60)
    print(f"Agents: {num_agents}")
    print(f"Max Episodes: {MAX_EPISODES}")
    print(f"Checkpoint every: {CHECKPOINT_EVERY} episodes")
    print(f"Resume mode: {args.resume}")
    if args.view:
        print("View mode: enabled for first agent")
    print(f"=" * 60)

    # Initialize
    env = SubprocVecEnv(num_agents=num_agents, view_first=args.view, nickname="dzaczekAI")
    agent = DDQNAgent()
    supervisor = TrainingSupervisor()
    
    # Starting episode (0 or loaded from checkpoint)
    start_episode = 0
    
    # Load checkpoint if resuming
    if args.resume:
        if os.path.exists(checkpoint_path):
            start_episode, sup_state = agent.load_checkpoint(checkpoint_path)
            supervisor.load_state(sup_state)
            print(f"Resuming from episode {start_episode}")
        else:
            print("No checkpoint found, starting fresh.")
    
    # Init or append to stats file
    if not args.resume or not os.path.exists(stats_file):
        with open(stats_file, 'w') as f:
            f.write("Episode,Steps,Reward,Epsilon,Cause,MemorySize,LearningRate,AvgReward50\n")
    else:
        print(f"Appending to existing stats file: {stats_file}")

    # Reset all environments
    states = env.reset()

    # Tracking per agent
    current_rewards = [0] * num_agents
    current_steps = [0] * num_agents
    total_episodes_finished = start_episode

    try:
        while total_episodes_finished < MAX_EPISODES:
            # 1. Select actions
            actions = agent.select_actions(states)

            # 2. Step
            next_states, rewards, dones, infos = env.step(actions)

            # 3. Store and update
            for i in range(num_agents):
                if dones[i]:
                    terminal_state = infos[i]['terminal_observation']
                    agent.memory.append((states[i], actions[i], rewards[i], terminal_state, dones[i]))

                    # Log
                    total_episodes_finished += 1
                    current_rewards[i] += rewards[i]
                    current_steps[i] += 1

                    eps = agent.get_epsilon()
                    cause = infos[i].get('cause', 'Unknown')
                    print(f"Ep {total_episodes_finished} | Agent {i} | Steps: {current_steps[i]} | Reward: {current_rewards[i]:.2f} | Eps: {eps:.3f} | {cause}")

                    # Supervisor Step
                    sup_action = supervisor.step(current_rewards[i])
                    if sup_action == "SAVE_CHECKPOINT":
                        print(">> SUPERVISOR: New Best! Saving best model...")
                        torch.save(agent.policy_net.state_dict(), best_model_path)
                    elif sup_action == "DECAY_LR":
                        print(">> SUPERVISOR: Stagnation. Decaying LR.")
                        for param_group in agent.optimizer.param_groups:
                            param_group['lr'] *= 0.5
                            print(f"   New LR: {param_group['lr']}")
                    elif sup_action == "WARNING_DEGRADATION":
                        print(">> SUPERVISOR: CRITICAL DEGRADATION! Rolling back to best model...")
                        # Load best model weights to policy net
                        if os.path.exists(best_model_path):
                            agent.policy_net.load_state_dict(torch.load(best_model_path))
                            agent.update_target() # Sync target net immediately
                            # Reset optimizer to prevent momentum from pushing in the wrong direction again
                            # (Optional, but usually safer to keep LR but clear state, here we just keep going)
                            print("   >> Rollback complete. Resumed from best state.")
                        else:
                            print("   >> Error: No best model to rollback to!")

                    # Write stats
                    mem_size = len(agent.memory)
                    lr = agent.optimizer.param_groups[0]['lr']
                    avg_reward = np.mean(list(supervisor.rewards_window)) if len(supervisor.rewards_window) > 0 else 0
                    with open(stats_file, 'a') as f:
                        f.write(f"{total_episodes_finished},{current_steps[i]},{current_rewards[i]:.2f},{eps:.4f},{cause},{mem_size},{lr:.6f},{avg_reward:.2f}\n")

                    # Periodic checkpoint save
                    if total_episodes_finished % CHECKPOINT_EVERY == 0:
                        agent.save_checkpoint(checkpoint_path, total_episodes_finished, supervisor.get_state())

                    # Reset trackers
                    current_rewards[i] = 0
                    current_steps[i] = 0
                    states[i] = next_states[i]
                else:
                    agent.memory.append((states[i], actions[i], rewards[i], next_states[i], dones[i]))
                    current_rewards[i] += rewards[i]
                    current_steps[i] += 1
                    states[i] = next_states[i]

            # 4. Train
            agent.optimize_model()

            # 5. Target update
            if agent.steps_done % TARGET_UPDATE == 0:
                agent.update_target()
                print(">> Target Network Updated")

    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("Training interrupted by user (Ctrl+C)")
        print("Saving checkpoint before exit...")
        agent.save_checkpoint(checkpoint_path, total_episodes_finished, supervisor.get_state())
        print("=" * 60)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        # Try to save on error too
        try:
            agent.save_checkpoint(checkpoint_path, total_episodes_finished, supervisor.get_state())
        except:
            pass
    finally:
        env.close()
        print(f"Training ended at episode {total_episodes_finished}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slither.io DDQN Training")
    parser.add_argument("--num_agents", type=int, default=1, help="Number of parallel agents")
    parser.add_argument("--view", action="store_true", help="Show browser for the first agent")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    args = parser.parse_args()

    # On Windows, spawn is default. On Unix, fork.
    # multiprocessing needs this protection.
    try:
        train(args)
    except Exception as e:
        print(f"Main loop error: {e}")
        import traceback
        traceback.print_exc()
