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

BATCH_SIZE = 64 # Smaller batch for slower Selenium
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 10000
TARGET_UPDATE = 1000
MEMORY_SIZE = 10000

class TrainingSupervisor:
    """
    Monitors training progress to detect stagnation or degradation.
    """
    def __init__(self, patience=50, improvement_threshold=0.01, degradation_threshold=0.5):
        self.patience = patience
        self.improvement_threshold = improvement_threshold
        self.degradation_threshold = degradation_threshold

        self.rewards_window = deque(maxlen=50) # Moving average window
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
             # Just a warning or stop? For now, let's just log.
             # Returning STOP might be too aggressive if random variance.
             return "WARNING_DEGRADATION"

        return None

def worker(remote, parent_remote, worker_id, headless, nickname):
    parent_remote.close()
    try:
        # Initialize environment inside the worker process
        env = SlitherEnv(headless=headless, nickname=f"{nickname}_{worker_id}")

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
    def __init__(self, num_agents, view_first=False, nickname="MatrixAI"):
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
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = MatrixPolicy().to(self.device)
        self.target_net = MatrixPolicy().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.steps_done = 0

    def get_epsilon(self):
        return EPS_END + (EPS_START - EPS_END) * \
            np.exp(-1. * self.steps_done / EPS_DECAY)

    def select_actions(self, states):
        # states: list of numpy arrays (3, 64, 64)
        sample = random.random()
        eps_threshold = self.get_epsilon()
        self.steps_done += len(states) # Increment by batch size

        # Convert to tensor batch
        # Stack states
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

def train(args):
    # Use SubprocVecEnv instead of single SlitherEnv
    num_agents = args.num_agents
    print(f"Starting Matrix-based Slither.io Training with {num_agents} agents...")
    if args.view:
        print("View mode enabled for the first agent.")

    env = SubprocVecEnv(num_agents=num_agents, view_first=args.view, nickname="MatrixAI")
    agent = DDQNAgent()

    supervisor = TrainingSupervisor()
    best_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_model.pth')

    # Init stats file
    stats_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'matrix_stats.csv')
    with open(stats_file, 'w') as f:
        f.write("Episode,Steps,Reward,Epsilon\n")

    # Reset all environments
    states = env.reset() # List of states

    # Tracking per agent
    current_rewards = [0] * num_agents
    current_steps = [0] * num_agents
    total_episodes_finished = 0

    MAX_EPISODES = 500

    try:
        while total_episodes_finished < MAX_EPISODES:
            # 1. Select actions
            actions = agent.select_actions(states)

            # 2. Step
            next_states, rewards, dones, infos = env.step(actions)

            # 3. Store and update
            for i in range(num_agents):
                # If done, next_states[i] is the RESET state.
                # The terminal state is in infos[i]['terminal_observation']
                if dones[i]:
                    terminal_state = infos[i]['terminal_observation']
                    agent.memory.append((states[i], actions[i], rewards[i], terminal_state, dones[i]))

                    # Log
                    total_episodes_finished += 1
                    current_rewards[i] += rewards[i]
                    current_steps[i] += 1

                    eps = agent.get_epsilon()
                    cause = infos[i].get('cause', 'Unknown')
                    print(f"Episode finished | Agent {i} | Steps: {current_steps[i]} | Reward: {current_rewards[i]:.2f} | Cause: {cause}")

                    # Supervisor Step
                    sup_action = supervisor.step(current_rewards[i])
                    if sup_action == "SAVE_CHECKPOINT":
                        print(">> SUPERVISOR: New Best Average Reward! Saving model...")
                        torch.save(agent.policy_net.state_dict(), best_model_path)
                    elif sup_action == "DECAY_LR":
                        print(">> SUPERVISOR: Stagnation detected. Decaying Learning Rate.")
                        for param_group in agent.optimizer.param_groups:
                            param_group['lr'] *= 0.5
                            print(f"   New LR: {param_group['lr']}")
                    elif sup_action == "WARNING_DEGRADATION":
                        print(">> SUPERVISOR: Warning! Performance degradation detected.")

                    with open(stats_file, 'a') as f:
                        f.write(f"{total_episodes_finished},{current_steps[i]},{current_rewards[i]:.2f},{eps:.4f}\n")

                    # Reset trackers for this agent
                    current_rewards[i] = 0
                    current_steps[i] = 0

                    # Update state to the new reset state
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
                print("Target Network Updated")

    except KeyboardInterrupt:
        print("Training stopped.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, default=1, help="Number of parallel agents")
    parser.add_argument("--view", action="store_true", help="Show browser for the first agent")
    args = parser.parse_args()

    try:
        train(args)
    except Exception as e:
        print(f"Main loop error: {e}")
