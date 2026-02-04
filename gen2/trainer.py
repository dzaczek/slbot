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
                    # Note: We return the transition result (done=True)
                    # The next step from main process will be start of new episode.
                    # But for the Agent, we need the "next_state" for the buffer.
                    # Standard practice: Return info about reset?
                    # Usually: return next_state (which is terminal), and maybe extra info['terminal_observation']
                    # Here, let's just reset immediately so the worker is ready.
                    reset_state = env.reset()
                    # We return next_state (terminal) so the agent learns from death.
                    # But we also need to tell main process that we reset.
                    # Let's attach the reset state to info or return it separately?
                    # For simplicity: Just return the terminal state.
                    # The main loop will see done=True.
                    # BUT: The NEXT step's state needs to be the reset_state.
                    # So we should return (reset_state, reward, done, info) where reset_state is the new start?
                    # NO. The transition is (s, a, r, s', done). s' must be the terminal state.
                    # So we return terminal state.
                    # The worker needs to store the reset state for the *next* step call?
                    # Or we send 'reset' command explicitly from main?
                    # Explicit reset is safer for synchronization but slower.
                    # Let's do auto-reset and return (reset_state, reward, done, info)
                    # but stash the terminal state in info?
                    # Standard Gym VectorEnv returns (reset_state, reward, done, info) where info has "terminal_observation".

                    # Let's do that.
                    info['terminal_observation'] = next_state
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

    # Init stats file
    stats_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'matrix_stats.csv')
    # If restarting, maybe append? For now, overwrite as per original logic,
    # but maybe we should check if file exists? Original overwrote. I'll stick to original behavior.
    with open(stats_file, 'w') as f:
        f.write("Episode,Steps,Reward,Epsilon\n")

    # Reset all environments
    states = env.reset() # List of states

    # Tracking per agent
    current_rewards = [0] * num_agents
    current_steps = [0] * num_agents
    total_episodes_finished = 0

    MAX_EPISODES = 500 # Total episodes across all agents? Or per agent?
    # Let's say total episodes.

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

    # On Windows, spawn is default. On Unix, fork.
    # multiprocessing needs this protection.
    try:
        train(args)
    except Exception as e:
        print(f"Main loop error: {e}")
