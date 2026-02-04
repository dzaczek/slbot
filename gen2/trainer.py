import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import os
import sys
from collections import deque
import time

# Add gen2 to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from slither_env import SlitherEnv
from model import MatrixPolicy

BATCH_SIZE = 64 # Smaller batch for slower Selenium
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 10000
TARGET_UPDATE = 1000
MEMORY_SIZE = 10000

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

    def select_action(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            np.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                return self.policy_net(state_t).max(1)[1].item() # index
        else:
            return random.randrange(6) # 6 actions

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

def train():
    env = SlitherEnv(headless=True, nickname="MatrixAI")
    agent = DDQNAgent()

    num_episodes = 500

    print("Starting Matrix-based Slither.io Training...")

    for i_episode in range(num_episodes):
        # Reset environment
        state = env.reset()

        total_reward = 0

        # Limit episode length? Slither can go on forever.
        # But step is slow.
        for t in range(1000):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            # Store in memory
            agent.memory.append((state, action, reward, next_state, done))

            state = next_state
            total_reward += reward

            # Train
            agent.optimize_model()

            if done:
                print(f"Episode {i_episode} finished after {t+1} steps. Reward: {total_reward:.2f}")
                break

            # Target update
            if agent.steps_done % TARGET_UPDATE == 0:
                agent.update_target()
                print("Target Network Updated")

    env.close()

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("Training stopped.")
    except Exception as e:
        print(f"Error: {e}")
