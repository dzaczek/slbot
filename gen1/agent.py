import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import DDQN
import torch.optim as optim
import torch.nn as nn

MAX_MEMORY = 100_000
BATCH_SIZE = 2048
LR = 0.001
GAMMA = 0.95
TAU = 0.005 # For soft update

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 1.0 # Randomness
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DDQN().to(self.device)
        self.target_net = DDQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.criterion = nn.MSELoss()

    def get_state(self, game):
        grid, orientation = game.get_state_data()
        # Convert to tensor and add batch dimension if needed,
        # but here we just return numpy arrays, conversion happens in train step or get_action
        return grid, orientation

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.train_step([state], [action], [reward], [next_state], [done])

    def train_step(self, states, actions, rewards, next_states, dones):
        # states is list of tuples (grid, orient)
        # We need to unzip them
        state_grids = np.array([s[0] for s in states])
        state_orients = np.array([s[1] for s in states])

        next_state_grids = np.array([s[0] for s in next_states])
        next_state_orients = np.array([s[1] for s in next_states])

        state_grids = torch.tensor(state_grids, dtype=torch.float).unsqueeze(1).to(self.device) # (B, 1, 20, 20)
        state_orients = torch.tensor(state_orients, dtype=torch.float).to(self.device)

        next_state_grids = torch.tensor(next_state_grids, dtype=torch.float).unsqueeze(1).to(self.device)
        next_state_orients = torch.tensor(next_state_orients, dtype=torch.float).to(self.device)

        actions = torch.tensor(actions, dtype=torch.long).to(self.device) # (B, 3) usually one-hot or index?
        # If action passed is [1,0,0], it's one-hot.
        # We need indices for gather if they are one-hot
        if len(actions.shape) == 2 and actions.shape[1] == 3:
             action_indices = torch.argmax(actions, dim=1).unsqueeze(1)
        else:
             action_indices = actions.unsqueeze(1)

        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device) # (B)

        # 1. Predicted Q values with current state
        pred = self.policy_net(state_grids, state_orients)
        pred_q = pred.gather(1, action_indices).squeeze(1)

        # 2. Double DQN Target
        # a. Select best action using Policy Network on Next State
        with torch.no_grad():
            next_actions_policy = self.policy_net(next_state_grids, next_state_orients).argmax(dim=1).unsqueeze(1)

            # b. Evaluate that action using Target Network
            next_q_target = self.target_net(next_state_grids, next_state_orients).gather(1, next_actions_policy).squeeze(1)

        target = rewards + (1 - dones.float()) * GAMMA * next_q_target

        # 3. Loss and Optimize
        loss = self.criterion(pred_q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 4. Soft Update Target Network
        self.soft_update()

    def soft_update(self):
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(TAU * policy_param.data + (1.0 - TAU) * target_param.data)

    def get_action(self, state):
        # state is (grid, orient)
        grid, orient = state

        # Random moves: tradeoff exploration / exploitation
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move = [0, 0, 0]
            final_move[move] = 1
        else:
            grid_t = torch.tensor(grid, dtype=torch.float).unsqueeze(0).unsqueeze(0).to(self.device) # (1, 1, 20, 20)
            orient_t = torch.tensor(orient, dtype=torch.float).unsqueeze(0).to(self.device)

            with torch.no_grad():
                prediction = self.policy_net(grid_t, orient_t)

            move = torch.argmax(prediction).item()
            final_move = [0, 0, 0]
            final_move[move] = 1

        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    print("Starting training...")

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.policy_net.save()

            print(f'Game {agent.n_games} Score {score} Record {record} Eps {agent.epsilon:.2f}')

            # Log to file
            with open('ddqn_stats.csv', 'a') as f:
                f.write(f'{agent.n_games},{score},{record},{agent.epsilon:.4f}\n')

            # Decay epsilon
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay

if __name__ == '__main__':
    train()
