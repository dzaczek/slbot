import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI
from model import DDQN
import torch.optim as optim
import torch.nn as nn
import multiprocessing as mp
import time

MAX_MEMORY = 100_000
BATCH_SIZE = 2048
LR = 0.001
GAMMA = 0.95
TAU = 0.005
NUM_ENVS = 8  # Number of parallel agents

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999 # Slower decay because more updates
        self.memory = deque(maxlen=MAX_MEMORY)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DDQN().to(self.device)
        self.target_net = DDQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.train_step(states, actions, rewards, next_states, dones)

    def train_step(self, states, actions, rewards, next_states, dones):
        state_grids = np.array([s[0] for s in states])
        state_orients = np.array([s[1] for s in states])

        next_state_grids = np.array([s[0] for s in next_states])
        next_state_orients = np.array([s[1] for s in next_states])

        state_grids = torch.tensor(state_grids, dtype=torch.float).unsqueeze(1).to(self.device)
        state_orients = torch.tensor(state_orients, dtype=torch.float).to(self.device)

        next_state_grids = torch.tensor(next_state_grids, dtype=torch.float).unsqueeze(1).to(self.device)
        next_state_orients = torch.tensor(next_state_orients, dtype=torch.float).to(self.device)

        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        if len(actions.shape) == 2 and actions.shape[1] == 3:
             action_indices = torch.argmax(actions, dim=1).unsqueeze(1)
        else:
             action_indices = actions.unsqueeze(1)

        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)

        # Q values
        pred = self.policy_net(state_grids, state_orients)
        pred_q = pred.gather(1, action_indices).squeeze(1)

        # Double DQN Target
        with torch.no_grad():
            next_actions_policy = self.policy_net(next_state_grids, next_state_orients).argmax(dim=1).unsqueeze(1)
            next_q_target = self.target_net(next_state_grids, next_state_orients).gather(1, next_actions_policy).squeeze(1)

        target = rewards + (1 - dones.float()) * GAMMA * next_q_target

        loss = self.criterion(pred_q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update()

    def soft_update(self):
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(TAU * policy_param.data + (1.0 - TAU) * target_param.data)

    def get_actions(self, states):
        # states is list of (grid, orient)
        # Batch inference
        grids = np.array([s[0] for s in states])
        orients = np.array([s[1] for s in states])

        grids_t = torch.tensor(grids, dtype=torch.float).unsqueeze(1).to(self.device) # (B, 1, 20, 20)
        orients_t = torch.tensor(orients, dtype=torch.float).to(self.device)

        with torch.no_grad():
            predictions = self.policy_net(grids_t, orients_t) # (B, 3)

        final_moves = []
        for i in range(len(states)):
            if random.random() < self.epsilon:
                move_idx = random.randint(0, 2)
            else:
                move_idx = torch.argmax(predictions[i]).item()

            move = [0, 0, 0]
            move[move_idx] = 1
            final_moves.append(move)

        return final_moves

# Worker Process
def worker(remote, parent_remote, worker_id):
    parent_remote.close()
    try:
        game = SnakeGameAI(render=False) # Headless

        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                action = data
                reward, done, score = game.play_step(action)
                state = game.get_state_data()
                if done:
                    game.reset()

                remote.send((state, reward, done, score))
            elif cmd == 'reset':
                state = game.reset()
                remote.send(state)
            elif cmd == 'close':
                break
    except Exception as e:
        print(f"Worker {worker_id} crashed: {e}")
        import traceback
        traceback.print_exc()

class VecEnv:
    def __init__(self, n_envs):
        self.n_envs = n_envs
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(n_envs)])
        self.ps = [mp.Process(target=worker, args=(work_remote, remote, i))
                   for i, (work_remote, remote) in enumerate(zip(self.work_remotes, self.remotes))]
        for p in self.ps:
            p.daemon = True # if main crashes, workers die
            p.start()
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
        states, rewards, dones, scores = zip(*results)
        return states, rewards, dones, scores

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

def train_parallel():
    envs = VecEnv(NUM_ENVS)
    agent = Agent()

    print(f"Starting PARALLEL training with {NUM_ENVS} workers...")

    # Initialize all envs
    states = envs.reset() # List of N states

    total_games = 0
    scores_window = []

    try:
        while True:
            # 1. Get Actions for all N states
            actions = agent.get_actions(states) # List of N moves

            # 2. Step all environments
            next_states, rewards, dones, scores = envs.step(actions)

            # 3. Store transitions
            for i in range(NUM_ENVS):
                agent.remember(states[i], actions[i], rewards[i], next_states[i], dones[i])

                if dones[i]:
                    total_games += 1
                    scores_window.append(scores[i])

                    # Log stats every 10 games or so
                    if total_games % 10 == 0:
                        avg_score = np.mean(scores_window[-50:]) if scores_window else 0
                        print(f"Game {total_games} | Worker {i} | Score {scores[i]} | Avg {avg_score:.2f} | Eps {agent.epsilon:.2f}")

                        # Log to file
                        with open('vector_stats.csv', 'a') as f:
                            f.write(f'{total_games},{scores[i]},{avg_score},{agent.epsilon:.4f}\n')

                    # We could train long memory here, or periodically
                    # agent.train_long_memory()

            # 4. Train
            # In Vectorized, we usually train once per step on the buffer,
            # or once every few steps.
            agent.train_long_memory() # Use a batch from replay buffer

            # 5. Update states
            # If done, next_state is the reset state (handled by worker implicitly? No.)
            # Wait, my worker logic:
            # if done: game.reset() -> internal state reset.
            # But the 'state' returned in 'step' was the one AFTER the move (Death state?).
            # Actually, standard VecEnv returns the NEW state (reset) if done is True.
            # My worker sends 'state' = game.get_state_data().
            # If done, I called game.reset() inside worker?
            # Let's check worker logic:
            # if done: game.reset()
            # Wait, if I reset, get_state_data() returns the START state.
            # So next_state IS the start state of the new game.
            # BUT for training, the transition (s, a, r, s') needs s' to be the terminal state (or masked).
            # The 'next_state' in my memory will be the START of the new game.
            # This is technically wrong for the Q-value update: Q(s,a) ~ r + gamma * maxQ(s').
            # If s' is the start of a new game, it's unrelated.
            # Correct logic:
            # If done, target = r. (gamma * 0).
            # My Agent code handles this: target = rewards + (1 - dones) * ...
            # So if done=True, it ignores next_state's Q-value.
            # So it DOES NOT MATTER what next_state is, as long as 'done' is True.
            # So my worker logic (sending start state as next_state) is fine for the Agent,
            # and convenient for the loop (states = next_states).

            states = next_states

            # Decay epsilon (slower because we step N times)
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay

    except KeyboardInterrupt:
        print("Stopping...")
        envs.close()

if __name__ == '__main__':
    # Initialize stats file
    with open('vector_stats.csv', 'w') as f:
        f.write("Game,Score,AvgScore,Epsilon\n")

    # mp.set_start_method('spawn') # Commented out to use default (fork)
    train_parallel()
