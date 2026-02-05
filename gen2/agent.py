import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
from collections import deque

from config import Config
from model import DuelingDQN
from per import PrioritizedReplayBuffer

class DDQNAgent:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Calculate input channels (3 base channels * frame_stack)
        self.input_channels = 3 * config.env.frame_stack

        self.policy_net = DuelingDQN(
            input_channels=self.input_channels,
            action_dim=6
        ).to(self.device)

        self.target_net = DuelingDQN(
            input_channels=self.input_channels,
            action_dim=6
        ).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=config.opt.lr
        )

        if config.buffer.prioritized:
            self.memory = PrioritizedReplayBuffer(
                capacity=config.buffer.capacity,
                alpha=config.buffer.alpha,
                beta_start=config.buffer.beta_start,
                beta_frames=config.buffer.beta_frames
            )
        else:
            # Fallback to simple deque if prioritized is disabled (not implemented in per.py but keeping structure open)
            raise NotImplementedError("Only Prioritized Buffer is supported currently.")

        self.steps_done = 0

        # Reward Normalization Stats
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 1e-5 # avoid div by zero

    def get_epsilon(self):
        return self.config.opt.eps_end + (self.config.opt.eps_start - self.config.opt.eps_end) * \
            np.exp(-1. * self.steps_done / self.config.opt.eps_decay)

    def _stack_frames(self, frames):
        """
        Stacks list of frames into a single numpy array.
        Each frame is (3, H, W).
        Output is (12, H, W).
        """
        return np.concatenate(frames, axis=0)

    def select_action(self, state_stack):
        """
        state_stack: numpy array (12, 84, 84)
        """
        eps_threshold = self.get_epsilon()
        self.steps_done += 1

        if random.random() > eps_threshold:
            with torch.no_grad():
                state_t = torch.tensor(state_stack, dtype=torch.float32).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_t)
                return q_values.max(1)[1].item()
        else:
            return random.randrange(6)

    def remember(self, state, action, reward, next_state, done):
        """
        Stores transition.
        state/next_state should be single frames (3, H, W) usually,
        BUT PER/LazyFrames logic handles stacking.

        However, standard PER implementation expects full states if we don't have a specialized LazyFrame wrapper.
        For simplicity in this step, we will assume the caller passes the FULL STACKED STATE.

        Wait, the plan said "Lazy Frames logic – store single frames, stack on sampling".
        To do this, I need to store the history in the buffer or agent.

        Simpler approach for now: Store full stacks. It uses more RAM (12 channels vs 3),
        but implementation is robust. 4x RAM usage.
        Given 84x84x12 float32 is ~340KB per item. 100k items = 34GB. Too big.

        We MUST implement Lazy Stacking or store unit8.
        The frames are currently float32 (0.0 to 1.0) from `slither_env`.
        We should convert to uint8 (0-255) for storage.

        Let's assume the Trainer manages the frame history (deque) and passes the CURRENT frame to the agent,
        and the agent stores a sequence.

        Actually, `PrioritizedReplayBuffer` stores whatever we pass to `push`.

        Refined Plan for Memory:
        The `SlitherEnv` returns a single frame (3, 84, 84).
        The `Trainer` maintains a `deque(maxlen=4)` of frames.
        When `remember` is called, we pass the *current frame*.
        BUT to reconstruct state, we need context.

        If we use `LazyFrames`, the buffer needs to be aware of the sequence.

        Given the constraints and complexity, I will implement **Explicit Stacking in Agent** but store **Compressed** data if possible.
        Or, I will stick to storing stacked frames but reduce buffer size if needed.

        Actually, let's implement the standard approach:
        The Agent stores `(state, action, reward, next_state, done)`.
        If we want to save memory, we can cast to `uint8` before storing.
        (3, 84, 84) float32 -> 84KB. 100k = 8.4GB. Manageable.
        (12, 84, 84) float32 -> 338KB. 100k = 33GB. Too big.

        Decision: Store stacked frames as `np.uint8`.
        The env returns float (0.0, 0.5, 1.0).
        We scale by 255 and cast to uint8.
        """
        # Compress to uint8
        state_u8 = (state * 255).astype(np.uint8)
        next_state_u8 = (next_state * 255).astype(np.uint8)

        self.memory.push(state_u8, action, reward, next_state_u8, done)

    def optimize_model(self):
        if len(self.memory) < self.config.opt.batch_size:
            return None

        # Sample
        transitions, idxs, is_weights = self.memory.sample(self.config.opt.batch_size)

        # Unzip
        batch_state, batch_action, batch_reward, batch_next, batch_done = zip(*transitions)

        # Convert to tensors
        # Note: They are uint8, need to convert back to float and normalize
        state_batch = torch.tensor(np.array(batch_state), dtype=torch.float32).to(self.device) / 255.0
        action_batch = torch.tensor(batch_action, dtype=torch.long).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(batch_reward, dtype=torch.float32).to(self.device)
        next_batch = torch.tensor(np.array(batch_next), dtype=torch.float32).to(self.device)
        done_batch = torch.tensor(batch_done, dtype=torch.float32).to(self.device)
        weights_batch = torch.tensor(is_weights, dtype=torch.float32).to(self.device)

        # Reward Normalization (Batch-wise)
        # Update running stats
        batch_mean = reward_batch.mean()
        batch_std = reward_batch.std()
        self.reward_mean = 0.99 * self.reward_mean + 0.01 * batch_mean.item()
        self.reward_std = 0.99 * self.reward_std + 0.01 * (batch_std.item() + 1e-5)

        # Normalize rewards for training
        norm_rewards = (reward_batch - self.reward_mean) / (self.reward_std + 1e-5)
        # Clamp for stability
        norm_rewards = torch.clamp(norm_rewards, -10, 10)

        # Q(s, a)
        q_values = self.policy_net(state_batch).gather(1, action_batch)

        # Double DQN Target
        with torch.no_grad():
            next_actions = self.policy_net(next_batch).max(1)[1].unsqueeze(1)
            next_q_values = self.target_net(next_batch).gather(1, next_actions).squeeze(1)
            expected_q_values = (next_q_values * self.config.opt.gamma * (1 - done_batch)) + norm_rewards

        # TD Error for PER
        td_errors = (q_values.squeeze(1) - expected_q_values).abs().detach().cpu().numpy()
        self.memory.update_priorities(idxs, td_errors)

        # Loss with IS weights
        loss = (weights_batch * F.mse_loss(q_values, expected_q_values.unsqueeze(1), reduction='none').squeeze()).mean()

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.opt.grad_clip)

        self.optimizer.step()

        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_checkpoint(self, filepath, episode, supervisor_state=None):
        checkpoint = {
            'episode': episode,
            'steps_done': self.steps_done,
            'policy_net_state': self.policy_net.state_dict(),
            'target_net_state': self.target_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            # Saving memory is heavy, maybe skip or save separately?
            # For now skip saving memory to save disk/time
            'supervisor_state': supervisor_state,
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        if not os.path.exists(filepath):
            return 0, None

        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state'])
        self.target_net.load_state_dict(checkpoint['target_net_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.steps_done = checkpoint['steps_done']

        return checkpoint['episode'], checkpoint.get('supervisor_state')
