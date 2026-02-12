import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
import os
from collections import deque

from config import Config
from model import DuelingDQN, HybridDuelingDQN
from per import PrioritizedReplayBuffer

class DDQNAgent:
    def __init__(self, config: Config):
        self.config = config

        # Device selection: CUDA -> MPS -> CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"Agent running on device: {self.device}")

        # Calculate input channels (3 base channels * frame_stack)
        self.input_channels = 3 * config.env.frame_stack
        self.input_size = config.env.resolution
        self.use_hybrid = config.model.architecture == 'HybridDuelingDQN'

        if self.use_hybrid:
            self.policy_net = HybridDuelingDQN(
                input_channels=self.input_channels,
                action_dim=10,
                input_size=self.input_size,
                sector_dim=config.model.sector_dim,
            ).to(self.device)
            self.target_net = HybridDuelingDQN(
                input_channels=self.input_channels,
                action_dim=10,
                input_size=self.input_size,
                sector_dim=config.model.sector_dim,
            ).to(self.device)
        else:
            self.policy_net = DuelingDQN(
                input_channels=self.input_channels,
                action_dim=10,
                input_size=self.input_size,
            ).to(self.device)
            self.target_net = DuelingDQN(
                input_channels=self.input_channels,
                action_dim=10,
                input_size=self.input_size,
            ).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(
            self.policy_net.parameters(),
            lr=config.opt.lr,
            weight_decay=config.opt.weight_decay
        )

        # Learning Rate Scheduler (Autonomy)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=config.opt.scheduler_factor,
            patience=config.opt.scheduler_patience,
            min_lr=config.opt.scheduler_min_lr
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

    def step_scheduler(self, metric):
        """Update LR based on metric (e.g. avg reward)."""
        self.scheduler.step(metric)

    def boost_exploration(self, target_eps=0.5):
        """Resets steps_done to boost epsilon back to target_eps."""
        start = self.config.opt.eps_start
        end = self.config.opt.eps_end
        decay = self.config.opt.eps_decay

        # Clamp target to be valid
        target_eps = max(end + 0.01, min(start, target_eps))

        ratio = (target_eps - end) / (start - end)
        if ratio <= 0:
            new_steps = decay * 10
        else:
            new_steps = -decay * np.log(ratio)

        print(f"  [Autonomy] Boosting Exploration: Eps {self.get_epsilon():.3f} -> {target_eps:.3f} (Reset steps to {int(new_steps)})")
        self.steps_done = int(new_steps)

    def _stack_frames(self, frames):
        """
        Stacks list of frames into a single numpy array.
        Each frame is (3, H, W).
        Output is (12, H, W).
        """
        return np.concatenate(frames, axis=0)

    def select_action(self, state):
        """
        state: dict {'matrix': (12, H, W), 'sectors': (75,)} or numpy array (12, H, W) for legacy.
        Note: steps_done is incremented externally by the trainer (once per batch step)
        to avoid N× decay with N parallel agents.
        """
        eps_threshold = self.get_epsilon()

        if random.random() > eps_threshold:
            with torch.no_grad():
                if self.use_hybrid:
                    mat_t = torch.tensor(state['matrix'], dtype=torch.float32).unsqueeze(0).to(self.device)
                    sec_t = torch.tensor(state['sectors'], dtype=torch.float32).unsqueeze(0).to(self.device)
                    q_values = self.policy_net(mat_t, sec_t)
                else:
                    state_arr = state['matrix'] if isinstance(state, dict) else state
                    state_t = torch.tensor(state_arr, dtype=torch.float32).unsqueeze(0).to(self.device)
                    q_values = self.policy_net(state_t)
                return q_values.max(1)[1].item()
        else:
            return random.randrange(10)

    def remember(self, state, action, reward, next_state, done):
        """
        Stores transition with compression.
        state/next_state: dict {'matrix': (12,H,W) float32, 'sectors': (75,) float32}
        Stored as: (matrix_u8, sectors_f32) tuple for memory efficiency.
        """
        if isinstance(state, dict):
            state_compressed = (
                (state['matrix'] * 255).astype(np.uint8),
                state['sectors'].astype(np.float32),
            )
            next_compressed = (
                (next_state['matrix'] * 255).astype(np.uint8),
                next_state['sectors'].astype(np.float32),
            )
        else:
            # Legacy: plain numpy array
            state_compressed = (state * 255).astype(np.uint8)
            next_compressed = (next_state * 255).astype(np.uint8)

        self.memory.push(state_compressed, action, reward, next_compressed, done)

    def optimize_model(self):
        if len(self.memory) < self.config.opt.batch_size:
            return None

        # Sample
        transitions, idxs, is_weights = self.memory.sample(self.config.opt.batch_size)

        # Unzip
        batch_state, batch_action, batch_reward, batch_next, batch_done = zip(*transitions)

        action_batch = torch.tensor(batch_action, dtype=torch.long).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(batch_reward, dtype=torch.float32).to(self.device)
        done_batch = torch.tensor(batch_done, dtype=torch.float32).to(self.device)
        weights_batch = torch.tensor(is_weights, dtype=torch.float32).to(self.device)

        # Reward scaling
        reward_scale = max(self.config.opt.reward_scale, 1.0)
        norm_rewards = torch.clamp(reward_batch / reward_scale, -5.0, 5.0)

        if self.use_hybrid:
            # Unpack tuples: (matrix_u8, sectors_f32)
            s_matrices = torch.tensor(np.array([s[0] for s in batch_state]), dtype=torch.float32).to(self.device) / 255.0
            s_sectors = torch.tensor(np.array([s[1] for s in batch_state]), dtype=torch.float32).to(self.device)
            n_matrices = torch.tensor(np.array([s[0] for s in batch_next]), dtype=torch.float32).to(self.device) / 255.0
            n_sectors = torch.tensor(np.array([s[1] for s in batch_next]), dtype=torch.float32).to(self.device)

            q_values = self.policy_net(s_matrices, s_sectors).gather(1, action_batch)

            with torch.no_grad():
                next_actions = self.policy_net(n_matrices, n_sectors).max(1)[1].unsqueeze(1)
                next_q_values = self.target_net(n_matrices, n_sectors).gather(1, next_actions).squeeze(1)
                expected_q_values = (next_q_values * self.config.opt.gamma * (1 - done_batch)) + norm_rewards
        else:
            # Legacy: plain uint8 arrays
            state_batch = torch.tensor(np.array(batch_state), dtype=torch.float32).to(self.device) / 255.0
            next_batch = torch.tensor(np.array(batch_next), dtype=torch.float32).to(self.device) / 255.0

            q_values = self.policy_net(state_batch).gather(1, action_batch)

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

    def save_checkpoint(self, filepath, episode, max_steps=None, supervisor_state=None, run_uid=None, parent_uid=None):
        checkpoint = {
            'episode': episode,
            'steps_done': self.steps_done,
            'policy_net_state': self.policy_net.state_dict(),
            'target_net_state': self.target_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            # Saving memory is heavy, maybe skip or save separately?
            # For now skip saving memory to save disk/time
            'max_steps': max_steps,  # Curriculum state
            'supervisor_state': supervisor_state,
            'run_uid': run_uid,
            'parent_uid': parent_uid,
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        if not os.path.exists(filepath):
            return 0, 200, None, None  # episode, max_steps (default), supervisor_state, run_uid

        checkpoint = torch.load(filepath, map_location=self.device)
        # strict=False allows loading old DuelingDQN weights into HybridDuelingDQN
        # (matching CNN layers transfer, new sector/merge layers stay randomly initialized)
        missing_p, unexpected_p = self.policy_net.load_state_dict(checkpoint['policy_net_state'], strict=False)
        missing_t, unexpected_t = self.target_net.load_state_dict(checkpoint['target_net_state'], strict=False)
        if missing_p:
            print(f"  [Checkpoint] Policy net - new layers (randomly init): {len(missing_p)} params")
        if unexpected_p:
            print(f"  [Checkpoint] Policy net - dropped layers: {len(unexpected_p)} params")
        # Only load optimizer if architectures match (no missing keys)
        if not missing_p and not unexpected_p:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        else:
            print(f"  [Checkpoint] Architecture changed - optimizer reset to fresh state")
        self.steps_done = checkpoint['steps_done']

        max_steps = checkpoint.get('max_steps', 200)  # Default to 200 for old checkpoints
        run_uid = checkpoint.get('run_uid', None)
        return checkpoint['episode'], max_steps, checkpoint.get('supervisor_state'), run_uid
