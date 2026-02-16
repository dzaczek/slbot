import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
import os
import logging
from collections import deque

from config import Config
from model import DuelingDQN, HybridDuelingDQN
from per import PrioritizedReplayBuffer

logger = logging.getLogger("slitherbot")

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

        logger.info(f"Agent running on device: {self.device}")

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

        # LR scheduler removed — ReduceLROnPlateau is incompatible with RL
        # (rolling reward is too noisy to detect a real plateau early in training)

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

        # Dynamic gamma (set per curriculum stage)
        self.current_gamma = config.opt.gamma

        # N-step returns
        self.n_step = 3
        self.n_step_buffers = {}  # per-agent buffers: {agent_id: deque}

        # Reward Normalization Stats
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 1e-5 # avoid div by zero

    def get_epsilon(self):
        return self.config.opt.eps_end + (self.config.opt.eps_start - self.config.opt.eps_end) * \
            np.exp(-1. * self.steps_done / self.config.opt.eps_decay)

    def step_scheduler(self, metric):
        """No-op: LR scheduler removed (incompatible with RL)."""
        pass

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

        logger.info(f"  Boosting Exploration: Eps {self.get_epsilon():.3f} -> {target_eps:.3f} (Reset steps to {int(new_steps)})")
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
        # --- REFLEX LAYER ---
        # Hardcoded survival reflexes that override the network when danger is imminent.
        # The network still learns from the outcomes — reflexes just keep the bot alive
        # long enough to generate useful training data.
        if isinstance(state, dict) and 'sectors' in state:
            reflex_action = self._check_reflexes(state['sectors'])
            if reflex_action is not None:
                return reflex_action

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

    def _check_reflexes(self, sectors):
        """
        Emergency reflexes based on sector vector. Returns action or None.

        Sector layout (99 floats, alpha-5):
          [0..23]   food_score per sector (0=ahead, clockwise 15° each)
          [24..47]  obstacle_score per sector (1.0=touching, 0.0=clear)
          [48..71]  obstacle_type per sector (-1=none, 0=body/wall, 1=head)
          [72..95]  enemy_approach per sector (dot product, -1..+1)
          [96]      wall_dist_norm (dist_to_wall / 2000)
          [97]      snake_length_norm
          [98]      speed_norm

        Actions: 0=straight, 1/2=gentle L/R, 3/4=medium L/R,
                 5/6=sharp L/R, 7/8=uturn L/R, 9=boost
        """
        ns = 24  # num_sectors
        obstacle = sectors[ns:ns*2]       # obstacle_score per sector
        obs_type = sectors[ns*2:ns*3]     # obstacle_type per sector
        # Globals start at index ns*4 (after 4 per-sector features)
        wall_norm = sectors[ns * 4]       # [96] wall_dist_norm

        # --- REFLEX 1: Obstacle directly ahead (sectors 0, 23 = front ±15°) ---
        # If something is close in front, turn away hard
        front_danger = max(obstacle[0], obstacle[23], obstacle[1])
        if front_danger > 0.6:  # >0.6 means within ~800 units (40% of 2000 scope)
            # Pick the safer side — check left vs right obstacle density
            # Left = sectors 20-23 (−60° to 0°), Right = sectors 1-4 (0° to +60°)
            left_danger = sum(obstacle[20:24]) / 4.0
            right_danger = sum(obstacle[1:5]) / 4.0

            if front_danger > 0.85:  # Very close — U-turn
                return 7 if left_danger <= right_danger else 8
            else:  # Medium close — sharp turn
                return 5 if left_danger <= right_danger else 6

        # --- REFLEX 2: Wall proximity emergency ---
        # wall_norm < 0.15 means within 300 units of wall (out of 2000 scope)
        if wall_norm < 0.15:
            # Turn toward center — check which side has more open space
            left_obs = sum(obstacle[18:24]) / 6.0
            right_obs = sum(obstacle[0:6]) / 6.0
            return 7 if left_obs <= right_obs else 8

        # --- REFLEX 3: Enemy head approaching from front ---
        # Enemy heads (type=1) in front sectors are the most dangerous
        for s_i in [0, 23, 1, 22]:  # front ±30°
            if obs_type[s_i] == 1 and obstacle[s_i] > 0.4:  # head within ~1200 units
                left_danger = sum(obstacle[20:24]) / 4.0
                right_danger = sum(obstacle[1:5]) / 4.0
                return 5 if left_danger <= right_danger else 6

        return None  # No reflex triggered — let the network decide

    def remember(self, state, action, reward, next_state, done, gamma=None):
        """
        Stores transition with compression.
        state/next_state: dict {'matrix': (12,H,W) float32, 'sectors': (75,) float32}
        Stored as: (matrix_u8, sectors_f32) tuple for memory efficiency.
        gamma: the gamma used to compute n-step return (stored for consistency across stage changes).
        """
        if gamma is None:
            gamma = self.current_gamma

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

        self.memory.push(state_compressed, action, reward, next_compressed, done, gamma)

    def set_gamma(self, gamma):
        """Set gamma for current curriculum stage."""
        self.current_gamma = gamma
        logger.info(f"  Gamma set to {gamma} (effective n-step gamma: {gamma**self.n_step:.3f})")

    def remember_nstep(self, state, action, reward, next_state, done, agent_id=0):
        """
        N-step return buffer. Accumulates transitions and pushes
        n-step returns to PER when buffer is full or episode ends.
        """
        if agent_id not in self.n_step_buffers:
            self.n_step_buffers[agent_id] = deque(maxlen=self.n_step)

        buf = self.n_step_buffers[agent_id]
        buf.append((state, action, reward, next_state, done))

        if done:
            # Flush all remaining transitions in buffer
            self._flush_nstep(agent_id)
        elif len(buf) == self.n_step:
            # Buffer full: compute n-step return for oldest transition
            self._push_nstep_transition(agent_id)

    def _push_nstep_transition(self, agent_id):
        """Compute n-step return for oldest transition and push to PER."""
        buf = self.n_step_buffers[agent_id]
        if not buf:
            return

        # Oldest transition provides (state, action)
        state_0, action_0, _, _, _ = buf[0]

        # Snapshot gamma at write time — stored in PER for consistency
        gamma_used = self.current_gamma

        # Compute n-step discounted return: R = r1 + gamma*r2 + gamma^2*r3
        R = 0.0
        last_next_state = None
        last_done = False
        for i, (_, _, r, ns, d) in enumerate(buf):
            R += (gamma_used ** i) * r
            last_next_state = ns
            last_done = d
            if d:
                break

        self.remember(state_0, action_0, R, last_next_state, last_done, gamma=gamma_used)

    def _flush_nstep(self, agent_id):
        """Flush all remaining transitions at episode end."""
        buf = self.n_step_buffers[agent_id]
        while buf:
            self._push_nstep_transition(agent_id)
            buf.popleft()

    def optimize_model(self):
        if len(self.memory) < self.config.opt.batch_size:
            return None

        # Sample
        transitions, idxs, is_weights = self.memory.sample(self.config.opt.batch_size)

        # Unzip (6-element tuples: state, action, reward, next_state, done, gamma)
        batch_state, batch_action, batch_reward, batch_next, batch_done, batch_gamma = zip(*transitions)

        action_batch = torch.tensor(batch_action, dtype=torch.long).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(batch_reward, dtype=torch.float32).to(self.device)
        done_batch = torch.tensor(batch_done, dtype=torch.float32).to(self.device)
        weights_batch = torch.tensor(is_weights, dtype=torch.float32).to(self.device)
        gamma_batch = torch.tensor(batch_gamma, dtype=torch.float32).to(self.device)

        # Reward scaling (scale=1.0 preserves signal, clamp [-30,30] to preserve n-step returns)
        reward_scale = max(self.config.opt.reward_scale, 1.0)
        norm_rewards = torch.clamp(reward_batch / reward_scale, -30.0, 30.0)

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
                # Use per-transition gamma from PER (consistent with n-step return computation)
                gamma_n = gamma_batch ** self.n_step
                expected_q_values = (next_q_values * gamma_n * (1 - done_batch)) + norm_rewards
        else:
            # Legacy: plain uint8 arrays
            state_batch = torch.tensor(np.array(batch_state), dtype=torch.float32).to(self.device) / 255.0
            next_batch = torch.tensor(np.array(batch_next), dtype=torch.float32).to(self.device) / 255.0

            q_values = self.policy_net(state_batch).gather(1, action_batch)

            with torch.no_grad():
                next_actions = self.policy_net(next_batch).max(1)[1].unsqueeze(1)
                next_q_values = self.target_net(next_batch).gather(1, next_actions).squeeze(1)
                gamma_n = gamma_batch ** self.n_step
                expected_q_values = (next_q_values * gamma_n * (1 - done_batch)) + norm_rewards

        # TD Error for PER
        td_errors_raw = (q_values.squeeze(1) - expected_q_values).detach()
        td_errors = td_errors_raw.abs().cpu().numpy()
        self.memory.update_priorities(idxs, td_errors)

        # Loss with IS weights
        loss = (weights_batch * F.mse_loss(q_values, expected_q_values.unsqueeze(1), reduction='none').squeeze()).mean()

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient norm BEFORE clipping (diagnostic)
        grad_norm = nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.opt.grad_clip)

        self.optimizer.step()

        # Collect training metrics
        with torch.no_grad():
            q_vals_np = q_values.squeeze(1).detach().cpu().numpy()
            metrics = {
                'loss': loss.item(),
                'q_mean': float(np.mean(q_vals_np)),
                'q_max': float(np.max(q_vals_np)),
                'td_error_mean': float(np.mean(td_errors)),
                'grad_norm': float(grad_norm) if isinstance(grad_norm, (int, float)) else float(grad_norm.item()) if hasattr(grad_norm, 'item') else float(grad_norm),
            }

        return metrics

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
            logger.info(f"  Checkpoint: Policy net - new layers (randomly init): {len(missing_p)} params")
        if unexpected_p:
            logger.info(f"  Checkpoint: Policy net - dropped layers: {len(unexpected_p)} params")
        # Only load optimizer if architectures match (no missing keys)
        if not missing_p and not unexpected_p:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        else:
            logger.info(f"  Checkpoint: Architecture changed - optimizer reset to fresh state")
        self.steps_done = checkpoint['steps_done']

        max_steps = checkpoint.get('max_steps', 200)  # Default to 200 for old checkpoints
        run_uid = checkpoint.get('run_uid', None)
        return checkpoint['episode'], max_steps, checkpoint.get('supervisor_state'), run_uid
