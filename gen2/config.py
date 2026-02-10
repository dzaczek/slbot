from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class EnvironmentConfig:
    frame_stack: int = 4
    frame_skip: int = 4
    resolution: Tuple[int, int] = (84, 84)
    grayscale: bool = False
    num_agents: int = 1
    view_first: bool = False

@dataclass
class ModelConfig:
    architecture: str = 'DuelingDQN'
    activation: str = 'LeakyReLU'
    negative_slope: float = 0.01
    fc_units: int = 512
    input_channels: int = 3

@dataclass
class OptimizationConfig:
    lr: float = 1e-4               # Increased from 6.25e-5
    weight_decay: float = 1e-5
    batch_size: int = 64
    gamma: float = 0.99
    grad_clip: float = 10.0
    eps_start: float = 1.0
    eps_end: float = 0.1           # Higher minimum exploration (was 0.05)
    eps_decay: int = 100000        # Faster decay (was 300000)
    target_update_freq: int = 1000 # More frequent updates (was 2000)
    max_episodes: int = 5000000
    checkpoint_every: int = 50
    reward_scale: float = 10.0

    # Autonomy / Stabilization
    scheduler_patience: int = 50
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6
    adaptive_eps_patience: int = 100
    super_pattern_enabled: bool = True
    super_pattern_window: int = 50
    super_pattern_wall_ratio: float = 0.55
    super_pattern_snake_ratio: float = 0.55
    super_pattern_food_ratio_low: float = 0.07
    super_pattern_food_ratio_high: float = 0.12
    super_pattern_penalty_step: float = 0.02
    super_pattern_reward_step: float = 0.5
    super_pattern_wall_penalty_cap: float = 0.3
    super_pattern_enemy_penalty_cap: float = 0.3
    super_pattern_straight_penalty_cap: float = 0.1
    super_pattern_food_reward_cap: float = 20.0

@dataclass
class ReplayBufferConfig:
    capacity: int = 100000
    prioritized: bool = True
    alpha: float = 0.6
    beta_start: float = 0.4
    beta_frames: int = 100000

@dataclass
class Config:
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    opt: OptimizationConfig = field(default_factory=OptimizationConfig)
    buffer: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)
