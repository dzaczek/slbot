from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class EnvironmentConfig:
    frame_stack: int = 4
    frame_skip: int = 4
    resolution: Tuple[int, int] = (64, 64)
    grayscale: bool = False
    num_agents: int = 1
    view_first: bool = False

@dataclass
class ModelConfig:
    architecture: str = 'HybridDuelingDQN'
    activation: str = 'LeakyReLU'
    negative_slope: float = 0.01
    fc_units: int = 512
    input_channels: int = 3
    sector_dim: int = 75
    sector_scope: float = 1500.0
    num_sectors: int = 24

@dataclass
class OptimizationConfig:
    lr: float = 1e-4               # Increased from 6.25e-5
    weight_decay: float = 1e-5
    batch_size: int = 64
    gamma: float = 0.8             # Starting gamma (Stage 1), overridden per-stage
    grad_clip: float = 10.0
    eps_start: float = 1.0
    eps_end: float = 0.08          # Less randomness at convergence
    eps_decay: int = 40000         # Faster transition to exploitation
    target_update_freq: int = 500  # More frequent target updates
    max_episodes: int = 5000000
    checkpoint_every: int = 50
    reward_scale: float = 10.0

    # Autonomy / Stabilization
    scheduler_patience: int = 150
    scheduler_factor: float = 0.7
    scheduler_min_lr: float = 1e-5
    adaptive_eps_patience: int = 200
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
    super_pattern_food_reward_cap: float = 15.0

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
