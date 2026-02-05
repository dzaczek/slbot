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
    lr: float = 6.25e-5
    batch_size: int = 64
    gamma: float = 0.99
    grad_clip: float = 10.0
    eps_start: float = 1.0
    eps_end: float = 0.1           # Higher minimum exploration (was 0.05)
    eps_decay: int = 300000        # Slower decay - 3x longer exploration (was 100000)
    target_update_freq: int = 2000
    max_episodes: int = 5000000
    checkpoint_every: int = 50

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
