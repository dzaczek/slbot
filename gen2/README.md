# Gen2: Matrix-Based Slither.io Bot

A Deep Reinforcement Learning agent for Slither.io using a Dueling DQN architecture with Prioritized Experience Replay (PER).

## Architecture

### Model (`gen2/model.py`)
- **Type**: Dueling Deep Q-Network (Dueling DQN).
- **Input**: `(Batch, 12, 84, 84)`
  - **Spatial**: 84x84 grid with 3 channels (Food, Danger/Enemies, Self).
  - **Temporal**: Stack of 4 frames (handled by `VecFrameStack`), resulting in 12 input channels.
- **CNN Backbone**:
  - Conv2d: 32 filters, 8x8 kernel, stride 4.
  - Conv2d: 64 filters, 4x4 kernel, stride 2.
  - Conv2d: 64 filters, 3x3 kernel, stride 1.
- **Dueling Heads**:
  - **Advantage Stream**: FC 512 -> Action Space (6).
  - **Value Stream**: FC 512 -> 1.
  - **Aggregation**: `Q = V + (A - mean(A))`.

### Algorithm
- **Policy**: Double DQN (DDQN) to reduce maximization bias.
- **Replay Buffer**: Prioritized Experience Replay (PER) using a SumTree structure to sample high-error transitions more frequently.
- **Optimization**: AdamW optimizer with Gradient Clipping (max_norm=10).

## Environment (`gen2/slither_env.py`)

The environment interfaces with the Slither.io game via Selenium and injected JavaScript.

### Observation Space
A 3-channel matrix (84x84) representing the agent's local view:
1. **Channel 0 (Food)**: Pellet locations (value 1.0).
2. **Channel 1 (Danger)**: Enemy snake bodies/heads (0.5/1.0) and Map Boundaries (1.0).
3. **Channel 2 (Self)**: Own snake body/head (0.5/1.0).

*Note: The view radius is dynamic. The matrix scales to fit the game's zoom level.*

### Action Space (Discrete: 6)
0. **Straight**: Maintain current heading.
1. **Turn Left (Small)**: ~40 degrees.
2. **Turn Right (Small)**: ~40 degrees.
3. **Turn Left (Big)**: ~103 degrees.
4. **Turn Right (Big)**: ~103 degrees.
5. **Boost**: Activate speed boost (consumes mass).

### Reward System (Curriculum Learning)
The agent progresses through stages defined in `gen2/trainer.py`:
1. **Stage 1 (EAT)**: High reward for food (+10), penalty for death (-15). Goal: Learn to gather mass.
2. **Stage 2 (SURVIVE)**: Increased death penalties (-100 wall, -20 enemy). Goal: Avoid collision.
3. **Stage 3 (GROW)**: Long-term survival and length maximization.

## Key Technical Features

### Sensor Logic (`gen2/browser_engine.py`)
- **Map Boundary**: Uses precise Polygon Boundary X (`window.pbx`) data from the game server if available, falling back to the exact Game Radius (`window.grd`). This eliminates "phantom wall" deaths caused by inaccurate safety buffers.
- **Snake Detection**:
  - Iterates `window.slithers` with an expanded search radius (5x view) to populate the matrix with off-screen enemies.
  - Supports up to 50 concurrent enemies (increased from legacy 15) to prevent "invisible snake" collisions in crowded areas.
- **Performance**: Game rendering is disabled/optimized via JS injection to maximize FPS for the headless browser.

### Visualization
- **View Plus**: A client-side canvas overlay (`--view-plus`) renders the agent's perception grid (Food/Enemies/Wall) in real-time at 60fps on top of the game view.

## Usage

### Prerequisites
- Python 3.8+
- `pip install torch selenium numpy pandas plotext`
- Chrome/Chromium installed.

### Training
Run the vectorized trainer (default 1 agent):
```bash
python gen2/trainer.py
```

### Options
- `--num_agents N`: Run `N` parallel browser instances.
- `--view`: Show the browser window for the first agent (others headless).
- `--view-plus`: Show browser + Debug Overlay (Perception Grid).
- `--resume`: Load checkpoint from `gen2/checkpoint.pth`.
- `--stage N`: Force start at specific curriculum stage (1, 2, or 3).

### Monitoring
Training stats are logged to `gen2/training_stats.csv`.
Visualize progress:
```bash
python gen2/analyze_matrix.py
```
