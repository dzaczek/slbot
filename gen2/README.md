# Gen2: Slither.io Deep RL Bot

A Deep Reinforcement Learning agent for Slither.io using a Hybrid Dueling DQN architecture with Prioritized Experience Replay (PER), egocentric observation, and curriculum learning.

## Architecture

### Model (`model.py`) — HybridDuelingDQN
- **Hybrid Input**:
  - **CNN Branch**: 64x64 matrix, 4-frame stack (12 input channels)
    - Conv2d: 32 filters, 8x8, stride 4
    - Conv2d: 64 filters, 4x4, stride 2
    - Conv2d: 64 filters, 3x3, stride 1
    - Flatten: 1024
  - **Sector Branch**: 75-float vector (24 egocentric sectors x 3 features + 3 globals)
    - FC 75 → 128 → 128
  - **Merge**: concat(1152) → 512 → Dueling heads
- **Dueling Heads**:
  - Advantage: 512 → 10 actions
  - Value: 512 → 1
  - Q = V + (A - mean(A))
- **Activation**: LeakyReLU throughout

### Algorithm
- **Double DQN** to reduce maximization bias
- **Prioritized Experience Replay** (PER) with SumTree
- **N-step returns** with variable gamma stored in replay buffer
- **AdamW** optimizer, fixed LR=1e-4, gradient clipping (max_norm=1)

### Observation
- **Matrix** (3 channels, 64x64): Food / Danger / Self — egocentric (heading = up)
- **Sectors** (75 floats): 24 sectors x 15° x (food_score, obstacle_score, obstacle_type) + 3 globals
- Frame stack of 4 → 12-channel CNN input

### Action Space (10 discrete)
| Action | Description | Angle |
|--------|-------------|-------|
| 0 | Straight | 0° |
| 1-2 | Gentle L/R | ~20° |
| 3-4 | Medium L/R | ~40° |
| 5-6 | Sharp L/R | ~69° |
| 7-8 | U-turn L/R | ~103° |
| 9 | Boost | - |

## Environment (`slither_env.py`)

### Browser Engine (`browser_engine.py`)
Selenium-based headless Chrome automation with injected JavaScript bridge.

**Optimizations** (v2.0):
- Persistent JS function injection — `_botGetState()` and `_botActAndRead()` injected once at startup, subsequent calls send ~30 bytes instead of ~8KB inline JS
- Combined action+read in single `execute_script` call
- Data caching between steps (reuses post-action data as next pre-action state)
- Single-send frame skip (1 send + 40ms wait vs 4x send + 80ms)
- **Step time: ~52ms** (was ~110ms, 2.1x improvement)

### Egocentric Rotation
All observations are rotated so the snake's heading points "up" in the matrix. This ensures consistent input regardless of absolute direction.

### Sensor Features
- **Map Boundary**: Circle boundary via `grd` (map grid radius), `cst` (custom scale)
- **Food**: Up to 300 items, sorted by distance, within view radius
- **Enemies**: Up to 50 snakes with up to 150 body points each, expanded search radius (5x view)
- **Wall Distance**: Calculated from snake position and map radius

## Training (`trainer.py`)

### Curriculum Learning
Progression through stages with configurable reward parameters:

| Stage | Name | Focus | Max Steps |
|-------|------|-------|-----------|
| 1 | FOOD_VECTOR | Eat food, basic movement | 300 |
| 2 | WALL_AVOID | Wall avoidance + food | 500 |
| 3 | SURVIVE | Full survival + growth | 1000+ |

Promotion criteria: compound (avg_steps threshold AND wall_death rate).

### Styles (`styles.py`)
Training styles with different reward configurations:
- **Standard** (Curriculum) — progressive difficulty
- **Aggressive** — high food reward, low survival
- **Defensive** — high survival, wall avoidance
- **Explorer** — exploration bonus

### Multi-Agent Training
- `SubprocVecEnv` for parallel agents (multiprocessing)
- Auto-scaling based on CPU/RAM/step-time metrics
- Each agent runs in a separate Chrome instance

### Backend Selection
```bash
# Selenium (default)
python trainer.py --backend selenium

# WebSocket (experimental — currently blocked by server anti-bot)
python trainer.py --backend websocket --ws-server-url ws://IP:PORT/slither
```

### UID System
Each training run gets a unique ID (`YYYYMMDD-8hexchars`). Stored in checkpoints and CSV for lineage tracking.

## Usage

### Prerequisites
- Python 3.10+
- Chrome/Chromium installed
- Dependencies:
```bash
pip install torch selenium numpy pandas matplotlib
```

### Training
```bash
# Basic (1 agent, Selenium)
python trainer.py

# Multiple agents with auto-scaling
python trainer.py --num_agents 3 --auto-num-agents --max-agents 10

# With visualization
python trainer.py --view-plus

# Resume from checkpoint
python trainer.py --resume

# Force curriculum stage
python trainer.py --stage 2

# Custom game server
python trainer.py --url http://slither.io

# Fresh start (delete all logs/checkpoints)
python trainer.py --reset
```

### CLI Options
| Flag | Description |
|------|-------------|
| `--num_agents N` | Number of parallel browser agents |
| `--view` | Show browser for first agent |
| `--view-plus` | Browser + debug overlay |
| `--resume` | Load from `checkpoint.pth` |
| `--stage N` | Force curriculum stage (1/2/3) |
| `--style-name NAME` | Training style |
| `--url URL` | Game server URL |
| `--vision-size N` | Override resolution |
| `--backend selenium\|websocket` | Browser backend |
| `--ws-server-url URL` | WebSocket server override |
| `--auto-num-agents` | Enable auto-scaling |
| `--max-agents N` | Max agents for auto-scale |
| `--reset` | Full reset (logs, CSV, checkpoints) |

### Analysis
```bash
# Generate training progress report with charts
python training_progress_analyzer.py --latest

# Filter by run UID
python training_progress_analyzer.py --uid 20260214-a3f7b2c1
```

## File Structure

| File | Description |
|------|-------------|
| `trainer.py` | Main training loop, SubprocVecEnv, auto-scaling |
| `agent.py` | DQN agent (select_action, optimize_model) |
| `model.py` | HybridDuelingDQN network |
| `slither_env.py` | RL environment (obs processing, rewards, curriculum) |
| `browser_engine.py` | Selenium Chrome automation + JS bridge |
| `browser_engine_ws.py` | WebSocket browser adapter (experimental) |
| `ws_engine.py` | Native WebSocket client (experimental) |
| `ws_protocol.py` | Binary protocol encoding/decoding |
| `config.py` | Configuration dataclasses |
| `styles.py` | Training style definitions |
| `per.py` | Prioritized Experience Replay (SumTree) |
| `coord_transform.py` | World-to-grid coordinate transforms |
| `training_progress_analyzer.py` | Training analysis + chart generation |
| `training_stats.csv` | Episode-level training metrics |

## Performance

| Metric | Value |
|--------|-------|
| Step time (Selenium optimized) | ~52ms |
| Step time (Selenium original) | ~110ms |
| RAM per agent (Selenium) | ~500MB |
| Agents per machine (8GB RAM) | 5-10 |
| Training speed (1 agent) | ~19 steps/sec |
