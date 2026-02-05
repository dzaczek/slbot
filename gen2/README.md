# Slither.io Gen2: Matrix-based Reinforcement Learning

## Overview
Gen2 implements a high-performance Reinforcement Learning (RL) agent for Slither.io, utilizing a "Matrix" representation of the game state (visual grid) rather than raw coordinates. It addresses the non-stationary and partially observable nature of the game through advanced Deep Q-Learning techniques.

## Architecture & Algorithms

### 1. Dueling Deep Q-Network (D3QN)
- **Concept:** Splits the neural network into two streams:
  - **Value Stream $V(s)$:** Estimates the intrinsic quality of the state.
  - **Advantage Stream $A(s, a)$:** Estimates the relative value of each action.
- **Aggregation:** $Q(s, a) = V(s) + (A(s, a) - \text{mean}(A(s, a)))$.
- **Benefit:** Faster convergence by learning state values regardless of action selection.
- **Activation:** `LeakyReLU` (slope 0.01) to prevent "Dying ReLU" in sparse reward environments.

### 2. Prioritized Experience Replay (PER)
- **Concept:** Samples transitions based on their **TD Error** (surprise factor) rather than uniformly.
- **Implementation:** SumTree data structure ($O(\log N)$ sampling).
- **Bias Correction:** Uses Importance Sampling weights in the loss function to correct for non-uniform sampling.
- **Benefit:** Agent learns significantly faster from rare, crucial events (e.g., death, eating).

### 3. Frame Stacking (POMDP Solution)
- **Problem:** A single frame lacks velocity and trajectory information (Partially Observable Markov Decision Process).
- **Solution:** Stacks the last **4 frames** as input channels.
- **Input Shape:** $(12, 84, 84)$ (3 channels [Food, Enemy, Self] $\times$ 4 frames).
- **Benefit:** Restores Markov property; allows detection of enemy speed and turn radius.

### 4. Optimization & Stability
- **Double DQN (DDQN):** Decouples action selection (Policy Net) from evaluation (Target Net) to reduce overestimation bias.
- **Reward Normalization:** Batch-wise normalization (mean 0, std 1) prevents gradient explosion from large food rewards.
- **Hardware Acceleration:** Auto-detects **CUDA** (NVIDIA), **MPS** (Apple Silicon), or **CPU**.

## Configuration (`config.py`)

Configuration is strictly typed via `dataclasses`.

- **Environment:**
  - `frame_stack`: 4 (Standard).
  - `resolution`: (84, 84).
- **Optimization:**
  - `lr`: $6.25 \times 10^{-5}$ (Precise rate for stability).
  - `batch_size`: 64.
  - `gamma`: 0.99 (Discount factor).
- **Buffer:**
  - `capacity`: 100,000 transitions.
  - `prioritized`: True.

## Optimization Scenarios & Troubleshooting

### Scenario A: Loss Does Not Decrease
- **Diagnosis:** Learning rate too high or data distribution too noisy.
- **Action:**
  1. Reduce `lr` in `config.py` to $1e-5$.
  2. Increase `batch_size` to 128 to smooth gradients.

### Scenario B: Q-Values Explode (Exponential Growth)
- **Diagnosis:** Gradient explosion or Reward Scaling issue.
- **Action:**
  1. Verify Reward Normalization is active.
  2. Reduce `grad_clip` (default 10.0) to 1.0.

### Scenario C: Agent Spins in Circles
- **Diagnosis:** "Reward Hacking" or Catastrophic Forgetting.
- **Action:**
  1. Increase `frame_stack` to 8 to better capture long-term context.
  2. Increase `gamma` to 0.995 to prioritize long-term survival over immediate movement.
  3. Ensure `epsilon` decay is not too fast (check `eps_decay`).

### Scenario D: Training is Too Slow
- **Diagnosis:** CPU bottleneck in environment or small Batch Size.
- **Action:**
  1. Increase `num_agents` in `train(args)` to utilize all CPU cores.
  2. Enable `MPS`/`CUDA` acceleration.

## Usage

```bash
# Start Training (Single Agent)
python gen2/trainer.py

# Start Training (8 Parallel Agents)
python gen2/trainer.py --num_agents 8

# Resume from Checkpoint
python gen2/trainer.py --resume
```
