# Slither.io Bot Project

This repository contains two generations of AI bots designed to play Slither.io.

*   **Gen 1**: Evolution-based (NEAT) - Good for discovering basic behaviors.
*   **Gen 2**: Deep Reinforcement Learning (DQN) - Modern approach using Convolutional Neural Networks (CNN) on visual grid data.

---

## Gen 2: Deep Q-Network (Recommended)

Located in `gen2/`. This is the current, active development branch.

### Features
*   **Matrix Vision**: The bot "sees" the game world as a 64x64 pixel grid with 3 channels (Food, Enemies/Walls, Self).
*   **Double DQN**: Uses a Dueling Double Deep Q-Network for stable learning.
*   **Parallel Training**: Runs multiple browser instances in parallel to gather experience faster.
*   **Supervisor System**: Automatically detects stagnation and adjusts learning rates or rolls back to previous best models.

### Quick Start (Gen 2)

```bash
cd gen2
python trainer.py --num_agents 2 --view
```
*   `--num_agents`: How many browsers to run in parallel.
*   `--view`: Shows the browser window for the first agent (others run headless).

---

## Gen 1: NEAT (Legacy)

Located in `gen1/`. This was the first attempt using NeuroEvolution.

### Features
*   **Sensor Inputs**: Uses 24 "rays" (sectors) to sense distance to food and enemies.
*   **Genetic Algorithm**: Evolves a population of neural networks over generations.
*   **Lightweight**: Runs on CPU efficiently.

### Usage (Gen 1)

```bash
cd gen1
python training_manager.py
```

---

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd slbot
    ```

2.  **Install Dependencies**:
    Requires Python 3.8+.
    ```bash
    pip install torch torchvision numpy selenium neat-python
    ```
    *Note: You also need Google Chrome installed.*

3.  **ChromeDriver**:
    Ensure `chromedriver` is compliant with your Chrome version. Selenium usually manages this automatically.

## Deployment (VPS / Headless)

For long-term training, we recommend running on a VPS.
See the **[VPS Deployment Guide](VPS_GUIDE.md)** for detailed instructions on setting up Xvfb and headless Chrome.

