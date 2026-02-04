# Slither.io NEAT Bot

A robust AI bot for Slither.io using NeuroEvolution of Augmenting Topologies (NEAT) and Selenium.

## Features

*   **Selenium Integration**: uses a "JS Bridge" to communicate with the game client efficiently.
*   **Optimized Performance**: Fetches all game data (player, enemies, food) in a single JavaScript execution per frame.
*   **Spatial Awareness**: 24-sector vision system (123 inputs total) sensing food density, wall distance, and enemy relative heading.
*   **Recurrent Neural Network**: Uses memory to perform complex maneuvers and plan paths.
*   **Dynamic Rewards**: Incentivizes survival (5x) and eating (20x) to balance growth and safety.
*   **NEAT Algorithm**: Evolves the neural network topology and weights to improve survival and length over generations.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd slbot
    ```

2.  **Install Dependencies**:
    Requires Python 3.8+.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: You also need Google Chrome installed.*

3.  **ChromeDriver**:
    Ensure `chromedriver` is compliant with your Chrome version. Selenium usually manages this, or you can use `webdriver-manager`.

## Usage

### Starting Training

**Option 1: Interactive Restart (RECOMMENDED)**
```bash
./restart_training.sh
```
This script will:
- Detect existing checkpoints
- Offer to backup old training and start fresh
- Or continue from last checkpoint

**Option 2: Direct Training**
```bash
python training_manager.py
```
Starts training and automatically resumes from the latest checkpoint if available.

**Option 3: Fresh Start (Manual)**
```bash
# Backup old training
mkdir backup_old
mv neat-checkpoint-* backup_old/
mv best_genome.pkl backup_old/

# Start fresh
python training_manager.py
```

### Playing the Best Bot

Watch your trained bot play:
```bash
python play_best.py
```

Or play from a specific checkpoint:
```bash
python play_best.py neat-checkpoint-50
```

### Analyzing Progress

Check training statistics:
```bash
python analyze_training.py
```

This shows:
- Average/max fitness over time
- Food eating statistics
- Causes of death
- Improvement trends

**Resuming Training**:
The bot automatically saves checkpoints (`neat-checkpoint-X`) every generation.
To resume, run `python training_manager.py` again. The script will detect the latest checkpoint and continue from where it left off.

## NEW: Double DQN Snake Agent (PyTorch)

A new, modern reinforcement learning agent has been added for a localized 20x20 Grid Snake environment.

### Prerequisites

You need to install PyTorch and Pygame:
```bash
pip install torch torchvision pygame numpy
```

### Running the DDQN Agent

To start training the new Double DQN agent:
```bash
python agent.py
```

### Architecture

*   **Algorithm**: Double Deep Q-Network (DDQN) with Polyack soft updates.
*   **Input**: Hybrid Input Model.
    *   **Visual**: A Convolutional Neural Network (CNN) processes the 20x20 Grid state (Head, Body, Food, Walls).
    *   **Orientation**: A Vector input captures the snake's current heading.
*   **Reward System**: Dense rewards with distance shaping (+/- 0.05 per step relative to food) and heavy penalties/rewards for death/eating.

### Configuration

Hyperparameters (Batch Size, Gamma, Epsilon) and Grid settings are located at the top of `agent.py`.

---

## Deployment (VPS / Headless)

For long-term training, we recommend running on a VPS.
See the **[VPS Deployment Guide](VPS_GUIDE.md)** for detailed instructions on setting up Xvfb and headless Chrome.

## Configuration (NEAT)

*   **`config_neat.txt`**: Adjust NEAT parameters (population size, mutation rates, etc.).
*   **`spatial_awareness.py`**: Modify logic for input calculation.
*   **`training_manager.py`**: Adjust fitness rewards, timeouts, and penalties.

### Recent Improvements (See IMPROVEMENTS.md for details)

Key changes for better learning:
- **150x reward for eating food** (was 25x) - bot now REALLY wants to eat
- **60s starvation timeout** (was 20s) - more time to learn
- **Better wall detection** - each sector independently checked
- **Penalties for bad behavior** - collision and starvation are punished
- **Larger population** - 50 genomes (was 30) for more diversity

## Architecture (NEAT)

*   **`browser_engine.py`**: Handles Selenium driver and JS injection.
*   **`spatial_awareness.py`**: Processes raw game data into neural network inputs.
*   **`ai_brain.py`**: Wrapper for the NEAT neural network.
*   **`training_manager.py`**: Main entry point and evolution loop.

## Contributing

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request
