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

To start the training loop:

```bash
python training_manager.py
```

3.  Start the evolution process.
4.  Automatically restart the game upon death.

**Resuming Training**:
The bot automatically saves checkpoints (`neat-checkpoint-X`) every 5 generations.
To resume, simply run `python training_manager.py` again. The script will detect the latest checkpoint and continue from where it left off.

## Deployment (VPS / Headless)

For long-term training, we recommend running on a VPS.
See the **[VPS Deployment Guide](VPS_GUIDE.md)** for detailed instructions on setting up Xvfb and headless Chrome.

## Configuration

*   **`config_neat.txt`**: Adjust NEAT parameters (population size, mutation rates, etc.).
*   **`spatial_awareness.py`**: Modify logic for input calculation.

## Architecture

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
