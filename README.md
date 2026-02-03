# Slither.io NEAT Bot

A robust AI bot for Slither.io using NeuroEvolution of Augmenting Topologies (NEAT) and Selenium.

## Features

*   **Selenium Integration**: uses a "JS Bridge" to communicate with the game client efficiently.
*   **Optimized Performance**: Fetches all game data (player, enemies, food) in a single JavaScript execution per frame.
*   **Spatial Awareness**: Input system divides vision into 24 sectors and detects complex states like "encirclement/traps".
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

The bot will:
1.  Open a Chrome window navigating to Slither.io.
2.  Inject custom JavaScript to disable high-quality graphics and override mouse controls.
3.  Start the evolution process.
4.  Automatically restart the game upon death.

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
