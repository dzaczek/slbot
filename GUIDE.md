# Slither.io NEAT Bot (Gen 1) - Complete Step-by-Step Guide

**NOTE: This guide applies to the Legacy NEAT Bot located in the `gen1/` directory.**
For the modern Deep Q-Network (DQN) bot, please refer to the `gen2/` directory and `README.md`.

This guide explains every component of the Gen 1 bot in detail, including how JavaScript injection works, how control is hijacked, and how to run training.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Project Structure](#project-structure)
3. [How JavaScript Injection Works](#how-javascript-injection-works)
4. [Control Hijacking Explained](#control-hijacking-explained)
5. [Data Extraction (JS Bridge)](#data-extraction-js-bridge)
6. [Auto-Restart Mechanism](#auto-restart-mechanism)
7. [Neural Network Inputs](#neural-network-inputs)
8. [Running Training](#running-training)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before running the bot, ensure you have:

1. **Python 3.8+** installed
2. **Google Chrome** browser installed
3. **ChromeDriver** matching your Chrome version (Selenium 4.6+ auto-manages this)

### Setup Virtual Environment

```bash
cd /path/to/slbot
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Project Structure

```
slbot/
├── gen1/
│   ├── config_neat.txt        # NEAT algorithm configuration
│   ├── browser_engine.py      # Selenium + JS injection logic
│   ├── spatial_awareness.py   # Converts game state to NN inputs
│   ├── ai_brain.py            # NEAT network wrapper
│   ├── training_manager.py    # Main training loop
│   └── smoke_test.py          # Verification script
├── gen2/                      # New DQN Bot (see README)
└── README.md                  # Main documentation
```

---

## How JavaScript Injection Works

### What is JS Injection?

Slither.io has no official API. The game runs entirely in JavaScript in your browser. We use **Selenium WebDriver** to:

1. Open a real Chrome browser
2. Navigate to `slither.io`
3. Execute custom JavaScript code inside the game's context

### Automatic Injection

The injection happens **automatically** when the browser starts. Here's the flow:

```
SlitherBrowser.__init__()
    │
    ├── Opens Chrome with Selenium
    ├── Navigates to slither.io
    ├── Waits 2 seconds for game to load
    └── Calls inject_override_script()  ← JS is injected here
```

### The Injection Code (Explained Line by Line)

Located in `gen1/browser_engine.py`, method `inject_override_script()`:

```javascript
// STEP 1: Reduce graphics quality (optional optimization)
if (window.want_quality) { window.want_quality = 0; }
if (window.high_quality) { window.high_quality = false; }
```
- The game stores quality settings in `window.want_quality` and `window.high_quality`
- Setting them to 0/false reduces GPU load during training

```javascript
// STEP 2: Disable real mouse control
window.onmousemove = function(e) { 
    return;  // Do nothing - ignore actual mouse movements
};
```
- Slither.io normally tracks your mouse via `onmousemove` event
- By overwriting this function to do nothing, we "disconnect" the real mouse
- Now the game only responds to our programmatic controls

```javascript
// STEP 3: Initialize control variables
if (typeof window.xm === 'undefined') window.xm = 0;
if (typeof window.ym === 'undefined') window.ym = 0;
```
- `window.xm` and `window.ym` are the target coordinates the snake moves toward
- We ensure they exist so we can write to them later

```javascript
// STEP 4: Create restart helper
window.force_connect_game = function() {
    if (window.connect) {
        window.connect();
    }
};
```
- `window.connect()` is the game's internal function to start/restart a game
- We wrap it for easy access from Python

---

## Control Hijacking Explained

### How Slither.io Steering Works

The game's internal loop does something like:

```javascript
// Simplified game loop (internal to slither.io)
function gameLoop() {
    // Calculate direction based on xm, ym relative to snake head
    var dx = window.xm - snake.xx;
    var dy = window.ym - snake.yy;
    var targetAngle = Math.atan2(dy, dx);
    
    // Gradually turn snake toward targetAngle
    snake.ang = lerpAngle(snake.ang, targetAngle, turnSpeed);
}
```

### Our Control Method

In `gen1/browser_engine.py`, method `send_action(angle, boost)`:

```python
def send_action(self, angle, boost):
    # Convert neural network output (0-1) to radians (0 to 2π)
    target_angle = angle * 2 * math.pi
    
    # Inject JS to set control variables
    control_js = f"""
    var ang = {target_angle};
    var boost = {1 if boost > 0.5 else 0};
    
    if (window.snake) {{
        // Calculate a point 200 units in the desired direction
        var dist = 200;
        window.xm = window.snake.xx + Math.cos(ang) * dist;
        window.ym = window.snake.yy + Math.sin(ang) * dist;
        
        // Activate boost if requested
        window.setAcceleration(boost); 
    }}
    """
    self.driver.execute_script(control_js)
```

**How it works:**
1. Neural network outputs a value 0-1
2. We convert to radians: `angle * 2π`
3. We calculate a target point 200 units away in that direction
4. We set `window.xm` and `window.ym` to that point
5. The game's loop sees these values and steers the snake accordingly

---

## Data Extraction (JS Bridge)

### The Problem

Calling `driver.execute_script()` has overhead (~5-20ms per call). If we called it 10 times per frame for different data, we'd be too slow.

### The Solution: Single JSON Batch

In `gen1/browser_engine.py`, method `get_game_data()`:

```javascript
function getGameState() {
    // Check if dead
    var is_dead = false;
    if (!window.snake || (window.dead_mtm && window.dead_mtm !== -1)) {
         is_dead = true;
    }
    if (is_dead) { return { dead: true }; }

    // 1. My Snake Data
    var my_snake = {
        x: window.snake.xx,      // X position (interpolated)
        y: window.snake.yy,      // Y position
        ang: window.snake.ang,   // Current heading (radians)
        sp: window.snake.sp,     // Speed
        len: window.snake.pts ? window.snake.pts.length : 0
    };

    // 2. Visible Foods
    var visible_foods = [];
    if (window.foods && window.foods.length) {
        for (var i = 0; i < window.foods.length; i++) {
            var f = window.foods[i];
            if (f && f.rx) {
                 visible_foods.push([f.rx, f.ry, f.sz || 1]);
            }
        }
    }

    // 3. Enemy Snakes
    var visible_enemies = [];
    if (window.snakes && window.snakes.length) {
        for (var i = 0; i < window.snakes.length; i++) {
            var s = window.snakes[i];
            if (s === window.snake) continue;  // Skip self
            
            var pts = [];
            if (s.pts) {
                for (var j = 0; j < s.pts.length; j++) {
                    var p = s.pts[j];
                    if (p.x !== undefined) pts.push([p.x, p.y]);
                    else if (p.xx !== undefined) pts.push([p.xx, p.yy]);
                }
            }
            visible_enemies.push({
                id: s.id, x: s.xx, y: s.yy,
                ang: s.ang, sp: s.sp, pts: pts
            });
        }
    }

    return {
        dead: false,
        self: my_snake,
        foods: visible_foods,
        enemies: visible_enemies
    };
}
return getGameState();
```

**Result:** One call returns ALL data as a Python dictionary.

---

## Auto-Restart Mechanism

When the snake dies, we detect it and restart instantly without reloading the page:

```python
# In gen1/training_manager.py
data = browser.get_game_data()
if data.get('dead', False):
    # Snake died - calculate fitness and restart
    break  # Exit game loop

# After loop ends:
browser.force_restart()  # Calls window.connect() via JS
```

**Detection method:**
- `window.snake` becomes `null` when dead
- `window.dead_mtm` is set to a timestamp when death occurs

---

## Neural Network Inputs

The network receives **145 inputs** (24 sectors × 6 features + 1 bias):

### Sector Division

The 360° view around the snake is divided into 24 equal sectors (15° each):

```
        Sector 0 (front)
             ▲
    Sec 23  │  Sec 1
        ╲   │   ╱
         ╲  │  ╱
   Sec 22 ──●── Sec 2   (snake head at center)
         ╱  │  ╲
        ╱   │   ╲
    Sec 21  │  Sec 3
             ▼
        Sector 12 (back)
```

### Features per Sector (6 total)

| Index | Feature | Value Range | Meaning |
|-------|---------|-------------|---------|
| 0 | Food Distance | 0.0 - 1.0 | 1.0 = food very close, 0.0 = no food |
| 1 | Enemy Distance | 0.0 - 1.0 | 1.0 = enemy head very close |
| 2 | Is Enemy Big | 0 or 1 | 1 = enemy has >50 body segments |
| 3 | Is Enemy Boosting | 0 or 1 | 1 = enemy is accelerating |
| 4 | Trap Detected | 0 or 1 | 1 = being encircled by enemy body |
| 5 | Wall Distance | 0.0 - 1.0 | 1.0 = very close to map border |

### Trap Detection Algorithm

Located in `gen1/spatial_awareness.py`, method `detect_encirclement()`:

```python
def detect_encirclement(self, my_pos, enemy_pts):
    """
    Checks if enemy body points surround us.
    
    Algorithm:
    1. Calculate angle from us to each enemy body segment
    2. Sort all angles
    3. Find the largest gap between consecutive angles
    4. If largest gap < 90° → we're surrounded by >270° → TRAP!
    """
    angles = []
    for point in enemy_pts:
        if distance(my_pos, point) < 600:  # Only nearby points
            angle = atan2(point.y - my_pos.y, point.x - my_pos.x)
            angles.append(angle)
    
    angles.sort()
    
    # Find largest angular gap
    max_gap = 0
    for i in range(len(angles)):
        gap = angles[(i+1) % len(angles)] - angles[i]
        if gap < 0: gap += 2π
        max_gap = max(max_gap, gap)
    
    # If gap < 90°, we're encircled
    return 1.0 if max_gap < (π/2) else 0.0
```

---

## Running Training

### Step 1: Activate Virtual Environment

```bash
cd /path/to/slbot
source venv/bin/activate
```

### Step 2: Start Training

```bash
cd gen1
python training_manager.py
```

### What Happens:

1. Chrome window opens and navigates to slither.io
2. JS injection occurs automatically
3. For each genome in the population (150 by default):
   - Game starts via `window.connect()`
   - Bot plays until death or timeout
   - Fitness = survival_time + (snake_length × 10)
4. After all genomes evaluated, NEAT evolves the population
5. Process repeats for 50 generations

### Expected Output:

```
 ****** Running generation 0 ****** 
Genome 1 Fitness: 45.2
Genome 2 Fitness: 112.7
Genome 3 Fitness: 23.1
...
Population's average fitness: 67.4
Best fitness: 325.8
...
```

---

## Troubleshooting

### "No module named 'neat'"
```bash
source venv/bin/activate
pip install neat-python
```

### Chrome doesn't open
- Ensure Chrome is installed
- Try: `pip install webdriver-manager` and modify `gen1/browser_engine.py` to use it

### Game doesn't start
- The "Play" button might need clicking first time
- Modify `force_restart()` to click the play button if needed

### Snake doesn't move
- Check browser console (F12) for JS errors
- Verify `window.snake` exists after game starts

---

## Next Steps

After basic training works:

1. **Tune fitness function** - Add rewards for eating, penalties for collisions
2. **Adjust NEAT config** - Modify mutation rates, population size
3. **Save checkpoints** - Use `neat.Checkpointer` to save progress
4. **Visualize best genome** - Use `neat.visualize` module

---

*Created for the Slither.io NEAT Bot project (Gen 1)*