# Reward Seeking Improvement Plan

## Goal
The user requested the bot to "actively seek the nearest food".

## Changes Implemented

### 1. Vision Enhancement (Food Compass)
- **File:** `gen2/slither_env.py`
- **Method:** `_process_data_to_matrix`
- **Description:**
    - The environment now identifies the **nearest food pellet**.
    - If it is **on-screen**, it is drawn as a **large circle (radius 2.0)** on Channel 0 (Food), making it visually distinct from other food (single pixels).
    - If it is **off-screen**, a **Compass Marker** (radius 1.5) is drawn on the **edge of the 84x84 grid** pointing towards the food's location.
    - This provides a constant visual signal for the bot to follow, even when the screen appears empty.

### 2. Reward Shaping Boost
- **File:** `gen2/styles.py`
- **Description:**
    - Increased `food_shaping` (reward for reducing distance to food) from ~0.015 to **0.05** for "Standard" (Eat/Grow) and "Explorer" styles.
    - This creates a much stronger gradient, rewarding the bot significantly for every step taken towards food.

## Expected Outcome
The bot should now:
1.  **Turn towards food** immediately upon spawning or seeing the compass marker.
2.  **Prioritize nearest food** due to the visual highlighting.
3.  **Explore less randomly** and move with purpose towards off-screen food sources.
