# Repair Plan: Gen2 Slither Agent (DQN)

This document outlines a concrete repair plan for the Gen2 (DQN-based) Slither.io agent. The current system suffers from inaccurate death classification, incorrect arena mapping, and non-proportional snake rendering.

## Goal
Create a robust, deterministic environment for training the Dueling DQN agent, ensuring >95% accuracy in death classification and correct spatial representation of the game world.

## Problems Observed
1.  **Death Classification:** Agent cannot reliably distinguish between Wall Death (penalty -100) and Enemy Collision (penalty -20). Current heuristic relies on `dist_to_wall` and `min_enemy_dist` which are noisy.
2.  **Arena Mapping:** Wall boundary is mapped incorrectly. `window.pbx` (polygon) and `window.grd` (radius) are available in JS but not consistently used in the Python environment for wall masks.
3.  **Snake Sizing:** Snake thickness on the minimap (matrix) is a rough approximation (`29 * scale`), leading to false positives/negatives in collision learning.
4.  **Input Resolution:** Vision size is fixed at 84x84. Need configurable sizes (64, 128, 256) to test impact on performance.
5.  **Logging:** Application logs (browser/selenium) and Training logs (rewards/loss) are mixed in stdout.

## Acceptance Criteria
-   **Death Classification Accuracy:** ≥ 95% on sampled episodes (Wall vs Enemy).
-   **Wall Mask:** Correctly aligns with the playable area boundary across all zoom levels.
-   **Snake Thickness:** Reflects actual collision geometry (World Units -> Grid Pixels).
-   **Configurable Vision:** `--vision_size 64|128|256` supported.
-   **Separated Logs:** `logs/app.log` and `logs/train.log`.
-   **Debug Artifacts:** Save "Death Event Packet" (JSON + Image) for every terminal event.

## Implementation Phases

### Phase 0: Instrumentation (Debug without Guessing)
*Objective: establish ground truth for death events.*
-   [ ] Define `DeathEvent` structure (JSON):
    -   Timestamp, Episode ID
    -   Cause (Wall/Enemy/Unknown)
    -   Location (x, y)
    -   Nearest Enemy Dist, Wall Dist
    -   Game State (Scale, View Radius)
-   [ ] Implement `save_death_packet` in `SlitherEnv`.
    -   Save JSON + PNG of the last frame (grid overlay).
-   [ ] Add counters in `Trainer` for `deaths_wall`, `deaths_enemy`, `deaths_unknown`.

### Phase 1: Arena Perimeter Mapping (Core)
*Objective: Fix the spatial representation of the world.*
-   [ ] Create `gen2/coord_transform.py`:
    -   Centralize World <-> Grid coordinate transformations.
    -   Implement `WorldToGrid(wx, wy, center_x, center_y, scale) -> (gx, gy)`.
-   [ ] Update `browser_engine.py`:
    -   Ensure `window.pbx`, `window.pby` (Polygon) and `window.grd` (Radius) are extracted reliably in `get_game_data`.
    -   Pass exact boundary data to `SlitherEnv`.
-   [ ] Update `SlitherEnv._process_data_to_matrix`:
    -   Render the Wall Channel (Channel 1) using the exact boundary data.
    -   Use `cv2.fillPoly` or robust rasterization for polygon walls.
    -   Add a visual safety margin (e.g., 5-10 units) to the wall mask.

### Phase 2: Fix Snake Sizing on Grid
*Objective: align visual representation with collision physics.*
-   [ ] Research Slither.io collision radius:
    -   Verify if `width = 29 * sc` is accurate.
    -   Investigate `sp` (speed) impact on width (boosting makes snake thinner).
-   [ ] Update `_draw_thick_line` in `SlitherEnv`:
    -   Accept `width_world` instead of `width_pixels`.
    -   Calculate `width_pixels = width_world * scale`.
    -   Ensure segments overlap correctly to form a continuous body.
-   [ ] Validation:
    -   Overlay the generated grid on top of the actual game screenshot (using `view-plus` or post-hoc analysis) to verify alignment.

### Phase 3: Deterministic Death Reason
*Objective: Eliminate ambiguity in reward assignment.*
-   [ ] JS Hook (Preferred):
    -   Attempt to hook `window.on_death` or similar in `browser_engine.py` to get the game's internal death reason.
-   [ ] Geometric Fallback (Python):
    -   If JS hook fails, implement strict geometric check at the last alive frame:
        -   `if dist_to_wall <= head_radius + buffer: CAUSE = WALL`
        -   `elif dist_to_enemy <= head_radius + buffer: CAUSE = ENEMY`
        -   `else: CAUSE = UNKNOWN` (or fallback to Enemy if very close).

### Phase 4: Increase Input Resolution / Multi-Size Training
*Objective: Enable higher fidelity perception.*
-   [ ] Update `Config` in `gen2/config.py`:
    -   Add `resolution: Tuple[int, int]` to `EnvironmentConfig`.
-   [ ] Update `DuelingDQN` in `gen2/model.py`:
    -   Ensure `flat_size` calculation in `__init__` correctly handles variable input sizes (64, 84, 128, 256).
    -   Test with different resolutions.
-   [ ] Update `Trainer` CLI:
    -   Add `--vision-size` argument (default 84).
-   [ ] Replay Buffer Consideration:
    -   Store observations at the captured resolution. (Resizing on the fly might be too slow for training loop, better to capture at desired res).

### Phase 5: Training Improvements
*Objective: Leverage the accurate environment for better learning.*
-   [ ] Enforce distinct penalties:
    -   Wall Death: -100 (Prevent suicide)
    -   Enemy Death: -20 (Acceptable risk)
-   [ ] Add Auxiliary Inputs (Optional):
    -   Distance to Wall (normalized scalar) as a separate input or channel?
    -   Current Length (normalized scalar).

### Phase 6: Logging Split and Run Structure
*Objective: Clean up operations.*
-   [ ] Configure Python `logging`:
    -   `logs/app.log`: Browser events, connection issues, specific game vars.
    -   `logs/train.log`: Training metrics (Episode, Reward, Loss, Epsilon).
-   [ ] Update `Trainer`:
    -   Use `logger.info()` instead of `print()`.
    -   Implement RotatingFileHandler to prevent huge logs.
-   [ ] Standardize Run Directory:
    -   `runs/YYYYMMDD_HHMMSS/`
        -   `config.yaml`
        -   `checkpoint.pth`
        -   `events/` (Death packets)
        -   `logs/`

## Execution Strategy
1.  **Phase 0** is the immediate next step.
2.  **Phase 1 & 2** can be developed in parallel as they touch different rendering logic.
3.  **Phase 3** depends on Phase 1 & 2 being reliable.
4.  **Phase 4, 5, 6** are enhancements once the core loop is stable.
