#!/usr/bin/env python3
"""
Steering Validation Test for SlitherBot.

Executes a deterministic sequence of turns and validates that the snake's
actual movement direction matches the commanded direction. No model, no
training — pure deterministic test.

Usage: python test_steering.py
"""

import math
import time
import logging
import os
import sys

# ── Logging ──────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
log_path = os.path.join("logs", "steering_test.log")

logger = logging.getLogger("steering_test")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(log_path, mode="w")
fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(fh)
logger.addHandler(ch)

# ── Action → angle-change mapping (from slither_env.py) ─────────────────────
ACTION_ANGLE = {
    0: 0.0,      # straight
    1: -0.35,    # L gentle  (~20°)
    2:  0.35,    # R gentle  (~20°)
    3: -0.7,     # L medium  (~40°)
    4:  0.7,     # R medium  (~40°)
    5: -1.2,     # L sharp   (~69°)
    6:  1.2,     # R sharp   (~69°)
    7: -1.8,     # L u-turn  (~103°)
    8:  1.8,     # R u-turn  (~103°)
    9: 0.0,      # boost (straight)
}

ACTION_NAMES = {
    0: "FWD", 1: "L1", 2: "R1", 3: "L2", 4: "R2",
    5: "L3", 6: "R3", 7: "LU", 8: "RU", 9: "BST",
}

# ── Test sequence ────────────────────────────────────────────────────────────
# (action_id, num_steps, description)
TEST_PHASES = [
    (0, 30, "Forward baseline"),
    (4, 15, "Medium right (+40°)"),
    (0, 20, "Hold heading"),
    (5, 15, "Sharp left (-69°)"),
    (0, 20, "Hold heading"),
    (8, 10, "Right u-turn (+103°)"),
    (0, 30, "Hold heading"),
    (7, 10, "Left u-turn (-103°)"),
    (0, 20, "Hold heading"),
]

FRAME_SKIP = 8
TOTAL_DURATION = 120  # seconds
PASS_THRESHOLD_FWD = 30.0   # degrees
PASS_THRESHOLD_TURN = 45.0  # degrees


def normalize_angle(a):
    """Normalize angle to [-pi, pi]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def angle_diff_deg(a, b):
    """Signed angular difference in degrees, normalized to [-180, 180]."""
    d = normalize_angle(a - b)
    return math.degrees(d)


def wait_for_alive(browser, timeout=20):
    """Block until the snake is alive or timeout."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        data = browser.get_game_data()
        if data and not data.get("dead", True):
            return True
        time.sleep(0.5)
    return False


def main():
    from browser_engine import SlitherBrowser

    logger.info("=" * 60)
    logger.info("STEERING VALIDATION TEST")
    logger.info("=" * 60)

    browser = SlitherBrowser(headless=False, nickname="SteerTest")
    logger.info("Browser opened, logging in...")

    if not browser._handle_login():
        logger.error("Login failed!")
        browser.driver.quit()
        return

    logger.info("Logged in. Waiting for game...")
    if not wait_for_alive(browser):
        logger.error("Snake never spawned!")
        browser.driver.quit()
        return

    # ── Diagnostics: check JS steering infrastructure ────────────────
    diag = browser.driver.execute_script("""
        var result = {};
        result.hasSlither = (window.slither !== undefined && window.slither !== null);
        result.hasBotSetWang = (typeof window._botSetWang === 'function');
        result.hasBotSteering = !!window._botSteering;

        if (result.hasSlither) {
            result.ang = window.slither.ang;
            result.wang = window.slither.wang;
            result.eang = window.slither.eang;

            // Check if defineProperty is installed on wang
            var desc = Object.getOwnPropertyDescriptor(window.slither, 'wang');
            result.wangHasGetter = !!(desc && desc.get);
            result.wangHasSetter = !!(desc && desc.set);

            // List all angle-like properties on slither
            var angleProps = {};
            for (var k in window.slither) {
                var v = window.slither[k];
                if (typeof v === 'number' && k.length <= 5) {
                    angleProps[k] = v;
                }
            }
            result.angleProps = angleProps;

            // Test _botSetWang round-trip
            if (result.hasBotSetWang) {
                var before = window.slither.wang;
                window._botSetWang(1.234);
                var after = window.slither.wang;
                window._botSetWang(before);  // restore
                result.setWangTest = {before: before, setTo: 1.234, readBack: after};
            }
        }
        return result;
    """)
    logger.info("=== DIAGNOSTICS ===")
    for k, v in sorted(diag.items()):
        logger.info(f"  {k}: {v}")
    logger.info("===================\n")

    logger.info("Game active. Starting test sequence.\n")

    # ── Header ───────────────────────────────────────────────────────────
    logger.info(f"{'Step':>5} {'Phase':>6} {'Act':>4} {'TargΔ°':>7} "
                f"{'PreAng°':>8} {'PostAng°':>9} {'MoveAng°':>9} "
                f"{'Err°':>6} {'wang°':>7} {'ehang°':>7}")
    logger.info("-" * 90)

    # ── Phase statistics ─────────────────────────────────────────────────
    phase_stats = []  # list of dicts per phase
    global_step = 0
    start_time = time.time()
    deaths = 0

    phase_idx = 0
    while time.time() - start_time < TOTAL_DURATION:
        action, num_steps, desc = TEST_PHASES[phase_idx % len(TEST_PHASES)]
        angle_change = ACTION_ANGLE[action]
        act_name = ACTION_NAMES[action]
        is_turn = action != 0

        errors = []
        steps_done = 0

        for s in range(num_steps):
            if time.time() - start_time >= TOTAL_DURATION:
                break

            # ── Pre-action read ──────────────────────────────────────
            pre_data = browser.get_game_data()
            if not pre_data or pre_data.get("dead", True):
                deaths += 1
                logger.info(f"  *** DEATH #{deaths} at global step {global_step} — respawning ***")
                browser.force_restart()
                time.sleep(2)
                if not wait_for_alive(browser):
                    logger.error("Could not respawn, aborting.")
                    break
                continue

            pre_x = pre_data["self"]["x"]
            pre_y = pre_data["self"]["y"]
            pre_ang = pre_data["self"]["ang"]

            # Compute target angle
            target_ang = normalize_angle(pre_ang + angle_change)

            # ── Send action ──────────────────────────────────────────
            browser.send_action(target_ang, boost=0)
            time.sleep(FRAME_SKIP * 0.010)

            # ── Post-action read ─────────────────────────────────────
            post_data = browser.get_game_data()
            if not post_data or post_data.get("dead", True):
                deaths += 1
                logger.info(f"  *** DEATH #{deaths} at global step {global_step} — respawning ***")
                browser.force_restart()
                time.sleep(2)
                if not wait_for_alive(browser):
                    logger.error("Could not respawn, aborting.")
                    break
                continue

            post_x = post_data["self"]["x"]
            post_y = post_data["self"]["y"]
            post_ang = post_data["self"]["ang"]
            wang = post_data["self"].get("wang", 0)
            ehang = post_data["self"].get("ehang", post_data["self"].get("eang", 0))

            dx = post_x - pre_x
            dy = post_y - pre_y
            dist = math.hypot(dx, dy)

            if dist > 0.5:
                move_ang = math.atan2(dy, dx)
                err = abs(angle_diff_deg(move_ang, target_ang))
            else:
                move_ang = pre_ang
                err = 0.0  # no movement, skip

            errors.append(err)
            steps_done += 1
            global_step += 1

            logger.info(
                f"{global_step:5d} {phase_idx%len(TEST_PHASES)+1:>6d} "
                f"{act_name:>4s} {math.degrees(angle_change):>+7.1f} "
                f"{math.degrees(pre_ang):>8.1f} {math.degrees(post_ang):>9.1f} "
                f"{math.degrees(move_ang):>9.1f} {err:>6.1f} "
                f"{math.degrees(wang):>7.1f} {math.degrees(ehang):>7.1f}"
            )

        # Record phase stats
        if errors:
            avg_err = sum(errors) / len(errors)
            max_err = max(errors)
            threshold = PASS_THRESHOLD_TURN if is_turn else PASS_THRESHOLD_FWD
            passed = avg_err < threshold
        else:
            avg_err = max_err = 0.0
            passed = True

        phase_stats.append({
            "phase": phase_idx % len(TEST_PHASES) + 1,
            "action": act_name,
            "desc": desc,
            "steps": steps_done,
            "avg_err": avg_err,
            "max_err": max_err,
            "passed": passed,
            "is_turn": is_turn,
        })

        phase_idx += 1

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("STEERING TEST SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total steps: {global_step}  |  Deaths: {deaths}  |  "
                f"Duration: {time.time()-start_time:.0f}s\n")

    logger.info(f"{'Phase':>5} {'Action':>6} {'Steps':>5} {'AvgErr°':>8} "
                f"{'MaxErr°':>8} {'Thresh°':>8} {'Result':>8}")
    logger.info("-" * 55)

    all_pass = True
    for ps in phase_stats:
        threshold = PASS_THRESHOLD_TURN if ps["is_turn"] else PASS_THRESHOLD_FWD
        result = "PASS" if ps["passed"] else "FAIL"
        if not ps["passed"]:
            all_pass = False
        logger.info(
            f"{ps['phase']:>5d} {ps['action']:>6s} {ps['steps']:>5d} "
            f"{ps['avg_err']:>7.1f}° {ps['max_err']:>7.1f}° "
            f"{threshold:>7.0f}° {result:>8s}  {ps['desc']}"
        )

    logger.info("-" * 55)
    overall = "ALL PASS" if all_pass else "SOME FAILED"
    logger.info(f"Overall: {overall}")
    logger.info(f"Log saved to: {log_path}")

    browser.driver.quit()
    logger.info("Browser closed. Done.")


if __name__ == "__main__":
    main()
