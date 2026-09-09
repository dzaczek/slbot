"""Worker-side env loop: death returns immediately, reset runs in the background.

SubprocVecEnv is lock-step — the parent waits for every worker before anyone
gets another action. If a dead snake's reset (browser reconnect, up to ~10s)
runs inside `step`, every other live snake flies with its last action.

This session starts reset on a background thread and answers subsequent `step`
commands with `spawning` / `spawned` until the new episode is ready.
"""

import threading

import numpy as np


SECTOR_DIM = 99


def spawning_obs(matrix_size):
    """Placeholder observation while Chrome/boot/reset is still running."""
    return {
        'matrix': np.zeros((3, matrix_size, matrix_size), dtype=np.float32),
        'sectors': np.zeros(SECTOR_DIM, dtype=np.float32),
        'spawning': True,
    }


def spawning_step_result(matrix_size):
    """Lock-step `step` payload that does not wait for env.reset()."""
    obs = spawning_obs(matrix_size)
    info = {
        'food_eaten': 0,
        'pos': (0, 0),
        'wall_dist': -1,
        'enemy_dist': -1,
        'length': 0,
        'spawning': True,
    }
    return (obs, 0.0, False, info)


class WorkerSession:
    """Serialize env.step / env.reset and respawn without blocking the vecenv."""

    def __init__(self, env, matrix_size):
        self.env = env
        self.matrix_size = matrix_size
        self._reset_thread = None
        self._reset_obs = None
        self._reset_error = None
        self._last_obs = None
        self._pending_stage = None
        self._env_lock = threading.Lock()

    def _dummy_obs(self):
        return {
            'matrix': np.zeros((3, self.matrix_size, self.matrix_size), dtype=np.float32),
            'sectors': np.zeros(SECTOR_DIM, dtype=np.float32),
        }

    def _copy_obs(self, obs):
        return {
            'matrix': np.array(obs['matrix'], copy=True),
            'sectors': np.array(obs['sectors'], copy=True),
        }

    def _remember_obs(self, obs):
        if obs is None:
            return
        self._last_obs = self._copy_obs(obs)

    def _placeholder_obs(self):
        """Non-destructive stand-in while a reset is running (keeps last frames)."""
        src = self._last_obs if self._last_obs is not None else self._dummy_obs()
        out = self._copy_obs(src)
        out['spawning'] = True
        return out

    def _idle_info(self, **extra):
        info = {
            'food_eaten': 0,
            'pos': (0, 0),
            'wall_dist': -1,
            'enemy_dist': -1,
            'length': 0,
        }
        info.update(extra)
        return info

    def _do_reset(self):
        try:
            with self._env_lock:
                self._reset_obs = self.env.reset()
        except Exception as e:
            self._reset_error = e

    def _start_reset(self):
        self._reset_obs = None
        self._reset_error = None
        self._reset_thread = threading.Thread(
            target=self._do_reset, daemon=True, name="env-reset",
        )
        self._reset_thread.start()

    def _join_reset(self, timeout=30):
        """Wait for an in-flight reset. Returns True only if it has fully finished."""
        if self._reset_thread is None:
            return True
        self._reset_thread.join(timeout=timeout)
        if self._reset_thread.is_alive():
            return False
        self._reset_thread = None
        self._reset_obs = None
        self._reset_error = None
        return True

    def _take_finished_reset(self):
        """Collect a reset thread that has already exited. Raises on reset failure."""
        self._reset_thread.join(timeout=1)
        self._reset_thread = None
        if self._reset_error is not None:
            err = self._reset_error
            self._reset_error = None
            raise err
        obs = self._reset_obs if self._reset_obs is not None else self._dummy_obs()
        self._reset_obs = None
        self._remember_obs(obs)
        self._apply_pending_stage()
        return obs

    def _apply_pending_stage(self):
        cfg = self._pending_stage
        if cfg is None:
            return
        self._pending_stage = None
        with self._env_lock:
            self.env.set_curriculum_stage(cfg)

    def handle(self, cmd, data):
        if cmd == 'step':
            return self.step(data)
        if cmd == 'reset':
            return self.reset_sync()
        if cmd == 'reset_one':
            return self.reset_async()
        if cmd == 'set_stage':
            # Never block the vecenv barrier on curriculum updates. If a reset
            # is in flight, apply the new stage when it finishes.
            if self._reset_thread is not None and self._reset_thread.is_alive():
                self._pending_stage = data
                return 'ok'
            with self._env_lock:
                self.env.set_curriculum_stage(data)
            return 'ok'
        raise ValueError(f"Unknown worker command: {cmd}")

    def step(self, action):
        if self._reset_thread is not None:
            if self._reset_thread.is_alive():
                return (self._placeholder_obs(), 0.0, False, self._idle_info(spawning=True))
            obs = self._take_finished_reset()
            return (obs, 0.0, False, self._idle_info(spawned=True))

        with self._env_lock:
            next_state, reward, done, info = self.env.step(action)
        self._remember_obs(next_state)
        if done:
            info['terminal_observation'] = next_state
            self._start_reset()
        return (next_state, reward, done, info)

    def reset_sync(self):
        """Startup / full reset — wait for a real observation."""
        if not self._join_reset():
            raise TimeoutError("background reset did not finish")
        self._apply_pending_stage()
        with self._env_lock:
            obs = self.env.reset()
        self._remember_obs(obs)
        return obs

    def reset_async(self):
        """Force-respawn without blocking the vecenv barrier (max-steps)."""
        if self._reset_thread is not None and self._reset_thread.is_alive():
            return self._placeholder_obs()
        if self._reset_thread is not None:
            return self._take_finished_reset()
        self._start_reset()
        return self._placeholder_obs()

    def close(self):
        self._join_reset(timeout=60)
        self.env.close()
