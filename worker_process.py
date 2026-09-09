"""Lightweight worker process entry.

Kept in its own module so `multiprocessing` spawn does not re-import trainer
(torch, rich, …) on every scale-up. Chrome/env construction runs on a
background thread; the command loop answers `step` with a spawning placeholder
until the session exists.
"""

import os
import sys
import threading
import time
import traceback

from worker_session import (
    WorkerSession,
    spawning_obs,
    spawning_step_result,
)


READY_MSG = 'ready'

AGENT_NAMES = [
    "Picard", "Riker", "Data", "Worf", "Troi", "LaForge",
    "Crusher", "Q", "Seven", "Raffi", "Rios", "Jurati",
]


def _log_crash(worker_id, exc):
    crash_msg = f"Worker {worker_id} crashed: {exc}\n{traceback.format_exc()}"
    try:
        os.makedirs("logs", exist_ok=True)
        with open("logs/worker_crashes.log", "a") as wf:
            wf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {crash_msg}\n")
    except Exception:
        pass
    try:
        print(crash_msg)
    except Exception:
        pass


def run_worker_loop(remote, matrix_size, boot_fn, autoreset=False, initial_stage=None,
                    boot_timeout=120):
    """Command loop that stays responsive while `boot_fn` creates the session.

    Sends READY_MSG as soon as it is listening. `boot_fn()` runs on a
    background thread and must return a WorkerSession.
    """
    remote.send(READY_MSG)

    session = None
    boot_error = None
    queued_stage = initial_stage
    lock = threading.Lock()

    def boot():
        nonlocal session, boot_error, queued_stage
        try:
            sess = boot_fn()
            with lock:
                stage = queued_stage
            if stage is not None:
                sess.handle('set_stage', stage)
            if autoreset:
                sess._start_reset()
            with lock:
                session = sess
        except Exception as e:
            boot_error = e

    boot_thread = threading.Thread(target=boot, daemon=True, name="env-boot")
    boot_thread.start()

    def wait_session():
        deadline = time.time() + boot_timeout
        while True:
            with lock:
                sess = session
                err = boot_error
            if err is not None:
                raise err
            if sess is not None:
                return sess
            if time.time() > deadline:
                raise TimeoutError("worker env boot timed out")
            time.sleep(0.05)

    while True:
        cmd, data = remote.recv()
        if cmd == 'close':
            with lock:
                sess = session
            if sess is not None:
                sess.close()
            break

        with lock:
            sess = session
            err = boot_error
            if cmd == 'set_stage' and sess is None:
                queued_stage = data

        if err is not None:
            raise err

        if sess is None:
            if cmd == 'step':
                remote.send(spawning_step_result(matrix_size))
            elif cmd == 'reset':
                sess = wait_session()
                remote.send(sess.reset_sync())
            elif cmd == 'reset_one':
                remote.send(spawning_obs(matrix_size))
            elif cmd == 'set_stage':
                remote.send('ok')
            else:
                raise ValueError(f"Unknown worker command: {cmd}")
            continue

        remote.send(sess.handle(cmd, data))


def worker(remote, parent_remote, worker_id, headless, nickname_prefix, matrix_size,
           frame_skip, view_plus=False, base_url="http://slither.io", backend="selenium",
           ws_server_url="", suppress_stdout=False, stage_config=None, autoreset=False):
    parent_remote.close()

    if suppress_stdout:
        devnull = open(os.devnull, 'w')
        sys.stdout = devnull
        sys.stderr = devnull

    chosen_name = AGENT_NAMES[worker_id % len(AGENT_NAMES)]

    def boot():
        from slither_env import SlitherEnv
        env = SlitherEnv(
            headless=headless,
            nickname=chosen_name,
            matrix_size=matrix_size,
            view_plus=view_plus,
            base_url=base_url,
            frame_skip=frame_skip,
            backend=backend,
            ws_server_url=ws_server_url,
        )
        return WorkerSession(env, matrix_size)

    try:
        run_worker_loop(
            remote,
            matrix_size,
            boot,
            autoreset=autoreset,
            initial_stage=stage_config,
        )
    except Exception as e:
        _log_crash(worker_id, e)
    finally:
        try:
            remote.close()
        except Exception:
            pass


def echo_worker(remote, parent_remote, worker_id, headless, nickname, matrix_size,
                frame_skip, view_plus, base_url, backend, ws_server_url,
                suppress_stdout, stage_config=None, autoreset=False):
    """Test double: handshake + command loop, no Chrome.

    Worker id >= 1 delays READY so tests can assert the parent does not wait.
    `autoreset` workers also delay close, so remove_agent must not join on the
    training thread.
    """
    import numpy as np

    parent_remote.close()
    if worker_id >= 1:
        time.sleep(0.4)
    remote.send(READY_MSG)
    while True:
        cmd, data = remote.recv()
        if cmd == 'close':
            if autoreset:
                time.sleep(1.0)
            break
        if cmd == 'step':
            remote.send(spawning_step_result(matrix_size))
        elif cmd == 'set_stage':
            remote.send('ok')
        elif cmd == 'reset':
            remote.send({
                'matrix': np.zeros((3, matrix_size, matrix_size), dtype=np.float32),
                'sectors': np.zeros(99, dtype=np.float32),
            })
        elif cmd == 'reset_one':
            remote.send(spawning_obs(matrix_size))
        else:
            raise ValueError(f"Unknown command: {cmd}")
