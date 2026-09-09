import os
import sys
import threading
import time
import unittest

from multiprocessing import Pipe

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from worker_process import READY_MSG, echo_worker, run_worker_loop
from worker_session import WorkerSession

from test_worker_session import FakeEnv


class TestWorkerBootLoop(unittest.TestCase):
    def test_step_returns_spawning_before_boot_finishes(self):
        release = threading.Event()
        parent, child = Pipe()

        def boot():
            release.wait(timeout=2)
            env = FakeEnv()
            env.reset_release.set()
            return WorkerSession(env, 8)

        t = threading.Thread(
            target=run_worker_loop, args=(child, 8, boot), daemon=True,
        )
        t.start()
        self.assertEqual(parent.recv(), READY_MSG)

        t0 = time.perf_counter()
        parent.send(('step', 0))
        obs, reward, done, info = parent.recv()
        self.assertLess(time.perf_counter() - t0, 0.1)
        self.assertTrue(info.get('spawning'))
        self.assertTrue(obs.get('spawning'))

        parent.send(('set_stage', {'food_reward': 3}))
        self.assertEqual(parent.recv(), 'ok')

        release.set()
        parent.send(('close', None))
        t.join(timeout=2)
        self.assertFalse(t.is_alive())

    def test_reset_waits_for_boot_then_returns_obs(self):
        release = threading.Event()
        parent, child = Pipe()

        def boot():
            release.wait(timeout=2)
            env = FakeEnv()
            env.reset_release.set()
            return WorkerSession(env, 8)

        t = threading.Thread(
            target=run_worker_loop, args=(child, 8, boot), daemon=True,
        )
        t.start()
        self.assertEqual(parent.recv(), READY_MSG)

        def do_reset():
            parent.send(('reset', None))
            return parent.recv()

        result = {}

        def run_reset():
            result['obs'] = do_reset()

        rt = threading.Thread(target=run_reset)
        rt.start()
        time.sleep(0.05)
        self.assertTrue(rt.is_alive(), "reset should wait until env boot finishes")
        release.set()
        rt.join(timeout=2)
        self.assertIn('obs', result)
        np.testing.assert_array_equal(result['obs']['matrix'], np.full((3, 8, 8), 1.0, dtype=np.float32))

        parent.send(('close', None))
        t.join(timeout=2)


class TestSubprocVecEnvScale(unittest.TestCase):
    def setUp(self):
        from trainer import SubprocVecEnv
        self.SubprocVecEnv = SubprocVecEnv

    def _make_env(self, n=1):
        return self.SubprocVecEnv(
            num_agents=n,
            matrix_size=8,
            frame_skip=1,
            worker_fn=echo_worker,
        )

    def test_add_agent_does_not_wait_for_worker_boot(self):
        env = self._make_env(1)
        try:
            t0 = time.perf_counter()
            obs = env.add_agent()
            elapsed = time.perf_counter() - t0
            self.assertLess(elapsed, 0.25, f"add_agent blocked for {elapsed:.2f}s")
            self.assertTrue(obs.get('spawning'))
            self.assertEqual(env.num_agents, 2)

            t0 = time.perf_counter()
            _, _, _, infos = env.step([0, 0])
            step_elapsed = time.perf_counter() - t0
            self.assertLess(step_elapsed, 0.25, f"step blocked on booting worker for {step_elapsed:.2f}s")
            self.assertTrue(infos[1].get('spawning'))
        finally:
            env.close()

    def test_remove_agent_does_not_wait_for_worker_close(self):
        env = self._make_env(1)
        try:
            env.add_agent()
            # Wait until the scaled worker is in the recv loop so close is processed.
            deadline = time.time() + 2
            while time.time() < deadline and not env._boot_ready[1]:
                time.sleep(0.02)
            self.assertTrue(env._boot_ready[1])

            t0 = time.perf_counter()
            self.assertTrue(env.remove_agent())
            elapsed = time.perf_counter() - t0
            self.assertLess(elapsed, 0.25, f"remove_agent blocked for {elapsed:.2f}s")
            self.assertEqual(env.num_agents, 1)
        finally:
            env.close()

    def test_vecframestack_add_agent_marks_spawning(self):
        from trainer import VecFrameStack

        class FakeVenv:
            def __init__(self):
                self.num_agents = 1

            def add_agent(self):
                self.num_agents += 1
                return {
                    'matrix': np.zeros((3, 4, 4), dtype=np.float32),
                    'sectors': np.zeros(99, dtype=np.float32),
                    'spawning': True,
                }

        stack = VecFrameStack(FakeVenv(), k=4)
        t0 = time.perf_counter()
        obs = stack.add_agent()
        self.assertLess(time.perf_counter() - t0, 0.05)
        self.assertTrue(obs.get('spawning'))
        self.assertEqual(stack.num_agents, 2)
        self.assertEqual(len(stack.frames[1]), 4)


if __name__ == '__main__':
    unittest.main()
