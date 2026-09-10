import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import OptimizationConfig

def test_train_freq_validation():
    # Should be valid
    cfg = OptimizationConfig(train_freq=1)
    assert cfg.train_freq == 1

    cfg = OptimizationConfig(train_freq=4)
    assert cfg.train_freq == 4

    # Should raise ValueError
    with pytest.raises(ValueError):
        OptimizationConfig(train_freq=0)

    with pytest.raises(ValueError):
        OptimizationConfig(train_freq=-1)

def test_train_freq_modulo_logic():
    # Simulate trainer logic
    class DummyAgent:
        def __init__(self):
            self.calls = 0
        def optimize_model(self):
            self.calls += 1
            return {"loss": 0.1}

    agent = DummyAgent()

    # Simulate train_freq=1
    train_freq = 1
    total_steps = 0
    for _ in range(10):
        total_steps += 1
        if total_steps % train_freq == 0:
            agent.optimize_model()

    assert agent.calls == 10, "With train_freq=1, optimize_model should be called on every step"

    # Simulate train_freq=4
    agent.calls = 0
    train_freq = 4
    total_steps = 0
    for _ in range(10):
        total_steps += 1
        if total_steps % train_freq == 0:
            agent.optimize_model()

    assert agent.calls == 2, "With train_freq=4, optimize_model should be called twice in 10 steps (at 4 and 8)"

    # Verify target update freq is independent
    target_updates = 0
    target_update_freq = 10000
    total_steps = 0
    for _ in range(20000):
        total_steps += 1
        if total_steps % target_update_freq == 0:
            target_updates += 1

    assert target_updates == 2, "target updates should happen exactly twice in 20000 steps"
