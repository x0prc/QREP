"""
Unit tests for the ContextAwareDP module.
Tests:
1. Privacy budget calculation.
2. Laplace noise addition.
3. Synthetic data generation for high-risk data.
"""

import numpy as np
import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.differential.differential_privacy import ContextAwareDP

@pytest.fixture
def dp():
    return ContextAwareDP(epsilon_base=0.3)

def test_privacy_budget_calculation(dp):
    epsilon = dp.calculate_privacy_budget(7, 4.2)
    assert 1.0 < epsilon < 1.05

def test_add_laplace_noise(dp):
    data = np.array([100.0, 200.0, 50.0])
    noisy = dp.add_laplace_noise(data, sensitivity=1, epsilon=0.5)
    assert noisy.shape == data.shape
    assert not np.allclose(noisy, data)

def test_apply_differential_privacy(dp):
    data = np.array([100.0, 200.0, 50.0])
    noisy = dp.apply_differential_privacy(data, epsilon=0.1)
    assert noisy.shape == data.shape

def test_process_data_high_risk(dp, monkeypatch):
    # Patch model to always return high context score
    class DummyOutput:
        logits = torch.tensor([[0,0,0,0,0,0,0,0,0,10]])
    dp.model = lambda **kwargs: DummyOutput()
    dp.tokenizer = lambda text, return_tensors: {}
    result = dp.process_data("suspicious transaction", diversity_metric=2.0)
    assert "synthetic_data" in result
    assert "adjusted_epsilon" in result

def test_process_data_low_risk(dp, monkeypatch):
    # Patch model to always return low context score
    class DummyOutput:
        logits = torch.tensor([[10,0,0,0,0,0,0,0,0,0]])
    dp.model = lambda **kwargs: DummyOutput()
    dp.tokenizer = lambda text, return_tensors: {}
    result = dp.process_data("normal transaction", diversity_metric=2.0)
    assert "synthetic_data" not in result
    assert "adjusted_epsilon" in result

if __name__ == "__main__":
    pytest.main()
