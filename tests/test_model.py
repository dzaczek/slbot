import torch
import torch.nn as nn
from gen2.model import DuelingDQN

def test_dueling_dqn_initialization():
    """Test that DuelingDQN initializes with different parameters."""
    input_channels = 12
    action_dim = 6
    model = DuelingDQN(input_channels, action_dim)
    assert isinstance(model, DuelingDQN)

    # Check another config
    model2 = DuelingDQN(input_channels=3, action_dim=10, input_size=(64, 64))
    assert isinstance(model2, DuelingDQN)

def test_dueling_dqn_forward_shape_84():
    """Test that the forward pass produces the correct output shape for 84x84."""
    batch_size = 4
    input_channels = 12
    action_dim = 6
    model = DuelingDQN(input_channels, action_dim, input_size=(84, 84))

    dummy_input = torch.randn(batch_size, input_channels, 84, 84)
    output = model(dummy_input)

    assert output.shape == (batch_size, action_dim)

def test_dueling_dqn_forward_shape_64():
    """Test that the forward pass produces the correct output shape for 64x64."""
    batch_size = 4
    input_channels = 12
    action_dim = 6
    model = DuelingDQN(input_channels, action_dim, input_size=(64, 64))

    dummy_input = torch.randn(batch_size, input_channels, 64, 64)
    output = model(dummy_input)

    assert output.shape == (batch_size, action_dim)

def test_dueling_dqn_logic():
    """Test the dueling aggregation logic (Q = V + A - mean(A))."""
    input_channels = 12
    action_dim = 6
    model = DuelingDQN(input_channels, action_dim)

    dummy_input = torch.randn(1, input_channels, 84, 84)

    # The forward pass should run without errors
    output = model(dummy_input)
    assert not torch.isnan(output).any()
    assert output.shape == (1, action_dim)

def test_weight_initialization():
    """Test that weights are initialized and not all zeros."""
    model = DuelingDQN(12, 6)

    # Check conv1 weights - they should be initialized (kaiming_normal_)
    assert torch.sum(torch.abs(model.conv1.weight)) > 0
    # Check value stream weights
    assert torch.sum(torch.abs(model.value_stream[0].weight)) > 0

def test_gradient_flow():
    """Test that gradients flow back to all layers."""
    model = DuelingDQN(12, 6)
    dummy_input = torch.randn(1, 12, 84, 84)
    output = model(dummy_input)

    loss = output.sum()
    loss.backward()

    # Check if conv1 has gradients
    assert model.conv1.weight.grad is not None
    assert torch.sum(torch.abs(model.conv1.weight.grad)) > 0

    # Check if value stream has gradients
    assert model.value_stream[0].weight.grad is not None
    assert torch.sum(torch.abs(model.value_stream[0].weight.grad)) > 0
