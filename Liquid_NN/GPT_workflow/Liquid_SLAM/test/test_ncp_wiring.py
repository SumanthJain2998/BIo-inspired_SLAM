import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'  # to avoid potential macOS issues with OpenMP

# test_ncp_wiring.py
import torch
import numpy as np
import pytest

# Adjust import path if module name differs; assumed wiring.py module exposes NCPWiring, NCPCell, WiringConfig, create_standard_wiring
from GPT_workflow.Liquid_SLAM.src.core.ncp_wiring import NCPWiring, NCPCell, WiringConfig, create_standard_wiring

torch.manual_seed(0)
np.random.seed(0)


def test_wiring_and_masks():
    # Build a small wiring config and wiring
    config = WiringConfig(
        sensory_size=4,
        inter_size=6,
        command_size=3,
        motor_size=2,
        sensory_to_inter=0.4,
        inter_to_command=0.5,
        command_to_motor=1.0,
        inter_recurrent=0.2,
        command_recurrent=0.3,
        use_sensory_to_command=True,
        use_sensory_to_motor=False
    )
    wiring = NCPWiring(config, seed=123)
    N = config.total_neurons()

    # adjacency matrix shape and dtype
    adj = wiring.adjacency_matrix
    assert adj.shape == (N, N)
    assert adj.dtype == np.float32

    # mask returned as torch tensor
    mask = wiring.get_weight_mask()
    assert isinstance(mask, torch.Tensor)
    assert mask.shape == (N, N)

    # layer masks
    layer_masks = wiring.get_layer_masks()
    for name in ("sensory", "inter", "command", "motor"):
        assert name in layer_masks
        m = layer_masks[name]
        assert m.dtype == torch.bool
        assert m.sum() > 0

    # sparsity should be > 0.5 given the chosen probabilities (probabilistic test)
    sparsity = 1.0 - (adj.sum() / float(N * N))
    assert 0.0 <= sparsity <= 1.0
    assert sparsity >= 0.2  # loose lower bound to detect trivial fully connected graphs


def test_W_masking_applied():
    config = WiringConfig(4, 6, 3, 2)
    wiring = NCPWiring(config, seed=7)
    cell = NCPCell(wiring)
    N = config.total_neurons()

    # set W to ones, then masked W should equal mask
    with torch.no_grad():
        cell.W.data.fill_(1.0)
        masked = cell.W * cell.mask
        # masked should be equal to mask as float
        mask_float = cell.mask.float()
        assert torch.allclose(masked, mask_float, atol=1e-6)

    # set W to random, compute masked manually
    with torch.no_grad():
        cell.W.data = torch.randn(N, N) * 0.1
        manual_masked = cell.W.data * cell.mask
        forward_masked = (cell.W * cell.mask)
        assert torch.allclose(manual_masked, forward_masked)


def manual_semiimplicit_step(state, W_masked, sensory_indices, sensory_input, tau, delta):
    """
    Compute the vectorized semi-implicit Euler step described in the test,
    matching the cell's forward implementation (Cmi=1, Eij=1, xleak=0).
    state: (B,N)
    W_masked: (N,N)
    sensory_indices: list
    sensory_input: (B, sensory_size)
    tau: (N,) tensor
    delta: (B,1) tensor
    """
    B, N = state.shape
    # clamp tau and compute gli = 1.0 / tau
    tau_clamped = torch.clamp(tau, min=1e-6)
    gli = (1.0 / tau_clamped).view(1, -1)  # (1,N)

    # form state_work with sensory neurons clamped
    state_work = state.clone()
    state_work[:, sensory_indices] = sensory_input

    # sigma(pre) = sigmoid(state_work)
    sigma_pre = torch.sigmoid(state_work)  # (B,N)

    # weighted inputs per target: sigma_pre @ W_masked.T -> (B, N)
    weighted_inputs = torch.matmul(sigma_pre, W_masked.t())

    numerator = (state / delta) + weighted_inputs  # x_leak=0, gli*x_leak=0
    denominator = (1.0 / delta) + gli + weighted_inputs

    denominator = torch.clamp(denominator, min=1e-6)
    new_state = numerator / denominator

    # clamp sensory neurons to sensory_input
    new_state[:, sensory_indices] = sensory_input

    return new_state


def test_ncpcell_semiimplicit_forward_matches_manual():
    # small deterministic wiring and cell
    config = WiringConfig(sensory_size=3, inter_size=4, command_size=2, motor_size=2,
                          sensory_to_inter=0.5, inter_to_command=0.5, command_to_motor=1.0,
                          inter_recurrent=0.2, command_recurrent=0.3,
                          use_sensory_to_command=True, use_sensory_to_motor=False)
    wiring = NCPWiring(config, seed=42)
    cell = NCPCell(wiring)
    cell.eval()

    B = 5
    N = config.total_neurons()

    # random input and initial state
    x = torch.randn(B, config.sensory_size)
    s0 = torch.randn(B, N)

    # set deterministic W and tau for test reproducibility
    torch.manual_seed(0)
    with torch.no_grad():
        cell.W.data = torch.randn(N, N) * 0.05
        # ensure mask applied won't change test determinism
        cell.W.data *= cell.mask
        # set tau to known values (avoid too small)
        cell.tau.data = torch.linspace(0.2, 1.2, steps=N)

    # scalar dt test
    dt_scalar = 0.2
    out, new_state = cell(x, s0, elapsed_time=dt_scalar)

    # manual computation
    W_masked = (cell.W * cell.mask).detach()
    delta = torch.full((B, 1), float(dt_scalar), dtype=s0.dtype)
    manual_new = manual_semiimplicit_step(s0, W_masked, wiring.sensory_indices, cell.input_proj(x), cell.tau.detach(), delta)

    assert new_state.shape == manual_new.shape
    assert torch.allclose(new_state, manual_new, atol=1e-6), "Semi-implicit forward (scalar dt) mismatch"

    # per-sample dt test (batch-wise dt)
    dt_batch = torch.linspace(0.1, 0.5, steps=B)
    out2, new_state2 = cell(x, s0, elapsed_time=dt_batch)

    delta_batch = dt_batch.view(B, 1)
    manual_new2 = manual_semiimplicit_step(s0, W_masked, wiring.sensory_indices, cell.input_proj(x), cell.tau.detach(), delta_batch)

    assert torch.allclose(new_state2, manual_new2, atol=1e-6), "Semi-implicit forward (per-sample dt) mismatch"
