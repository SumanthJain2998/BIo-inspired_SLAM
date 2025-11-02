import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'  # to avoid potential macOS issues with OpenMP

# test_cfc_eq10_strict.py
import torch
import torch.nn as nn
import pytest

# Adjust this import if your implementation resides in a different file
# Example: from cfc_modified import CfCCell, CfCLayer
from GPT_workflow.Liquid_SLAM.src.core.cfc_modified import CfCCell, CfCLayer

torch.manual_seed(0)


def compute_eq10_from_heads(g, f, h, dt):
    """
    Compute Eq.10: s = sigmoid(-f * dt), new = s * g + (1-s) * h
    g,f,h: (batch, hidden)
    dt: scalar or (batch,)
    returns: (new, s)
    """
    # Normalize dt to shape (batch, 1)
    if isinstance(dt, (float, int)):
        dt_t = torch.full((g.shape[0], 1), float(dt), device=g.device, dtype=g.dtype)
    else:
        dt_t = torch.as_tensor(dt, device=g.device, dtype=g.dtype)
        dt_t = dt_t.view(g.shape[0], -1)[:, 0].unsqueeze(-1)
    s = torch.sigmoid(- f * dt_t)  # broadcast to (batch, hidden)
    new = s * g + (1.0 - s) * h
    return new, s


def extract_heads_from_cell(cell: CfCCell, x: torch.Tensor, state: torch.Tensor):
    """
    Extract backbone features and heads (g,f,h) for a given CfCCell and inputs.
    """
    combined = torch.cat([x, state], dim=-1)
    feat = cell.backbone(combined) if len(cell.backbone) > 0 else combined
    g = cell.head_g(feat)
    f = cell.head_f(feat)
    h = cell.head_h(feat)
    return g, f, h


def assert_allclose_msg(a, b, atol=1e-6, rtol=1e-5, msg=""):
    assert torch.allclose(a, b, atol=atol, rtol=rtol), msg + f"\nMax abs diff: {(a-b).abs().max():.6g}"


def test_cfccell_forward_matches_eq10():
    """
    Strict test: CfCCell.forward(...) must return the Eq.10 value exactly (within tolerance).
    - Create a CfCCell, produce input and state, compute g,f,h via backbone+heads,
      compute Eq.10, then call cell.forward(...) and compare returned new state to Eq.10.
    """
    batch = 5
    input_size = 7
    hidden = 9
    dt = 0.37  # scalar dt test, also try per-sample dt below

    cell = CfCCell(input_size, hidden, backbone_units=32, backbone_layers=1, readout=False)
    cell.eval()

    x = torch.randn(batch, input_size)
    s0 = torch.randn(batch, hidden)

    # Compute heads manually
    g, f, h = extract_heads_from_cell(cell, x, s0)
    expected_new, s_gate = compute_eq10_from_heads(g, f, h, dt)

    # Call cell.forward
    out, new_state = cell(x, s0, elapsed_time=dt, return_output=True)

    # If the cell has a readout/projection, out could be different; we required readout=False above
    # so out == new_state (the implementation used earlier sometimes returns (output, state))
    # We'll compare new_state (returned) to expected Eq.10 result.
    assert new_state.shape == expected_new.shape, "shape mismatch between computed Eq.10 and cell output"

    assert_allclose_msg(
        new_state,
        expected_new,
        atol=1e-5,
        msg="CfCCell.forward did NOT match Eq.10 for scalar dt."
    )

    # Also test per-sample dt (batch-wise different dt)
    dt_batch = torch.linspace(0.05, 1.0, steps=batch)
    expected_new_b, s_gate_b = compute_eq10_from_heads(g, f, h, dt_batch)
    out_b, new_state_b = cell(x, s0, elapsed_time=dt_batch, return_output=True)
    assert_allclose_msg(
        new_state_b,
        expected_new_b,
        atol=1e-5,
        msg="CfCCell.forward did NOT match Eq.10 for per-sample dt."
    )


def test_cfclayer_forward_equals_manual_step_and_eq10():
    """
    1) Validate CfCLayer.forward equals a manual stepping through its internal CfCCell objects.
    2) For each cell invocation in that manual stepping, assert that CfCCell.forward matches Eq.10
       (i.e., CfCCell implements Eq.10).
    This ensures both the layer wiring plus the per-cell update follow Eq.10.
    """
    batch = 3
    seq_len = 4
    input_size = 6
    hidden = 8
    dt = 0.25  # scalar dt for simplicity (also test per-step dt below)

    # Build a layer with cells that do NOT apply readout projection, so outputs are cell states
    layer = CfCLayer(input_size, hidden, num_layers=3, dropout=0.0, backbone_units=24, backbone_layers=1, readout=False)
    layer.eval()

    xseq = torch.randn(batch, seq_len, input_size)

    # Run layer forward
    out_seq, final_state = layer(xseq, elapsed_time=dt, return_output=True)

    # Manual stepping: use the same cell objects inside layer to step through sequence
    # state shape: (num_layers, batch, hidden)
    manual_state = torch.zeros(layer.num_layers, batch, hidden, dtype=xseq.dtype, device=xseq.device)
    manual_outputs = []

    for t in range(seq_len):
        x_t = xseq[:, t]  # (batch, input_size)
        new_states = []
        h_t = x_t
        for li, cell in enumerate(layer.cells):
            s_prev = manual_state[li]
            # call the cell to get its output and new state
            out_cell, new_s = cell(h_t, s_prev, elapsed_time=dt, return_output=True)

            # --- STRICT Eq.10 check for this cell invocation ---
            g, f, h = extract_heads_from_cell(cell, h_t, s_prev)
            expected_new_cell, s_gate = compute_eq10_from_heads(g, f, h, dt)
            # compare new_s to expected_new_cell
            assert_allclose_msg(
                new_s,
                expected_new_cell,
                atol=1e-5,
                msg=f"CfCCell at layer {li} time {t} did NOT match Eq.10."
            )
            # end Eq.10 check

            new_states.append(new_s)
            # for the next layer, the input is the output of this cell
            h_t = out_cell
        # after all layers, collect output (the output of last cell)
        manual_outputs.append(h_t)
        manual_state = torch.stack(new_states, dim=0)

    manual_out_seq = torch.stack(manual_outputs, dim=1)  # (batch, seq_len, hidden)

    # Compare manual_out_seq to layer's out_seq
    assert_allclose_msg(
        manual_out_seq,
        out_seq,
        atol=1e-6,
        msg="CfCLayer.forward outputs differ from manual stepping through cells."
    )

    # And final states match
    assert_allclose_msg(
        manual_state,
        final_state,
        atol=1e-6,
        msg="CfCLayer.final_state differs from manual-stepped state."
    )

    # Additional check: per-step per-sample dt broadcasting (batch x seq_len)
    dt_seq = torch.ones(batch, seq_len) * 0.5
    dt_seq[0] = torch.linspace(0.1, 0.9, steps=seq_len)  # vary first sample
    out_seq2, final_state2 = layer(xseq, elapsed_time=dt_seq, return_output=True)

    # Manual step with dt per-step per-sample
    manual_state2 = torch.zeros(layer.num_layers, batch, hidden, dtype=xseq.dtype, device=xseq.device)
    manual_outputs2 = []
    for t in range(seq_len):
        x_t = xseq[:, t]
        dt_t = dt_seq[:, t]
        new_states = []
        h_t = x_t
        for li, cell in enumerate(layer.cells):
            s_prev = manual_state2[li]
            out_cell, new_s = cell(h_t, s_prev, elapsed_time=dt_t, return_output=True)

            # Eq.10 check for this cell invocation with per-sample dt
            g, f, h = extract_heads_from_cell(cell, h_t, s_prev)
            expected_new_cell, s_gate = compute_eq10_from_heads(g, f, h, dt_t)
            assert_allclose_msg(
                new_s,
                expected_new_cell,
                atol=1e-5,
                msg=f"CfCCell at layer {li} time {t} with per-sample dt did NOT match Eq.10."
            )

            new_states.append(new_s)
            h_t = out_cell
        manual_outputs2.append(h_t)
        manual_state2 = torch.stack(new_states, dim=0)

    manual_out_seq2 = torch.stack(manual_outputs2, dim=1)
    assert_allclose_msg(manual_out_seq2, out_seq2, atol=1e-6, msg="Per-step dt: layer outputs differ from manual stepping.")
    assert_allclose_msg(manual_state2, final_state2, atol=1e-6, msg="Per-step dt: final states differ.")

