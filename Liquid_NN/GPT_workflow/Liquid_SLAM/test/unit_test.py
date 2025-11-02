import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'  # to avoid potential macOS issues with OpenMP

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from GPT_workflow.Liquid_SLAM.src.core.cfc_modified import CfCCell, CfCLayer


_eps = 1e-6


# =======================
# Unit tests
# =======================
def run_unit_tests():
    torch.manual_seed(0)
    batch = 4
    seq_len = 5
    input_size = 8
    hidden = 16
    
    print("Create CfCLayer with 3 stacked CfC layers (default)")
    layer = CfCLayer(input_size, hidden, num_layers=3, dropout=0.0, backbone_units=32, backbone_layers=1, readout=True)
    layer.eval()
    
    # single-step forward (scalar dt)
    x = torch.randn(batch, input_size)
    out, state = layer(x, elapsed_time=0.5)
    assert out.shape == (batch, hidden), f"single-step output shape {out.shape}"
    assert state.shape == (3, batch, hidden)
    print("Single-step forward shapes OK.")
    
    # sequence forward (scalar dt)
    xseq = torch.randn(batch, seq_len, input_size)
    out_seq, final_state = layer(xseq, elapsed_time=1.0)
    assert out_seq.shape == (batch, seq_len, hidden)
    assert final_state.shape == (3, batch, hidden)
    print("Sequence forward shapes OK.")
    
    # per-sample dt (tensor of shape (batch,))
    dt_batch = torch.linspace(0.1, 2.0, steps=batch)
    out_dt, st_dt = layer(x, elapsed_time=dt_batch)
    assert out_dt.shape == (batch, hidden)
    print("Per-sample dt (batch) forward OK.")
    
    # per-sample per-step dt (batch, seq_len)
    dt_seq = torch.ones(batch, seq_len) * 0.5
    dt_seq[0] = torch.linspace(0.1, 1.0, steps=seq_len)
    out_seq_dt, st_seq_dt = layer(xseq, elapsed_time=dt_seq)
    assert out_seq_dt.shape == (batch, seq_len, hidden)
    print("Per-sample per-step dt forward OK.")
    
    # check tau positivity
    cell0 = layer.cells[0]
    tau_vals = cell0.tau().detach()
    assert torch.all(tau_vals > 0), "tau must be positive"
    print(f"Tau stats: min={tau_vals.min():.4f}, max={tau_vals.max():.4f}, mean={tau_vals.mean():.4f}")
    
    # Check dt effect
    cell = CfCCell(input_size, hidden, backbone_units=32, backbone_layers=1, readout=False)
    cell.eval()
    xin = torch.randn(batch, input_size)
    s0 = torch.randn(batch, hidden)
    _, new_small = cell(xin, s0, elapsed_time=0.01)
    _, new_large = cell(xin, s0, elapsed_time=10.0)
    with torch.no_grad():
        feat = cell.backbone(torch.cat([xin, s0], dim=-1)) if len(cell.backbone) > 0 else torch.cat([xin, s0], dim=-1)
        f_target = cell.head_f(feat)
        ds_small = torch.norm(new_small - f_target, dim=-1).mean()
        ds_large = torch.norm(new_large - f_target, dim=-1).mean()
    assert ds_large < ds_small + 1e-4, "Larger dt should move state closer to f_target"
    print("Behavioral test for dt effect OK.")
    
    # test return_output flag in cell
    out_only, st_out = cell(xin, s0, elapsed_time=1.0, return_output=True)
    st_only, _ = cell(xin, s0, elapsed_time=1.0, return_output=False)
    assert torch.allclose(st_only, st_out)
    print("return_output flag behavior OK.")
    
    print("All unit tests passed.")

if __name__ == "__main__":
    run_unit_tests()