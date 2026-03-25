'''
Cf-S: detailed, meticulous plan + implementation notes (PyTorch-ready)
Design choices & rationale
Keep everything elementwise (hidden dim = D). No dense D×D matrix exponentials → cheap.
Use a small backbone net φ_shared (e.g. single Linear + LayerNorm + activation) 
that consumes [x_t, u_t] and produces a feature vector which splits into heads: 
f_head, A_head (or learnable global A), and optionally a candidate B (but we can use x_t and A as in the exact formula).
Ensure positivity for rate terms: 

param → softplus(param) or F.softplus(head_output) for f and w_tau.
Clamp exponent argument: arg = -(w + f) * dt; 
compute arg = arg.clamp(min=-50.0, max=50.0) (you may pick different limits) before torch.exp(arg). 

This avoids overflow/underflow.
Numerical safety for tiny denominators: Cf-S doesn’t require matrix inverses when elementwise; avoid explicit inverse formulas.
Per-step algorithm (elementwise tensors of shape (batch, D))
Given x (batch,D), u (batch, U), dt (scalar or batch):
inp = concat(x, u)
feat = φ_shared(inp)
f_raw = f_head(feat) → f = softplus(f_raw) (ensures ≥0)
w_raw = self.w_raw (learnable vector shape (D,)) → w = softplus(w_raw)
A = self.A (learnable vector (D,)) — can be constant across time or a head A_head(feat)
exp_term = torch.exp(torch.clamp(-(w + f) * dt, min=-50.0, max=50.0)) (broadcast shapes)
x_next = (x - A) * exp_term + A
Initialization tips
Initialize w_raw such that softplus(w_raw) ~ 1 / typical Δt so dynamics are neither too fast nor too slow. For example w_raw = torch.log(torch.exp(1.0) - 1.0) if you expect w≈1.0.
Initialize A using small values or zeros depending on desired bias.
Initialize backbone weights with small scale; consider LayerNorm on the backbone features to keep head inputs well-conditioned.'''

# file: cfs_cell_test.py
# Run with: python cfs_cell_test.py

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

class CfSCell(nn.Module):
    """
    Closed-form scalar (elementwise) CfS cell.
    Implements per-step update:
        x_next = (x - A) * exp(-(w + f) * dt) + A
    """
    def __init__(self, input_dim, hidden_dim, shared_hidden=64, use_A_head=False, exp_clamp=(-50.0,50.0)):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_A_head = use_A_head
        self.exp_clamp = exp_clamp
        self.shared = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, shared_hidden),
            nn.GELU(),
            nn.LayerNorm(shared_hidden)
        )
        self.f_head = nn.Linear(shared_hidden, hidden_dim)
        if use_A_head:
            self.A_head = nn.Linear(shared_hidden, hidden_dim)
        else:
            self.A = nn.Parameter(torch.zeros(hidden_dim))
        self.w_raw = nn.Parameter(torch.ones(hidden_dim) * 0.5)

    def forward(self, x, u, dt):
        batch = x.shape[0]
        inp = torch.cat([x, u], dim=-1)
        feat = self.shared(inp)
        f_raw = self.f_head(feat)
        f = F.softplus(f_raw)                   # ensures f >= 0
        w = F.softplus(self.w_raw).unsqueeze(0) # shape (1, D)
        if self.use_A_head:
            A = self.A_head(feat)
        else:
            A = self.A.unsqueeze(0).expand(batch, -1)
        arg = - (w + f) * dt
        arg = torch.clamp(arg, self.exp_clamp[0], self.exp_clamp[1])
        exp_term = torch.exp(arg)
        x_next = (x - A) * exp_term + A
        return x_next, {"f": f, "w": w, "A": A, "exp_term": exp_term}

# helpers for tests
def manual_closed_form(x, A, w, f, dt):
    arg = - (w + f) * dt
    arg = torch.clamp(arg, -50.0, 50.0)
    exp_term = torch.exp(arg)
    return (x - A) * exp_term + A, exp_term

def euler_integrate_constant_coeff(x0, A, w, f, dt, n_steps=10000):
    sub_dt = dt / float(n_steps)
    x = x0.clone().detach().float()
    for _ in range(n_steps):
        dx = - (w + f) * x + (w + f) * A
        x = x + sub_dt * dx
    return x

def run_unit_tests():
    batch = 2
    D = 3
    U = 4

    cell = CfSCell(input_dim=U, hidden_dim=D, shared_hidden=64, use_A_head=False)
    x0 = torch.randn(batch, D, requires_grad=True)
    u = torch.randn(batch, U)
    dt = torch.tensor(0.1)

    x_next, saved = cell(x0, u, dt)
    x_manual, _ = manual_closed_form(x0, saved["A"], saved["w"], saved["f"], dt)
    x_euler = euler_integrate_constant_coeff(x0.detach(), saved["A"].detach(),
                                            saved["w"].detach(), saved["f"].detach(),
                                            float(dt.item()), n_steps=20000)

    print("x_next (cell):\n", x_next.detach().numpy())
    print("\nmanual closed-form:\n", x_manual.detach().numpy())
    print("\nmax abs diff between cell and manual closed-form:", float(torch.max(torch.abs(x_next - x_manual)).item()))
    print("max abs diff between manual closed-form and Euler approx:", float(torch.max(torch.abs(x_manual - x_euler)).item()))

    loss = x_next.sum()
    loss.backward()
    print("\nGradient check: x0.grad is not None and finite? ->", x0.grad is not None and torch.isfinite(x0.grad).all().item())

    # large dt check
    dt_large = torch.tensor(10.0)
    x_next_large, _ = cell(x0, u, dt_large)
    print("\nLarge-dt forward ok? NaNs present? ->", not torch.isnan(x_next_large).any().item())

    print("\nUnit tests finished. Expected: max diffs ~1e-6..1e-4 and gradient finite.")

if __name__ == "__main__":
    run_unit_tests()
