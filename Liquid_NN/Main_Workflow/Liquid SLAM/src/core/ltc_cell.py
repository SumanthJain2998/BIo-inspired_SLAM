"""
src/core/ltc_cell.py

Liquid Time-Constant (LTC) Cell Implementation
Based on: "Liquid Time-constant Networks" (Hasani et al., 2021)

This module implements LTC cells that use numerical ODE solvers.
Included for comparison and hybrid approaches.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # For macOS compatibility

import torch #type: ignore
import torch.nn as nn #type: ignore
from typing import Tuple, Optional, Literal
import math


class LTCCell(nn.Module):
    """
    Liquid Time-Constant (LTC) Recurrent Neural Network Cell.
    
    LTC networks model continuous-time dynamics with adaptive time constants:
        dx/dt = -1/τ(x,I) * [x(t) - f(x(t), I(t); θ)]
    
    Uses numerical ODE solvers (Euler or RK4) for integration.
    
    Args:
        input_size: Number of input features
        hidden_size: Number of hidden units
        ode_solver: Integration method - "euler" or "rk4"
        ode_steps: Number of ODE solver steps per forward pass
        activation: Activation function for dynamics
        
    Note:
        This is slower than CfC but more general for complex dynamics.
        Useful for hybrid approaches or when closed-form is not applicable.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        ode_solver: Literal["euler", "rk4"] = "euler",
        ode_steps: int = 6,
        activation: str = "tanh",
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ode_solver = ode_solver
        self.ode_steps = ode_steps
        
        # State-dependent time constants
        self.tau_network = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Softplus(),  # Ensures positive time constants
        )
        
        # Dynamics function f(x, I)
        self.dynamics = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size * 2),
            self._get_activation(activation),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        
        self._init_weights()
    
    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        """Get activation function."""
        activations = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "sigmoid": nn.Sigmoid(),
        }
        return activations.get(name, nn.Tanh())
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def compute_derivatives(
        self,
        x: torch.Tensor,
        input: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute dx/dt = -1/τ(x,I) * [x - f(x, I)]
        
        Args:
            x: Current state (batch, hidden_size)
            input: Input (batch, input_size)
        
        Returns:
            dx_dt: Time derivative (batch, hidden_size)
        """
        combined = torch.cat([input, x], dim=-1)
        
        # Compute adaptive time constants
        tau = self.tau_network(combined)
        tau = torch.clamp(tau, min=0.1, max=10.0)  # Bound time constants
        
        # Compute target dynamics
        f_x = self.dynamics(combined)
        
        # Compute derivative: dx/dt = -1/τ * (x - f)
        dx_dt = -(1.0 / tau) * (x - f_x)
        
        return dx_dt
    
    def _euler_step(
        self,
        x: torch.Tensor,
        input: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """
        Single Euler integration step: x(t+dt) = x(t) + dt * dx/dt
        
        Args:
            x: Current state
            input: Input
            dt: Time step
        
        Returns:
            Updated state
        """
        dx_dt = self.compute_derivatives(x, input)
        return x + dt * dx_dt
    
    def _rk4_step(
        self,
        x: torch.Tensor,
        input: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """
        Fourth-order Runge-Kutta integration step.
        More accurate but computationally expensive.
        
        RK4 stages:
            k1 = f(x, t)
            k2 = f(x + dt/2 * k1, t + dt/2)
            k3 = f(x + dt/2 * k2, t + dt/2)
            k4 = f(x + dt * k3, t + dt)
            x_new = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        """
        k1 = self.compute_derivatives(x, input)
        k2 = self.compute_derivatives(x + 0.5 * dt * k1, input)
        k3 = self.compute_derivatives(x + 0.5 * dt * k2, input)
        k4 = self.compute_derivatives(x + dt * k3, input)
        
        return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    def forward(
        self,
        input: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        elapsed_time: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with numerical ODE integration.
        
        Args:
            input: Input tensor (batch, input_size)
            state: Previous state (batch, hidden_size)
            elapsed_time: Total time to integrate
        
        Returns:
            new_state: Updated state (batch, hidden_size)
            output: Cell output (batch, hidden_size)
        """
        batch_size = input.size(0)
        
        if state is None:
            state = torch.zeros(
                batch_size,
                self.hidden_size,
                device=input.device,
                dtype=input.dtype
            )
        
        # Time step for numerical integration
        dt = elapsed_time / self.ode_steps
        
        # Integrate over multiple steps
        x = state
        for _ in range(self.ode_steps):
            if self.ode_solver == "euler":
                x = self._euler_step(x, input, dt)
            elif self.ode_solver == "rk4":
                x = self._rk4_step(x, input, dt)
            else:
                raise ValueError(f"Unknown ODE solver: {self.ode_solver}")
        
        return x, x


class LTCLayer(nn.Module):
    """
    Multi-layer LTC network for sequence processing.
    
    Args:
        input_size: Size of input features
        hidden_size: Size of hidden state
        num_layers: Number of stacked layers
        dropout: Dropout between layers
        **kwargs: Additional arguments for LTCCell
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Create stacked LTC cells
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.cells.append(
                LTCCell(layer_input_size, hidden_size, **kwargs)
            )
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(
        self,
        input: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        elapsed_time: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process sequence through stacked LTC layers.
        
        Args:
            input: Input (batch, seq_len, input_size) or (batch, input_size)
            state: Initial states (num_layers, batch, hidden_size)
            elapsed_time: Integration time
        
        Returns:
            output: Output sequence (batch, seq_len, hidden_size)
            state: Final states (num_layers, batch, hidden_size)
        """
        is_sequence = input.dim() == 3
        if not is_sequence:
            input = input.unsqueeze(1)
        
        batch_size, seq_len, _ = input.shape
        
        if state is None:
            state = torch.zeros(
                self.num_layers,
                batch_size,
                self.hidden_size,
                device=input.device,
                dtype=input.dtype
            )
        
        outputs = []
        for t in range(seq_len):
            x = input[:, t]
            new_states = []
            
            for layer_idx, cell in enumerate(self.cells):
                x, new_state = cell(x, state[layer_idx], elapsed_time)
                new_states.append(new_state)
                
                if self.dropout is not None and layer_idx < self.num_layers - 1:
                    x = self.dropout(x)
            
            outputs.append(x)
            state = torch.stack(new_states)
        
        output = torch.stack(outputs, dim=1)
        
        if not is_sequence:
            output = output.squeeze(1)
        
        return output, state


# Performance comparison utilities
def benchmark_cfc_vs_ltc():
    """
    Benchmark CfC vs LTC performance.
    Demonstrates the speed advantage of closed-form solutions.
    """
    import time
    from cfc_cell import CfCLayer
    
    print("CfC vs LTC Performance Benchmark")
    print("=" * 60)
    
    batch_size = 32
    seq_len = 100
    input_size = 64
    hidden_size = 128
    
    # Create models
    cfc = CfCLayer(input_size, hidden_size, num_layers=2)
    ltc = LTCLayer(input_size, hidden_size, num_layers=2, ode_steps=6)
    
    # Test data
    x = torch.randn(batch_size, seq_len, input_size)
    
    # Warm up
    with torch.no_grad():
        _ = cfc(x)
        _ = ltc(x)
    
    # Benchmark CfC
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = cfc(x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    cfc_time = (time.time() - start) / 10
    
    # Benchmark LTC
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = ltc(x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    ltc_time = (time.time() - start) / 10
    
    print(f"CfC time: {cfc_time*1000:.2f} ms")
    print(f"LTC time: {ltc_time*1000:.2f} ms")
    print(f"Speedup: {ltc_time/cfc_time:.2f}x")


if __name__ == "__main__":
    # Example usage
    print("LTC Cell Implementation Test")
    print("=" * 60)
    
    batch_size = 4
    seq_len = 10
    input_size = 32
    hidden_size = 64
    
    # Test Euler solver
    ltc_euler = LTCLayer(input_size, hidden_size, ode_solver="euler")
    x = torch.randn(batch_size, seq_len, input_size)
    output, state = ltc_euler(x)
    print(f"Euler - Output shape: {output.shape}, State shape: {state.shape}")
    
    # Test RK4 solver
    ltc_rk4 = LTCLayer(input_size, hidden_size, ode_solver="rk4")
    output, state = ltc_rk4(x)
    print(f"RK4 - Output shape: {output.shape}, State shape: {state.shape}")
    
    # Run benchmark
    print("\n")
    benchmark_cfc_vs_ltc()