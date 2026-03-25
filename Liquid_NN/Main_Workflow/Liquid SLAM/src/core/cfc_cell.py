"""
src/core/cfc_cell.py

Closed-form Continuous-time (CfC) Cell Implementation
Based on: "Closed-form continuous-time neural networks" (Hasani et al., 2022)

This module implements the CfC cell which eliminates the computational bottleneck
of numerical ODE solvers by deriving closed-form solutions.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # For macOS compatibility

import torch # type: ignore
import torch.nn as nn #type: ignore
from typing import Tuple, Optional, Literal
import math


class CfCCell(nn.Module):
    """
    Closed-form Continuous-time (CfC) Recurrent Neural Network Cell.
    
    The CfC cell solves the differential equation in closed form:
        dx/dt = -1/τ * x(t) + f(x(t), I(t); θ)
    
    Key advantages:
        - 1-5 orders of magnitude faster than ODE-based methods
        - No numerical approximation errors
        - Better scalability and stability
    
    Args:
        input_size: Number of input features
        hidden_size: Number of hidden units
        mode: CfC mode - "default", "pure", or "no_gate"
        activation: Backbone activation function
        backbone_units: Number of units in backbone network
        backbone_layers: Number of layers in backbone
        backbone_dropout: Dropout rate for backbone
        
    Mathematical formulation:
        x(t) = e^(-t/τ) * x₀ + (1 - e^(-t/τ)) * [A + B·Φ(I, t, τ)]
    where:
        - τ: time constants (liquid component)
        - A, B: learnable parameters
        - Φ: bounded approximation of integral term
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        mode: Literal["default", "pure", "no_gate"] = "default",
        activation: str = "lecun_tanh",
        backbone_units: int = 128,
        backbone_layers: int = 1,
        backbone_dropout: float = 0.0,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mode = mode
        
        # Time constants (the "liquid" component)
        # Initialized with log-uniform distribution for stability
        self._init_time_constants()
        
        # Backbone network for computing f(x, I)
        self.backbone = self._build_backbone(
            input_size + hidden_size,
            backbone_units,
            backbone_layers,
            backbone_dropout,
            activation
        )
        
        # Output projection from backbone to hidden state
        self.backbone_output = nn.Linear(backbone_units, hidden_size)
        
        # Mode-specific parameters
        if mode == "default":
            # Full CfC with gating
            self.gate = nn.Linear(input_size + hidden_size, hidden_size)
        elif mode == "no_gate":
            # CfC without gating mechanism
            pass
        elif mode == "pure":
            # Pure continuous-time without any modifications
            pass
        
        self._init_weights()
    
    def _init_time_constants(self):
        """Initialize time constants with log-uniform distribution."""
        # Time constants in range [0.1, 10.0]
        log_tau = torch.rand(self.hidden_size) * math.log(100) + math.log(0.1)
        self.tau = nn.Parameter(torch.exp(log_tau))
    
    def _build_backbone(
        self,
        input_dim: int,
        units: int,
        layers: int,
        dropout: float,
        activation: str
    ) -> nn.Sequential:
        """Build the backbone neural network."""
        act_fn = self._get_activation(activation)
        
        modules = []
        in_features = input_dim
        
        for i in range(layers):
            modules.append(nn.Linear(in_features, units))
            modules.append(act_fn())
            if dropout > 0:
                modules.append(nn.Dropout(dropout))
            in_features = units
        
        return nn.Sequential(*modules)
    
    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "tanh": nn.Tanh,
            "lecun_tanh": lambda: nn.Tanh(),  # LeCun initialization
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "sigmoid": nn.Sigmoid,
        }
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}")
        return activations[name]
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        elapsed_time: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through CfC cell.
        
        Args:
            input: Input tensor of shape (batch, input_size)
            state: Previous hidden state of shape (batch, hidden_size)
                   If None, initialized to zeros
            elapsed_time: Time elapsed since last update (dt)
        
        Returns:
            new_state: Updated hidden state (batch, hidden_size)
            output: Cell output (batch, hidden_size)
        
        Mathematical operations:
            1. Compute backbone: f = backbone([x, h])
            2. Apply decay: h_decay = exp(-dt/τ) * h₀
            3. Compute integral approximation: Φ
            4. Update: h_new = h_decay + (1 - exp(-dt/τ)) * f
        """
        batch_size = input.size(0)
        
        # Initialize state if not provided
        if state is None:
            state = torch.zeros(
                batch_size,
                self.hidden_size,
                device=input.device,
                dtype=input.dtype
            )
        
        # Concatenate input and previous state
        combined = torch.cat([input, state], dim=-1)
        
        # Compute backbone function f(x, h)
        backbone_output = self.backbone(combined)
        f_output = self.backbone_output(backbone_output)
        
        # Compute exponential decay factor
        # decay = exp(-dt/τ)
        decay = torch.exp(-elapsed_time / torch.clamp(self.tau, min=1e-6))
        
        # Closed-form solution
        # h(t) = exp(-t/τ) * h₀ + (1 - exp(-t/τ)) * f
        new_state = decay * state + (1.0 - decay) * f_output
        
        # Apply gating if in default mode
        if self.mode == "default":
            gate_values = torch.sigmoid(self.gate(combined))
            new_state = gate_values * new_state + (1.0 - gate_values) * state
        
        return new_state, new_state
    
    def get_time_constants(self) -> torch.Tensor:
        """Return current time constants."""
        return self.tau.data
    
    def set_time_constants(self, tau: torch.Tensor):
        """Set time constants manually."""
        assert tau.shape == self.tau.shape
        self.tau.data = tau


class CfCLayer(nn.Module):
    """
    Multi-layer CfC network for sequence processing.
    
    This wraps CfCCell for handling sequences and multiple layers.
    
    Args:
        input_size: Size of input features
        hidden_size: Size of hidden state
        num_layers: Number of stacked CfC layers
        dropout: Dropout probability between layers
        **kwargs: Additional arguments passed to CfCCell
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
        
        # Create stacked CfC cells
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.cells.append(
                CfCCell(layer_input_size, hidden_size, **kwargs)
            )
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(
        self,
        input: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        elapsed_time: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process sequence through stacked CfC layers.
        
        Args:
            input: Input sequence (batch, seq_len, input_size)
                   or single step (batch, input_size)
            state: Initial states for all layers (num_layers, batch, hidden_size)
            elapsed_time: Time step size
        
        Returns:
            output: Output sequence (batch, seq_len, hidden_size)
            state: Final states (num_layers, batch, hidden_size)
        """
        # Handle both sequence and single-step inputs
        is_sequence = input.dim() == 3
        if not is_sequence:
            input = input.unsqueeze(1)  # Add sequence dimension
        
        batch_size, seq_len, _ = input.shape
        
        # Initialize states if not provided
        if state is None:
            state = torch.zeros(
                self.num_layers,
                batch_size,
                self.hidden_size,
                device=input.device,
                dtype=input.dtype
            )
        
        # Process sequence
        outputs = []
        for t in range(seq_len):
            x = input[:, t]
            new_states = []
            
            for layer_idx, cell in enumerate(self.cells):
                x, new_state = cell(x, state[layer_idx], elapsed_time)
                new_states.append(new_state)
                
                # Apply dropout between layers (not after last layer)
                if self.dropout is not None and layer_idx < self.num_layers - 1:
                    x = self.dropout(x)
            
            outputs.append(x)
            state = torch.stack(new_states)
        
        output = torch.stack(outputs, dim=1)
        
        # Remove sequence dimension if input was single-step
        if not is_sequence:
            output = output.squeeze(1)
        
        return output, state
    
    def get_all_time_constants(self) -> list:
        """Get time constants from all layers."""
        return [cell.get_time_constants() for cell in self.cells]


if __name__ == "__main__":
    # Example usage
    print("CfC Cell Implementation Test")
    print("=" * 60)
    
    batch_size = 4
    seq_len = 10
    input_size = 32
    hidden_size = 64
    
    # Create CfC layer
    cfc = CfCLayer(input_size, hidden_size, num_layers=2)
    
    # Test single-step forward
    x = torch.randn(batch_size, input_size)
    output, state = cfc(x)
    print(f"Single-step output shape: {output.shape}")
    print(f"State shape: {state.shape}")
    
    # Test sequence forward
    x_seq = torch.randn(batch_size, seq_len, input_size)
    output_seq, final_state = cfc(x_seq)
    print(f"Sequence output shape: {output_seq.shape}")
    print(f"Final state shape: {final_state.shape}")
    
    # Check time constants
    tau_values = cfc.get_all_time_constants()
    print(f"\nTime constants statistics:")
    for i, tau in enumerate(tau_values):
        print(f"Layer {i}: min={tau.min():.3f}, max={tau.max():.3f}, mean={tau.mean():.3f}")