"""
src/core/wiring.py

Neural Circuit Policies (NCP) Wiring Implementation
Based on: "Liquid Structural State-Space Models" (Lechner et al., 2020)

Provides structured, sparse connectivity patterns inspired by biological
neural circuits (C. elegans). Reduces parameters and improves interpretability.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # For macOS compatibility

import torch #type: ignore
import torch.nn as nn #type: ignore
import numpy as np #type: ignore
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class WiringConfig:
    """Configuration for NCP wiring architecture."""
    sensory_size: int  # Input neurons
    inter_size: int    # Interneurons
    command_size: int  # Command neurons
    motor_size: int    # Output neurons
    
    # Connectivity ratios (fraction of possible connections)
    sensory_to_inter: float = 0.5
    inter_to_command: float = 0.5
    command_to_motor: float = 1.0
    inter_recurrent: float = 0.3
    command_recurrent: float = 0.3
    
    # Connectivity types
    use_sensory_to_command: bool = True
    use_sensory_to_motor: bool = False
    
    def total_neurons(self) -> int:
        """Total number of neurons in the circuit."""
        return (self.sensory_size + self.inter_size + 
                self.command_size + self.motor_size)


class NCPWiring:
    """
    Neural Circuit Policy wiring structure.
    
    Defines a hierarchical connectivity pattern:
        Sensory → Inter → Command → Motor
        
    With recurrent connections in inter and command layers.
    Inspired by C. elegans nervous system architecture.
    
    Args:
        config: Wiring configuration
        seed: Random seed for reproducible connectivity
    """
    
    def __init__(self, config: WiringConfig, seed: int = 42):
        self.config = config
        self.seed = seed
        np.random.seed(seed)
        
        # Neuron type indices
        self.sensory_indices = list(range(config.sensory_size))
        
        self.inter_indices = list(range(
            config.sensory_size,
            config.sensory_size + config.inter_size
        ))
        
        self.command_indices = list(range(
            config.sensory_size + config.inter_size,
            config.sensory_size + config.inter_size + config.command_size
        ))
        
        self.motor_indices = list(range(
            config.sensory_size + config.inter_size + config.command_size,
            config.total_neurons()
        ))
        
        # Build connectivity matrix
        self.adjacency_matrix = self._build_connectivity()
        
    def _build_connectivity(self) -> np.ndarray:
        """
        Build sparse connectivity matrix following NCP principles.
        
        Returns:
            adjacency_matrix: (N, N) binary matrix where N = total neurons
                             adjacency[i,j] = 1 if connection from j to i exists
        """
        N = self.config.total_neurons()
        adj = np.zeros((N, N), dtype=np.float32)
        
        # 1. Sensory → Inter connections
        self._connect_layers(
            adj,
            self.sensory_indices,
            self.inter_indices,
            self.config.sensory_to_inter
        )
        
        # 2. Sensory → Command connections (optional)
        if self.config.use_sensory_to_command:
            self._connect_layers(
                adj,
                self.sensory_indices,
                self.command_indices,
                self.config.sensory_to_inter * 0.3  # Sparse
            )
        
        # 3. Sensory → Motor connections (optional, usually not used)
        if self.config.use_sensory_to_motor:
            self._connect_layers(
                adj,
                self.sensory_indices,
                self.motor_indices,
                0.1
            )
        
        # 4. Inter → Command connections
        self._connect_layers(
            adj,
            self.inter_indices,
            self.command_indices,
            self.config.inter_to_command
        )
        
        # 5. Command → Motor connections (usually fully connected)
        self._connect_layers(
            adj,
            self.command_indices,
            self.motor_indices,
            self.config.command_to_motor
        )
        
        # 6. Inter recurrent connections
        self._connect_recurrent(
            adj,
            self.inter_indices,
            self.config.inter_recurrent
        )
        
        # 7. Command recurrent connections
        self._connect_recurrent(
            adj,
            self.command_indices,
            self.config.command_recurrent
        )
        
        return adj
    
    def _connect_layers(
        self,
        adj: np.ndarray,
        source_indices: List[int],
        target_indices: List[int],
        connection_prob: float
    ):
        """Create feedforward connections between two layers."""
        for target_idx in target_indices:
            for source_idx in source_indices:
                if np.random.rand() < connection_prob:
                    adj[target_idx, source_idx] = 1.0
    
    def _connect_recurrent(
        self,
        adj: np.ndarray,
        indices: List[int],
        connection_prob: float
    ):
        """Create recurrent connections within a layer."""
        for i in indices:
            for j in indices:
                if i != j and np.random.rand() < connection_prob:
                    adj[i, j] = 1.0
    
    def get_weight_mask(self) -> torch.Tensor:
        """
        Get binary mask for weight matrix.
        
        Returns:
            mask: (N, N) binary tensor for masking weight connections
        """
        return torch.from_numpy(self.adjacency_matrix)
    
    def get_layer_masks(self) -> Dict[str, torch.Tensor]:
        """
        Get masks for different neuron types.
        
        Returns:
            Dictionary of masks for each layer type
        """
        N = self.config.total_neurons()
        
        masks = {}
        for name, indices in [
            ("sensory", self.sensory_indices),
            ("inter", self.inter_indices),
            ("command", self.command_indices),
            ("motor", self.motor_indices)
        ]:
            mask = torch.zeros(N, dtype=torch.bool)
            mask[indices] = True
            masks[name] = mask
        
        return masks
    
    def visualize(self, figsize=(12, 8)):
        """
        Visualize the wiring diagram.
        
        Args:
            figsize: Figure size for matplotlib
        """
        import matplotlib.pyplot as plt #type: ignore
        import matplotlib.patches as mpatches #type: ignore
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Adjacency matrix
        im = ax1.imshow(self.adjacency_matrix, cmap='binary', aspect='auto')
        ax1.set_title("Connectivity Matrix")
        ax1.set_xlabel("Source Neuron")
        ax1.set_ylabel("Target Neuron")
        
        # Add grid lines separating layers
        boundaries = [
            0,
            self.config.sensory_size,
            self.config.sensory_size + self.config.inter_size,
            self.config.sensory_size + self.config.inter_size + self.config.command_size,
            self.config.total_neurons()
        ]
        
        for boundary in boundaries[1:-1]:
            ax1.axhline(boundary - 0.5, color='red', linewidth=2)
            ax1.axvline(boundary - 0.5, color='red', linewidth=2)
        
        plt.colorbar(im, ax=ax1)
        
        # Plot 2: Network statistics
        stats = self.get_statistics()
        
        labels = list(stats.keys())
        values = list(stats.values())
        
        ax2.barh(labels, values, color='steelblue')
        ax2.set_xlabel("Value")
        ax2.set_title("Network Statistics")
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("ncp_wiring.png", dpi=150, bbox_inches='tight')
        print("Saved visualization to ncp_wiring.png")
        
        return fig
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Compute network statistics.
        
        Returns:
            Dictionary of statistics
        """
        total_possible = self.config.total_neurons() ** 2
        total_connections = self.adjacency_matrix.sum()
        
        stats = {
            "Total Neurons": self.config.total_neurons(),
            "Total Connections": int(total_connections),
            "Sparsity": 1.0 - (total_connections / total_possible),
            "Sensory Neurons": self.config.sensory_size,
            "Interneurons": self.config.inter_size,
            "Command Neurons": self.config.command_size,
            "Motor Neurons": self.config.motor_size,
        }
        
        return stats


class NCPCell(nn.Module):
    """
    NCP Cell with structured connectivity.
    
    Combines CfC dynamics with NCP wiring for efficient,
    interpretable neural circuits.
    
    Args:
        wiring: NCPWiring object defining connectivity
        mode: CfC mode (default, pure, no_gate)
    """
    
    def __init__(
        self,
        wiring: NCPWiring,
        mode: str = "default",
    ):
        super().__init__()
        
        self.wiring = wiring
        self.config = wiring.config
        self.mode = mode
        
        N = self.config.total_neurons()
        
        # Time constants for each neuron
        self.tau = nn.Parameter(torch.rand(N) * 2.0 + 0.1)
        
        # Weight matrix (will be masked)
        self.W = nn.Parameter(torch.randn(N, N) * 0.1)
        
        # Register connectivity mask as buffer (not a parameter)
        self.register_buffer('mask', wiring.get_weight_mask())
        
        # Layer-specific masks
        layer_masks = wiring.get_layer_masks()
        for name, mask in layer_masks.items():
            self.register_buffer(f'{name}_mask', mask)
        
        # Input and output projections
        self.input_proj = nn.Linear(
            self.config.sensory_size,
            self.config.sensory_size,
            bias=False
        )
        
        self.output_proj = nn.Linear(
            self.config.motor_size,
            self.config.motor_size
        )
        
    def forward(
        self,
        input: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        elapsed_time: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Semi-implicit Euler update (vectorized) following Eq.3 (Methods) of Lechner et al. 2020.
        Δ may be scalar or a per-sample tensor of shape (batch,) or (batch,1).
        """
        batch_size = input.size(0)
        N = self.config.total_neurons()

        if state is None:
            state = torch.zeros(batch_size, N, device=input.device, dtype=input.dtype)

        # Project input onto sensory neurons (unchanged)
        sensory_input = self.input_proj(input)  # (B, sensory_size)

        # Prepare working state where sensory neurons are clamped to input
        state_work = state.clone()
        # USE wiring-owned lists directly (robust)
        state_work[:, self.wiring.sensory_indices] = sensory_input

        # Masked weight matrix (targets x sources)
        W_masked = self.W * self.mask  # (N, N)

        # Compute presynaptic activation sigma(x_j) -- shape (B, N)
        sigma_pre = torch.sigmoid(state_work)  # (B, N)

        # Weighted presynaptic currents per target neuron:
        weighted_inputs = torch.matmul(sigma_pre, W_masked.t())  # (B, N)

        # Convert elapsed_time Δ to shape (B,1) for broadcasting
        if isinstance(elapsed_time, (float, int)):
            delta = torch.full((batch_size, 1), float(elapsed_time), device=input.device, dtype=input.dtype)
        else:
            dt = torch.as_tensor(elapsed_time, device=input.device, dtype=input.dtype)
            # allow shape (B,) or (B,1)
            if dt.dim() == 1:
                delta = dt.view(batch_size, 1)
            else:
                delta = dt

        # Map tau -> gli (leak conductance)
        tau_clamped = torch.clamp(self.tau, min=1e-6)  # (N,)
        gli = (1.0 / tau_clamped).view(1, -1)  # shape (1, N), broadcast over batch

        # Numerator and denominator (Cmi=1, x_leak=0)
        numerator = (state / delta) + weighted_inputs  # (B, N)
        denominator = (1.0 / delta) + gli + weighted_inputs  # (B, N)
        denominator = torch.clamp(denominator, min=1e-6)

        # New state by semi-implicit Euler step (vectorized)
        new_state = numerator / denominator  # (B, N)

        # Sensory neurons: overwrite with sensory_input (sensory neurons are driven externally)
        new_state[:, self.wiring.sensory_indices] = sensory_input

        # Compute motor output using wiring motor indices
        motor_state = new_state[:, self.wiring.motor_indices]
        output = self.output_proj(motor_state)

        return output, new_state




def create_standard_wiring(
    input_size: int,
    output_size: int,
    hidden_size: int = 64,
    command_size: int = 16
) -> NCPWiring:
    """
    Create a standard NCP wiring configuration.
    
    Args:
        input_size: Number of input features
        output_size: Number of output features
        hidden_size: Number of interneurons
        command_size: Number of command neurons
    
    Returns:
        NCPWiring object with standard configuration
    """
    config = WiringConfig(
        sensory_size=input_size,
        inter_size=hidden_size,
        command_size=command_size,
        motor_size=output_size,
        sensory_to_inter=0.4,
        inter_to_command=0.5,
        command_to_motor=1.0,
        inter_recurrent=0.2,
        command_recurrent=0.4,
        use_sensory_to_command=True,
        use_sensory_to_motor=False
    )
    
    return NCPWiring(config)


if __name__ == "__main__":
    print("NCP Wiring Test")
    print("=" * 60)
    
    # Create wiring
    wiring = create_standard_wiring(
        input_size=32,
        output_size=8,
        hidden_size=64,
        command_size=16
    )
    
    # Print statistics
    stats = wiring.get_statistics()
    print("\nNetwork Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test NCP cell
    print("\nTesting NCP Cell:")
    cell = NCPCell(wiring)
    
    batch_size = 4
    x = torch.randn(batch_size, 32)
    output, state = cell(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  State shape: {state.shape}")
    
    # Visualize wiring
    print("\nGenerating wiring visualization...")
    wiring.visualize()