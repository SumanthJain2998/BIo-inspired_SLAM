"""
src/models/cfc_network.py

Complete CfC Network implementations for various tasks.
Includes models for time-series prediction, classification, and sequence-to-sequence.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
import sys
sys.path.insert(0, '../src')

from core.cfc_cell import CfCLayer
from core.ltc_cell import LTCLayer
from core.wiring import NCPWiring, NCPCell, create_standard_wiring


class CfCSequenceModel(nn.Module):
    """
    CfC-based sequence model for time-series tasks.
    
    Architecture:
        Input → CfC Layers → Dense → Output
    
    Suitable for:
        - Time series forecasting
        - Sequence classification
        - Anomaly detection
    
    Args:
        input_size: Input feature dimension
        hidden_size: Hidden state dimension
        output_size: Output dimension
        num_layers: Number of CfC layers
        dropout: Dropout rate
        use_ncp: Whether to use NCP wiring
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_ncp: bool = False,
        **kwargs
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # Input projection (optional normalization)
        self.input_norm = nn.LayerNorm(input_size)
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Recurrent backbone
        if use_ncp:
            # Use NCP wiring for structured sparsity
            wiring = create_standard_wiring(
                input_size=hidden_size,
                output_size=hidden_size,
                hidden_size=hidden_size * 2,
                command_size=hidden_size // 4
            )
            self.rnn = nn.ModuleList([NCPCell(wiring) for _ in range(num_layers)])
        else:
            # Standard CfC layers
            self.rnn = CfCLayer(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                **kwargs
            )
        
        self.use_ncp = use_ncp
        
        # Output head
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        return_states: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input (batch, seq_len, input_size) or (batch, input_size)
            state: Initial hidden state
            return_states: Whether to return all hidden states
        
        Returns:
            output: Predictions (batch, seq_len, output_size)
            states: Hidden states if return_states=True
        """
        # Input projection
        x = self.input_norm(x)
        x = self.input_proj(x)
        
        # Recurrent processing
        if self.use_ncp:
            # Process through NCP cells
            is_sequence = x.dim() == 3
            if not is_sequence:
                x = x.unsqueeze(1)
            
            batch_size, seq_len, _ = x.shape
            outputs = []
            states_list = []
            
            for t in range(seq_len):
                h = x[:, t]
                for cell in self.rnn:
                    h, _ = cell(h, state)
                outputs.append(h)
                if return_states:
                    states_list.append(h)
            
            h = torch.stack(outputs, dim=1)
            if not is_sequence:
                h = h.squeeze(1)
        else:
            h, state = self.rnn(x, state)
        
        # Output projection
        h = self.output_norm(h)
        output = self.output_head(h)
        
        if return_states:
            return output, h
        return output, state
    
    def predict_step(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        num_steps: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-step prediction (autoregressive).
        
        Args:
            x: Initial input (batch, input_size)
            state: Initial state
            num_steps: Number of steps to predict
        
        Returns:
            predictions: (batch, num_steps, output_size)
            final_state: Final hidden state
        """
        predictions = []
        current_input = x
        
        for _ in range(num_steps):
            pred, state = self.forward(current_input, state)
            predictions.append(pred)
            
            # Use prediction as next input (requires compatible dims)
            if pred.size(-1) == self.input_size:
                current_input = pred
            else:
                # Pad or project to match input_size
                current_input = torch.cat([
                    pred,
                    torch.zeros(
                        pred.size(0), self.input_size - pred.size(-1),
                        device=pred.device
                    )
                ], dim=-1)
        
        return torch.stack(predictions, dim=1), state


class CfCAutoencoder(nn.Module):
    """
    CfC-based autoencoder for sequence representation learning.
    
    Architecture:
        Encoder: Input → CfC → Latent
        Decoder: Latent → CfC → Reconstruction
    
    Args:
        input_size: Input feature dimension
        hidden_size: Hidden state dimension
        latent_size: Latent representation dimension
        num_layers: Number of encoder/decoder layers
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        latent_size: int,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        
        # Encoder
        self.encoder = CfCLayer(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        self.encoder_output = nn.Linear(hidden_size, latent_size)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_size, hidden_size)
        self.decoder = CfCLayer(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        self.decoder_output = nn.Linear(hidden_size, input_size)
    
    def encode(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode sequence to latent representation.
        
        Args:
            x: Input sequence (batch, seq_len, input_size)
        
        Returns:
            latent: Latent representation (batch, latent_size)
            hidden_states: Encoder hidden states
        """
        h, states = self.encoder(x)
        
        # Use final hidden state
        if h.dim() == 3:
            h = h[:, -1, :]  # Take last time step
        
        latent = self.encoder_output(h)
        return latent, states
    
    def decode(
        self,
        latent: torch.Tensor,
        seq_len: int
    ) -> torch.Tensor:
        """
        Decode latent representation to sequence.
        
        Args:
            latent: Latent representation (batch, latent_size)
            seq_len: Length of output sequence
        
        Returns:
            reconstruction: (batch, seq_len, input_size)
        """
        h = self.decoder_input(latent)
        
        # Repeat latent for each time step
        h = h.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Decode
        h, _ = self.decoder(h)
        reconstruction = self.decoder_output(h)
        
        return reconstruction
    
    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full autoencoder forward pass.
        
        Args:
            x: Input (batch, seq_len, input_size)
        
        Returns:
            reconstruction: (batch, seq_len, input_size)
            latent: (batch, latent_size)
        """
        seq_len = x.size(1)
        latent, _ = self.encode(x)
        reconstruction = self.decode(latent, seq_len)
        return reconstruction, latent


class EventCfCModel(nn.Module):
    """
    CfC model specifically designed for event camera data.
    
    Processes asynchronous event streams with continuous-time dynamics.
    Ideal for event-based vision tasks.
    
    Args:
        input_channels: Number of input channels (typically 2 for polarity)
        feature_dim: Intermediate feature dimension
        hidden_size: CfC hidden size
        output_size: Output dimension
        spatial_size: Input spatial size (H, W)
    """
    
    def __init__(
        self,
        input_channels: int = 2,
        feature_dim: int = 64,
        hidden_size: int = 128,
        output_size: int = 6,  # e.g., 6-DOF pose
        spatial_size: Tuple[int, int] = (128, 128),
        num_cfc_layers: int = 2
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.spatial_size = spatial_size
        
        # Event feature extractor (spatial processing)
        self.event_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Temporal processing with CfC
        cfc_input_size = feature_dim * 4 * 4
        self.temporal_encoder = CfCLayer(
            input_size=cfc_input_size,
            hidden_size=hidden_size,
            num_layers=num_cfc_layers,
            dropout=0.1
        )
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(
        self,
        events: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process event stream.
        
        Args:
            events: Event tensor (batch, seq_len, C, H, W) or (batch, C, H, W)
            state: Previous hidden state
            timestamps: Time differences for adaptive integration
        
        Returns:
            output: Predictions (batch, seq_len, output_size)
            state: Updated hidden state
        """
        is_sequence = events.dim() == 5
        
        if is_sequence:
            batch, seq_len, C, H, W = events.shape
            # Reshape for batch processing
            events = events.view(batch * seq_len, C, H, W)
        
        # Extract spatial features
        features = self.event_encoder(events)
        features = features.flatten(1)  # (batch * seq_len, feature_dim)
        
        if is_sequence:
            # Reshape back to sequence
            features = features.view(batch, seq_len, -1)
        
        # Temporal processing
        # Use timestamps for adaptive time constants if provided
        elapsed_time = 1.0
        if timestamps is not None:
            elapsed_time = timestamps.mean().item()
        
        h, state = self.temporal_encoder(features, state, elapsed_time)
        
        # Output
        output = self.output_head(h)
        
        return output, state


class HybridCfCLTC(nn.Module):
    """
    Hybrid model combining CfC (fast) and LTC (accurate) layers.
    
    Uses CfC for most processing and LTC for critical dynamics.
    Balances speed and accuracy.
    
    Args:
        input_size: Input dimension
        hidden_size: Hidden dimension
        output_size: Output dimension
        num_cfc_layers: Number of fast CfC layers
        num_ltc_layers: Number of accurate LTC layers
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_cfc_layers: int = 3,
        num_ltc_layers: int = 1
    ):
        super().__init__()
        
        # Fast CfC layers for initial processing
        self.cfc_layers = CfCLayer(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_cfc_layers,
            dropout=0.1
        )
        
        # Accurate LTC layers for critical dynamics
        self.ltc_layers = LTCLayer(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_ltc_layers,
            ode_solver='rk4',
            ode_steps=10
        )
        
        # Output
        self.output = nn.Linear(hidden_size, output_size)
    
    def forward(
        self,
        x: torch.Tensor,
        cfc_state: Optional[torch.Tensor] = None,
        ltc_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through hybrid model.
        
        Args:
            x: Input tensor
            cfc_state: CfC hidden state
            ltc_state: LTC hidden state
        
        Returns:
            output: Predictions
            states: (cfc_state, ltc_state)
        """
        # Fast processing with CfC
        h, cfc_state = self.cfc_layers(x, cfc_state)
        
        # Accurate processing with LTC
        h, ltc_state = self.ltc_layers(h, ltc_state)
        
        # Output
        output = self.output(h)
        
        return output, (cfc_state, ltc_state)


def create_model(
    model_type: str,
    config: Dict[str, Any]
) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: Type of model to create
        config: Model configuration dictionary
    
    Returns:
        Instantiated model
    """
    models = {
        'sequence': CfCSequenceModel,
        'autoencoder': CfCAutoencoder,
        'event': EventCfCModel,
        'hybrid': HybridCfCLTC
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return models[model_type](**config)


if __name__ == "__main__":
    print("Testing Complete Network Models")
    print("=" * 60)
    
    # Test CfCSequenceModel
    print("\n1. Testing CfCSequenceModel")
    model = CfCSequenceModel(
        input_size=32,
        hidden_size=64,
        output_size=10,
        num_layers=2
    )
    x = torch.randn(4, 20, 32)
    output, state = model(x)
    print(f"   Input: {x.shape}")
    print(f"   Output: {output.shape}")
    print(f"   State: {state.shape}")
    
    # Test EventCfCModel
    print("\n2. Testing EventCfCModel")
    event_model = EventCfCModel(
        input_channels=2,
        feature_dim=64,
        hidden_size=128,
        output_size=6
    )
    events = torch.randn(4, 10, 2, 128, 128)  # batch, seq, channels, H, W
    output, state = event_model(events)
    print(f"   Events: {events.shape}")
    print(f"   Output: {output.shape}")
    
    # Test CfCAutoencoder
    print("\n3. Testing CfCAutoencoder")
    ae = CfCAutoencoder(
        input_size=32,
        hidden_size=64,
        latent_size=16
    )
    x = torch.randn(4, 20, 32)
    recon, latent = ae(x)
    print(f"   Input: {x.shape}")
    print(f"   Reconstruction: {recon.shape}")
    print(f"   Latent: {latent.shape}")
    
    print("\n✓ All model tests passed!")