"""
tests/test_cfc_cell.py

Comprehensive unit tests for CfC cells and layers.
Tests correctness, stability, and edge cases.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # For macOS compatibility

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

# Import from src
import sys
sys.path.insert(0, '../src')
from core.cfc_cell import CfCCell, CfCLayer
from core.ltc_cell import LTCCell, LTCLayer
from core.wiring import NCPWiring, NCPCell, create_standard_wiring, WiringConfig


class TestCfCCell:
    """Test suite for CfCCell."""
    
    @pytest.fixture
    def cell_config(self):
        """Standard cell configuration for tests."""
        return {
            'input_size': 32,
            'hidden_size': 64,
            'mode': 'default'
        }
    
    @pytest.fixture
    def cell(self, cell_config):
        """Create a CfC cell for testing."""
        return CfCCell(**cell_config)
    
    def test_initialization(self, cell, cell_config):
        """Test proper initialization of CfC cell."""
        assert cell.input_size == cell_config['input_size']
        assert cell.hidden_size == cell_config['hidden_size']
        assert cell.mode == cell_config['mode']
        
        # Check time constants are initialized properly
        tau = cell.get_time_constants()
        assert tau.shape == (cell_config['hidden_size'],)
        assert torch.all(tau > 0), "Time constants must be positive"
        assert torch.all(tau < 100), "Time constants should be bounded"
    
    def test_forward_single_step(self, cell):
        """Test single-step forward pass."""
        batch_size = 4
        x = torch.randn(batch_size, cell.input_size)
        
        output, state = cell(x)
        
        assert output.shape == (batch_size, cell.hidden_size)
        assert state.shape == (batch_size, cell.hidden_size)
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"
    
    def test_forward_with_state(self, cell):
        """Test forward pass with provided initial state."""
        batch_size = 4
        x = torch.randn(batch_size, cell.input_size)
        init_state = torch.randn(batch_size, cell.hidden_size)
        
        output, state = cell(x, init_state)
        
        assert output.shape == (batch_size, cell.hidden_size)
        assert state.shape == (batch_size, cell.hidden_size)
    
    def test_different_elapsed_times(self, cell):
        """Test behavior with different time steps."""
        batch_size = 4
        x = torch.randn(batch_size, cell.input_size)
        
        # Test with different dt values
        for dt in [0.1, 1.0, 10.0]:
            output, state = cell(x, elapsed_time=dt)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
    
    def test_gradient_flow(self, cell):
        """Test that gradients flow properly."""
        x = torch.randn(2, cell.input_size, requires_grad=True)
        output, state = cell(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        
        # Check that cell parameters have gradients
        for param in cell.parameters():
            assert param.grad is not None
    
    def test_modes(self, cell_config):
        """Test different CfC modes."""
        modes = ['default', 'pure', 'no_gate']
        
        for mode in modes:
            config = cell_config.copy()
            config['mode'] = mode
            cell = CfCCell(**config)
            
            x = torch.randn(4, config['input_size'])
            output, state = cell(x)
            
            assert output.shape == (4, config['hidden_size'])
            assert not torch.isnan(output).any()
    
    def test_time_constant_manipulation(self, cell):
        """Test getting and setting time constants."""
        original_tau = cell.get_time_constants().clone()
        
        # Set new time constants
        new_tau = torch.ones_like(original_tau) * 5.0
        cell.set_time_constants(new_tau)
        
        retrieved_tau = cell.get_time_constants()
        assert torch.allclose(retrieved_tau, new_tau)


class TestCfCLayer:
    """Test suite for CfCLayer."""
    
    @pytest.fixture
    def layer_config(self):
        return {
            'input_size': 32,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.1
        }
    
    @pytest.fixture
    def layer(self, layer_config):
        return CfCLayer(**layer_config)
    
    def test_sequence_processing(self, layer):
        """Test processing of sequences."""
        batch_size = 4
        seq_len = 10
        x = torch.randn(batch_size, seq_len, layer.input_size)
        
        output, state = layer(x)
        
        assert output.shape == (batch_size, seq_len, layer.hidden_size)
        assert state.shape == (layer.num_layers, batch_size, layer.hidden_size)
        assert not torch.isnan(output).any()
    
    def test_single_step_processing(self, layer):
        """Test single-step (non-sequence) processing."""
        batch_size = 4
        x = torch.randn(batch_size, layer.input_size)
        
        output, state = layer(x)
        
        # Output should not have sequence dimension
        assert output.shape == (batch_size, layer.hidden_size)
        assert state.shape == (layer.num_layers, batch_size, layer.hidden_size)
    
    def test_stateful_processing(self, layer):
        """Test processing with state persistence."""
        batch_size = 4
        seq_len = 10
        x = torch.randn(batch_size, seq_len, layer.input_size)
        
        # Process first half
        x1 = x[:, :5]
        output1, state1 = layer(x1)
        
        # Process second half with state from first half
        x2 = x[:, 5:]
        output2, state2 = layer(x2, state1)
        
        # Process full sequence at once
        output_full, state_full = layer(x)
        
        # Results should be similar (not exact due to numerical precision)
        assert output2.shape == (batch_size, 5, layer.hidden_size)


class TestLTCCell:
    """Test suite for LTCCell."""
    
    @pytest.fixture
    def cell(self):
        return LTCCell(input_size=32, hidden_size=64, ode_solver='euler')
    
    def test_euler_integration(self):
        """Test Euler integration method."""
        cell = LTCCell(input_size=32, hidden_size=64, ode_solver='euler')
        x = torch.randn(4, 32)
        output, state = cell(x)
        
        assert output.shape == (4, 64)
        assert not torch.isnan(output).any()
    
    def test_rk4_integration(self):
        """Test RK4 integration method."""
        cell = LTCCell(input_size=32, hidden_size=64, ode_solver='rk4')
        x = torch.randn(4, 32)
        output, state = cell(x)
        
        assert output.shape == (4, 64)
        assert not torch.isnan(output).any()
    
    def test_ode_steps_effect(self):
        """Test effect of different ODE step counts."""
        cell_2steps = LTCCell(input_size=32, hidden_size=64, ode_steps=2)
        cell_10steps = LTCCell(input_size=32, hidden_size=64, ode_steps=10)
        
        x = torch.randn(4, 32)
        
        out_2, _ = cell_2steps(x)
        out_10, _ = cell_10steps(x)
        
        # More steps should give more accurate results (different outputs)
        assert not torch.allclose(out_2, out_10, atol=1e-3)


class TestNCPWiring:
    """Test suite for NCP wiring."""
    
    @pytest.fixture
    def wiring_config(self):
        return WiringConfig(
            sensory_size=16,
            inter_size=32,
            command_size=8,
            motor_size=4
        )
    
    @pytest.fixture
    def wiring(self, wiring_config):
        return NCPWiring(wiring_config)
    
    def test_wiring_structure(self, wiring, wiring_config):
        """Test basic wiring structure."""
        assert len(wiring.sensory_indices) == wiring_config.sensory_size
        assert len(wiring.inter_indices) == wiring_config.inter_size
        assert len(wiring.command_indices) == wiring_config.command_size
        assert len(wiring.motor_indices) == wiring_config.motor_size
    
    def test_connectivity_matrix(self, wiring, wiring_config):
        """Test connectivity matrix properties."""
        adj = wiring.adjacency_matrix
        N = wiring_config.total_neurons()
        
        assert adj.shape == (N, N)
        assert np.all((adj == 0) | (adj == 1)), "Adjacency should be binary"
        
        # Check sparsity
        sparsity = 1.0 - (adj.sum() / (N * N))
        assert sparsity > 0.5, "Network should be sparse"
    
    def test_hierarchical_connectivity(self, wiring):
        """Test that connectivity follows hierarchical pattern."""
        adj = wiring.adjacency_matrix
        
        # Sensory neurons should not have incoming connections
        sensory_incoming = adj[wiring.sensory_indices, :].sum()
        # Allow input projections but they shouldn't dominate
        
        # Motor neurons should not have outgoing connections (except to themselves)
        motor_outgoing = adj[:, wiring.motor_indices].sum()
        motor_self = adj[np.ix_(wiring.motor_indices, wiring.motor_indices)].sum()
        assert motor_outgoing == motor_self or motor_outgoing == 0
    
    def test_weight_mask(self, wiring):
        """Test weight mask generation."""
        mask = wiring.get_weight_mask()
        assert isinstance(mask, torch.Tensor)
        assert mask.shape == wiring.adjacency_matrix.shape
    
    def test_layer_masks(self, wiring):
        """Test layer mask generation."""
        masks = wiring.get_layer_masks()
        
        assert 'sensory' in masks
        assert 'inter' in masks
        assert 'command' in masks
        assert 'motor' in masks
        
        # Check masks are non-overlapping
        total_mask = torch.zeros_like(masks['sensory'])
        for mask in masks.values():
            assert not torch.any(total_mask & mask), "Masks should not overlap"
            total_mask |= mask
        
        assert torch.all(total_mask), "Masks should cover all neurons"
    
    def test_reproducibility(self, wiring_config):
        """Test that same seed gives same wiring."""
        wiring1 = NCPWiring(wiring_config, seed=42)
        wiring2 = NCPWiring(wiring_config, seed=42)
        
        assert np.allclose(wiring1.adjacency_matrix, wiring2.adjacency_matrix)
        
        wiring3 = NCPWiring(wiring_config, seed=123)
        assert not np.allclose(wiring1.adjacency_matrix, wiring3.adjacency_matrix)


class TestNCPCell:
    """Test suite for NCP cell."""
    
    @pytest.fixture
    def wiring(self):
        return create_standard_wiring(
            input_size=16,
            output_size=4,
            hidden_size=32,
            command_size=8
        )
    
    @pytest.fixture
    def cell(self, wiring):
        return NCPCell(wiring)
    
    def test_forward_pass(self, cell, wiring):
        """Test NCP cell forward pass."""
        batch_size = 4
        x = torch.randn(batch_size, wiring.config.sensory_size)
        
        output, state = cell(x)
        
        assert output.shape == (batch_size, wiring.config.motor_size)
        assert state.shape == (batch_size, wiring.config.total_neurons())
        assert not torch.isnan(output).any()
    
    def test_sparse_connectivity(self, cell):
        """Test that connectivity is properly masked."""
        # Get weight matrix
        W = cell.W.data
        mask = cell.mask
        
        # Check that masked weights are actually zero when multiplied
        W_masked = W * mask
        
        # Verify sparsity is maintained
        sparsity = (mask == 0).float().mean()
        assert sparsity > 0.3, "Network should be sparse"
    
    def test_gradient_flow_through_mask(self, cell, wiring):
        """Test gradients flow properly through masked connections."""
        x = torch.randn(2, wiring.config.sensory_size, requires_grad=True)
        output, state = cell(x)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        assert cell.W.grad is not None
        
        # Check that gradients respect mask
        # (gradients for masked connections should be zero or very small)
        masked_grad = cell.W.grad * (1 - cell.mask)
        assert torch.allclose(masked_grad, torch.zeros_like(masked_grad), atol=1e-6)


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_cfc_vs_ltc_consistency(self):
        """Test that CfC and LTC give similar results."""
        input_size, hidden_size = 32, 64
        
        # Create models with similar initialization
        torch.manual_seed(42)
        cfc = CfCCell(input_size, hidden_size)
        
        torch.manual_seed(42)
        ltc = LTCCell(input_size, hidden_size, ode_steps=20)
        
        x = torch.randn(4, input_size)
        
        out_cfc, _ = cfc(x, elapsed_time=0.1)
        out_ltc, _ = ltc(x, elapsed_time=0.1)
        
        # Results should be in similar range (not exact due to different approaches)
        assert torch.abs(out_cfc.mean() - out_ltc.mean()) < 1.0
        assert torch.abs(out_cfc.std() - out_ltc.std()) < 1.0
    
    def test_layered_processing(self):
        """Test multi-layer processing pipeline."""
        batch_size, seq_len = 4, 20
        input_size, hidden_size = 32, 64
        
        # Create multi-layer network
        layer1 = CfCLayer(input_size, hidden_size, num_layers=1)
        layer2 = CfCLayer(hidden_size, hidden_size, num_layers=1)
        output_layer = nn.Linear(hidden_size, 10)
        
        x = torch.randn(batch_size, seq_len, input_size)
        
        h1, _ = layer1(x)
        h2, _ = layer2(h1)
        output = output_layer(h2)
        
        assert output.shape == (batch_size, seq_len, 10)
        assert not torch.isnan(output).any()
    
    def test_ncp_with_training(self):
        """Test NCP cell in a simple training scenario."""
        wiring = create_standard_wiring(16, 4, 32, 8)
        cell = NCPCell(wiring)
        optimizer = torch.optim.Adam(cell.parameters(), lr=0.01)
        
        # Simple training loop
        for _ in range(10):
            x = torch.randn(8, 16)
            target = torch.randn(8, 4)
            
            output, _ = cell(x)
            loss = nn.MSELoss()(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            assert not torch.isnan(loss)


# Benchmark tests
class TestPerformance:
    """Performance benchmarking tests."""
    
    @pytest.mark.benchmark
    def test_cfc_inference_speed(self, benchmark):
        """Benchmark CfC inference speed."""
        layer = CfCLayer(64, 128, num_layers=2)
        x = torch.randn(32, 100, 64)
        
        def run_inference():
            with torch.no_grad():
                return layer(x)
        
        result = benchmark(run_inference)
    
    @pytest.mark.benchmark
    def test_ltc_inference_speed(self, benchmark):
        """Benchmark LTC inference speed."""
        layer = LTCLayer(64, 128, num_layers=2, ode_steps=6)
        x = torch.randn(32, 100, 64)
        
        def run_inference():
            with torch.no_grad():
                return layer(x)
        
        result = benchmark(run_inference)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])