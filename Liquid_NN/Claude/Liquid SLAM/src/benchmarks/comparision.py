"""
src/benchmarks/comparison.py

Comprehensive benchmarking suite comparing LNN models against baselines.
Includes performance metrics, speed tests, and visualization.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # For MacOS compatibility

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import pandas as pd


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    model_name: str
    accuracy: float
    inference_time: float  # milliseconds
    memory_usage: float  # MB
    parameters: int
    flops: Optional[float] = None
    extra_metrics: Optional[Dict[str, float]] = None


class ModelBenchmark:
    """
    Benchmark suite for comparing models.
    
    Compares:
        - CfC vs LTC vs Standard RNN/LSTM/GRU
        - Accuracy metrics
        - Inference speed
        - Memory usage
        - Parameter count
    """
    
    def __init__(
        self,
        device: torch.device = None,
        save_dir: str = './benchmark_results'
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[BenchmarkResult] = []
    
    def benchmark_model(
        self,
        model: nn.Module,
        dataloader,
        model_name: str,
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> BenchmarkResult:
        """
        Benchmark a single model.
        
        Args:
            model: Model to benchmark
            dataloader: Test data loader
            model_name: Name for identification
            num_iterations: Number of inference iterations
            warmup_iterations: Warmup iterations
        
        Returns:
            BenchmarkResult with all metrics
        """
        print(f"\nBenchmarking {model_name}...")
        model = model.to(self.device)
        model.eval()
        
        # Get sample batch
        sample_batch = next(iter(dataloader))
        sample_input = sample_batch['events'].to(self.device)
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = model(sample_input)
        
        # Synchronize GPU
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Measure inference time
        inference_times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.time()
                _ = model(sample_input)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                inference_times.append((time.time() - start) * 1000)  # ms
        
        avg_inference_time = np.mean(inference_times)
        
        # Measure memory usage
        memory_mb = self._measure_memory(model, sample_input)
        
        # Measure accuracy
        accuracy, extra_metrics = self._measure_accuracy(model, dataloader)
        
        result = BenchmarkResult(
            model_name=model_name,
            accuracy=accuracy,
            inference_time=avg_inference_time,
            memory_usage=memory_mb,
            parameters=num_params,
            extra_metrics=extra_metrics
        )
        
        self.results.append(result)
        
        print(f"  ✓ Accuracy: {accuracy:.4f}")
        print(f"  ✓ Inference Time: {avg_inference_time:.2f} ms")
        print(f"  ✓ Memory Usage: {memory_mb:.2f} MB")
        print(f"  ✓ Parameters: {num_params:,}")
        
        return result
    
    def _measure_memory(self, model: nn.Module, sample_input: torch.Tensor) -> float:
        """Measure model memory usage."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = model(sample_input)
            memory_bytes = torch.cuda.max_memory_allocated()
            return memory_bytes / (1024 ** 2)  # Convert to MB
        else:
            # Approximate CPU memory
            param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
            return param_memory / (1024 ** 2)
    
    def _measure_accuracy(
        self,
        model: nn.Module,
        dataloader,
        max_batches: int = 50
    ) -> Tuple[float, Dict[str, float]]:
        """Measure model accuracy on test set."""
        model.eval()
        correct = 0
        total = 0
        all_losses = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= max_batches:
                    break
                
                inputs = batch['events'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs, _ = model(inputs)
                
                # Assuming classification task
                if outputs.dim() > 2:
                    outputs = outputs.mean(dim=1)  # Average over sequence
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Compute loss
                loss = nn.CrossEntropyLoss()(outputs, labels)
                all_losses.append(loss.item())
        
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = np.mean(all_losses) if all_losses else 0.0
        
        extra_metrics = {
            'loss': avg_loss,
            'correct': correct,
            'total': total
        }
        
        return accuracy, extra_metrics
    
    def compare_models(
        self,
        models: Dict[str, nn.Module],
        dataloader,
        save_plots: bool = True
    ) -> pd.DataFrame:
        """
        Compare multiple models.
        
        Args:
            models: Dictionary of {name: model}
            dataloader: Test data loader
            save_plots: Whether to save comparison plots
        
        Returns:
            DataFrame with comparison results
        """
        print("=" * 60)
        print("BENCHMARKING MODELS")
        print("=" * 60)
        
        # Benchmark each model
        for name, model in models.items():
            self.benchmark_model(model, dataloader, name)
        
        # Create comparison DataFrame
        df = self._create_comparison_df()
        
        # Print comparison table
        print("\n" + "=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)
        print(df.to_string(index=False))
        
        # Save results
        csv_path = self.save_dir / 'benchmark_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")
        
        # Generate plots
        if save_plots:
            self.plot_comparisons()
        
        return df
    
    def _create_comparison_df(self) -> pd.DataFrame:
        """Create comparison DataFrame from results."""
        data = []
        for result in self.results:
            row = {
                'Model': result.model_name,
                'Accuracy (%)': result.accuracy * 100,
                'Inference Time (ms)': result.inference_time,
                'Memory (MB)': result.memory_usage,
                'Parameters (M)': result.parameters / 1e6
            }
            
            if result.extra_metrics:
                row['Loss'] = result.extra_metrics.get('loss', 0.0)
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def plot_comparisons(self):
        """Generate comparison plots."""
        if len(self.results) < 2:
            print("Need at least 2 models for comparison plots")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        model_names = [r.model_name for r in self.results]
        
        # 1. Accuracy comparison
        accuracies = [r.accuracy * 100 for r in self.results]
        axes[0, 0].bar(model_names, accuracies, color='steelblue', alpha=0.7)
        axes[0, 0].set_ylabel('Accuracy (%)')
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. Inference time comparison
        inference_times = [r.inference_time for r in self.results]
        axes[0, 1].bar(model_names, inference_times, color='coral', alpha=0.7)
        axes[0, 1].set_ylabel('Inference Time (ms)')
        axes[0, 1].set_title('Inference Speed Comparison')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. Memory usage comparison
        memory_usage = [r.memory_usage for r in self.results]
        axes[1, 0].bar(model_names, memory_usage, color='lightgreen', alpha=0.7)
        axes[1, 0].set_ylabel('Memory Usage (MB)')
        axes[1, 0].set_title('Memory Usage Comparison')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 4. Parameter count comparison
        param_counts = [r.parameters / 1e6 for r in self.results]
        axes[1, 1].bar(model_names, param_counts, color='plum', alpha=0.7)
        axes[1, 1].set_ylabel('Parameters (Millions)')
        axes[1, 1].set_title('Model Size Comparison')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_path = self.save_dir / 'model_comparison.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plots saved to {save_path}")
        plt.close()
        
        # Additional: Efficiency plot (accuracy vs speed)
        self._plot_efficiency()
    
    def _plot_efficiency(self):
        """Plot accuracy vs inference time efficiency."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for result in self.results:
            ax.scatter(
                result.inference_time,
                result.accuracy * 100,
                s=result.parameters / 1e5,  # Size proportional to params
                alpha=0.6,
                label=result.model_name
            )
            ax.annotate(
                result.model_name,
                (result.inference_time, result.accuracy * 100),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9
            )
        
        ax.set_xlabel('Inference Time (ms)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Model Efficiency (Accuracy vs Speed)\nBubble size = parameter count')
        ax.grid(alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        save_path = self.save_dir / 'efficiency_plot.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Efficiency plot saved to {save_path}")
        plt.close()


class BaselineModels:
    """Factory for creating baseline models for comparison."""
    
    @staticmethod
    def create_lstm_baseline(
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2
    ) -> nn.Module:
        """Create LSTM baseline."""
        class LSTMModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size, hidden_size, num_layers,
                    batch_first=True, dropout=0.1
                )
                self.fc = nn.Linear(hidden_size, output_size)
            
            def forward(self, x, state=None):
                if x.dim() == 4:  # Event tensor
                    b, t, c, h, w = x.shape
                    x = x.view(b, t, -1)
                h, state = self.lstm(x, state)
                out = self.fc(h)
                return out, state
        
        return LSTMModel()
    
    @staticmethod
    def create_gru_baseline(
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2
    ) -> nn.Module:
        """Create GRU baseline."""
        class GRUModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = nn.GRU(
                    input_size, hidden_size, num_layers,
                    batch_first=True, dropout=0.1
                )
                self.fc = nn.Linear(hidden_size, output_size)
            
            def forward(self, x, state=None):
                if x.dim() == 4:
                    b, t, c, h, w = x.shape
                    x = x.view(b, t, -1)
                h, state = self.gru(x, state)
                out = self.fc(h)
                return out, state
        
        return GRUModel()
    
    @staticmethod
    def create_transformer_baseline(
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        nhead: int = 4
    ) -> nn.Module:
        """Create Transformer baseline."""
        class TransformerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_proj = nn.Linear(input_size, hidden_size)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=nhead,
                    dim_feedforward=hidden_size * 4,
                    dropout=0.1,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
                self.fc = nn.Linear(hidden_size, output_size)
            
            def forward(self, x, state=None):
                if x.dim() == 4:
                    b, t, c, h, w = x.shape
                    x = x.view(b, t, -1)
                x = self.input_proj(x)
                h = self.transformer(x)
                out = self.fc(h)
                return out, None
        
        return TransformerModel()


def run_comprehensive_benchmark(
    test_dataloader,
    save_dir: str = './benchmark_results'
):
    """
    Run comprehensive benchmark comparing all model types.
    
    Args:
        test_dataloader: Test data loader
        save_dir: Directory to save results
    """
    import sys
    sys.path.insert(0, '../src')
    from models.cfc_network import CfCSequenceModel, EventCfCModel
    from core.cfc_cell import CfCLayer
    from core.ltc_cell import LTCLayer
    
    print("=" * 60)
    print("COMPREHENSIVE MODEL BENCHMARK")
    print("=" * 60)
    
    # Get sample batch to determine sizes
    sample_batch = next(iter(test_dataloader))
    sample_events = sample_batch['events']
    batch_size, seq_len, channels, height, width = sample_events.shape
    input_size = channels * height * width
    output_size = 10  # Assuming 10 classes
    hidden_size = 128
    
    # Create models
    models = {
        'CfC (Ours)': CfCSequenceModel(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=2
        ),
        '''
        'LTC': nn.Sequential(
            nn.Flatten(2),
            nn.Linear(input_size, hidden_size),
            nn.Flatten(LTCLayer(hidden_size, hidden_size, ode_solver='euler')),
            nn.Linear(hidden_size, output_size)
        ),
        '''
        'LSTM': BaselineModels.create_lstm_baseline(
            input_size, hidden_size, output_size
        ),
        'GRU': BaselineModels.create_gru_baseline(
            input_size, hidden_size, output_size
        ),
        'Transformer': BaselineModels.create_transformer_baseline(
            input_size, hidden_size, output_size
        )
    }
    
    # Run benchmark
    benchmark = ModelBenchmark(save_dir=save_dir)
    results_df = benchmark.compare_models(models, test_dataloader)
    
    return results_df, benchmark


if __name__ == "__main__":
    print("Testing Benchmark Suite")
    print("=" * 60)
    import sys
    sys.path.insert(0, '../src')
    # Create synthetic test data
    from data.event_dataset import create_event_dataloader
    
    test_loader = create_event_dataloader(
        'synthetic',
        {
            'num_samples': 100,
            'sequence_length': 10,
            'spatial_size': (32, 32)
        },
        batch_size=8,
        num_workers=0
    )
    
    print("\nRunning comprehensive benchmark...")
    results_df, benchmark = run_comprehensive_benchmark(test_loader)
    
    print("\n✓ Benchmark completed successfully")
    print("\nSummary:")
    print(results_df)