"""
src/utils/visualization.py

Comprehensive visualization utilities for Liquid Neural Networks.
Includes network dynamics, time constants, activations, and training metrics.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


class LNNVisualizer:
    """
    Comprehensive visualizer for Liquid Neural Networks.
    
    Provides visualization for:
    - Network architecture and connectivity
    - Time constant distributions
    - Hidden state dynamics
    - Activation patterns
    - Training metrics
    """
    
    def __init__(self, save_dir: str = "./visualizations"):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save visualization outputs
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def plot_time_constants(
        self,
        model,
        title: str = "Time Constant Distribution",
        save_name: Optional[str] = None
    ):
        """
        Visualize time constant distribution across network.
        
        Args:
            model: CfC or LTC model
            title: Plot title
            save_name: Filename to save (if None, uses title)
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Collect time constants from all layers
        if hasattr(model, 'get_all_time_constants'):
            tau_list = model.get_all_time_constants()
        elif hasattr(model, 'get_time_constants'):
            tau_list = [model.get_time_constants()]
        else:
            raise ValueError("Model does not have time constant methods")
        
        all_tau = torch.cat([tau.flatten() for tau in tau_list]).cpu().numpy()
        
        # 1. Histogram
        axes[0, 0].hist(all_tau, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        axes[0, 0].set_xlabel('Time Constant (τ)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Distribution of Time Constants')
        axes[0, 0].axvline(all_tau.mean(), color='red', linestyle='--', label=f'Mean: {all_tau.mean():.2f}')
        axes[0, 0].legend()
        
        # 2. Log-scale histogram
        axes[0, 1].hist(np.log10(all_tau), bins=50, alpha=0.7, color='coral', edgecolor='black')
        axes[0, 1].set_xlabel('Log₁₀(Time Constant)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Log-Scale Distribution')
        
        # 3. Layer-wise boxplot
        if len(tau_list) > 1:
            layer_data = [tau.cpu().numpy() for tau in tau_list]
            axes[1, 0].boxplot(layer_data, labels=[f'Layer {i}' for i in range(len(layer_data))])
            axes[1, 0].set_ylabel('Time Constant (τ)')
            axes[1, 0].set_title('Time Constants by Layer')
            axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 4. Statistics table
        stats_text = f"""
        Statistics:
        ────────────────
        Mean:    {all_tau.mean():.3f}
        Std:     {all_tau.std():.3f}
        Min:     {all_tau.min():.3f}
        Max:     {all_tau.max():.3f}
        Median:  {np.median(all_tau):.3f}
        Q1:      {np.percentile(all_tau, 25):.3f}
        Q3:      {np.percentile(all_tau, 75):.3f}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
                       verticalalignment='center')
        axes[1, 1].axis('off')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        save_path = self.save_dir / (save_name or f"{title.replace(' ', '_')}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
        plt.close()
    
    def plot_hidden_states(
        self,
        states: torch.Tensor,
        title: str = "Hidden State Evolution",
        max_neurons: int = 50,
        save_name: Optional[str] = None
    ):
        """
        Visualize evolution of hidden states over time.
        
        Args:
            states: Hidden states (time_steps, batch, hidden_size) or (time_steps, hidden_size)
            title: Plot title
            max_neurons: Maximum number of neurons to plot
            save_name: Filename to save
        """
        if states.dim() == 3:
            states = states[:, 0, :]  # Take first batch element
        
        states_np = states.detach().cpu().numpy()
        time_steps, hidden_size = states_np.shape
        
        # Subsample neurons if too many
        if hidden_size > max_neurons:
            indices = np.linspace(0, hidden_size-1, max_neurons, dtype=int)
            states_np = states_np[:, indices]
            hidden_size = max_neurons
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # 1. Heatmap of all neurons over time
        im = axes[0, 0].imshow(states_np.T, aspect='auto', cmap='RdBu_r', 
                               interpolation='nearest')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Neuron Index')
        axes[0, 0].set_title('State Heatmap (All Neurons)')
        plt.colorbar(im, ax=axes[0, 0])
        
        # 2. Individual neuron trajectories (subset)
        num_plot = min(10, hidden_size)
        plot_indices = np.linspace(0, hidden_size-1, num_plot, dtype=int)
        for idx in plot_indices:
            axes[0, 1].plot(states_np[:, idx], alpha=0.6, linewidth=1)
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Activation')
        axes[0, 1].set_title(f'Sample Neuron Trajectories (n={num_plot})')
        axes[0, 1].grid(alpha=0.3)
        
        # 3. Mean and std over neurons
        mean_activity = states_np.mean(axis=1)
        std_activity = states_np.std(axis=1)
        time_axis = np.arange(time_steps)
        
        axes[1, 0].plot(time_axis, mean_activity, color='blue', label='Mean')
        axes[1, 0].fill_between(time_axis, 
                                mean_activity - std_activity,
                                mean_activity + std_activity,
                                alpha=0.3, color='blue', label='±1 Std')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Activation')
        axes[1, 0].set_title('Population Activity')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # 4. Activation distribution over time
        axes[1, 1].hist(states_np[0, :], bins=30, alpha=0.5, label='t=0', color='blue')
        axes[1, 1].hist(states_np[time_steps//2, :], bins=30, alpha=0.5, 
                       label=f't={time_steps//2}', color='green')
        axes[1, 1].hist(states_np[-1, :], bins=30, alpha=0.5, 
                       label=f't={time_steps-1}', color='red')
        axes[1, 1].set_xlabel('Activation Value')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Activation Distribution')
        axes[1, 1].legend()
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.save_dir / (save_name or f"{title.replace(' ', '_')}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
        plt.close()
    
    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        title: str = "Training Progress",
        save_name: Optional[str] = None
    ):
        """
        Plot training and validation metrics.
        
        Args:
            history: Dictionary with 'train_loss', 'val_loss', etc.
            title: Plot title
            save_name: Filename to save
        """
        n_metrics = len(history)
        n_cols = 2
        n_rows = (n_metrics + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, (metric_name, values) in enumerate(history.items()):
            if idx >= len(axes):
                break
                
            axes[idx].plot(values, linewidth=2)
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel(metric_name.replace('_', ' ').title())
            axes[idx].set_title(metric_name.replace('_', ' ').title())
            axes[idx].grid(alpha=0.3)
            
            # Add best value annotation
            best_idx = np.argmin(values) if 'loss' in metric_name.lower() else np.argmax(values)
            best_val = values[best_idx]
            axes[idx].scatter([best_idx], [best_val], color='red', s=100, 
                            zorder=5, label=f'Best: {best_val:.4f}')
            axes[idx].legend()
        
        # Hide unused subplots
        for idx in range(len(history), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.save_dir / (save_name or "training_curves.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
        plt.close()
    
    def plot_gradient_flow(
        self,
        model,
        title: str = "Gradient Flow",
        save_name: Optional[str] = None
    ):
        """
        Visualize gradient magnitudes through the network.
        
        Args:
            model: Neural network model
            title: Plot title
            save_name: Filename to save
        """
        ave_grads = []
        max_grads = []
        layers = []
        
        for name, param in model.named_parameters():
            if param.grad is not None and 'bias' not in name:
                layers.append(name)
                ave_grads.append(param.grad.abs().mean().item())
                max_grads.append(param.grad.abs().max().item())
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        x = np.arange(len(layers))
        width = 0.35
        
        ax.bar(x - width/2, ave_grads, width, label='Mean Gradient', alpha=0.7)
        ax.bar(x + width/2, max_grads, width, label='Max Gradient', alpha=0.7)
        
        ax.set_xlabel('Layers')
        ax.set_ylabel('Gradient Magnitude')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(layers, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        
        save_path = self.save_dir / (save_name or "gradient_flow.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
        plt.close()
    
    def create_state_animation(
        self,
        states: torch.Tensor,
        output_file: str = "state_evolution.gif",
        fps: int = 10,
        max_neurons: int = 30
    ):
        """
        Create animated visualization of state evolution.
        
        Args:
            states: Hidden states (time_steps, batch, hidden_size)
            output_file: Output filename
            fps: Frames per second
            max_neurons: Maximum neurons to animate
        """
        if states.dim() == 3:
            states = states[:, 0, :]
        
        states_np = states.detach().cpu().numpy()
        time_steps, hidden_size = states_np.shape
        
        if hidden_size > max_neurons:
            indices = np.linspace(0, hidden_size-1, max_neurons, dtype=int)
            states_np = states_np[:, indices]
            hidden_size = max_neurons
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Initialize plots
        im = ax1.imshow(states_np[0:1, :].T, aspect='auto', cmap='RdBu_r',
                       vmin=states_np.min(), vmax=states_np.max())
        ax1.set_ylabel('Neuron Index')
        ax1.set_xlabel('Time Window')
        plt.colorbar(im, ax=ax1)
        
        lines = [ax2.plot([], [], alpha=0.6)[0] for _ in range(hidden_size)]
        ax2.set_xlim(0, time_steps)
        ax2.set_ylim(states_np.min(), states_np.max())
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Activation')
        ax2.grid(alpha=0.3)
        
        time_text = ax2.text(0.02, 0.95, '', transform=ax2.transAxes)
        
        def update(frame):
            # Update heatmap
            window_size = min(50, frame + 1)
            start = max(0, frame - window_size + 1)
            im.set_array(states_np[start:frame+1, :].T)
            ax1.set_title(f'State Heatmap (t={frame})')
            
            # Update line plots
            for idx, line in enumerate(lines):
                line.set_data(np.arange(frame + 1), states_np[:frame+1, idx])
            
            time_text.set_text(f'Time: {frame}/{time_steps}')
            
            return [im] + lines + [time_text]
        
        anim = animation.FuncAnimation(fig, update, frames=time_steps,
                                      interval=1000/fps, blit=True)
        
        output_path = self.save_dir / output_file
        anim.save(output_path, writer='pillow', fps=fps)
        print(f"Saved animation to {output_path}")
        plt.close()
    
    def plot_state_space_trajectory(
        self,
        states: torch.Tensor,
        dims: Tuple[int, int, int] = (0, 1, 2),
        title: str = "State Space Trajectory",
        save_name: Optional[str] = None
    ):
        """
        Plot 3D trajectory in state space.
        
        Args:
            states: Hidden states (time_steps, hidden_size)
            dims: Three dimensions to plot
            title: Plot title
            save_name: Filename to save
        """
        if states.dim() == 3:
            states = states[:, 0, :]
        
        states_np = states.detach().cpu().numpy()
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectory
        x, y, z = states_np[:, dims[0]], states_np[:, dims[1]], states_np[:, dims[2]]
        
        # Color by time
        colors = plt.cm.viridis(np.linspace(0, 1, len(x)))
        
        for i in range(len(x) - 1):
            ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], 
                   color=colors[i], linewidth=2, alpha=0.7)
        
        # Mark start and end
        ax.scatter([x[0]], [y[0]], [z[0]], color='green', s=100, 
                  marker='o', label='Start')
        ax.scatter([x[-1]], [y[-1]], [z[-1]], color='red', s=100, 
                  marker='*', label='End')
        
        ax.set_xlabel(f'Neuron {dims[0]}')
        ax.set_ylabel(f'Neuron {dims[1]}')
        ax.set_zlabel(f'Neuron {dims[2]}')
        ax.set_title(title)
        ax.legend()
        
        plt.tight_layout()
        
        save_path = self.save_dir / (save_name or "state_space_trajectory.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
        plt.close()
    
    def plot_ncp_architecture(
        self,
        wiring,
        save_name: str = "ncp_architecture.png"
    ):
        """
        Visualize NCP architecture with improved layout.
        
        Args:
            wiring: NCPWiring object
            save_name: Filename to save
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: Adjacency matrix
        im = axes[0].imshow(wiring.adjacency_matrix, cmap='Blues', aspect='auto')
        axes[0].set_title("Connectivity Matrix", fontsize=12, fontweight='bold')
        axes[0].set_xlabel("Source Neuron")
        axes[0].set_ylabel("Target Neuron")
        
        # Add layer boundaries
        config = wiring.config
        boundaries = [
            0,
            config.sensory_size,
            config.sensory_size + config.inter_size,
            config.sensory_size + config.inter_size + config.command_size,
            config.total_neurons()
        ]
        
        labels = ['Sensory', 'Inter', 'Command', 'Motor']
        for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            mid = (start + end) / 2
            axes[0].axhline(end - 0.5, color='red', linewidth=2, alpha=0.5)
            axes[0].axvline(end - 0.5, color='red', linewidth=2, alpha=0.5)
        
        plt.colorbar(im, ax=axes[0], label='Connection')
        
        # Plot 2: Statistics
        stats = wiring.get_statistics()
        stats_items = list(stats.items())
        
        axes[1].barh([item[0] for item in stats_items], 
                    [item[1] for item in stats_items], 
                    color='steelblue')
        axes[1].set_xlabel("Value")
        axes[1].set_title("Network Statistics", fontsize=12, fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)
        
        # Plot 3: Connectivity pattern
        layer_sizes = [
            config.sensory_size,
            config.inter_size,
            config.command_size,
            config.motor_size
        ]
        layer_names = ['Sensory', 'Inter', 'Command', 'Motor']
        
        axes[2].bar(layer_names, layer_sizes, color=['green', 'blue', 'orange', 'red'], alpha=0.7)
        axes[2].set_ylabel("Number of Neurons")
        axes[2].set_title("Layer Sizes", fontsize=12, fontweight='bold')
        axes[2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
        plt.close()


# Standalone visualization functions
def visualize_event_stream(
    events: np.ndarray,
    height: int = 128,
    width: int = 128,
    time_window: float = 0.05,
    save_path: Optional[str] = None
):
    """
    Visualize event camera stream.
    
    Args:
        events: Event array with columns [x, y, t, p]
        height: Image height
        width: Image width
        time_window: Time window for accumulation
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Event accumulation
    frame = np.zeros((height, width))
    x, y, p = events[:, 0].astype(int), events[:, 1].astype(int), events[:, 3]
    
    for xi, yi, pi in zip(x, y, p):
        if 0 <= xi < width and 0 <= yi < height:
            frame[yi, xi] += 1 if pi > 0 else -1
    
    im1 = axes[0, 0].imshow(frame, cmap='RdBu_r')
    axes[0, 0].set_title('Event Accumulation')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 2. Temporal distribution
    axes[0, 1].hist(events[:, 2], bins=50, alpha=0.7)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Event Count')
    axes[0, 1].set_title('Temporal Distribution')
    
    # 3. Polarity distribution
    pos_events = (events[:, 3] > 0).sum()
    neg_events = (events[:, 3] < 0).sum()
    axes[1, 0].bar(['Positive', 'Negative'], [pos_events, neg_events], 
                  color=['red', 'blue'], alpha=0.7)
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Polarity Distribution')
    
    # 4. Spatial distribution
    axes[1, 1].hist2d(events[:, 0], events[:, 1], bins=50, cmap='hot')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Y')
    axes[1, 1].set_title('Spatial Distribution')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    print("Testing visualization utilities...")
    
    # Test with dummy data
    visualizer = LNNVisualizer()
    
    # Simulate hidden states
    time_steps = 100
    hidden_size = 64
    states = torch.randn(time_steps, hidden_size).cumsum(0) * 0.1
    
    visualizer.plot_hidden_states(states, title="Test Hidden States")
    print("✓ Hidden state visualization test complete")
    
    # Simulate training history
    history = {
        'train_loss': list(np.exp(-np.linspace(0, 3, 50)) + np.random.rand(50) * 0.1),
        'val_loss': list(np.exp(-np.linspace(0, 2.5, 50)) + np.random.rand(50) * 0.15),
    }
    visualizer.plot_training_curves(history)
    print("✓ Training curves visualization test complete")
    
    print("\nAll visualization tests passed!")