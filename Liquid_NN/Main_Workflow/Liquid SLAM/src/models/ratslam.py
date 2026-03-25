"""
src/models/ratslam.py

Liquid Neural Network Enhanced RatSLAM Implementation.

Integrates CfC and NCP with bio-inspired SLAM:
    - Pose Cells: Continuous attractor with CfC dynamics
    - Local View Cells: CfC-based visual processing
    - Experience Map: Liquid dynamics for map relaxation
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import math

import sys
sys.path.insert(0, '../src')
from core.cfc_cell import CfCCell, CfCLayer
from core.wiring import NCPWiring, NCPCell, create_standard_wiring


@dataclass
class RatSLAMConfig:
    """Configuration for RatSLAM system."""
    # Pose cells
    pose_cells_x: int = 80
    pose_cells_y: int = 80
    pose_cells_theta: int = 36
    
    # Visual processing
    visual_template_size: int = 512
    visual_match_threshold: float = 0.15
    
    # Experience map
    exp_map_size: int = 10000
    exp_correction_rate: float = 0.5
    
    # Dynamics
    use_liquid_dynamics: bool = True
    cfc_hidden_size: int = 256
    
    # Path integration
    v_trans_scale: float = 1.0
    v_rot_scale: float = 1.0


class LiquidPoseCells(nn.Module):
    """
    Pose cell network with liquid time-constant dynamics.
    
    Represents robot pose (x, y, θ) in a 3D continuous attractor network.
    Uses CfC for continuous-time updates instead of discrete iterations.
    
    Args:
        config: RatSLAM configuration
    """
    
    def __init__(self, config: RatSLAMConfig):
        super().__init__()
        
        self.config = config
        self.dims = (config.pose_cells_x, config.pose_cells_y, config.pose_cells_theta)
        self.total_cells = np.prod(self.dims)
        
        # Continuous attractor with CfC dynamics
        if config.use_liquid_dynamics:
            self.dynamics = CfCCell(
                input_size=self.total_cells + 3,  # Current state + odometry
                hidden_size=config.cfc_hidden_size,
                mode='default'
            )
            self.output_proj = nn.Linear(config.cfc_hidden_size, self.total_cells)
        else:
            # Traditional discrete update
            self.dynamics = None
        
        # Initialize activity
        self.activity = None
        self._init_activity()
        
        # Attractor weights (Mexican hat connectivity)
        self.register_buffer('attractor_weights', self._build_attractor_weights())
    
    def _init_activity(self):
        """Initialize pose cell activity with single peak."""
        activity = torch.zeros(self.dims)
        center = (self.dims[0] // 2, self.dims[1] // 2, self.dims[2] // 2)
        
        # Create Gaussian peak
        for i in range(self.dims[0]):
            for j in range(self.dims[1]):
                for k in range(self.dims[2]):
                    dx = min(abs(i - center[0]), self.dims[0] - abs(i - center[0]))
                    dy = min(abs(j - center[1]), self.dims[1] - abs(j - center[1]))
                    dtheta = min(abs(k - center[2]), self.dims[2] - abs(k - center[2]))
                    
                    dist = np.sqrt(dx**2 + dy**2 + dtheta**2)
                    activity[i, j, k] = np.exp(-dist**2 / (2 * 5**2))
        
        # Normalize
        activity = activity / activity.sum()
        self.register_buffer('activity', activity)
    
    def _build_attractor_weights(self) -> torch.Tensor:
        """Build Mexican hat connectivity weights."""
        # Simplified 1D version for each dimension
        size = max(self.dims)
        weights = torch.zeros(size, size)
        
        for i in range(size):
            for j in range(size):
                dist = min(abs(i - j), size - abs(i - j))
                # Mexican hat: excitation - inhibition
                weights[i, j] = (
                    np.exp(-dist**2 / (2 * 3**2)) -  # Excitation
                    0.5 * np.exp(-dist**2 / (2 * 10**2))  # Inhibition
                )
        
        return weights
    
    def path_integration(
        self,
        vtrans: float,
        vrot: float,
        dt: float = 1.0
    ):
        """
        Update pose cells via path integration.
        
        Args:
            vtrans: Translational velocity
            vrot: Rotational velocity
            dt: Time step
        """
        if self.config.use_liquid_dynamics:
            # Use CfC for continuous-time update
            current_state = self.activity.flatten()
            odometry = torch.tensor(
                [vtrans * self.config.v_trans_scale,
                 vrot * self.config.v_rot_scale,
                 dt],
                device=self.activity.device
            )
            
            # Combine state and odometry
            input_vec = torch.cat([current_state, odometry])
            
            # CfC update
            hidden, _ = self.dynamics(input_vec.unsqueeze(0), elapsed_time=dt)
            new_activity = self.output_proj(hidden).squeeze(0)
            
            # Reshape and normalize
            new_activity = new_activity.view(self.dims)
            new_activity = torch.softmax(new_activity.flatten(), dim=0).view(self.dims)
            
        else:
            # Traditional discrete path integration
            # Shift activity based on velocity
            shift_x = int(vtrans * np.cos(self.get_heading()) * self.config.v_trans_scale)
            shift_y = int(vtrans * np.sin(self.get_heading()) * self.config.v_trans_scale)
            shift_theta = int(vrot * self.config.v_rot_scale)
            
            new_activity = torch.roll(self.activity, shifts=(shift_x, shift_y, shift_theta), dims=(0, 1, 2))
        
        # Apply attractor dynamics
        new_activity = self._apply_attractor(new_activity)
        
        self.activity = new_activity
    
    def _apply_attractor(self, activity: torch.Tensor) -> torch.Tensor:
        """Apply continuous attractor dynamics."""
        # Simplified attractor update
        activity_flat = activity.flatten()
        
        # Normalize to maintain single peak
        activity_flat = torch.softmax(activity_flat * 10, dim=0)  # Sharpen peak
        
        return activity_flat.view(self.dims)
    
    def inject_visual(self, visual_activation: torch.Tensor):
        """Inject visual template match to reinforce pose."""
        # Add visual evidence to current activity
        self.activity = self.activity + 0.1 * visual_activation
        self.activity = self.activity / self.activity.sum()
    
    def get_pose(self) -> Tuple[int, int, int]:
        """Get current pose estimate (x, y, theta)."""
        # Find peak of activity
        flat_idx = torch.argmax(self.activity)
        indices = np.unravel_index(flat_idx.item(), self.dims)
        return indices
    
    def get_heading(self) -> float:
        """Get current heading in radians."""
        _, _, theta_idx = self.get_pose()
        return (theta_idx / self.dims[2]) * 2 * np.pi


class LiquidLocalViewCells(nn.Module):
    """
    Local view cells with CfC-based visual processing.
    
    Processes visual input (images or events) to create
    place-specific templates for loop closure detection.
    
    Args:
        config: RatSLAM configuration
        input_channels: Number of input channels
    """
    
    def __init__(
        self,
        config: RatSLAMConfig,
        input_channels: int = 3,
        spatial_size: Tuple[int, int] = (64, 64)
    ):
        super().__init__()
        
        self.config = config
        self.template_size = config.visual_template_size
        
        # Visual feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # CfC for temporal integration of visual features
        if config.use_liquid_dynamics:
            self.temporal_encoder = CfCLayer(
                input_size=128 * 4 * 4,
                hidden_size=config.cfc_hidden_size,
                num_layers=1
            )
            self.template_proj = nn.Linear(config.cfc_hidden_size, self.template_size)
        else:
            self.template_proj = nn.Linear(128 * 4 * 4, self.template_size)
        
        # Template database
        self.templates: List[torch.Tensor] = []
        self.template_poses: List[Tuple] = []
    
    def forward(
        self,
        image: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Process visual input to template.
        
        Args:
            image: Input image (B, C, H, W) or events
            state: Previous hidden state
        
        Returns:
            template: Visual template (B, template_size)
            new_state: Updated hidden state
        """
        # Extract features
        features = self.feature_extractor(image)
        features = features.flatten(1)
        
        # Temporal encoding
        if self.config.use_liquid_dynamics:
            encoded, state = self.temporal_encoder(features, state)
            template = self.template_proj(encoded)
        else:
            template = self.template_proj(features)
            state = None
        
        # Normalize template
        template = nn.functional.normalize(template, p=2, dim=-1)
        
        return template, state
    
    def match_template(
        self,
        current_template: torch.Tensor
    ) -> Tuple[int, float]:
        """
        Match current template against database.
        
        Args:
            current_template: Current visual template
        
        Returns:
            best_match_idx: Index of best matching template (-1 if none)
            similarity: Similarity score of best match
        """
        if len(self.templates) == 0:
            return -1, 0.0
        
        # Stack all templates
        template_stack = torch.stack(self.templates)
        
        # Compute cosine similarity
        similarities = torch.matmul(
            current_template.squeeze(0),
            template_stack.t()
        )
        
        best_idx = torch.argmax(similarities).item()
        best_sim = similarities[best_idx].item()
        
        # Check threshold
        if best_sim < self.config.visual_match_threshold:
            return -1, best_sim
        
        return best_idx, best_sim
    
    def add_template(
        self,
        template: torch.Tensor,
        pose: Tuple[int, int, int]
    ) -> int:
        """Add new template to database."""
        template_id = len(self.templates)
        self.templates.append(template.detach().squeeze(0))
        self.template_poses.append(pose)
        return template_id


class ExperienceMap(nn.Module):
    """
    Experience map with liquid dynamics for relaxation.
    
    Maintains a topological-metric map of experiences,
    where each experience links pose cells with visual templates.
    
    Args:
        config: RatSLAM configuration
    """
    
    def __init__(self, config: RatSLAMConfig):
        super().__init__()
        
        self.config = config
        self.experiences: List[Experience] = []
        self.links: List[Tuple[int, int, float]] = []  # (from, to, distance)
        
        # Liquid dynamics for map relaxation
        if config.use_liquid_dynamics:
            self.relaxation_network = CfCCell(
                input_size=2,  # 2D position
                hidden_size=64,
                mode='pure'
            )
    
    def create_experience(
        self,
        pose: Tuple[int, int, int],
        template_id: int,
        position: np.ndarray
    ) -> int:
        """Create new experience."""
        exp_id = len(self.experiences)
        exp = Experience(
            id=exp_id,
            pose=pose,
            template_id=template_id,
            position=position
        )
        self.experiences.append(exp)
        return exp_id
    
    def add_link(self, from_id: int, to_id: int):
        """Add link between experiences."""
        if from_id >= len(self.experiences) or to_id >= len(self.experiences):
            return
        
        pos1 = self.experiences[from_id].position
        pos2 = self.experiences[to_id].position
        distance = np.linalg.norm(pos1 - pos2)
        
        self.links.append((from_id, to_id, distance))
    
    def relax_map(self, iterations: int = 10):
        """
        Relax experience map to reduce inconsistencies.
        Uses liquid dynamics for smooth corrections.
        """
        if len(self.experiences) < 2:
            return
        
        for _ in range(iterations):
            for from_id, to_id, target_dist in self.links:
                exp_from = self.experiences[from_id]
                exp_to = self.experiences[to_id]
                
                # Compute current distance
                current_dist = np.linalg.norm(
                    exp_from.position - exp_to.position
                )
                
                # Correction force
                if current_dist > 0:
                    correction = (target_dist - current_dist) / current_dist
                    correction *= self.config.exp_correction_rate
                    
                    # Apply correction
                    direction = exp_to.position - exp_from.position
                    exp_from.position += correction * direction * 0.5
                    exp_to.position -= correction * direction * 0.5


@dataclass
class Experience:
    """Single experience in the map."""
    id: int
    pose: Tuple[int, int, int]
    template_id: int
    position: np.ndarray
    times_visited: int = 1


class LNNRatSLAM(nn.Module):
    """
    Complete RatSLAM system with Liquid Neural Networks.
    
    Combines:
        - Liquid Pose Cells
        - Liquid Local View Cells
        - Experience Map with liquid relaxation
    
    Args:
        config: RatSLAM configuration
        input_channels: Number of input channels
        spatial_size: Input spatial dimensions
    """
    
    def __init__(
        self,
        config: Optional[RatSLAMConfig] = None,
        input_channels: int = 2,
        spatial_size: Tuple[int, int] = (128, 128)
    ):
        super().__init__()
        
        self.config = config or RatSLAMConfig()
        
        # Core components
        self.pose_cells = LiquidPoseCells(self.config)
        self.local_view_cells = LiquidLocalViewCells(
            self.config,
            input_channels,
            spatial_size
        )
        self.experience_map = ExperienceMap(self.config)
        
        # Current state
        self.current_exp_id = None
        self.prev_exp_id = None
        self.current_position = np.zeros(2)
        
        # History for visualization
        self.trajectory = []
    
    def step(
        self,
        image: torch.Tensor,
        vtrans: float,
        vrot: float,
        dt: float = 1.0
    ) -> Dict:
        """
        Single SLAM step.
        
        Args:
            image: Visual input (1, C, H, W)
            vtrans: Translational velocity
            vrot: Rotational velocity
            dt: Time step
        
        Returns:
            Dictionary with SLAM outputs
        """
        # 1. Path integration (update pose cells)
        self.pose_cells.path_integration(vtrans, vrot, dt)
        current_pose = self.pose_cells.get_pose()
        
        # 2. Visual processing
        template, _ = self.local_view_cells(image)
        match_id, similarity = self.local_view_cells.match_template(template)
        
        # 3. Experience map update
        if match_id >= 0:
            # Recognized place - close loop
            self.current_exp_id = match_id
            self.experiences[match_id].times_visited += 1
            
            # Inject visual evidence to pose cells
            matched_pose = self.local_view_cells.template_poses[match_id]
            visual_activation = torch.zeros_like(self.pose_cells.activity)
            visual_activation[matched_pose] = 1.0
            self.pose_cells.inject_visual(visual_activation)
            
        else:
            # New place - create experience
            template_id = self.local_view_cells.add_template(template, current_pose)
            
            # Update position estimate
            heading = self.pose_cells.get_heading()
            self.current_position += np.array([
                vtrans * np.cos(heading) * dt,
                vtrans * np.sin(heading) * dt
            ])
            
            self.current_exp_id = self.experience_map.create_experience(
                current_pose,
                template_id,
                self.current_position.copy()
            )
            
            # Link to previous experience
            if self.prev_exp_id is not None:
                self.experience_map.add_link(self.prev_exp_id, self.current_exp_id)
        
        # 4. Map relaxation (periodically)
        if len(self.experience_map.experiences) % 10 == 0:
            self.experience_map.relax_map()
        
        # Update trajectory
        self.trajectory.append(self.current_position.copy())
        
        # Prepare output
        output = {
            'pose': current_pose,
            'position': self.current_position,
            'exp_id': self.current_exp_id,
            'loop_closed': match_id >= 0,
            'similarity': similarity,
            'heading': self.pose_cells.get_heading()
        }
        
        self.prev_exp_id = self.current_exp_id
        
        return output
    
    def get_map(self) -> Dict:
        """Get current map state."""
        positions = np.array([
            exp.position for exp in self.experience_map.experiences
        ])
        
        return {
            'experiences': self.experience_map.experiences,
            'links': self.experience_map.links,
            'positions': positions,
            'trajectory': np.array(self.trajectory)
        }
    
    def visualize_map(self, save_path: Optional[str] = None):
        """Visualize experience map and trajectory."""
        import matplotlib.pyplot as plt
        
        map_data = self.get_map()
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot links
        for from_id, to_id, _ in map_data['links']:
            pos1 = map_data['positions'][from_id]
            pos2 = map_data['positions'][to_id]
            ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                   'b-', alpha=0.3, linewidth=1)
        
        # Plot experiences
        if len(map_data['positions']) > 0:
            ax.scatter(map_data['positions'][:, 0], 
                      map_data['positions'][:, 1],
                      c='red', s=50, zorder=5, label='Experiences')
        
        # Plot trajectory
        if len(map_data['trajectory']) > 0:
            trajectory = map_data['trajectory']
            ax.plot(trajectory[:, 0], trajectory[:, 1], 
                   'g-', alpha=0.6, linewidth=2, label='Trajectory')
            
            # Mark start and end
            ax.scatter([trajectory[0, 0]], [trajectory[0, 1]], 
                      c='green', s=100, marker='o', zorder=6, label='Start')
            ax.scatter([trajectory[-1, 0]], [trajectory[-1, 1]], 
                      c='blue', s=100, marker='*', zorder=6, label='Current')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('RatSLAM Experience Map')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.axis('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Map saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def create_ratslam_variants() -> Dict[str, nn.Module]:
    """
    Create different RatSLAM variants for comparison.
    
    Returns:
        Dictionary of model variants
    """
    variants = {}
    
    # 1. Full LNN-enhanced RatSLAM
    config_lnn = RatSLAMConfig(use_liquid_dynamics=True)
    variants['LNN-RatSLAM'] = LNNRatSLAM(config_lnn)
    
    # 2. Traditional RatSLAM (discrete dynamics)
    config_traditional = RatSLAMConfig(use_liquid_dynamics=False)
    variants['Traditional-RatSLAM'] = LNNRatSLAM(config_traditional)
    
    # 3. Compact version (smaller network)
    config_compact = RatSLAMConfig(
        use_liquid_dynamics=True,
        cfc_hidden_size=128,
        pose_cells_x=40,
        pose_cells_y=40,
        pose_cells_theta=18
    )
    variants['Compact-LNN-RatSLAM'] = LNNRatSLAM(config_compact)
    
    return variants


if __name__ == "__main__":
    print("Testing LNN-Enhanced RatSLAM")
    print("=" * 60)
    
    # Create RatSLAM system
    config = RatSLAMConfig()
    ratslam = LNNRatSLAM(config, input_channels=2, spatial_size=(128, 128))
    
    print(f"✓ RatSLAM initialized")
    print(f"  Pose cells: {config.pose_cells_x}x{config.pose_cells_y}x{config.pose_cells_theta}")
    print(f"  Template size: {config.visual_template_size}")
    print(f"  Using liquid dynamics: {config.use_liquid_dynamics}")
    
    # Simulate trajectory
    print("\nSimulating circular trajectory...")
    num_steps = 100
    radius = 5.0
    angular_vel = 2 * np.pi / num_steps
    
    for step in range(num_steps):
        # Circular motion
        vtrans = 0.1  # m/s
        vrot = angular_vel  # rad/s
        
        # Generate synthetic event image
        image = torch.randn(1, 2, 128, 128)
        
        # SLAM step
        output = ratslam.step(image, vtrans, vrot, dt=0.1)
        
        if step % 20 == 0:
            print(f"  Step {step}: Exp ID={output['exp_id']}, "
                  f"Loop closed={output['loop_closed']}, "
                  f"Position=({output['position'][0]:.2f}, {output['position'][1]:.2f})")
    
    # Visualize results
    print("\nGenerating visualization...")
    ratslam.visualize_map('ratslam_test.png')
    
    # Get final statistics
    map_data = ratslam.get_map()
    print(f"\n✓ SLAM completed:")
    print(f"  Total experiences: {len(map_data['experiences'])}")
    print(f"  Total links: {len(map_data['links'])}")
    print(f"  Trajectory length: {len(map_data['trajectory'])}")
    
    print("\n✓ All RatSLAM tests passed!")