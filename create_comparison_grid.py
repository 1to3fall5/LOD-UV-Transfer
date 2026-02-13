"""
Create a 2x2 comparison grid of UV transfer algorithms.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from uv_transfer import FBXHandler
from test_uv_algorithms import UVTransferAlgorithms
from uv_transfer.fbx.fbx_handler import UVChannel


def create_comparison_grid():
    """Create 2x2 comparison grid."""
    print("=" * 70)
    print("Creating 2x2 Comparison Grid")
    print("=" * 70)
    
    handler = FBXHandler()
    
    base_dir = Path(__file__).parent
    lod0_file = base_dir / "SM_LostTomb01_Down_A10_Build_03_LOD0.fbx"
    lod1_file = base_dir / "SM_LostTomb01_Down_A10_Build_03_LOD1.fbx"
    
    output_dir = base_dir / "output" / "algorithm_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load meshes
    print("\n--- Loading meshes ---")
    source_scene = handler.load(str(lod0_file))
    target_scene = handler.load(str(lod1_file))
    
    source_mesh = list(source_scene.meshes.values())[0]
    target_mesh = list(target_scene.meshes.values())[0]
    source_uv = source_mesh.get_uv_channel(2)
    
    print(f"Source: {source_mesh.vertex_count} vertices, {len(source_mesh.faces)} faces")
    print(f"Target: {target_mesh.vertex_count} vertices, {len(target_mesh.faces)} faces")
    
    # Initialize algorithms
    algorithms = UVTransferAlgorithms(source_mesh, target_mesh, source_uv)
    
    # Create figure with 2x2 grid
    fig = plt.figure(figsize=(24, 24))
    gs = GridSpec(2, 2, figure=fig, hspace=0.15, wspace=0.15)
    
    # Plot 1: Source (LOD0)
    print("\n--- Processing: Source (LOD0) ---")
    ax1 = fig.add_subplot(gs[0, 0])
    _plot_uv_to_axis(ax1, source_uv, source_mesh.faces, "Source (LOD0)")
    
    # Plot 2: Algorithm 2 - Triangle Center
    print("\n--- Processing: Algorithm 2: Triangle Center ---")
    target_uvs, target_indices = algorithms.algorithm_2_triangle_center()
    uv_channel_2 = UVChannel(
        name="UVChannel_Transferred",
        index=2,
        uv_coordinates=target_uvs,
        uv_indices=target_indices
    )
    ax2 = fig.add_subplot(gs[0, 1])
    _plot_uv_to_axis(ax2, uv_channel_2, target_mesh.faces, "Algorithm 2: Triangle Center")
    
    # Plot 3: Algorithm 3 - Area Weighted
    print("\n--- Processing: Algorithm 3: Area Weighted ---")
    target_uvs, target_indices = algorithms.algorithm_3_area_weighted()
    uv_channel_3 = UVChannel(
        name="UVChannel_Transferred",
        index=2,
        uv_coordinates=target_uvs,
        uv_indices=target_indices
    )
    ax3 = fig.add_subplot(gs[1, 0])
    _plot_uv_to_axis(ax3, uv_channel_3, target_mesh.faces, "Algorithm 3: Area Weighted")
    
    # Plot 4: Algorithm 4 - Normal Aware
    print("\n--- Processing: Algorithm 4: Normal Aware ---")
    target_uvs, target_indices = algorithms.algorithm_4_normal_aware()
    uv_channel_4 = UVChannel(
        name="UVChannel_Transferred",
        index=2,
        uv_coordinates=target_uvs,
        uv_indices=target_indices
    )
    ax4 = fig.add_subplot(gs[1, 1])
    _plot_uv_to_axis(ax4, uv_channel_4, target_mesh.faces, "Algorithm 4: Normal Aware")
    
    # Save figure
    output_path = output_dir / "comparison_grid_2x2.png"
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n{'=' * 70}")
    print(f"Comparison grid saved to: {output_path}")
    print(f"{'=' * 70}")


def _plot_uv_to_axis(ax, uv_channel, faces, title):
    """Plot UV layout to matplotlib axis."""
    uvs = uv_channel.uv_coordinates
    
    # Plot faces
    for face in faces:
        if len(face) >= 3:
            face_uvs = []
            for vert_idx in face:
                if vert_idx < len(uvs):
                    face_uvs.append(uvs[vert_idx])
            
            if len(face_uvs) >= 3:
                face_uvs = np.array(face_uvs)
                # Draw filled polygon with transparency
                polygon = plt.Polygon(face_uvs, alpha=0.3, facecolor='lightblue', edgecolor='blue', linewidth=0.5)
                ax.add_patch(polygon)
    
    # Plot UV points
    ax.scatter(uvs[:, 0], uvs[:, 1], c='red', s=1, alpha=0.5)
    
    # Set labels and title
    ax.set_xlabel('U', fontsize=12)
    ax.set_ylabel('V', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Invert Y axis to match UV convention
    ax.invert_yaxis()


if __name__ == "__main__":
    create_comparison_grid()
