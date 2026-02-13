"""
UV Comparison - Compare UV data between meshes.
"""

import os
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

from ..utils.logger import get_logger
from ..utils.math_utils import calculate_uv_distance
from ..fbx.fbx_handler import MeshData, UVChannel


class UVComparison:
    """
    Compare UV data between source and target meshes.
    
    Provides:
    - Side-by-side comparison
    - Difference heatmaps
    - Statistical analysis
    """
    
    def __init__(self):
        self.logger = get_logger("uv_transfer.comparison")
        self._matplotlib_available = self._check_matplotlib()
    
    def _check_matplotlib(self) -> bool:
        """Check if matplotlib is available."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            return True
        except ImportError:
            self.logger.warning("matplotlib not available, comparison visualization disabled")
            return False
    
    def compare_uv_channels(
        self,
        source_uv: UVChannel,
        target_uv: UVChannel,
        vertex_mapping: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Compare two UV channels.
        
        Args:
            source_uv: Source UVChannel
            target_uv: Target UVChannel
            vertex_mapping: Optional vertex mapping
        
        Returns:
            Dictionary with comparison results
        """
        source_coords = source_uv.uv_coordinates
        target_coords = target_uv.uv_coordinates
        
        result = {
            'source_count': len(source_coords),
            'target_count': len(target_coords),
            'count_match': len(source_coords) == len(target_coords),
        }
        
        if vertex_mapping is not None and len(vertex_mapping) > 0:
            matched_mask = vertex_mapping >= 0
            matched_count = np.sum(matched_mask)
            
            result['matched_vertices'] = int(matched_count)
            result['match_rate'] = float(matched_count / len(vertex_mapping))
            
            if matched_count > 0:
                distances = []
                for i in range(len(vertex_mapping)):
                    if matched_mask[i]:
                        source_idx = vertex_mapping[i]
                        dist = np.sqrt(
                            np.sum((target_coords[i] - source_coords[source_idx]) ** 2)
                        )
                        distances.append(dist)
                
                distances = np.array(distances)
                result['mean_distance'] = float(np.mean(distances))
                result['max_distance'] = float(np.max(distances))
                result['min_distance'] = float(np.min(distances))
                result['std_distance'] = float(np.std(distances))
                result['median_distance'] = float(np.median(distances))
        
        return result
    
    def visualize_comparison(
        self,
        source_uv: UVChannel,
        target_uv: UVChannel,
        source_faces: Optional[np.ndarray] = None,
        target_faces: Optional[np.ndarray] = None,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (20, 10)
    ) -> Optional[Any]:
        """
        Visualize side-by-side UV comparison.
        
        Args:
            source_uv: Source UVChannel
            target_uv: Target UVChannel
            source_faces: Source face indices
            target_faces: Target face indices
            output_path: Path to save image
            figsize: Figure size
        
        Returns:
            matplotlib figure or None
        """
        if not self._matplotlib_available:
            return None
        
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        for ax, uv, faces, title in [
            (axes[0], source_uv, source_faces, "Source UV"),
            (axes[1], target_uv, target_faces, "Target UV")
        ]:
            uv_coords = uv.uv_coordinates
            
            if faces is not None:
                polygons = []
                for face in faces:
                    if len(face) >= 3:
                        poly = uv_coords[face[:3]]
                        polygons.append(poly)
                
                collection = PolyCollection(
                    polygons,
                    facecolors='lightblue',
                    edgecolors='blue',
                    linewidths=0.3,
                    alpha=0.7
                )
                ax.add_collection(collection)
            
            ax.scatter(uv_coords[:, 0], uv_coords[:, 1], s=0.5, c='red', alpha=0.3)
            
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)
            ax.set_aspect('equal')
            ax.set_title(title)
            ax.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved comparison to {output_path}")
        
        return fig
    
    def visualize_difference_heatmap(
        self,
        source_uv: UVChannel,
        target_uv: UVChannel,
        vertex_mapping: np.ndarray,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10)
    ) -> Optional[Any]:
        """
        Visualize UV difference as heatmap.
        
        Args:
            source_uv: Source UVChannel
            target_uv: Target UVChannel
            vertex_mapping: Vertex mapping array
            output_path: Path to save image
            figsize: Figure size
        
        Returns:
            matplotlib figure or None
        """
        if not self._matplotlib_available:
            return None
        
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        
        source_coords = source_uv.uv_coordinates
        target_coords = target_uv.uv_coordinates
        
        distances = np.zeros(len(target_coords))
        for i in range(len(target_coords)):
            if vertex_mapping[i] >= 0:
                source_idx = vertex_mapping[i]
                distances[i] = np.sqrt(
                    np.sum((target_coords[i] - source_coords[source_idx]) ** 2)
                )
            else:
                distances[i] = -1
        
        fig, ax = plt.subplots(figsize=figsize)
        
        matched_mask = distances >= 0
        
        scatter = ax.scatter(
            target_coords[matched_mask, 0],
            target_coords[matched_mask, 1],
            c=distances[matched_mask],
            cmap='hot',
            s=2,
            alpha=0.8
        )
        
        ax.scatter(
            target_coords[~matched_mask, 0],
            target_coords[~matched_mask, 1],
            c='gray',
            s=2,
            alpha=0.5,
            label='Unmatched'
        )
        
        plt.colorbar(scatter, ax=ax, label='UV Distance')
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal')
        ax.set_title("UV Difference Heatmap")
        ax.legend()
        
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved heatmap to {output_path}")
        
        return fig
    
    def generate_comparison_report(
        self,
        source_mesh: MeshData,
        target_mesh: MeshData,
        source_uv_channel: int,
        target_uv_channel: int,
        vertex_mapping: Optional[np.ndarray] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive comparison report.
        
        Args:
            source_mesh: Source MeshData
            target_mesh: Target MeshData
            source_uv_channel: Source UV channel index
            target_uv_channel: Target UV channel index
            vertex_mapping: Vertex mapping array
            output_dir: Directory for output files
        
        Returns:
            Dictionary with report data
        """
        source_uv = source_mesh.get_uv_channel(source_uv_channel)
        target_uv = target_mesh.get_uv_channel(target_uv_channel)
        
        if source_uv is None or target_uv is None:
            return {'error': 'UV channel not found'}
        
        report = {
            'source_mesh': source_mesh.name,
            'target_mesh': target_mesh.name,
            'source_uv_channel': source_uv_channel,
            'target_uv_channel': target_uv_channel,
        }
        
        comparison = self.compare_uv_channels(
            source_uv, target_uv, vertex_mapping
        )
        report['comparison'] = comparison
        
        if output_dir and self._matplotlib_available:
            os.makedirs(output_dir, exist_ok=True)
            
            comparison_path = os.path.join(
                output_dir,
                f"{source_mesh.name}_vs_{target_mesh.name}_comparison.png"
            )
            self.visualize_comparison(
                source_uv, target_uv,
                source_mesh.faces, target_mesh.faces,
                comparison_path
            )
            report['comparison_image'] = comparison_path
            
            if vertex_mapping is not None:
                heatmap_path = os.path.join(
                    output_dir,
                    f"{source_mesh.name}_vs_{target_mesh.name}_heatmap.png"
                )
                self.visualize_difference_heatmap(
                    source_uv, target_uv, vertex_mapping,
                    heatmap_path
                )
                report['heatmap_image'] = heatmap_path
        
        return report
    
    def compute_statistics(
        self,
        source_uv: UVChannel,
        target_uv: UVChannel,
        vertex_mapping: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute statistical measures for UV comparison.
        
        Args:
            source_uv: Source UVChannel
            target_uv: Target UVChannel
            vertex_mapping: Vertex mapping array
        
        Returns:
            Dictionary with statistics
        """
        source_coords = source_uv.uv_coordinates
        target_coords = target_uv.uv_coordinates
        
        matched_mask = vertex_mapping >= 0
        matched_count = np.sum(matched_mask)
        
        if matched_count == 0:
            return {'error': 'No matched vertices'}
        
        distances = []
        for i in range(len(target_coords)):
            if matched_mask[i]:
                source_idx = vertex_mapping[i]
                dist = np.sqrt(
                    np.sum((target_coords[i] - source_coords[source_idx]) ** 2)
                )
                distances.append(dist)
        
        distances = np.array(distances)
        
        return {
            'matched_count': int(matched_count),
            'total_count': len(target_coords),
            'match_rate': float(matched_count / len(target_coords)),
            'mean_error': float(np.mean(distances)),
            'max_error': float(np.max(distances)),
            'min_error': float(np.min(distances)),
            'std_error': float(np.std(distances)),
            'median_error': float(np.median(distances)),
            'rmse': float(np.sqrt(np.mean(distances ** 2))),
            'within_0.001': float(np.sum(distances < 0.001) / matched_count),
            'within_0.01': float(np.sum(distances < 0.01) / matched_count),
            'within_0.1': float(np.sum(distances < 0.1) / matched_count),
        }
