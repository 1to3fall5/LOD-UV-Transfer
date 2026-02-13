"""
UV Viewer - Visualization tool for UV data.
"""

import os
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

from ..utils.logger import get_logger
from ..fbx.fbx_handler import MeshData, UVChannel


class UVViewer:
    """
    Visualization tool for UV coordinates.
    
    Provides:
    - UV layout visualization
    - Texture overlay
    - Island detection
    """
    
    def __init__(self):
        self.logger = get_logger("uv_transfer.viewer")
        self._matplotlib_available = self._check_matplotlib()
    
    def _check_matplotlib(self) -> bool:
        """Check if matplotlib is available."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            return True
        except ImportError:
            self.logger.warning("matplotlib not available, visualization disabled")
            return False
    
    def visualize_uv(
        self,
        uv_channel: UVChannel,
        faces: Optional[np.ndarray] = None,
        output_path: Optional[str] = None,
        title: str = "UV Layout",
        show_grid: bool = True,
        figsize: Tuple[int, int] = (10, 10)
    ) -> Optional[Any]:
        """
        Visualize UV layout.
        
        Args:
            uv_channel: UVChannel to visualize
            faces: Face indices for drawing edges
            output_path: Path to save image (None for display)
            title: Plot title
            show_grid: Whether to show grid
            figsize: Figure size
        
        Returns:
            matplotlib figure or None
        """
        if not self._matplotlib_available:
            self.logger.warning("Cannot visualize: matplotlib not available")
            return None
        
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        
        uv_coords = uv_channel.uv_coordinates
        
        fig, ax = plt.subplots(figsize=figsize)
        
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
                linewidths=0.5,
                alpha=0.7
            )
            ax.add_collection(collection)
        
        ax.scatter(uv_coords[:, 0], uv_coords[:, 1], s=1, c='red', alpha=0.5)
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal')
        
        if show_grid:
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.axhline(y=0, color='k', linewidth=0.5)
            ax.axhline(y=1, color='k', linewidth=0.5)
            ax.axvline(x=0, color='k', linewidth=0.5)
            ax.axvline(x=1, color='k', linewidth=0.5)
        
        ax.set_xlabel('U')
        ax.set_ylabel('V')
        ax.set_title(title)
        
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved UV visualization to {output_path}")
        
        return fig
    
    def visualize_uv_with_texture(
        self,
        uv_channel: UVChannel,
        texture_path: str,
        faces: Optional[np.ndarray] = None,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 10)
    ) -> Optional[Any]:
        """
        Visualize UV layout with texture overlay.
        
        Args:
            uv_channel: UVChannel to visualize
            texture_path: Path to texture image
            faces: Face indices
            output_path: Path to save image
            figsize: Figure size
        
        Returns:
            matplotlib figure or None
        """
        if not self._matplotlib_available:
            return None
        
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        
        try:
            from PIL import Image
            texture = Image.open(texture_path)
        except Exception as e:
            self.logger.warning(f"Could not load texture: {e}")
            return self.visualize_uv(uv_channel, faces, output_path)
        
        uv_coords = uv_channel.uv_coordinates
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.imshow(texture, extent=[0, 1, 0, 1], origin='lower')
        
        if faces is not None:
            for face in faces:
                if len(face) >= 3:
                    poly = uv_coords[face[:3]]
                    closed_poly = np.vstack([poly, poly[0]])
                    ax.plot(closed_poly[:, 0], closed_poly[:, 1], 'r-', linewidth=0.5)
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal')
        ax.set_title("UV Layout with Texture")
        
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved UV+texture visualization to {output_path}")
        
        return fig
    
    def visualize_uv_islands(
        self,
        uv_channel: UVChannel,
        faces: np.ndarray,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 10)
    ) -> Optional[Any]:
        """
        Visualize UV islands with different colors.
        
        Args:
            uv_channel: UVChannel to visualize
            faces: Face indices
            output_path: Path to save image
            figsize: Figure size
        
        Returns:
            matplotlib figure or None
        """
        if not self._matplotlib_available:
            return None
        
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        
        uv_coords = uv_channel.uv_coordinates
        
        visited = np.zeros(len(uv_coords), dtype=bool)
        islands = []
        
        neighbors = [set() for _ in range(len(uv_coords))]
        for face in faces:
            for i in range(len(face)):
                neighbors[face[i]].add(face[(i + 1) % len(face)])
                neighbors[face[i]].add(face[(i - 1) % len(face)])
        
        for start in range(len(uv_coords)):
            if visited[start]:
                continue
            
            island = []
            stack = [start]
            
            while stack:
                vertex = stack.pop()
                if visited[vertex]:
                    continue
                
                visited[vertex] = True
                island.append(vertex)
                
                for neighbor in neighbors[vertex]:
                    if not visited[neighbor]:
                        stack.append(neighbor)
            
            islands.append(island)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(islands)))
        
        for i, island in enumerate(islands):
            island_uvs = uv_coords[island]
            ax.scatter(
                island_uvs[:, 0],
                island_uvs[:, 1],
                s=2,
                c=[colors[i]],
                alpha=0.7
            )
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal')
        ax.set_title(f"UV Islands ({len(islands)} islands)")
        ax.grid(True, linestyle='--', alpha=0.3)
        
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved UV islands visualization to {output_path}")
        
        return fig
    
    def generate_uv_report(
        self,
        mesh_data: MeshData,
        uv_channel_index: int = 0,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive UV report with visualizations.
        
        Args:
            mesh_data: MeshData to analyze
            uv_channel_index: UV channel to analyze
            output_dir: Directory for output files
        
        Returns:
            Dictionary with report data
        """
        uv_channel = mesh_data.get_uv_channel(uv_channel_index)
        
        if uv_channel is None:
            return {'error': f'UV channel {uv_channel_index} not found'}
        
        report = {
            'mesh_name': mesh_data.name,
            'uv_channel': uv_channel.name,
            'uv_count': len(uv_channel.uv_coordinates),
            'vertex_count': mesh_data.vertex_count,
        }
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            base_name = f"{mesh_data.name}_uv{uv_channel_index}"
            
            uv_layout_path = os.path.join(output_dir, f"{base_name}_layout.png")
            self.visualize_uv(
                uv_channel,
                mesh_data.faces,
                uv_layout_path,
                title=f"{mesh_data.name} - UV{uv_channel_index}"
            )
            report['uv_layout_image'] = uv_layout_path
            
            islands_path = os.path.join(output_dir, f"{base_name}_islands.png")
            self.visualize_uv_islands(
                uv_channel,
                mesh_data.faces,
                islands_path
            )
            report['islands_image'] = islands_path
        
        return report
