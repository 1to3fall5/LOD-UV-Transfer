"""
UV Image Exporter - Export UV layout as image files.
Provides detailed visualization of UV faces, edges, and coordinates.
"""

import os
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from pathlib import Path

from ..utils.logger import get_logger
from ..fbx.fbx_handler import MeshData, UVChannel


class UVImageExporter:
    """
    Export UV layouts as high-quality images.
    
    Features:
    - UV face rendering with filled polygons
    - Edge highlighting
    - UV coordinate point display
    - Grid overlay
    - Island coloring
    - Comparison mode
    """
    
    def __init__(self):
        self.logger = get_logger("uv_transfer.image_exporter")
        self._matplotlib_available = self._check_matplotlib()
    
    def _check_matplotlib(self) -> bool:
        """Check if matplotlib is available."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.collections import PolyCollection
            from matplotlib.patches import Polygon
            return True
        except ImportError:
            self.logger.warning("matplotlib not available")
            return False
    
    def export_uv_image(
        self,
        uv_channel: UVChannel,
        faces: np.ndarray,
        output_path: str,
        title: str = "UV Layout",
        image_size: Tuple[int, int] = (2048, 2048),
        show_grid: bool = True,
        show_faces: bool = True,
        show_edges: bool = True,
        show_points: bool = True,
        show_indices: bool = False,
        face_alpha: float = 0.6,
        edge_width: float = 1.0,
        point_size: float = 2.0,
        background_color: str = 'white',
        face_color: str = 'lightblue',
        edge_color: str = 'blue',
        point_color: str = 'red',
        grid_color: str = 'gray',
        dpi: int = 150
    ) -> bool:
        """
        Export UV layout as an image.
        
        Args:
            uv_channel: UVChannel to export
            faces: Face indices array (control point indices)
            output_path: Output image path
            title: Image title
            image_size: Image size in pixels (width, height)
            show_grid: Whether to show UV grid
            show_faces: Whether to fill UV faces
            show_edges: Whether to draw face edges
            show_points: Whether to show UV points
            show_indices: Whether to show vertex indices
            face_alpha: Face fill transparency
            edge_width: Edge line width
            point_size: Point size
            background_color: Background color
            face_color: Face fill color
            edge_color: Edge line color
            point_color: Point color
            grid_color: Grid line color
            dpi: Output DPI
        
        Returns:
            True if export successful
        """
        if not self._matplotlib_available:
            self.logger.error("matplotlib not available for image export")
            return False
        
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPolygon
        
        uv_coords = uv_channel.uv_coordinates
        uv_indices = uv_channel.uv_indices
        
        fig_size = (image_size[0] / dpi, image_size[1] / dpi)
        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
        
        ax.set_facecolor(background_color)
        
        if show_grid:
            self._draw_grid(ax, grid_color)
        
        if show_faces or show_edges:
            self._draw_faces(
                ax, uv_coords, faces, uv_indices,
                show_faces, show_edges,
                face_color, edge_color, face_alpha, edge_width
            )
        
        if show_points:
            self._draw_points(ax, uv_coords, point_color, point_size)
        
        if show_indices:
            self._draw_indices(ax, uv_coords)
        
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('U', fontsize=12)
        ax.set_ylabel('V', fontsize=12)
        
        plt.tight_layout()
        
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                    facecolor=background_color, edgecolor='none')
        plt.close(fig)
        
        self.logger.info(f"Exported UV image: {output_path}")
        return True
    
    def _draw_grid(self, ax, grid_color: str):
        """Draw UV grid."""
        import matplotlib.pyplot as plt
        
        for i in range(11):
            val = i / 10.0
            ax.axhline(y=val, color=grid_color, linewidth=0.3, alpha=0.5)
            ax.axvline(x=val, color=grid_color, linewidth=0.3, alpha=0.5)
        
        ax.axhline(y=0, color='black', linewidth=1.0)
        ax.axhline(y=1, color='black', linewidth=1.0)
        ax.axvline(x=0, color='black', linewidth=1.0)
        ax.axvline(x=1, color='black', linewidth=1.0)
        
        for i in range(11):
            val = i / 10.0
            ax.text(val, -0.02, f'{val:.1f}', ha='center', va='top', fontsize=8)
            ax.text(-0.02, val, f'{val:.1f}', ha='right', va='center', fontsize=8)
    
    def _draw_faces(
        self,
        ax,
        uv_coords: np.ndarray,
        faces: np.ndarray,
        uv_indices: Optional[np.ndarray],
        show_faces: bool,
        show_edges: bool,
        face_color: str,
        edge_color: str,
        face_alpha: float,
        edge_width: float
    ):
        """Draw UV faces using proper face-vertex mapping."""
        from matplotlib.patches import Polygon as MplPolygon
        
        patches = []
        
        # Build face-vertex UV mapping
        # In FBX, UVs are stored per face-vertex, not per control point
        face_vertex_uvs = self._build_face_vertex_uvs(uv_coords, faces, uv_indices)
        
        for face_idx in range(len(faces)):
            if face_idx >= len(face_vertex_uvs):
                continue
            
            uv_face = face_vertex_uvs[face_idx]
            if len(uv_face) < 3:
                continue
            
            polygon = MplPolygon(
                uv_face,
                closed=True,
                facecolor=face_color if show_faces else 'none',
                edgecolor=edge_color if show_edges else 'none',
                alpha=face_alpha if show_faces else 1.0,
                linewidth=edge_width
            )
            patches.append(polygon)
        
        for patch in patches:
            ax.add_patch(patch)
    
    def _build_face_vertex_uvs(
        self,
        uv_coords: np.ndarray,
        faces: np.ndarray,
        uv_indices: Optional[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Build per-face-vertex UV coordinates.
        
        After FBX extraction, uv_coords contains per-face-vertex UVs.
        We just need to group them by face.
        """
        face_vertex_uvs = []
        current_idx = 0
        
        for face in faces:
            face_size = len(face)
            if face_size < 3:
                current_idx += face_size
                continue
            
            # Get UVs for this face (consecutive in the array)
            end_idx = current_idx + face_size
            if end_idx <= len(uv_coords):
                face_uvs = uv_coords[current_idx:end_idx]
                face_vertex_uvs.append(face_uvs)
            
            current_idx += face_size
        
        return face_vertex_uvs
    
    def _draw_points(self, ax, uv_coords: np.ndarray, color: str, size: float):
        """Draw UV points."""
        ax.scatter(
            uv_coords[:, 0], uv_coords[:, 1],
            s=size, c=color, alpha=0.7, zorder=5
        )
    
    def _draw_indices(self, ax, uv_coords: np.ndarray):
        """Draw vertex indices."""
        for i, uv in enumerate(uv_coords):
            ax.text(uv[0], uv[1], str(i), fontsize=4, ha='center', va='center')
    
    def export_uv_islands(
        self,
        uv_channel: UVChannel,
        faces: np.ndarray,
        output_path: str,
        title: str = "UV Islands",
        image_size: Tuple[int, int] = (2048, 2048),
        dpi: int = 150
    ) -> bool:
        """
        Export UV layout with islands colored differently.
        
        Args:
            uv_channel: UVChannel to export
            faces: Face indices array
            output_path: Output image path
            title: Image title
            image_size: Image size in pixels
            dpi: Output DPI
        
        Returns:
            True if export successful
        """
        if not self._matplotlib_available:
            return False
        
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPolygon
        
        uv_coords = uv_channel.uv_coordinates
        uv_indices = uv_channel.uv_indices
        
        # Build face-vertex UVs
        face_vertex_uvs = self._build_face_vertex_uvs(uv_coords, faces, uv_indices)
        
        # Find islands
        islands = self._find_uv_islands(face_vertex_uvs)
        
        fig_size = (image_size[0] / dpi, image_size[1] / dpi)
        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
        
        self._draw_grid(ax, 'gray')
        
        colors = plt.cm.tab20(np.linspace(0, 1, max(len(islands), 1)))
        
        for island_idx, island_faces in enumerate(islands):
            color = colors[island_idx % len(colors)]
            
            for face_uv in island_faces:
                polygon = MplPolygon(
                    face_uv,
                    closed=True,
                    facecolor=color,
                    edgecolor='black',
                    alpha=0.7,
                    linewidth=0.5
                )
                ax.add_patch(polygon)
        
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.set_title(f"{title} ({len(islands)} islands)", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"Exported UV islands image: {output_path}")
        return True
    
    def _find_uv_islands(
        self,
        face_vertex_uvs: List[np.ndarray]
    ) -> List[List[np.ndarray]]:
        """Find UV islands based on shared edges with tolerance for floating point precision."""
        if not face_vertex_uvs:
            return []
        
        # Filter out degenerate faces (faces with zero area or duplicate vertices)
        def is_degenerate_face(face_uv):
            """Check if a face is degenerate (zero area or duplicate vertices)."""
            if len(face_uv) < 3:
                return True
            
            # Check for duplicate vertices
            unique_verts = set()
            for uv in face_uv:
                unique_verts.add((round(float(uv[0]), 10), round(float(uv[1]), 10)))
            if len(unique_verts) < 3:
                return True
            
            # Check for zero area using shoelace formula
            area = 0.0
            n = len(face_uv)
            for i in range(n):
                j = (i + 1) % n
                area += face_uv[i][0] * face_uv[j][1]
                area -= face_uv[j][0] * face_uv[i][1]
            area = abs(area) / 2.0
            
            return area < 1e-15
        
        # Filter degenerate faces
        valid_faces = []
        valid_face_indices = []
        for idx, face_uv in enumerate(face_vertex_uvs):
            if not is_degenerate_face(face_uv):
                valid_faces.append(face_uv)
                valid_face_indices.append(idx)
        
        self.logger.debug(f"Filtered {len(face_vertex_uvs) - len(valid_faces)} degenerate faces, {len(valid_faces)} remaining")
        
        if not valid_faces:
            return []
        
        # Build edge to face mapping using UV coordinates with tolerance
        # Use a spatial hash to handle floating point precision issues
        # Use larger tolerance (1e-6) to handle FBX floating point variations
        def uv_to_key(uv, tolerance=1e-6):
            """Convert UV to a hashable key with tolerance."""
            # Round to handle floating point precision issues
            return (round(float(uv[0]) / tolerance), round(float(uv[1]) / tolerance))
        
        def get_edge_key(uv1, uv2):
            # Create edge key with consistent ordering
            key1 = uv_to_key(uv1)
            key2 = uv_to_key(uv2)
            # Ensure consistent ordering (smaller first)
            if key1 > key2:
                key1, key2 = key2, key1
            return (key1, key2)
        
        edge_to_faces = {}
        for face_idx, face_uv in enumerate(valid_faces):
            n = len(face_uv)
            for i in range(n):
                edge_key = get_edge_key(face_uv[i], face_uv[(i + 1) % n])
                if edge_key not in edge_to_faces:
                    edge_to_faces[edge_key] = []
                edge_to_faces[edge_key].append(face_idx)
        
        # Find connected faces (islands) using BFS
        visited = [False] * len(valid_faces)
        islands = []
        
        for start_idx in range(len(valid_faces)):
            if visited[start_idx]:
                continue
            
            island = []
            stack = [start_idx]
            
            while stack:
                face_idx = stack.pop()
                
                if visited[face_idx]:
                    continue
                
                visited[face_idx] = True
                island.append(valid_faces[face_idx])
                
                # Find connected faces through shared edges
                face_uv = valid_faces[face_idx]
                n = len(face_uv)
                for i in range(n):
                    edge_key = get_edge_key(face_uv[i], face_uv[(i + 1) % n])
                    connected_faces = edge_to_faces.get(edge_key, [])
                    for connected_face in connected_faces:
                        if connected_face != face_idx and not visited[connected_face]:
                            stack.append(connected_face)
            
            if island:
                islands.append(island)
        
        return islands
    
    def export_comparison_image(
        self,
        source_uv: UVChannel,
        target_uv: UVChannel,
        source_faces: np.ndarray,
        target_faces: np.ndarray,
        output_path: str,
        title: str = "UV Comparison",
        image_size: Tuple[int, int] = (4096, 2048),
        dpi: int = 150
    ) -> bool:
        """
        Export side-by-side comparison of two UV layouts.
        
        Args:
            source_uv: Source UVChannel
            target_uv: Target UVChannel
            source_faces: Source face indices
            target_faces: Target face indices
            output_path: Output image path
            title: Image title
            image_size: Image size in pixels
            dpi: Output DPI
        
        Returns:
            True if export successful
        """
        if not self._matplotlib_available:
            return False
        
        import matplotlib.pyplot as plt
        
        fig_size = (image_size[0] / dpi, image_size[1] / dpi)
        fig, axes = plt.subplots(1, 2, figsize=fig_size, dpi=dpi)
        
        for ax, uv, faces, subtitle in [
            (axes[0], source_uv, source_faces, "Source UV"),
            (axes[1], target_uv, target_faces, "Target UV")
        ]:
            ax.set_facecolor('white')
            self._draw_grid(ax, 'gray')
            self._draw_faces(ax, uv.uv_coordinates, faces, uv.uv_indices,
                           True, True, 'lightblue', 'blue', 0.6, 0.5)
            self._draw_points(ax, uv.uv_coordinates, 'red', 1.0)
            
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_aspect('equal')
            ax.set_title(subtitle, fontsize=12, fontweight='bold')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"Exported comparison image: {output_path}")
        return True
    
    def export_heatmap_image(
        self,
        source_uv: UVChannel,
        target_uv: UVChannel,
        vertex_mapping: np.ndarray,
        output_path: str,
        title: str = "UV Distance Heatmap",
        image_size: Tuple[int, int] = (2048, 2048),
        dpi: int = 150
    ) -> bool:
        """
        Export UV distance heatmap showing transfer accuracy.
        
        Args:
            source_uv: Source UVChannel
            target_uv: Target UVChannel
            vertex_mapping: Vertex mapping array
            output_path: Output image path
            title: Image title
            image_size: Image size in pixels
            dpi: Output DPI
        
        Returns:
            True if export successful
        """
        if not self._matplotlib_available:
            return False
        
        import matplotlib.pyplot as plt
        
        source_coords = source_uv.uv_coordinates
        target_coords = target_uv.uv_coordinates
        
        num_target = len(target_coords)
        num_source = len(source_coords)
        
        distances = np.zeros(num_target)
        for i in range(num_target):
            if i < len(vertex_mapping):
                src_idx = vertex_mapping[i]
                if src_idx >= 0 and src_idx < num_source:
                    distances[i] = np.sqrt(
                        np.sum((target_coords[i] - source_coords[src_idx]) ** 2)
                    )
                else:
                    distances[i] = -1
            else:
                distances[i] = -1
        
        fig_size = (image_size[0] / dpi, image_size[1] / dpi)
        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
        
        ax.set_facecolor('white')
        self._draw_grid(ax, 'gray')
        
        matched_mask = distances >= 0
        
        scatter = ax.scatter(
            target_coords[matched_mask, 0],
            target_coords[matched_mask, 1],
            c=distances[matched_mask],
            cmap='hot_r',
            s=3,
            alpha=0.8,
            zorder=5
        )
        
        if np.any(~matched_mask):
            ax.scatter(
                target_coords[~matched_mask, 0],
                target_coords[~matched_mask, 1],
                c='gray',
                s=3,
                alpha=0.5,
                label='Unmatched',
                zorder=4
            )
        
        cbar = plt.colorbar(scatter, ax=ax, label='UV Distance')
        cbar.ax.tick_params(labelsize=10)
        
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        
        stats_text = f"Matched: {np.sum(matched_mask)}/{len(target_coords)}\n"
        if np.any(matched_mask):
            stats_text += f"Mean: {np.mean(distances[matched_mask]):.6f}\n"
            stats_text += f"Max: {np.max(distances[matched_mask]):.6f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"Exported heatmap image: {output_path}")
        return True
