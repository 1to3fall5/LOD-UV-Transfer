"""
UV Interpolator - Handles UV interpolation and smoothing operations.
"""

from typing import Optional, List, Tuple
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

from ..utils.logger import get_logger
from ..utils.math_utils import (
    barycentric_interpolation,
    find_nearest_vertices,
    smooth_uv_coordinates
)


class UVInterpolator:
    """
    Handles UV interpolation between meshes with different topologies.
    
    Provides methods for:
    - Barycentric interpolation
    - UV smoothing
    - Seam handling
    """
    
    def __init__(self):
        self.logger = get_logger("uv_transfer.interpolator")
    
    def interpolate_uvs(
        self,
        target_vertices: np.ndarray,
        source_vertices: np.ndarray,
        source_uvs: np.ndarray,
        source_faces: np.ndarray,
        method: str = 'barycentric'
    ) -> np.ndarray:
        """
        Interpolate UVs from source mesh to target positions.
        
        Args:
            target_vertices: Target vertex positions (M, 3)
            source_vertices: Source vertex positions (N, 3)
            source_uvs: Source UV coordinates (N, 2)
            source_faces: Source face indices (F, 3)
            method: Interpolation method ('barycentric', 'nearest', 'linear')
        
        Returns:
            Interpolated UV coordinates (M, 2)
        """
        target_vertices = np.asarray(target_vertices, dtype=np.float64)
        source_vertices = np.asarray(source_vertices, dtype=np.float64)
        source_uvs = np.asarray(source_uvs, dtype=np.float64)
        
        if method == 'nearest':
            return self._nearest_interpolation(
                target_vertices, source_vertices, source_uvs
            )
        elif method == 'linear':
            return self._linear_interpolation(
                target_vertices, source_vertices, source_uvs
            )
        else:
            return self._barycentric_interpolation(
                target_vertices, source_vertices, source_uvs, source_faces
            )
    
    def _nearest_interpolation(
        self,
        target_vertices: np.ndarray,
        source_vertices: np.ndarray,
        source_uvs: np.ndarray
    ) -> np.ndarray:
        """Nearest neighbor interpolation."""
        indices = find_nearest_vertices(target_vertices, source_vertices)
        return source_uvs[indices.flatten()]
    
    def _linear_interpolation(
        self,
        target_vertices: np.ndarray,
        source_vertices: np.ndarray,
        source_uvs: np.ndarray
    ) -> np.ndarray:
        """Linear interpolation using scipy griddata."""
        try:
            result = griddata(
                source_vertices,
                source_uvs,
                target_vertices,
                method='linear'
            )
            
            nan_mask = np.isnan(result[:, 0])
            if np.any(nan_mask):
                nearest = griddata(
                    source_vertices,
                    source_uvs,
                    target_vertices[nan_mask],
                    method='nearest'
                )
                result[nan_mask] = nearest
            
            return result
        except Exception as e:
            self.logger.warning(f"Linear interpolation failed: {e}, using nearest")
            return self._nearest_interpolation(
                target_vertices, source_vertices, source_uvs
            )
    
    def _barycentric_interpolation(
        self,
        target_vertices: np.ndarray,
        source_vertices: np.ndarray,
        source_uvs: np.ndarray,
        source_faces: np.ndarray
    ) -> np.ndarray:
        """Barycentric interpolation within triangles."""
        result = np.zeros((len(target_vertices), 2), dtype=np.float64)
        
        for i, target in enumerate(target_vertices):
            min_dist = float('inf')
            best_uv = source_uvs[0].copy()
            
            for face in source_faces:
                triangle = source_vertices[face]
                center = np.mean(triangle, axis=0)
                dist = np.linalg.norm(target - center)
                
                if dist < min_dist * 2:
                    uv = barycentric_interpolation(
                        target,
                        triangle,
                        source_uvs[face]
                    )
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_uv = uv
            
            result[i] = best_uv
        
        return result
    
    def smooth_uvs(
        self,
        uv_coords: np.ndarray,
        faces: np.ndarray,
        iterations: int = 1,
        factor: float = 0.5,
        preserve_boundary: bool = True
    ) -> np.ndarray:
        """
        Apply Laplacian smoothing to UV coordinates.
        
        Args:
            uv_coords: UV coordinates (N, 2)
            faces: Face indices (F, 3)
            iterations: Number of smoothing iterations
            factor: Smoothing factor (0-1)
            preserve_boundary: Whether to preserve boundary UVs
        
        Returns:
            Smoothed UV coordinates
        """
        uv_coords = np.asarray(uv_coords, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int32)
        
        neighbors = self._build_neighbor_list(len(uv_coords), faces)
        
        if preserve_boundary:
            boundary = self._find_uv_boundary(uv_coords, faces)
        else:
            boundary = np.zeros(len(uv_coords), dtype=bool)
        
        smoothed = uv_coords.copy()
        
        for _ in range(iterations):
            new_uv = smoothed.copy()
            
            for i in range(len(uv_coords)):
                if boundary[i]:
                    continue
                
                if neighbors[i]:
                    neighbor_uv = smoothed[list(neighbors[i])]
                    avg_uv = np.mean(neighbor_uv, axis=0)
                    new_uv[i] = smoothed[i] * (1 - factor) + avg_uv * factor
            
            smoothed = new_uv
        
        return smoothed
    
    def _build_neighbor_list(
        self,
        vertex_count: int,
        faces: np.ndarray
    ) -> List[set]:
        """Build adjacency list for vertices."""
        neighbors = [set() for _ in range(vertex_count)]
        
        for face in faces:
            for i in range(3):
                neighbors[face[i]].add(face[(i + 1) % 3])
                neighbors[face[i]].add(face[(i + 2) % 3])
        
        return neighbors
    
    def _find_uv_boundary(
        self,
        uv_coords: np.ndarray,
        faces: np.ndarray
    ) -> np.ndarray:
        """Find vertices on UV boundaries."""
        edge_count = {}
        
        for face in faces:
            uv_face = uv_coords[face]
            
            for i in range(3):
                j = (i + 1) % 3
                
                uv1 = uv_face[i]
                uv2 = uv_face[j]
                
                edge_key = tuple(sorted([
                    (round(uv1[0], 6), round(uv1[1], 6)),
                    (round(uv2[0], 6), round(uv2[1], 6))
                ]))
                
                edge_count[edge_key] = edge_count.get(edge_key, 0) + 1
        
        boundary_edges = [e for e, c in edge_count.items() if c == 1]
        
        boundary_vertices = np.zeros(len(uv_coords), dtype=bool)
        
        for edge in boundary_edges:
            uv1, uv2 = edge
            
            for i, uv in enumerate(uv_coords):
                if (abs(uv[0] - uv1[0]) < 1e-5 and abs(uv[1] - uv1[1]) < 1e-5) or \
                   (abs(uv[0] - uv2[0]) < 1e-5 and abs(uv[1] - uv2[1]) < 1e-5):
                    boundary_vertices[i] = True
        
        return boundary_vertices
    
    def fix_uv_seams(
        self,
        uv_coords: np.ndarray,
        faces: np.ndarray,
        tolerance: float = 0.001
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fix UV seams by merging close UV coordinates.
        
        Args:
            uv_coords: UV coordinates (N, 2)
            faces: Face indices (F, 3)
            tolerance: Distance tolerance for merging
        
        Returns:
            Tuple of (fixed_uvs, new_faces)
        """
        uv_coords = np.asarray(uv_coords, dtype=np.float64)
        
        unique_uvs = {}
        mapping = np.zeros(len(uv_coords), dtype=np.int32)
        
        for i, uv in enumerate(uv_coords):
            key = (round(uv[0] / tolerance) * tolerance,
                   round(uv[1] / tolerance) * tolerance)
            
            if key not in unique_uvs:
                unique_uvs[key] = len(unique_uvs)
            
            mapping[i] = unique_uvs[key]
        
        new_faces = mapping[faces]
        
        new_uv_coords = np.zeros((len(unique_uvs), 2), dtype=np.float64)
        for key, idx in unique_uvs.items():
            new_uv_coords[idx] = key
        
        return new_uv_coords, new_faces
    
    def unwrap_uv_islands(
        self,
        uv_coords: np.ndarray,
        faces: np.ndarray
    ) -> List[np.ndarray]:
        """
        Identify separate UV islands.
        
        Args:
            uv_coords: UV coordinates (N, 2)
            faces: Face indices (F, 3)
        
        Returns:
            List of vertex index arrays for each island
        """
        visited = np.zeros(len(uv_coords), dtype=bool)
        islands = []
        
        neighbors = self._build_neighbor_list(len(uv_coords), faces)
        
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
            
            islands.append(np.array(island, dtype=np.int32))
        
        return islands
