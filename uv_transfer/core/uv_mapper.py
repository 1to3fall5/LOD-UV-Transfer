"""
UV Mapper - Computes vertex mappings between different LOD meshes.
"""

from typing import Optional, Tuple, List
import numpy as np
from scipy.spatial import cKDTree

from ..utils.logger import get_logger
from ..utils.math_utils import compute_vertex_normals


class UVMapper:
    """
    Computes vertex mappings between source and target meshes.
    
    Supports multiple mapping strategies:
    - Direct: For identical topology
    - Spatial: Based on 3D position
    - Normal-weighted: Combines position and normal direction
    """
    
    def __init__(self):
        self.logger = get_logger("uv_transfer.mapper")
    
    def compute_mapping(
        self,
        source_vertices: np.ndarray,
        target_vertices: np.ndarray,
        source_normals: Optional[np.ndarray] = None,
        target_normals: Optional[np.ndarray] = None,
        threshold: float = 0.01,
        use_normals: bool = True
    ) -> np.ndarray:
        """
        Compute vertex mapping from target to source.
        
        Args:
            source_vertices: Source mesh vertices (N, 3)
            target_vertices: Target mesh vertices (M, 3)
            source_normals: Source mesh normals (N, 3)
            target_normals: Target mesh normals (M, 3)
            threshold: Maximum distance for direct mapping
            use_normals: Whether to use normals for better matching
        
        Returns:
            Mapping array where result[i] = source vertex index for target vertex i
            (-1 indicates no direct match found)
        """
        source_vertices = np.asarray(source_vertices, dtype=np.float64)
        target_vertices = np.asarray(target_vertices, dtype=np.float64)
        
        if source_normals is None or target_normals is None:
            use_normals = False
        
        if use_normals:
            return self._normal_weighted_mapping(
                source_vertices, target_vertices,
                source_normals, target_normals,
                threshold
            )
        else:
            return self._spatial_mapping(
                source_vertices, target_vertices, threshold
            )
    
    def _spatial_mapping(
        self,
        source_vertices: np.ndarray,
        target_vertices: np.ndarray,
        threshold: float
    ) -> np.ndarray:
        """Compute spatial position-based mapping."""
        tree = cKDTree(source_vertices)
        
        distances, indices = tree.query(target_vertices, k=1)
        
        mapping = indices.copy()
        mapping[distances > threshold] = -1
        
        matched_count = np.sum(mapping >= 0)
        self.logger.debug(
            f"Spatial mapping: {matched_count}/{len(target_vertices)} "
            f"vertices matched within threshold {threshold}"
        )
        
        return mapping
    
    def _normal_weighted_mapping(
        self,
        source_vertices: np.ndarray,
        target_vertices: np.ndarray,
        source_normals: np.ndarray,
        target_normals: np.ndarray,
        threshold: float
    ) -> np.ndarray:
        """Compute normal-weighted spatial mapping."""
        tree = cKDTree(source_vertices)
        
        distances, indices = tree.query(target_vertices, k=3)
        
        mapping = np.full(len(target_vertices), -1, dtype=np.int32)
        
        for i in range(len(target_vertices)):
            best_idx = -1
            best_score = float('inf')
            
            for j, source_idx in enumerate(indices[i]):
                if distances[i, j] > threshold * 3:
                    continue
                
                pos_score = distances[i, j]
                
                normal_dot = np.dot(
                    target_normals[i],
                    source_normals[source_idx]
                )
                normal_score = 1.0 - abs(normal_dot)
                
                combined_score = pos_score + normal_score * threshold
                
                if combined_score < best_score:
                    best_score = combined_score
                    best_idx = source_idx
            
            if best_idx >= 0 and distances[i, 0] <= threshold:
                mapping[i] = best_idx
        
        matched_count = np.sum(mapping >= 0)
        self.logger.debug(
            f"Normal-weighted mapping: {matched_count}/{len(target_vertices)} "
            f"vertices matched"
        )
        
        return mapping
    
    def compute_triangle_mapping(
        self,
        source_vertices: np.ndarray,
        source_faces: np.ndarray,
        target_vertices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute which source triangle each target vertex falls into.
        
        Args:
            source_vertices: Source mesh vertices (N, 3)
            source_faces: Source mesh faces (F, 3)
            target_vertices: Target mesh vertices (M, 3)
        
        Returns:
            Tuple of (triangle_indices, barycentric_coords)
        """
        from ..utils.math_utils import barycentric_interpolation
        
        triangle_indices = np.full(len(target_vertices), -1, dtype=np.int32)
        barycentric_coords = np.zeros((len(target_vertices), 3), dtype=np.float64)
        
        for i, vertex in enumerate(target_vertices):
            min_dist = float('inf')
            best_triangle = -1
            best_bary = np.array([1/3, 1/3, 1/3])
            
            for j, face in enumerate(source_faces):
                triangle = source_vertices[face]
                
                center = np.mean(triangle, axis=0)
                dist = np.linalg.norm(vertex - center)
                
                if dist < min_dist:
                    v0, v1, v2 = triangle
                    
                    v0v1 = v1 - v0
                    v0v2 = v2 - v0
                    v0p = vertex - v0
                    
                    d00 = np.dot(v0v1, v0v1)
                    d01 = np.dot(v0v1, v0v2)
                    d11 = np.dot(v0v2, v0v2)
                    d20 = np.dot(v0p, v0v1)
                    d21 = np.dot(v0p, v0v2)
                    
                    denom = d00 * d11 - d01 * d01
                    if abs(denom) > 1e-10:
                        v = (d11 * d20 - d01 * d21) / denom
                        w = (d00 * d21 - d01 * d20) / denom
                        u = 1.0 - v - w
                        
                        if u >= -0.01 and v >= -0.01 and w >= -0.01:
                            min_dist = dist
                            best_triangle = j
                            best_bary = np.array([u, v, w])
            
            triangle_indices[i] = best_triangle
            barycentric_coords[i] = best_bary
        
        return triangle_indices, barycentric_coords
    
    def find_boundary_vertices(
        self,
        vertices: np.ndarray,
        faces: np.ndarray
    ) -> np.ndarray:
        """
        Find boundary vertices of a mesh.
        
        Args:
            vertices: Mesh vertices (N, 3)
            faces: Mesh faces (F, 3)
        
        Returns:
            Boolean array indicating boundary vertices
        """
        edge_count = {}
        
        for face in faces:
            edges = [
                tuple(sorted([face[0], face[1]])),
                tuple(sorted([face[1], face[2]])),
                tuple(sorted([face[2], face[0]]))
            ]
            
            for edge in edges:
                edge_count[edge] = edge_count.get(edge, 0) + 1
        
        boundary_edges = [e for e, c in edge_count.items() if c == 1]
        
        boundary_vertices = set()
        for edge in boundary_edges:
            boundary_vertices.add(edge[0])
            boundary_vertices.add(edge[1])
        
        is_boundary = np.zeros(len(vertices), dtype=bool)
        for v in boundary_vertices:
            is_boundary[v] = True
        
        return is_boundary
    
    def compute_vertex_weights(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        boundary_weight: float = 2.0
    ) -> np.ndarray:
        """
        Compute weights for each vertex based on importance.
        
        Boundary vertices get higher weights for UV preservation.
        
        Args:
            vertices: Mesh vertices (N, 3)
            faces: Mesh faces (F, 3)
            boundary_weight: Weight multiplier for boundary vertices
        
        Returns:
            Weight array (N,)
        """
        weights = np.ones(len(vertices), dtype=np.float64)
        
        is_boundary = self.find_boundary_vertices(vertices, faces)
        weights[is_boundary] = boundary_weight
        
        return weights
