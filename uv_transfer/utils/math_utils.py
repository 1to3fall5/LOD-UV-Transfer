"""
Mathematical utilities for UV operations.
Provides functions for UV coordinate manipulation, interpolation, and spatial calculations.
"""

import numpy as np
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass


@dataclass
class UVCoordinate:
    """Represents a UV coordinate."""
    u: float
    v: float
    
    def to_array(self) -> np.ndarray:
        return np.array([self.u, self.v])
    
    def distance_to(self, other: 'UVCoordinate') -> float:
        """Calculate distance to another UV coordinate."""
        return np.sqrt((self.u - other.u) ** 2 + (self.v - other.v) ** 2)
    
    def is_valid(self, tolerance: float = 1e-6) -> bool:
        """Check if UV coordinate is valid."""
        return (
            not (np.isnan(self.u) or np.isnan(self.v)) and
            -tolerance <= self.u <= 1.0 + tolerance and
            -tolerance <= self.v <= 1.0 + tolerance
        )


def normalize_uv(uv: np.ndarray, clamp: bool = True) -> np.ndarray:
    """
    Normalize UV coordinates to [0, 1] range.
    
    Args:
        uv: UV coordinates array of shape (N, 2)
        clamp: Whether to clamp values to [0, 1]
    
    Returns:
        Normalized UV coordinates
    """
    uv = np.asarray(uv, dtype=np.float64)
    
    if uv.ndim == 1:
        uv = uv.reshape(1, -1)
    
    uv = uv - np.floor(uv)
    
    if clamp:
        uv = np.clip(uv, 0.0, 1.0)
    
    return uv


def calculate_uv_distance(uv1: np.ndarray, uv2: np.ndarray) -> np.ndarray:
    """
    Calculate Euclidean distance between UV coordinates.
    
    Args:
        uv1: First UV coordinates (N, 2)
        uv2: Second UV coordinates (N, 2)
    
    Returns:
        Distance array (N,)
    """
    uv1 = np.asarray(uv1, dtype=np.float64)
    uv2 = np.asarray(uv2, dtype=np.float64)
    
    return np.sqrt(np.sum((uv1 - uv2) ** 2, axis=-1))


def calculate_uv_area(uv_coords: np.ndarray) -> float:
    """
    Calculate the area of a UV triangle or polygon.
    
    Args:
        uv_coords: UV coordinates of vertices (N, 2) where N >= 3
    
    Returns:
        Area of the polygon
    """
    uv_coords = np.asarray(uv_coords, dtype=np.float64)
    
    if len(uv_coords) < 3:
        return 0.0
    
    if len(uv_coords) == 3:
        v0, v1, v2 = uv_coords
        return abs((v1[0] - v0[0]) * (v2[1] - v0[1]) - 
                   (v2[0] - v0[0]) * (v1[1] - v0[1])) / 2.0
    
    n = len(uv_coords)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += uv_coords[i][0] * uv_coords[j][1]
        area -= uv_coords[j][0] * uv_coords[i][1]
    
    return abs(area) / 2.0


def barycentric_interpolation(
    point: np.ndarray,
    triangle_vertices: np.ndarray,
    triangle_values: np.ndarray
) -> np.ndarray:
    """
    Interpolate values at a point using barycentric coordinates.
    
    Args:
        point: Point coordinates (2,) or (3,)
        triangle_vertices: Triangle vertex coordinates (3, D)
        triangle_values: Values at triangle vertices (3, V)
    
    Returns:
        Interpolated value
    """
    point = np.asarray(point, dtype=np.float64)
    vertices = np.asarray(triangle_vertices, dtype=np.float64)
    values = np.asarray(triangle_values, dtype=np.float64)
    
    v0 = vertices[1] - vertices[0]
    v1 = vertices[2] - vertices[0]
    v2 = point - vertices[0]
    
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-10:
        return np.mean(values, axis=0)
    
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    
    if u < -1e-6 or v < -1e-6 or w < -1e-6:
        u = max(0.0, min(1.0, u))
        v = max(0.0, min(1.0, v))
        w = max(0.0, min(1.0, w))
        total = u + v + w
        if total > 0:
            u, v, w = u / total, v / total, w / total
    
    return u * values[0] + v * values[1] + w * values[2]


def find_nearest_vertices(
    query_points: np.ndarray,
    target_points: np.ndarray,
    k: int = 1,
    return_distances: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Find the k nearest target points for each query point.
    
    Args:
        query_points: Query point coordinates (N, D)
        target_points: Target point coordinates (M, D)
        k: Number of nearest neighbors to find
        return_distances: Whether to return distances
    
    Returns:
        Indices of nearest neighbors (N, k) and optionally distances (N, k)
    """
    query_points = np.asarray(query_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    
    if query_points.ndim == 1:
        query_points = query_points.reshape(1, -1)
    if target_points.ndim == 1:
        target_points = target_points.reshape(1, -1)
    
    n_queries = len(query_points)
    k = min(k, len(target_points))
    
    indices = np.zeros((n_queries, k), dtype=np.int32)
    distances = np.zeros((n_queries, k), dtype=np.float64)
    
    for i, query in enumerate(query_points):
        dists = np.sqrt(np.sum((target_points - query) ** 2, axis=1))
        nearest_idx = np.argpartition(dists, k)[:k]
        sorted_idx = nearest_idx[np.argsort(dists[nearest_idx])]
        
        indices[i] = sorted_idx
        distances[i] = dists[sorted_idx]
    
    if return_distances:
        return indices, distances
    return indices


def compute_vertex_normals(
    vertices: np.ndarray,
    faces: np.ndarray
) -> np.ndarray:
    """
    Compute vertex normals from mesh geometry.
    
    Args:
        vertices: Vertex positions (N, 3)
        faces: Face indices (F, 3)
    
    Returns:
        Vertex normals (N, 3)
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    
    normals = np.zeros_like(vertices)
    
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    face_normals = np.cross(v1 - v0, v2 - v0)
    face_areas = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_areas = np.where(face_areas > 1e-10, face_areas, 1.0)
    face_normals = face_normals / face_areas
    
    for i, face in enumerate(faces):
        for vertex_idx in face:
            normals[vertex_idx] += face_normals[i]
    
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.where(lengths > 1e-10, lengths, 1.0)
    normals = normals / lengths
    
    return normals


def smooth_uv_coordinates(
    uv_coords: np.ndarray,
    vertex_neighbors: List[List[int]],
    iterations: int = 1,
    factor: float = 0.5
) -> np.ndarray:
    """
    Smooth UV coordinates using Laplacian smoothing.
    
    Args:
        uv_coords: UV coordinates (N, 2)
        vertex_neighbors: List of neighbor indices for each vertex
        iterations: Number of smoothing iterations
        factor: Smoothing factor (0-1)
    
    Returns:
        Smoothed UV coordinates
    """
    uv_coords = np.asarray(uv_coords, dtype=np.float64)
    smoothed = uv_coords.copy()
    
    for _ in range(iterations):
        new_uv = smoothed.copy()
        for i, neighbors in enumerate(vertex_neighbors):
            if neighbors:
                neighbor_uv = smoothed[neighbors]
                avg_uv = np.mean(neighbor_uv, axis=0)
                new_uv[i] = smoothed[i] * (1 - factor) + avg_uv * factor
        smoothed = new_uv
    
    return smoothed


def detect_uv_overlaps(uv_coords: np.ndarray, faces: np.ndarray) -> List[Tuple[int, int]]:
    """
    Detect overlapping UV faces.
    
    Args:
        uv_coords: UV coordinates (N, 2)
        faces: Face indices (F, 3)
    
    Returns:
        List of overlapping face pairs
    """
    uv_coords = np.asarray(uv_coords, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    
    overlaps = []
    
    for i in range(len(faces)):
        face1_uv = uv_coords[faces[i]]
        center1 = np.mean(face1_uv, axis=0)
        
        for j in range(i + 1, len(faces)):
            face2_uv = uv_coords[faces[j]]
            center2 = np.mean(face2_uv, axis=0)
            
            if np.linalg.norm(center1 - center2) < 0.1:
                overlaps.append((i, j))
    
    return overlaps


def compute_uv_stretching(
    vertices_3d: np.ndarray,
    uv_coords: np.ndarray,
    faces: np.ndarray
) -> np.ndarray:
    """
    Compute UV stretching factor for each face.
    
    Args:
        vertices_3d: 3D vertex positions (N, 3)
        uv_coords: UV coordinates (N, 2)
        faces: Face indices (F, 3)
    
    Returns:
        Stretching factors for each face
    """
    vertices_3d = np.asarray(vertices_3d, dtype=np.float64)
    uv_coords = np.asarray(uv_coords, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    
    stretching = np.zeros(len(faces))
    
    for i, face in enumerate(faces):
        v3d = vertices_3d[face]
        uv = uv_coords[face]
        
        area_3d = calculate_uv_area(v3d[:, :2])
        area_uv = calculate_uv_area(uv)
        
        if area_uv > 1e-10:
            stretching[i] = area_3d / area_uv
        else:
            stretching[i] = 0.0
    
    return stretching
