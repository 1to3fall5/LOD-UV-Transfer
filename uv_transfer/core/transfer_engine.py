"""
UV Transfer Engine - Core module for transferring UV data between LOD models.
"""

import os
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
from enum import Enum

from ..utils.logger import get_logger, OperationContext
from ..utils.error_handler import TransferError, ValidationError, FBXError
from ..utils.math_utils import (
    calculate_uv_distance,
    barycentric_interpolation,
    find_nearest_vertices,
    normalize_uv,
)
from ..fbx.fbx_handler import FBXHandler, FBXScene, MeshData, UVChannel
from .uv_mapper import UVMapper
from .interpolator import UVInterpolator
from .validator import UVValidator


class TransferMode(Enum):
    """UV transfer modes."""
    DIRECT = "direct"
    SPATIAL = "spatial"
    INTERPOLATED = "interpolated"


@dataclass
class TransferConfig:
    """Configuration for UV transfer operation."""
    source_uv_channel: int = 0
    target_uv_channel: int = 0
    mode: TransferMode = TransferMode.SPATIAL
    create_target_channel: bool = True
    validate_source: bool = True
    validate_result: bool = True
    accuracy_threshold: float = 0.001
    vertex_match_threshold: float = 0.01
    max_iterations: int = 100
    smooth_iterations: int = 0
    preserve_boundary: bool = True
    

@dataclass
class TransferResult:
    """Result of a UV transfer operation."""
    success: bool
    source_vertices: int
    target_vertices: int
    matched_vertices: int
    accuracy: float
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def match_rate(self) -> float:
        if self.target_vertices == 0:
            return 0.0
        return self.matched_vertices / self.target_vertices


class UVTransferEngine:
    """
    Core engine for transferring UV data between LOD models.
    
    Supports multiple transfer modes:
    - DIRECT: Direct vertex-to-vertex mapping (for identical topology)
    - SPATIAL: Spatial position-based mapping (for LOD variations)
    - INTERPOLATED: Interpolated mapping with smoothing
    """
    
    def __init__(
        self,
        fbx_handler: Optional[FBXHandler] = None,
        config: Optional[TransferConfig] = None
    ):
        self.logger = get_logger("uv_transfer.engine")
        self.fbx_handler = fbx_handler or FBXHandler()
        self.config = config or TransferConfig()
        
        self.mapper = UVMapper()
        self.interpolator = UVInterpolator()
        self.validator = UVValidator()
        
        self._source_scene: Optional[FBXScene] = None
        self._target_scene: Optional[FBXScene] = None
    
    def load_source(self, file_path: str) -> FBXScene:
        """Load source FBX file."""
        with OperationContext(self.logger, "load_source", f"Loading {file_path}"):
            self._source_scene = self.fbx_handler.load(file_path)
            self.logger.info(f"Loaded source: {len(self._source_scene.meshes)} meshes")
            return self._source_scene
    
    def load_target(self, file_path: str) -> FBXScene:
        """Load target FBX file."""
        with OperationContext(self.logger, "load_target", f"Loading {file_path}"):
            self._target_scene = self.fbx_handler.load(file_path)
            self.logger.info(f"Loaded target: {len(self._target_scene.meshes)} meshes")
            return self._target_scene
    
    def transfer(
        self,
        source_mesh_name: Optional[str] = None,
        target_mesh_name: Optional[str] = None,
        config: Optional[TransferConfig] = None
    ) -> TransferResult:
        """
        Execute UV transfer operation.
        
        Args:
            source_mesh_name: Name of source mesh (uses first if None)
            target_mesh_name: Name of target mesh (uses first if None)
            config: Transfer configuration (uses default if None)
        
        Returns:
            TransferResult with transfer details
        """
        config = config or self.config
        
        if self._source_scene is None:
            raise TransferError("Source scene not loaded", error_code=3002)
        if self._target_scene is None:
            raise TransferError("Target scene not loaded", error_code=3003)
        
        source_mesh = self._get_mesh(self._source_scene, source_mesh_name)
        target_mesh = self._get_mesh(self._target_scene, target_mesh_name)
        
        if source_mesh is None:
            raise TransferError(
                f"Source mesh not found: {source_mesh_name}",
                error_code=3002
            )
        if target_mesh is None:
            raise TransferError(
                f"Target mesh not found: {target_mesh_name}",
                error_code=3003
            )
        
        with OperationContext(
            self.logger,
            "uv_transfer",
            f"Transferring UV{config.source_uv_channel} -> UV{config.target_uv_channel}"
        ):
            return self._execute_transfer(source_mesh, target_mesh, config)
    
    def _get_mesh(
        self,
        scene: FBXScene,
        mesh_name: Optional[str]
    ) -> Optional[MeshData]:
        """Get mesh from scene by name or first available."""
        if mesh_name:
            return scene.get_mesh_by_name(mesh_name)
        
        meshes = scene.get_all_meshes()
        return meshes[0] if meshes else None
    
    def _execute_transfer(
        self,
        source_mesh: MeshData,
        target_mesh: MeshData,
        config: TransferConfig
    ) -> TransferResult:
        """Execute the actual UV transfer."""
        result = TransferResult(
            success=False,
            source_vertices=source_mesh.vertex_count,
            target_vertices=target_mesh.vertex_count,
            matched_vertices=0,
            accuracy=0.0
        )
        
        source_uv = source_mesh.get_uv_channel(config.source_uv_channel)
        if source_uv is None:
            result.errors.append(
                f"Source UV channel {config.source_uv_channel} not found"
            )
            return result
        
        if config.validate_source:
            validation = self.validator.validate_uv_channel(source_uv)
            if not validation.is_valid:
                result.errors.extend(validation.errors)
                return result
            result.warnings.extend(validation.warnings)
        
        self.logger.info(
            f"Source: {source_mesh.name} ({source_mesh.vertex_count} vertices, "
            f"UV channel {config.source_uv_channel}: {len(source_uv.uv_coordinates)} UVs)"
        )
        self.logger.info(
            f"Target: {target_mesh.name} ({target_mesh.vertex_count} vertices)"
        )
        
        vertex_mapping = self.mapper.compute_mapping(
            source_mesh.vertices,
            target_mesh.vertices,
            source_mesh.normals,
            target_mesh.normals,
            threshold=config.vertex_match_threshold
        )
        
        result.matched_vertices = np.sum(vertex_mapping >= 0)
        result.details['mapping_mode'] = config.mode.value
        
        self.logger.info(
            f"Vertex mapping: {result.matched_vertices}/{target_mesh.vertex_count} "
            f"({result.match_rate:.1%})"
        )
        
        target_uv_coords, target_uv_indices = self._compute_target_uvs(
            source_mesh,
            target_mesh,
            source_uv,
            vertex_mapping,
            config
        )
        
        if config.smooth_iterations > 0:
            target_uv_coords = self.interpolator.smooth_uvs(
                target_uv_coords,
                target_mesh.faces,
                iterations=config.smooth_iterations
            )
        
        self._apply_uv_to_mesh(
            target_mesh,
            config.target_uv_channel,
            target_uv_coords,
            target_uv_indices,
            config.create_target_channel
        )
        
        if config.validate_result:
            target_uv = target_mesh.get_uv_channel(config.target_uv_channel)
            if target_uv:
                accuracy = self.validator.compute_accuracy(
                    source_uv,
                    target_uv,
                    vertex_mapping,
                    source_mesh.vertices,
                    target_mesh.vertices
                )
                result.accuracy = accuracy
                
                if accuracy < config.accuracy_threshold:
                    result.warnings.append(
                        f"Transfer accuracy {accuracy:.6f} below threshold {config.accuracy_threshold}"
                    )
        
        result.success = True
        self.logger.info(
            f"Transfer complete: accuracy={result.accuracy:.6f}, "
            f"match_rate={result.match_rate:.1%}"
        )
        
        return result
    
    def _compute_target_uvs(
        self,
        source_mesh: MeshData,
        target_mesh: MeshData,
        source_uv: UVChannel,
        vertex_mapping: np.ndarray,
        config: TransferConfig
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute target UV coordinates based on transfer mode.
        
        For LOD models, UVs are independently unwrapped, so we need to match
        based on vertex position rather than vertex index.
        
        Returns:
            Tuple of (uv_coordinates, uv_indices) for per-face-vertex UV mapping.
        """
        # Calculate total face vertices for target mesh
        total_face_vertices = sum(len(face) for face in target_mesh.faces)
        
        # Build source face-vertex UVs and positions
        source_face_vertex_uvs = self._build_source_face_vertex_uvs(source_mesh, source_uv)
        source_face_vertex_positions = self._build_source_face_vertex_positions(source_mesh)
        
        # Build KD-tree for fast nearest neighbor search
        from scipy.spatial import cKDTree
        source_tree = cKDTree(source_face_vertex_positions)
        
        # Compute UVs for each target face vertex
        target_uv_coords = np.zeros((total_face_vertices, 2), dtype=np.float64)
        target_uv_indices = np.arange(total_face_vertices, dtype=np.int32)
        
        fv_idx = 0  # face-vertex index
        for face in target_mesh.faces:
            for vert_idx in face:
                target_pos = target_mesh.vertices[vert_idx]
                
                # Find nearest source face vertex by position
                distance, nearest_idx = source_tree.query(target_pos, k=1)
                
                if distance < 0.001:  # 1mm threshold
                    # Use the UV from the nearest source face vertex
                    target_uv_coords[fv_idx] = source_face_vertex_uvs[nearest_idx]
                else:
                    # No close match - interpolate from nearest vertices
                    uv = self._interpolate_uv_at_position(
                        target_pos,
                        source_mesh,
                        source_face_vertex_uvs
                    )
                    target_uv_coords[fv_idx] = uv
                
                fv_idx += 1
        
        return normalize_uv(target_uv_coords), target_uv_indices
    
    def _build_source_face_vertex_positions(
        self,
        source_mesh: MeshData
    ) -> np.ndarray:
        """Build per-face-vertex position array from source mesh."""
        total_face_vertices = sum(len(face) for face in source_mesh.faces)
        positions = np.zeros((total_face_vertices, 3), dtype=np.float64)
        
        fv_idx = 0
        for face in source_mesh.faces:
            for vert_idx in face:
                positions[fv_idx] = source_mesh.vertices[vert_idx]
                fv_idx += 1
        
        return positions
    
    def _build_source_face_vertex_uvs(
        self,
        source_mesh: MeshData,
        source_uv: UVChannel
    ) -> np.ndarray:
        """Build per-face-vertex UV array from source UV channel."""
        total_face_vertices = sum(len(face) for face in source_mesh.faces)
        
        if len(source_uv.uv_coordinates) == total_face_vertices:
            # Already per-face-vertex
            return source_uv.uv_coordinates
        elif source_uv.uv_indices is not None and len(source_uv.uv_indices) == total_face_vertices:
            # Use indices to lookup UVs
            result = np.zeros((total_face_vertices, 2), dtype=np.float64)
            for i, idx in enumerate(source_uv.uv_indices):
                if idx < len(source_uv.uv_coordinates):
                    result[i] = source_uv.uv_coordinates[idx]
            return result
        else:
            # Fallback: duplicate vertex UVs for each face vertex
            result = np.zeros((total_face_vertices, 2), dtype=np.float64)
            fv_idx = 0
            for face in source_mesh.faces:
                for vert_idx in face:
                    if vert_idx < len(source_uv.uv_coordinates):
                        result[fv_idx] = source_uv.uv_coordinates[vert_idx]
                    fv_idx += 1
            return result
    
    def _get_source_uv_at_vertex(
        self,
        source_mesh: MeshData,
        source_face_vertex_uvs: np.ndarray,
        vertex_idx: int,
        target_face_idx: int = -1,
        target_face_vert_idx: int = -1
    ) -> np.ndarray:
        """
        Get UV at a specific source vertex.
        
        For LOD models, we need to find the corresponding face vertex UV.
        If the vertex is used in multiple faces with different UVs, we need to
        find the best matching face based on the target face context.
        """
        # Collect all UVs for this vertex
        uvs_at_vertex = []
        fv_idx = 0
        for face in source_mesh.faces:
            for v_idx in face:
                if v_idx == vertex_idx and fv_idx < len(source_face_vertex_uvs):
                    uvs_at_vertex.append((fv_idx, source_face_vertex_uvs[fv_idx]))
                fv_idx += 1
        
        if not uvs_at_vertex:
            return np.array([0.0, 0.0])
        
        # If only one UV, return it
        if len(uvs_at_vertex) == 1:
            return uvs_at_vertex[0][1]
        
        # For vertices with multiple UVs (on UV seams), we need to choose the right one
        # For now, return the first one (this is a simplification)
        # A better approach would be to match based on face normal or position
        return uvs_at_vertex[0][1]
    
    def _interpolate_uv_at_position(
        self,
        position: np.ndarray,
        source_mesh: MeshData,
        source_face_vertex_uvs: np.ndarray
    ) -> np.ndarray:
        """Interpolate UV at a position using nearest vertices."""
        nearest_indices, distances = find_nearest_vertices(
            position.reshape(1, -1),
            source_mesh.vertices,
            k=3,
            return_distances=True
        )
        
        weights = 1.0 / (distances[0] + 1e-10)
        weights = weights / np.sum(weights)
        
        interpolated_uv = np.zeros(2)
        for j, vert_idx in enumerate(nearest_indices[0]):
            uv = self._get_source_uv_at_vertex(source_mesh, source_face_vertex_uvs, vert_idx)
            interpolated_uv += weights[j] * uv
        
        return interpolated_uv
    
    def _direct_transfer(
        self,
        source_uvs: np.ndarray,
        vertex_mapping: np.ndarray,
        target_count: int
    ) -> np.ndarray:
        """Direct vertex-to-vertex UV transfer."""
        target_uvs = np.zeros((target_count, 2), dtype=np.float64)
        
        for i, source_idx in enumerate(vertex_mapping):
            if source_idx >= 0:
                target_uvs[i] = source_uvs[source_idx]
        
        return target_uvs
    
    def _spatial_transfer(
        self,
        source_mesh: MeshData,
        target_mesh: MeshData,
        source_uv: UVChannel,
        vertex_mapping: np.ndarray
    ) -> np.ndarray:
        """Spatial position-based UV transfer."""
        target_uvs = np.zeros((target_mesh.vertex_count, 2), dtype=np.float64)
        
        source_uvs = source_uv.uv_coordinates
        if source_uv.uv_indices is not None and len(source_uv.uv_indices) > 0:
            source_uvs = source_uvs[source_uv.uv_indices] if len(source_uv.uv_indices) == len(source_mesh.faces) * 3 else source_uvs
        
        for i in range(target_mesh.vertex_count):
            source_idx = vertex_mapping[i]
            if source_idx >= 0 and source_idx < len(source_uvs):
                target_uvs[i] = source_uvs[source_idx % len(source_uvs)]
            elif source_idx >= 0:
                target_uvs[i] = source_uvs[0]
            else:
                nearest_indices, distances = find_nearest_vertices(
                    target_mesh.vertices[i:i+1],
                    source_mesh.vertices,
                    k=3,
                    return_distances=True
                )
                
                weights = 1.0 / (distances[0] + 1e-10)
                weights = weights / np.sum(weights)
                
                interpolated_uv = np.zeros(2)
                for j, idx in enumerate(nearest_indices[0]):
                    if idx < len(source_uvs):
                        interpolated_uv += weights[j] * source_uvs[idx]
                    else:
                        interpolated_uv += weights[j] * source_uvs[idx % len(source_uvs)]
                
                target_uvs[i] = interpolated_uv
        
        return target_uvs
    
    def _interpolated_transfer(
        self,
        source_mesh: MeshData,
        target_mesh: MeshData,
        source_uv: UVChannel,
        vertex_mapping: np.ndarray
    ) -> np.ndarray:
        """Interpolated UV transfer with barycentric interpolation."""
        target_uvs = np.zeros((target_mesh.vertex_count, 2), dtype=np.float64)
        
        for i in range(target_mesh.vertex_count):
            source_idx = vertex_mapping[i]
            if source_idx >= 0:
                target_uvs[i] = source_uv.uv_coordinates[source_idx]
            else:
                nearest_indices, distances = find_nearest_vertices(
                    target_mesh.vertices[i:i+1],
                    source_mesh.vertices,
                    k=3,
                    return_distances=True
                )
                
                triangle_vertices = source_mesh.vertices[nearest_indices[0]]
                triangle_uvs = source_uv.uv_coordinates[nearest_indices[0]]
                
                target_uvs[i] = barycentric_interpolation(
                    target_mesh.vertices[i],
                    triangle_vertices,
                    triangle_uvs
                )
        
        return target_uvs
    
    def _apply_uv_to_mesh(
        self,
        mesh: MeshData,
        channel_index: int,
        uv_coords: np.ndarray,
        uv_indices: np.ndarray,
        create_if_not_exists: bool
    ):
        """Apply UV coordinates to mesh."""
        self.fbx_handler._backend_instance.set_uv_channel(
            mesh,
            channel_index,
            uv_coords,
            uv_indices
        )
    
    def save_result(self, output_path: str) -> bool:
        """Save the modified target scene."""
        if self._target_scene is None:
            raise TransferError("No target scene to save", error_code=3006)
        
        with OperationContext(self.logger, "save_result", f"Saving to {output_path}"):
            return self.fbx_handler.save(self._target_scene, output_path)
    
    def batch_transfer(
        self,
        transfer_pairs: List[Dict[str, Any]],
        config: Optional[TransferConfig] = None
    ) -> List[TransferResult]:
        """
        Execute batch UV transfer operations.
        
        Args:
            transfer_pairs: List of transfer configurations
            config: Default configuration
        
        Returns:
            List of TransferResult objects
        """
        results = []
        
        for i, pair in enumerate(transfer_pairs):
            self.logger.info(f"Processing batch item {i+1}/{len(transfer_pairs)}")
            
            try:
                if 'source_file' in pair:
                    self.load_source(pair['source_file'])
                if 'target_file' in pair:
                    self.load_target(pair['target_file'])
                
                pair_config = config or self.config
                if 'config' in pair:
                    pair_config = TransferConfig(**pair['config'])
                
                result = self.transfer(
                    source_mesh_name=pair.get('source_mesh'),
                    target_mesh_name=pair.get('target_mesh'),
                    config=pair_config
                )
                
                if 'output_file' in pair:
                    self.save_result(pair['output_file'])
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Batch item {i+1} failed: {e}")
                results.append(TransferResult(
                    success=False,
                    source_vertices=0,
                    target_vertices=0,
                    matched_vertices=0,
                    accuracy=0.0,
                    errors=[str(e)]
                ))
        
        return results
