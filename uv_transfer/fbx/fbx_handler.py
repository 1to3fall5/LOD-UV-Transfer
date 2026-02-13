"""
FBX Handler - Unified interface for FBX file operations.
Supports multiple backends with automatic fallback.
"""

import os
import sys
from typing import Optional, List, Dict, Any, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
from enum import Enum

from ..utils.logger import get_logger, OperationContext
from ..utils.error_handler import FBXError, ValidationError


class FBXBackend(Enum):
    """Available FBX backend types."""
    OFFICIAL = "official"
    PYFBX = "pyfbx"
    ASSIMP = "assimp"
    NATIVE = "native"
    NONE = "none"


@dataclass
class UVChannel:
    """Represents a UV channel in a mesh."""
    name: str
    index: int
    uv_coordinates: np.ndarray
    uv_indices: Optional[np.ndarray] = None
    
    def __post_init__(self):
        self.uv_coordinates = np.asarray(self.uv_coordinates, dtype=np.float64)
        if self.uv_indices is not None:
            self.uv_indices = np.asarray(self.uv_indices, dtype=np.int32)


@dataclass
class MeshData:
    """Represents mesh data extracted from FBX."""
    name: str
    vertices: np.ndarray
    faces: np.ndarray
    normals: Optional[np.ndarray] = None
    uv_channels: Dict[str, UVChannel] = field(default_factory=dict)
    vertex_colors: Optional[np.ndarray] = None
    material_indices: Optional[np.ndarray] = None
    
    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        # Handle irregular faces (different number of vertices per face)
        faces_array = np.asarray(self.faces)
        if faces_array.dtype == object:
            # Irregular faces - keep as object array
            self.faces = faces_array
        else:
            # Regular faces - convert to int32
            self.faces = np.asarray(self.faces, dtype=np.int32)
        if self.normals is not None:
            self.normals = np.asarray(self.normals, dtype=np.float64)
    
    @property
    def vertex_count(self) -> int:
        return len(self.vertices)
    
    @property
    def face_count(self) -> int:
        return len(self.faces)
    
    def get_uv_channel(self, channel_index: int) -> Optional[UVChannel]:
        """Get UV channel by index."""
        channel_name = f"UVChannel_{channel_index}"
        if channel_name in self.uv_channels:
            return self.uv_channels[channel_name]
        for name, channel in self.uv_channels.items():
            if channel.index == channel_index:
                return channel
        return None
    
    def has_uv_channel(self, channel_index: int) -> bool:
        """Check if UV channel exists."""
        return self.get_uv_channel(channel_index) is not None


@dataclass
class FBXScene:
    """Represents an FBX scene."""
    file_path: str
    meshes: Dict[str, MeshData] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    _original_scene: Any = None  # Reference to original FBX scene object
    _backend: Any = None  # Reference to backend for saving
    
    def get_mesh_by_name(self, name: str) -> Optional[MeshData]:
        """Get mesh by name."""
        return self.meshes.get(name)
    
    def get_all_meshes(self) -> List[MeshData]:
        """Get all meshes in scene."""
        return list(self.meshes.values())


class FBXBackendBase:
    """Base class for FBX backends."""
    
    def __init__(self):
        self.logger = get_logger("uv_transfer.fbx")
        self._initialized = False
    
    def is_available(self) -> bool:
        """Check if backend is available."""
        raise NotImplementedError
    
    def initialize(self) -> bool:
        """Initialize the backend."""
        raise NotImplementedError
    
    def load(self, file_path: str) -> FBXScene:
        """Load FBX file."""
        raise NotImplementedError
    
    def save(self, scene: FBXScene, file_path: str) -> bool:
        """Save FBX file."""
        raise NotImplementedError
    
    def get_uv_channels(self, mesh: MeshData) -> List[UVChannel]:
        """Get all UV channels from mesh."""
        return list(mesh.uv_channels.values())
    
    def set_uv_channel(
        self,
        mesh: MeshData,
        channel_index: int,
        uv_coordinates: np.ndarray,
        uv_indices: Optional[np.ndarray] = None
    ) -> bool:
        """Set UV channel data."""
        raise NotImplementedError


class FBXHandler:
    """
    Unified FBX handler with automatic backend detection and fallback.
    """
    
    BACKEND_PRIORITY = [
        FBXBackend.OFFICIAL,
        FBXBackend.PYFBX,
        FBXBackend.ASSIMP,
        FBXBackend.NATIVE,
    ]
    
    def __init__(self, preferred_backend: Optional[FBXBackend] = None):
        self.logger = get_logger("uv_transfer.fbx")
        self.backends: Dict[FBXBackend, FBXBackendBase] = {}
        self.active_backend: Optional[FBXBackend] = None
        self._backend_instance: Optional[FBXBackendBase] = None
        
        self._discover_backends()
        
        if preferred_backend:
            self._set_backend(preferred_backend)
        else:
            self._auto_select_backend()
    
    def _discover_backends(self):
        """Discover available FBX backends."""
        self.logger.info("Discovering available FBX backends...")
        
        backends_to_try = [
            (FBXBackend.OFFICIAL, self._try_official_backend),
            (FBXBackend.PYFBX, self._try_pyfbx_backend),
            (FBXBackend.ASSIMP, self._try_assimp_backend),
            (FBXBackend.NATIVE, self._try_native_backend),
        ]
        
        for backend_type, try_func in backends_to_try:
            try:
                backend = try_func()
                if backend and backend.is_available():
                    self.backends[backend_type] = backend
                    self.logger.info(f"Backend available: {backend_type.value}")
            except Exception as e:
                self.logger.debug(f"Backend {backend_type.value} not available: {e}")
    
    def _try_official_backend(self) -> Optional[FBXBackendBase]:
        """Try to load official FBX SDK backend."""
        try:
            from .fbx_official import OfficialFBXBackend
            backend = OfficialFBXBackend()
            if backend.is_available():
                return backend
            else:
                self.logger.debug("Official FBX backend is not available")
                return None
        except Exception as e:
            self.logger.debug(f"Failed to load official FBX backend: {e}")
            return None
    
    def _try_pyfbx_backend(self) -> Optional[FBXBackendBase]:
        """Try to load pyfbx backend."""
        try:
            from .fbx_pyfbx import PyFBXBackend
            return PyFBXBackend()
        except ImportError:
            return None
    
    def _try_assimp_backend(self) -> Optional[FBXBackendBase]:
        """Try to load assimp backend."""
        try:
            from .fbx_assimp import AssimpFBXBackend
            return AssimpFBXBackend()
        except ImportError:
            return None
    
    def _try_native_backend(self) -> Optional[FBXBackendBase]:
        """Try to load native parser backend."""
        try:
            from .fbx_native import NativeFBXBackend
            return NativeFBXBackend()
        except ImportError:
            return None
    
    def _auto_select_backend(self):
        """Automatically select best available backend."""
        for backend_type in self.BACKEND_PRIORITY:
            if backend_type in self.backends:
                self._set_backend(backend_type)
                return
        
        self.logger.warning("No FBX backend available!")
        self.active_backend = FBXBackend.NONE
    
    def _set_backend(self, backend_type: FBXBackend):
        """Set the active backend."""
        if backend_type not in self.backends:
            raise FBXError(
                f"Backend {backend_type.value} is not available",
                error_code=1010
            )
        
        backend = self.backends[backend_type]
        if backend.initialize():
            self.active_backend = backend_type
            self._backend_instance = backend
            self.logger.info(f"Using FBX backend: {backend_type.value}")
        else:
            raise FBXError(
                f"Failed to initialize backend {backend_type.value}",
                error_code=1005
            )
    
    def get_available_backends(self) -> List[FBXBackend]:
        """Get list of available backends."""
        return list(self.backends.keys())
    
    def get_active_backend(self) -> FBXBackend:
        """Get currently active backend."""
        return self.active_backend or FBXBackend.NONE
    
    def load(self, file_path: str) -> FBXScene:
        """
        Load FBX file.
        
        Args:
            file_path: Path to FBX file
        
        Returns:
            FBXScene object containing mesh data
        
        Raises:
            FBXError: If file cannot be loaded
        """
        if not os.path.exists(file_path):
            raise FBXError(
                f"FBX file not found: {file_path}",
                error_code=1001,
                file_path=file_path
            )
        
        if self.active_backend == FBXBackend.NONE:
            raise FBXError(
                "No FBX backend available",
                error_code=1010
            )
        
        with OperationContext(self.logger, "fbx_load", f"Loading {file_path}"):
            try:
                scene = self._backend_instance.load(file_path)
                self.logger.info(
                    f"Loaded FBX: {len(scene.meshes)} meshes, "
                    f"file: {file_path}"
                )
                return scene
            except Exception as e:
                raise FBXError(
                    f"Failed to load FBX file: {str(e)}",
                    error_code=1002,
                    file_path=file_path
                )
    
    def save(self, scene: FBXScene, file_path: str) -> bool:
        """
        Save FBX file.
        
        Args:
            scene: FBXScene to save
            file_path: Output file path
        
        Returns:
            True if save successful
        
        Raises:
            FBXError: If file cannot be saved
        """
        if self.active_backend == FBXBackend.NONE:
            raise FBXError(
                "No FBX backend available",
                error_code=1010
            )
        
        output_dir = os.path.dirname(file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        with OperationContext(self.logger, "fbx_save", f"Saving {file_path}"):
            try:
                result = self._backend_instance.save(scene, file_path)
                if result:
                    self.logger.info(f"Saved FBX: {file_path}")
                return result
            except Exception as e:
                raise FBXError(
                    f"Failed to save FBX file: {str(e)}",
                    error_code=1003,
                    file_path=file_path
                )
    
    def transfer_uv(
        self,
        source_scene: FBXScene,
        target_scene: FBXScene,
        source_mesh_name: str,
        target_mesh_name: str,
        source_uv_channel: int,
        target_uv_channel: int,
        create_if_not_exists: bool = True
    ) -> bool:
        """
        Transfer UV channel from source mesh to target mesh.
        
        Args:
            source_scene: Source FBX scene
            target_scene: Target FBX scene
            source_mesh_name: Name of source mesh
            target_mesh_name: Name of target mesh
            source_uv_channel: Source UV channel index
            target_uv_channel: Target UV channel index
            create_if_not_exists: Create target channel if not exists
        
        Returns:
            True if transfer successful
        """
        source_mesh = source_scene.get_mesh_by_name(source_mesh_name)
        target_mesh = target_scene.get_mesh_by_name(target_mesh_name)
        
        if source_mesh is None:
            raise FBXError(
                f"Source mesh not found: {source_mesh_name}",
                error_code=1011
            )
        
        if target_mesh is None:
            raise FBXError(
                f"Target mesh not found: {target_mesh_name}",
                error_code=1011
            )
        
        source_uv = source_mesh.get_uv_channel(source_uv_channel)
        if source_uv is None:
            raise FBXError(
                f"Source UV channel not found: {source_uv_channel}",
                error_code=1008
            )
        
        return True
    
    def get_mesh_info(self, scene: FBXScene, mesh_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a mesh.
        
        Args:
            scene: FBX scene
            mesh_name: Name of mesh
        
        Returns:
            Dictionary with mesh information
        """
        mesh = scene.get_mesh_by_name(mesh_name)
        if mesh is None:
            return {}
        
        return {
            "name": mesh.name,
            "vertex_count": mesh.vertex_count,
            "face_count": mesh.face_count,
            "uv_channels": [
                {
                    "name": ch.name,
                    "index": ch.index,
                    "uv_count": len(ch.uv_coordinates)
                }
                for ch in mesh.uv_channels.values()
            ],
            "has_normals": mesh.normals is not None,
            "has_vertex_colors": mesh.vertex_colors is not None,
        }
