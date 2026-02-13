"""
PyFBX backend implementation.
Alternative pure Python FBX parser.
"""

import os
from typing import Optional, List, Dict, Any
import numpy as np

from .fbx_handler import (
    FBXBackendBase, FBXScene, MeshData, UVChannel, FBXBackend
)
from ..utils.logger import get_logger
from ..utils.error_handler import FBXError


class PyFBXBackend(FBXBackendBase):
    """
    Backend using pyfbx library.
    """
    
    def __init__(self):
        super().__init__()
        self._pyfbx = None
    
    def is_available(self) -> bool:
        """Check if pyfbx is available."""
        try:
            import pyfbx
            return True
        except ImportError:
            return False
    
    def initialize(self) -> bool:
        """Initialize pyfbx backend."""
        try:
            import pyfbx
            self._pyfbx = pyfbx
            self._initialized = True
            self.logger.info("PyFBX backend initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize pyfbx: {e}")
            return False
    
    def load(self, file_path: str) -> FBXScene:
        """Load FBX file using pyfbx."""
        if not self._initialized:
            raise FBXError("PyFBX not initialized", error_code=1005)
        
        try:
            scene = self._pyfbx.load(file_path)
            
            fbx_scene = FBXScene(file_path=file_path)
            
            for obj in scene.objects:
                if obj.type == "Mesh":
                    mesh_data = self._convert_mesh(obj)
                    fbx_scene.meshes[mesh_data.name] = mesh_data
            
            return fbx_scene
            
        except Exception as e:
            raise FBXError(
                f"Failed to load FBX with pyfbx: {e}",
                error_code=1002,
                file_path=file_path
            )
    
    def save(self, scene: FBXScene, file_path: str) -> bool:
        """Save FBX file using pyfbx."""
        if not self._initialized:
            raise FBXError("PyFBX not initialized", error_code=1005)
        
        try:
            fbx_scene = self._pyfbx.Scene()
            
            for mesh_data in scene.meshes.values():
                self._create_mesh(fbx_scene, mesh_data)
            
            fbx_scene.save(file_path)
            return True
            
        except Exception as e:
            raise FBXError(
                f"Failed to save FBX with pyfbx: {e}",
                error_code=1003,
                file_path=file_path
            )
    
    def _convert_mesh(self, obj) -> MeshData:
        """Convert pyfbx mesh object to MeshData."""
        vertices = np.array(obj.vertices, dtype=np.float64)
        faces = np.array(obj.faces, dtype=np.int32)
        
        uv_channels = {}
        if hasattr(obj, 'uv_layers'):
            for i, uv_layer in enumerate(obj.uv_layers):
                uv_coords = np.array(uv_layer.data, dtype=np.float64)
                uv_channels[f"UVChannel_{i}"] = UVChannel(
                    name=f"UVChannel_{i}",
                    index=i,
                    uv_coordinates=uv_coords
                )
        
        return MeshData(
            name=obj.name,
            vertices=vertices,
            faces=faces,
            uv_channels=uv_channels
        )
    
    def _create_mesh(self, fbx_scene, mesh_data: MeshData):
        """Create mesh in pyfbx scene."""
        mesh = self._pyfbx.Mesh(mesh_data.name)
        mesh.vertices = mesh_data.vertices.tolist()
        mesh.faces = mesh_data.faces.tolist()
        
        for channel in mesh_data.uv_channels.values():
            uv_layer = mesh.add_uv_layer(channel.name)
            uv_layer.data = channel.uv_coordinates.tolist()
        
        fbx_scene.add_object(mesh)
    
    def set_uv_channel(
        self,
        mesh: MeshData,
        channel_index: int,
        uv_coordinates: np.ndarray,
        uv_indices: Optional[np.ndarray] = None
    ) -> bool:
        """Set UV channel data."""
        channel_name = f"UVChannel_{channel_index}"
        uv_channel = UVChannel(
            name=channel_name,
            index=channel_index,
            uv_coordinates=uv_coordinates,
            uv_indices=uv_indices
        )
        mesh.uv_channels[channel_name] = uv_channel
        return True
