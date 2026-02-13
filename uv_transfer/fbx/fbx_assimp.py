"""
Assimp backend implementation.
Using pyassimp for FBX file handling.
"""

import os
from typing import Optional, List, Dict, Any
import numpy as np

from .fbx_handler import (
    FBXBackendBase, FBXScene, MeshData, UVChannel, FBXBackend
)
from ..utils.logger import get_logger
from ..utils.error_handler import FBXError


class AssimpFBXBackend(FBXBackendBase):
    """
    Backend using pyassimp library.
    """
    
    def __init__(self):
        super().__init__()
        self._assimp = None
    
    def is_available(self) -> bool:
        """Check if pyassimp is available."""
        try:
            import pyassimp
            return True
        except ImportError:
            return False
    
    def initialize(self) -> bool:
        """Initialize assimp backend."""
        try:
            import pyassimp
            self._assimp = pyassimp
            self._initialized = True
            self.logger.info("Assimp backend initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize pyassimp: {e}")
            return False
    
    def load(self, file_path: str) -> FBXScene:
        """Load FBX file using pyassimp."""
        if not self._initialized:
            raise FBXError("Assimp not initialized", error_code=1005)
        
        try:
            scene = self._assimp.load(file_path)
            
            fbx_scene = FBXScene(file_path=file_path)
            
            for mesh in scene.meshes:
                mesh_data = self._convert_mesh(mesh)
                fbx_scene.meshes[mesh_data.name] = mesh_data
            
            self._assimp.release(scene)
            
            return fbx_scene
            
        except Exception as e:
            raise FBXError(
                f"Failed to load FBX with assimp: {e}",
                error_code=1002,
                file_path=file_path
            )
    
    def save(self, scene: FBXScene, file_path: str) -> bool:
        """Save FBX file using pyassimp."""
        if not self._initialized:
            raise FBXError("Assimp not initialized", error_code=1005)
        
        self.logger.warning("Assimp backend has limited export support")
        
        raise FBXError(
            "Assimp backend does not support saving FBX files",
            error_code=1003,
            file_path=file_path
        )
    
    def _convert_mesh(self, mesh) -> MeshData:
        """Convert assimp mesh to MeshData."""
        vertices = np.array(mesh.vertices, dtype=np.float64)
        
        faces = []
        for face in mesh.faces:
            if len(face) == 3:
                faces.append(list(face))
            elif len(face) == 4:
                faces.append([face[0], face[1], face[2]])
                faces.append([face[0], face[2], face[3]])
        faces = np.array(faces, dtype=np.int32)
        
        normals = None
        if hasattr(mesh, 'normals') and mesh.normals is not None:
            normals = np.array(mesh.normals, dtype=np.float64)
        
        uv_channels = {}
        for i in range(8):
            uv_attr = f'texturecoords'
            if hasattr(mesh, uv_attr):
                texture_coords = getattr(mesh, uv_attr)
                if texture_coords and len(texture_coords) > i:
                    uv_data = texture_coords[i]
                    uv_coords = np.array(uv_data[:, :2], dtype=np.float64)
                    uv_channels[f"UVChannel_{i}"] = UVChannel(
                        name=f"UVChannel_{i}",
                        index=i,
                        uv_coordinates=uv_coords
                    )
        
        name = mesh.name if hasattr(mesh, 'name') else f"Mesh_{id(mesh)}"
        
        return MeshData(
            name=name,
            vertices=vertices,
            faces=faces,
            normals=normals,
            uv_channels=uv_channels
        )
    
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
