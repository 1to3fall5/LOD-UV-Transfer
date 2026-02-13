"""
FBX handler using official Autodesk FBX SDK.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import numpy as np

from ..utils.logger import get_logger
from .fbx_handler import FBXHandler as BaseFBXHandler, MeshData, UVChannel, FBXScene


class OfficialFBXBackend(BaseFBXHandler):
    """FBX handler using official Autodesk FBX SDK."""
    
    def __init__(self):
        self.logger = get_logger("uv_transfer.fbx.official")
        self._fbx_module = None
        self._sdk_manager = None
        self._io_settings = None
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize FBX SDK."""
        if self._initialized:
            return True
        
        try:
            import fbx
            self._fbx_module = fbx
            self._sdk_manager = fbx.FbxManager.Create()
            self._io_settings = fbx.FbxIOSettings.Create(self._sdk_manager, fbx.IOSROOT)
            self._sdk_manager.SetIOSettings(self._io_settings)
            self._initialized = True
            self.logger.debug("Official FBX SDK initialized successfully")
            return True
        except ImportError as e:
            self.logger.debug(f"Failed to import FBX SDK: {e}")
            return False
        except Exception as e:
            self.logger.debug(f"Failed to initialize FBX SDK: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if official SDK is available."""
        return self.initialize()
    
    def load(self, file_path: Union[str, Path]) -> FBXScene:
        """Load FBX file using official SDK."""
        file_path = str(file_path)
        self.logger.info(f"fbx_load", f"Starting: Loading {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"FBX file not found: {file_path}")
        
        importer = self._fbx_module.FbxImporter.Create(self._sdk_manager, "")
        
        if not importer.Initialize(file_path, -1, self._io_settings):
            error_msg = importer.GetStatus().GetErrorString()
            self.logger.error(f"Failed to initialize FBX importer: {error_msg}")
            raise RuntimeError(f"FBX import failed: {error_msg}")
        
        scene = self._fbx_module.FbxScene.Create(self._sdk_manager, "scene")
        importer.Import(scene)
        importer.Destroy()
        
        meshes = self._extract_meshes(scene)
        
        self.logger.info(f"Loaded FBX: {len(meshes)} meshes, file: {file_path}")
        
        # Store original scene and backend reference for saving
        fbx_scene = FBXScene(
            file_path=file_path,
            meshes={mesh.name: mesh for mesh in meshes}
        )
        fbx_scene._original_scene = scene
        fbx_scene._backend = self
        
        return fbx_scene
    
    def _extract_meshes(self, scene) -> List[MeshData]:
        """Extract mesh data from FBX scene."""
        meshes = []
        root = scene.GetRootNode()
        
        for i in range(root.GetChildCount()):
            child = root.GetChild(i)
            attr = child.GetNodeAttribute()
            
            if attr and self._is_mesh_attribute(attr):
                mesh_data = self._extract_mesh_data(child, attr)
                if mesh_data:
                    meshes.append(mesh_data)
        
        return meshes
    
    def _is_mesh_attribute(self, attr) -> bool:
        """Check if attribute is a mesh."""
        try:
            mesh_type = self._fbx_module.FbxNodeAttribute.eMesh
        except AttributeError:
            try:
                mesh_type = self._fbx_module.FbxNodeAttribute.EType.eMesh
            except AttributeError:
                mesh_type = 8
        return attr.GetAttributeType() == mesh_type
    
    def _extract_mesh_data(self, node, mesh) -> Optional[MeshData]:
        """Extract mesh data from FBX mesh node."""
        try:
            name = node.GetName()
            self.logger.debug(f"Extracting mesh data for: {name}")
            
            # Extract vertices
            self.logger.debug("Extracting vertices...")
            vertices = self._extract_vertices(mesh)
            self.logger.debug(f"Vertices: {len(vertices)}")
            
            # Extract faces
            self.logger.debug("Extracting faces...")
            faces = self._extract_faces(mesh)
            self.logger.debug(f"Faces: {len(faces)}")
            
            # Extract normals
            self.logger.debug("Extracting normals...")
            normals = self._extract_normals(mesh)
            self.logger.debug(f"Normals: {len(normals)}")
            
            # Extract UV channels
            self.logger.debug("Extracting UV channels...")
            uv_channels = self._extract_uv_channels(mesh)
            self.logger.debug(f"UV channels: {list(uv_channels.keys())}")
            
            return MeshData(
                name=name,
                vertices=vertices,
                faces=faces,
                normals=normals,
                uv_channels=uv_channels
            )
        except Exception as e:
            self.logger.error(f"Failed to extract mesh data: {e}")
            import traceback
            self.logger.error(f"Stack: {traceback.format_exc()}")
            return None
    
    def _extract_vertices(self, mesh) -> np.ndarray:
        """Extract vertex positions."""
        control_points = mesh.GetControlPoints()
        vertices = np.zeros((mesh.GetControlPointsCount(), 3))
        
        for i in range(mesh.GetControlPointsCount()):
            point = control_points[i]
            # FbxVector4 supports direct indexing
            try:
                vertices[i] = [float(point[0]), float(point[1]), float(point[2])]
            except (TypeError, IndexError, AttributeError):
                vertices[i] = [0.0, 0.0, 0.0]
        
        return vertices
    
    def _extract_faces(self, mesh) -> np.ndarray:
        """Extract face indices."""
        faces = []
        for i in range(mesh.GetPolygonCount()):
            face = []
            for j in range(mesh.GetPolygonSize(i)):
                face.append(mesh.GetPolygonVertex(i, j))
            faces.append(face)
        return np.array(faces, dtype=object)
    
    def _extract_normals(self, mesh) -> Optional[np.ndarray]:
        """Extract vertex normals."""
        normals = np.zeros((mesh.GetControlPointsCount(), 3))
        
        for i in range(mesh.GetControlPointsCount()):
            normal = mesh.GetElementNormal(0)
            if normal:
                try:
                    n = normal.GetDirectArray().GetAt(i)
                    normals[i] = [n[0], n[1], n[2]]
                except:
                    normals[i] = [0, 1, 0]
            else:
                normals[i] = [0, 1, 0]
        
        return normals
    
    def _extract_uv_channels(self, mesh) -> Dict[str, UVChannel]:
        """Extract all UV channels from mesh."""
        uv_channels = {}
        uv_count = mesh.GetElementUVCount()
        
        for i in range(uv_count):
            uv_channel = self._extract_uv_channel(mesh, i)
            if uv_channel:
                uv_channels[uv_channel.name] = uv_channel
        
        return uv_channels
    
    def _extract_uv_channel(self, mesh, uv_index: int) -> Optional[UVChannel]:
        """Extract UV channel from mesh using GetIndexArray."""
        fbx = self._fbx_module
        
        uv_element = mesh.GetElementUV(uv_index)
        if not uv_element:
            return None
        
        name = uv_element.GetName() or f"UVChannel_{uv_index}"
        
        try:
            # Get the direct array (unique UV coordinates)
            direct_array = uv_element.GetDirectArray()
            direct_count = direct_array.GetCount()
            
            # Get the index array (maps polygon vertices to direct array)
            index_array = uv_element.GetIndexArray()
            index_count = index_array.GetCount()
            
            # Build per-face-vertex UV coordinates using GetIndexArray
            # Note: GetTextureUVIndex returns different values than GetIndexArray
            uv_coords_list = []
            uv_indices_list = []
            
            # Iterate through all polygon vertices
            idx = 0
            polygon_count = mesh.GetPolygonCount()
            
            for poly_idx in range(polygon_count):
                poly_size = mesh.GetPolygonSize(poly_idx)
                for vert_idx in range(poly_size):
                    # Get the UV index from index array
                    if idx < index_count:
                        uv_idx = index_array.GetAt(idx)
                    else:
                        # Fallback to GetTextureUVIndex if index array is exhausted
                        uv_idx = mesh.GetTextureUVIndex(poly_idx, vert_idx)
                    
                    uv_indices_list.append(uv_idx)
                    
                    # Get the UV coordinate from direct array
                    if uv_idx < direct_count:
                        uv = direct_array.GetAt(uv_idx)
                        uv_coords_list.append([uv[0], uv[1]])
                    else:
                        # Index out of bounds - use default
                        self.logger.warning(f"UV {name}: Index {uv_idx} out of bounds (max {direct_count-1}) at poly {poly_idx}, vert {vert_idx}")
                        uv_coords_list.append([0.0, 0.0])
                    
                    idx += 1
            
            uv_coords = np.array(uv_coords_list, dtype=np.float64)
            uv_indices = np.array(uv_indices_list, dtype=np.int32)
            
            self.logger.debug(f"UV {name}: extracted {len(uv_coords)} UVs, {len(uv_indices)} indices")
            
        except Exception as e:
            self.logger.error(f"Failed to extract UV channel {name}: {e}")
            return None
        
        return UVChannel(
            name=name,
            index=uv_index,
            uv_coordinates=uv_coords,
            uv_indices=uv_indices
        )
    
    def save(self, scene: FBXScene, file_path: Union[str, Path]) -> bool:
        """Save FBX file using official SDK.
        
        If the scene has an original FBX scene reference, it will modify the UV data
        in the original scene and save it, preserving all other data (materials, normals, etc.).
        Otherwise, it creates a new scene from scratch.
        """
        file_path = str(file_path)
        self.logger.info(f"fbx_save", f"Starting: Saving to {file_path}")
        
        # Check if we have original scene to preserve all data
        if hasattr(scene, '_original_scene') and scene._original_scene is not None:
            self.logger.info("Using original FBX scene to preserve all data")
            fbx_scene = scene._original_scene
            
            # Get target UV channel if set
            target_uv_channel = None
            if hasattr(scene, 'get_target_uv_channel'):
                target_uv_channel = scene.get_target_uv_channel()
            
            if target_uv_channel is not None:
                self.logger.info(f"Will only update UV channel {target_uv_channel}")
            else:
                self.logger.info("Will update all UV channels")
            
            # Update UV data in original scene
            self._update_uv_in_scene(fbx_scene, scene, target_uv_channel)
        else:
            self.logger.warning("No original scene, creating new FBX (may lose materials/normals)")
            # Create new scene (fallback)
            fbx_scene = self._fbx_module.FbxScene.Create(self._sdk_manager, "export_scene")
            
            # Add meshes to scene
            for mesh_name, mesh_data in scene.meshes.items():
                self._add_mesh_to_scene(fbx_scene, mesh_data)
        
        # Create exporter
        exporter = self._fbx_module.FbxExporter.Create(self._sdk_manager, "")
        
        if not exporter.Initialize(file_path, -1, self._io_settings):
            error_msg = exporter.GetStatus().GetErrorString()
            self.logger.error(f"Failed to initialize FBX exporter: {error_msg}")
            return False
        
        # Export
        success = exporter.Export(fbx_scene)
        exporter.Destroy()
        
        if success:
            self.logger.info(f"Successfully saved FBX: {file_path}")
        else:
            self.logger.error(f"Failed to save FBX: {file_path}")
        
        return success
    
    def _update_uv_in_scene(self, fbx_scene, scene: FBXScene, target_uv_channel: int = None) -> None:
        """Update UV data in the original FBX scene.
        
        Args:
            fbx_scene: The original FBX scene
            scene: The FBXScene wrapper containing updated mesh data
            target_uv_channel: Specific UV channel index to update. If None, updates all channels.
        """
        root_node = fbx_scene.GetRootNode()
        if not root_node:
            return
        
        for i in range(root_node.GetChildCount()):
            node = root_node.GetChild(i)
            mesh_attr = node.GetMesh()
            
            if mesh_attr:
                mesh_name = node.GetName()
                
                # Find corresponding mesh data
                mesh_data = scene.meshes.get(mesh_name)
                if not mesh_data:
                    # Try to find by index or any mesh
                    for name, data in scene.meshes.items():
                        mesh_data = data
                        break
                
                if mesh_data:
                    # Update only the specified UV channel, or all if not specified
                    for uv_channel_name, uv_channel in mesh_data.uv_channels.items():
                        if target_uv_channel is None or uv_channel.index == target_uv_channel:
                            self._update_mesh_uv(mesh_attr, uv_channel)
                            self.logger.info(f"Updated UV channel {uv_channel.index} ({uv_channel_name})")
                        else:
                            self.logger.info(f"Skipped UV channel {uv_channel.index} ({uv_channel_name})")
    
    def _add_mesh_to_scene(self, scene, mesh_data: MeshData):
        """Add mesh to FBX scene."""
        fbx = self._fbx_module
        
        # Create mesh
        mesh = fbx.FbxMesh.Create(scene, mesh_data.name)
        
        # Set vertices
        mesh.InitControlPoints(mesh_data.vertex_count)
        for i, vertex in enumerate(mesh_data.vertices):
            mesh.SetControlPointAt(fbx.FbxVector4(vertex[0], vertex[1], vertex[2], 1), i)
        
        # Add polygons
        for face in mesh_data.faces:
            mesh.BeginPolygon()
            for idx in face:
                mesh.AddPolygon(idx)
            mesh.EndPolygon()
        
        # Add UV channels
        for uv_name, uv_channel in mesh_data.uv_channels.items():
            self._add_uv_layer(mesh, uv_channel)
        
        # Create node
        node = fbx.FbxNode.Create(scene, mesh_data.name)
        node.SetNodeAttribute(mesh)
        
        # Add to scene
        scene.GetRootNode().AddChild(node)
    
    def set_uv_channel(
        self,
        mesh: MeshData,
        channel_index: int,
        uv_coordinates: np.ndarray,
        uv_indices: Optional[np.ndarray] = None
    ) -> bool:
        """Set UV channel data for a mesh."""
        try:
            channel_name = f"UVChannel_{channel_index}"
            
            # Create or update UV channel
            uv_channel = UVChannel(
                name=channel_name,
                index=channel_index,
                uv_coordinates=uv_coordinates,
                uv_indices=uv_indices
            )
            
            mesh.uv_channels[channel_name] = uv_channel
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set UV channel {channel_index}: {e}")
            return False
    
    def _add_uv_layer(self, mesh, uv_channel: UVChannel):
        """Add UV layer to mesh."""
        fbx = self._fbx_module
        
        try:
            uv_layer = mesh.CreateElementUV(uv_channel.name)
            
            # Set mapping mode
            try:
                mapping_mode = fbx.FbxLayerElement.eByPolygonVertex
            except AttributeError:
                try:
                    mapping_mode = fbx.FbxLayerElement.EMappingMode.eByPolygonVertex
                except AttributeError:
                    mapping_mode = 2
            
            # Use IndexToDirect reference mode to support shared UV coordinates
            try:
                reference_mode = fbx.FbxLayerElement.eIndexToDirect
            except AttributeError:
                try:
                    reference_mode = fbx.FbxLayerElement.EReferenceMode.eIndexToDirect
                except AttributeError:
                    reference_mode = 1  # eIndexToDirect = 1
            
            uv_layer.SetMappingMode(mapping_mode)
            uv_layer.SetReferenceMode(reference_mode)
            
            # Add UV coordinates to direct array
            direct_array = uv_layer.GetDirectArray()
            
            # Check if we have uv_indices - if so, we need to deduplicate UVs
            if uv_channel.uv_indices is not None and len(uv_channel.uv_indices) > 0:
                # Use unique UV coordinates from the direct array
                unique_uvs = []
                uv_index_map = {}
                
                for i, uv in enumerate(uv_channel.uv_coordinates):
                    uv_tuple = (float(uv[0]), float(uv[1]))
                    if uv_tuple not in uv_index_map:
                        uv_index_map[uv_tuple] = len(unique_uvs)
                        unique_uvs.append(uv_tuple)
                        direct_array.Add(fbx.FbxVector2(uv[0], uv[1]))
                
                # Add indices to index array
                index_array = uv_layer.GetIndexArray()
                index_array.Resize(len(uv_channel.uv_indices))
                
                for i, uv_idx in enumerate(uv_channel.uv_indices):
                    if uv_idx < len(uv_channel.uv_coordinates):
                        uv = uv_channel.uv_coordinates[uv_idx]
                        uv_tuple = (float(uv[0]), float(uv[1]))
                        if uv_tuple in uv_index_map:
                            index_array.SetAt(i, uv_index_map[uv_tuple])
                        else:
                            index_array.SetAt(i, 0)
                    else:
                        index_array.SetAt(i, 0)
            else:
                # No indices - add all UV coordinates directly
                for uv in uv_channel.uv_coordinates:
                    direct_array.Add(fbx.FbxVector2(uv[0], uv[1]))
                
        except Exception as e:
            self.logger.warning(f"Failed to add UV layer: {e}")
    
    def _update_mesh_uv(self, mesh, uv_channel: UVChannel):
        """Update UV layer in existing mesh."""
        fbx = self._fbx_module
        
        try:
            # Find existing UV layer or create new one
            uv_layer = None
            uv_element_count = mesh.GetElementUVCount()
            
            # Try to find existing UV layer with matching name or index
            for i in range(uv_element_count):
                element = mesh.GetElementUV(i)
                if element:
                    element_name = element.GetName()
                    # Match by name or use index-based naming
                    if (element_name == uv_channel.name or 
                        element_name == f"UVChannel_{uv_channel.index}" or
                        (uv_channel.index == 0 and i == 0) or
                        (uv_channel.index == 1 and i == 1)):
                        uv_layer = element
                        self.logger.info(f"Found existing UV layer: {element_name}")
                        break
            
            # If no existing layer found, create new one
            if uv_layer is None:
                self.logger.info(f"Creating new UV layer: {uv_channel.name}")
                uv_layer = mesh.CreateElementUV(uv_channel.name)
                
                # Set mapping mode
                try:
                    mapping_mode = fbx.FbxLayerElement.eByPolygonVertex
                except AttributeError:
                    try:
                        mapping_mode = fbx.FbxLayerElement.EMappingMode.eByPolygonVertex
                    except AttributeError:
                        mapping_mode = 2
                
                # Use IndexToDirect reference mode
                try:
                    reference_mode = fbx.FbxLayerElement.eIndexToDirect
                except AttributeError:
                    try:
                        reference_mode = fbx.FbxLayerElement.EReferenceMode.eIndexToDirect
                    except AttributeError:
                        reference_mode = 1
                
                uv_layer.SetMappingMode(mapping_mode)
                uv_layer.SetReferenceMode(reference_mode)
            
            # Clear existing data
            direct_array = uv_layer.GetDirectArray()
            direct_array.Clear()
            
            index_array = uv_layer.GetIndexArray()
            index_array.Clear()
            
            # Add UV coordinates
            if uv_channel.uv_indices is not None and len(uv_channel.uv_indices) > 0:
                # Use unique UV coordinates
                unique_uvs = []
                uv_index_map = {}
                
                for i, uv in enumerate(uv_channel.uv_coordinates):
                    uv_tuple = (float(uv[0]), float(uv[1]))
                    if uv_tuple not in uv_index_map:
                        uv_index_map[uv_tuple] = len(unique_uvs)
                        unique_uvs.append(uv_tuple)
                        direct_array.Add(fbx.FbxVector2(uv[0], uv[1]))
                
                # Add indices
                index_array.Resize(len(uv_channel.uv_indices))
                for i, uv_idx in enumerate(uv_channel.uv_indices):
                    if uv_idx < len(uv_channel.uv_coordinates):
                        uv = uv_channel.uv_coordinates[uv_idx]
                        uv_tuple = (float(uv[0]), float(uv[1]))
                        if uv_tuple in uv_index_map:
                            index_array.SetAt(i, uv_index_map[uv_tuple])
                        else:
                            index_array.SetAt(i, 0)
                    else:
                        index_array.SetAt(i, 0)
            else:
                # Direct mapping - add all UVs
                for uv in uv_channel.uv_coordinates:
                    direct_array.Add(fbx.FbxVector2(uv[0], uv[1]))
                
                # Create sequential indices
                index_array.Resize(len(uv_channel.uv_coordinates))
                for i in range(len(uv_channel.uv_coordinates)):
                    index_array.SetAt(i, i)
            
            self.logger.info(f"Updated UV layer with {direct_array.GetCount()} unique UVs, "
                           f"{index_array.GetCount()} indices")
            
        except Exception as e:
            self.logger.error(f"Failed to update UV layer: {e}")
