"""
Native FBX parser backend implementation.
Pure Python implementation for FBX file parsing.
"""

import os
import struct
import zlib
from typing import Optional, List, Dict, Any, Tuple, BinaryIO
import numpy as np

from .fbx_handler import (
    FBXBackendBase, FBXScene, MeshData, UVChannel, FBXBackend
)
from ..utils.logger import get_logger
from ..utils.error_handler import FBXError


class FBXRecord:
    """Represents an FBX record in the file."""
    
    def __init__(self):
        self.name: str = ""
        self.properties: List[Any] = []
        self.nested_records: List['FBXRecord'] = []
    
    def find_record(self, name: str) -> Optional['FBXRecord']:
        """Find nested record by name."""
        for record in self.nested_records:
            if record.name == name:
                return record
        return None
    
    def find_all_records(self, name: str) -> List['FBXRecord']:
        """Find all nested records with given name."""
        return [r for r in self.nested_records if r.name == name]


class NativeFBXBackend(FBXBackendBase):
    """
    Native Python FBX parser backend.
    Supports FBX binary format (2014+).
    """
    
    MAGIC_BINARY = b'Kaydara FBX Binary  '
    MAGIC_ASCII = b'; FBX'
    
    def __init__(self):
        super().__init__()
        self._file_version: int = 0
    
    def is_available(self) -> bool:
        """Native backend is always available."""
        return True
    
    def initialize(self) -> bool:
        """Initialize native backend."""
        self._initialized = True
        self.logger.info("Native FBX backend initialized successfully")
        return True
    
    def load(self, file_path: str) -> FBXScene:
        """Load FBX file using native parser."""
        try:
            with open(file_path, 'rb') as f:
                magic = f.read(23)
                
                if magic.startswith(self.MAGIC_BINARY):
                    return self._load_binary(file_path, f)
                elif magic.startswith(self.MAGIC_ASCII):
                    return self._load_ascii(file_path, f)
                else:
                    raise FBXError(
                        f"Unknown FBX format: {magic[:20]}",
                        error_code=1004,
                        file_path=file_path
                    )
                    
        except FBXError:
            raise
        except Exception as e:
            raise FBXError(
                f"Failed to load FBX file: {e}",
                error_code=1002,
                file_path=file_path
            )
    
    def _load_binary(self, file_path: str, f: BinaryIO) -> FBXScene:
        """Load binary FBX file."""
        f.seek(23)
        
        version_data = f.read(4)
        self._file_version = struct.unpack('<I', version_data)[0]
        
        self.logger.debug(f"FBX version: {self._file_version}")
        
        root_record = FBXRecord()
        root_record.name = "FBXRoot"
        
        while True:
            record = self._read_record(f)
            if record is None:
                break
            root_record.nested_records.append(record)
        
        fbx_scene = FBXScene(file_path=file_path)
        
        self._parse_fbx_root(root_record, fbx_scene)
        
        return fbx_scene
    
    def _read_record(self, f: BinaryIO, end_offset: int = 0) -> Optional[FBXRecord]:
        """Read a single FBX record."""
        pos = f.tell()
        
        if end_offset > 0 and pos >= end_offset:
            return None
        
        header = f.read(12)
        if len(header) < 12:
            return None
        
        if self._file_version >= 7500:
            end_offset, num_properties, property_list_len, name_len = struct.unpack('<QQQI', header + f.read(4))
        else:
            end_offset, num_properties, property_list_len, name_len = struct.unpack('<IIII', header)
        
        if end_offset == 0:
            return None
        
        name = f.read(name_len).decode('utf-8', errors='ignore')
        
        record = FBXRecord()
        record.name = name
        
        for _ in range(num_properties):
            prop = self._read_property(f)
            record.properties.append(prop)
        
        if end_offset > f.tell():
            nested_end = end_offset
            while f.tell() < nested_end - 13:
                nested = self._read_record(f, nested_end)
                if nested:
                    record.nested_records.append(nested)
        
        if end_offset > f.tell():
            f.seek(end_offset)
        
        return record
    
    def _read_property(self, f: BinaryIO) -> Any:
        """Read a property value."""
        type_code = f.read(1)
        
        if type_code == b'Y':
            return struct.unpack('<h', f.read(2))[0]
        elif type_code == b'C':
            return struct.unpack('<?', f.read(1))[0]
        elif type_code == b'I':
            return struct.unpack('<i', f.read(4))[0]
        elif type_code == b'F':
            return struct.unpack('<f', f.read(4))[0]
        elif type_code == b'D':
            return struct.unpack('<d', f.read(8))[0]
        elif type_code == b'L':
            return struct.unpack('<q', f.read(8))[0]
        elif type_code == b'R':
            length = struct.unpack('<I', f.read(4))[0]
            return f.read(length)
        elif type_code == b'S':
            length = struct.unpack('<I', f.read(4))[0]
            return f.read(length).decode('utf-8', errors='ignore')
        elif type_code == b'f':
            return self._read_array(f, 'f', 4)
        elif type_code == b'd':
            return self._read_array(f, 'd', 8)
        elif type_code == b'l':
            return self._read_array(f, 'q', 8)
        elif type_code == b'i':
            return self._read_array(f, 'i', 4)
        elif type_code == b'b':
            return self._read_array(f, '?', 1)
        else:
            return None
    
    def _read_array(self, f: BinaryIO, fmt: str, size: int) -> np.ndarray:
        """Read array property."""
        length, encoding, compressed_size = struct.unpack('<III', f.read(12))
        
        if encoding == 0:
            data = f.read(length * size)
            return np.frombuffer(data, dtype=np.dtype(fmt))
        else:
            compressed_data = f.read(compressed_size)
            data = zlib.decompress(compressed_data)
            return np.frombuffer(data, dtype=np.dtype(fmt))
    
    def _load_ascii(self, file_path: str, f: BinaryIO) -> FBXScene:
        """Load ASCII FBX file."""
        raise FBXError(
            "ASCII FBX format not yet supported",
            error_code=1004,
            file_path=file_path
        )
    
    def _parse_fbx_root(self, root: FBXRecord, scene: FBXScene):
        """Parse FBX root record."""
        objects_record = root.find_record('Objects')
        if objects_record:
            self._parse_objects(objects_record, scene)
        
        connections_record = root.find_record('Connections')
        if connections_record:
            self._parse_connections(connections_record, scene)
    
    def _parse_objects(self, objects: FBXRecord, scene: FBXScene):
        """Parse Objects section."""
        for record in objects.nested_records:
            if len(record.properties) >= 3 and record.properties[2] == 'Mesh':
                mesh_data = self._parse_mesh(record)
                if mesh_data:
                    scene.meshes[mesh_data.name] = mesh_data
    
    def _parse_mesh(self, record: FBXRecord) -> Optional[MeshData]:
        """Parse mesh record."""
        mesh_id = record.properties[0] if record.properties else 0
        mesh_name = record.properties[1] if len(record.properties) > 1 else f"Mesh_{mesh_id}"
        
        if isinstance(mesh_name, bytes):
            mesh_name = mesh_name.decode('utf-8', errors='ignore')
        
        vertices = None
        faces = None
        normals = None
        uv_channels = {}
        
        vertices_record = record.find_record('Vertices')
        if vertices_record and vertices_record.properties:
            vertices = np.array(vertices_record.properties[0], dtype=np.float64).reshape(-1, 3)
        
        indices_record = record.find_record('PolygonVertexIndex')
        if indices_record and indices_record.properties:
            indices = np.array(indices_record.properties[0], dtype=np.int32)
            faces = self._convert_indices_to_faces(indices)
        
        normals_record = record.find_record('Normals')
        if normals_record and normals_record.properties:
            normals = np.array(normals_record.properties[0], dtype=np.float64).reshape(-1, 3)
        
        layer_uv_record = record.find_record('LayerElementUV')
        if layer_uv_record:
            uv_channel = self._parse_uv_layer(layer_uv_record, 0)
            if uv_channel:
                uv_channels[uv_channel.name] = uv_channel
        
        for i, layer_uv in enumerate(record.find_all_records('LayerElementUV')):
            if i > 0:
                uv_channel = self._parse_uv_layer(layer_uv, i)
                if uv_channel:
                    uv_channels[uv_channel.name] = uv_channel
        
        if vertices is not None and faces is not None:
            return MeshData(
                name=mesh_name,
                vertices=vertices,
                faces=faces,
                normals=normals,
                uv_channels=uv_channels
            )
        
        return None
    
    def _convert_indices_to_faces(self, indices: np.ndarray) -> np.ndarray:
        """Convert polygon vertex indices to faces."""
        faces = []
        current_face = []
        
        for idx in indices:
            if idx < 0:
                current_face.append(-idx - 1)
                if len(current_face) >= 3:
                    if len(current_face) == 3:
                        faces.append(current_face.copy())
                    else:
                        for i in range(1, len(current_face) - 1):
                            faces.append([current_face[0], current_face[i], current_face[i + 1]])
                current_face = []
            else:
                current_face.append(idx)
        
        return np.array(faces, dtype=np.int32)
    
    def _parse_uv_layer(self, record: FBXRecord, index: int) -> Optional[UVChannel]:
        """Parse UV layer."""
        name = f"UVChannel_{index}"
        
        name_record = record.find_record('Name')
        if name_record and name_record.properties:
            name = name_record.properties[0]
            if isinstance(name, bytes):
                name = name.decode('utf-8', errors='ignore')
        
        uv_data = None
        uv_record = record.find_record('UV')
        if uv_record and uv_record.properties:
            uv_data = np.array(uv_record.properties[0], dtype=np.float64).reshape(-1, 2)
        
        uv_indices = None
        uv_index_record = record.find_record('UVIndex')
        if uv_index_record and uv_index_record.properties:
            uv_indices = np.array(uv_index_record.properties[0], dtype=np.int32)
        
        if uv_data is not None:
            return UVChannel(
                name=name,
                index=index,
                uv_coordinates=uv_data,
                uv_indices=uv_indices
            )
        
        return None
    
    def _parse_connections(self, connections: FBXRecord, scene: FBXScene):
        """Parse Connections section."""
        pass
    
    def save(self, scene: FBXScene, file_path: str) -> bool:
        """Save FBX file using native writer."""
        try:
            with open(file_path, 'wb') as f:
                self._write_binary(f, scene)
            return True
        except Exception as e:
            raise FBXError(
                f"Failed to save FBX file: {e}",
                error_code=1003,
                file_path=file_path
            )
    
    def _write_binary(self, f: BinaryIO, scene: FBXScene):
        """Write binary FBX file."""
        f.write(self.MAGIC_BINARY)
        f.write(b'\x00\x00\x00')
        
        version = 7400
        f.write(struct.pack('<I', version))
        
        self._write_header_section(f)
        self._write_objects_section(f, scene)
        self._write_connections_section(f, scene)
        self._write_footer_section(f)
    
    def _write_header_section(self, f: BinaryIO):
        """Write FBX header section."""
        pass
    
    def _write_objects_section(self, f: BinaryIO, scene: FBXScene):
        """Write Objects section."""
        for mesh_data in scene.meshes.values():
            self._write_mesh(f, mesh_data)
    
    def _write_mesh(self, f: BinaryIO, mesh_data: MeshData):
        """Write mesh to file."""
        pass
    
    def _write_connections_section(self, f: BinaryIO, scene: FBXScene):
        """Write Connections section."""
        pass
    
    def _write_footer_section(self, f: BinaryIO):
        """Write FBX footer section."""
        pass
    
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
