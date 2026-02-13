"""
UV Validator - Validates UV data integrity and accuracy.
"""

from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
import numpy as np

from ..utils.logger import get_logger
from ..utils.error_handler import ValidationError
from ..utils.math_utils import (
    calculate_uv_distance,
    calculate_uv_area,
    normalize_uv
)


@dataclass
class ValidationResult:
    """Result of UV validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def uv_count(self) -> int:
        return self.details.get('uv_count', 0)
    
    @property
    def coverage(self) -> float:
        return self.details.get('coverage', 0.0)


class UVValidator:
    """
    Validates UV data for integrity and correctness.
    
    Checks performed:
    - Data completeness
    - Value range (0-1)
    - NaN/Inf values
    - Degenerate UVs (zero area)
    - Overlapping UVs
    """
    
    def __init__(self):
        self.logger = get_logger("uv_transfer.validator")
    
    def validate_uv_channel(
        self,
        uv_channel,
        check_overlap: bool = False,
        tolerance: float = 1e-6
    ) -> ValidationResult:
        """
        Validate a UV channel for data integrity.
        
        Args:
            uv_channel: UVChannel to validate
            check_overlap: Whether to check for overlapping UVs
            tolerance: Numerical tolerance
        
        Returns:
            ValidationResult with validation details
        """
        result = ValidationResult(is_valid=True)
        
        uv_coords = uv_channel.uv_coordinates
        
        result.details['uv_count'] = len(uv_coords)
        
        nan_count = np.sum(np.isnan(uv_coords))
        inf_count = np.sum(np.isinf(uv_coords))
        
        if nan_count > 0:
            result.errors.append(f"Found {nan_count} NaN values in UV coordinates")
            result.is_valid = False
        
        if inf_count > 0:
            result.errors.append(f"Found {inf_count} Inf values in UV coordinates")
            result.is_valid = False
        
        if result.is_valid:
            out_of_range = np.sum(
                (uv_coords < -tolerance) | (uv_coords > 1.0 + tolerance)
            )
            
            if out_of_range > 0:
                result.warnings.append(
                    f"Found {out_of_range} UV coordinates outside [0,1] range"
                )
        
        if result.is_valid:
            unique_uvs = len(np.unique(uv_coords, axis=0))
            duplicate_count = len(uv_coords) - unique_uvs
            
            if duplicate_count > len(uv_coords) * 0.5:
                result.warnings.append(
                    f"High number of duplicate UVs: {duplicate_count}"
                )
        
        result.details['coverage'] = self._compute_uv_coverage(uv_coords)
        
        self.logger.debug(
            f"UV validation: valid={result.is_valid}, "
            f"errors={len(result.errors)}, warnings={len(result.warnings)}"
        )
        
        return result
    
    def validate_mesh_uv(
        self,
        mesh_data,
        uv_channel_index: int = 0
    ) -> ValidationResult:
        """
        Validate UV data for a mesh.
        
        Args:
            mesh_data: MeshData to validate
            uv_channel_index: UV channel to validate
        
        Returns:
            ValidationResult
        """
        uv_channel = mesh_data.get_uv_channel(uv_channel_index)
        
        if uv_channel is None:
            return ValidationResult(
                is_valid=False,
                errors=[f"UV channel {uv_channel_index} not found"]
            )
        
        result = self.validate_uv_channel(uv_channel)
        
        if result.is_valid:
            self._validate_uv_vertex_ratio(
                result,
                len(uv_channel.uv_coordinates),
                mesh_data.vertex_count
            )
        
        return result
    
    def _validate_uv_vertex_ratio(
        self,
        result: ValidationResult,
        uv_count: int,
        vertex_count: int
    ):
        """Validate UV to vertex count ratio."""
        if uv_count < vertex_count:
            result.warnings.append(
                f"UV count ({uv_count}) less than vertex count ({vertex_count})"
            )
        elif uv_count > vertex_count * 4:
            result.warnings.append(
                f"UV count ({uv_count}) significantly higher than vertex count ({vertex_count})"
            )
    
    def _compute_uv_coverage(self, uv_coords: np.ndarray) -> float:
        """Compute UV space coverage (0-1)."""
        if len(uv_coords) == 0:
            return 0.0
        
        uv_coords = normalize_uv(uv_coords)
        
        min_u = np.min(uv_coords[:, 0])
        max_u = np.max(uv_coords[:, 0])
        min_v = np.min(uv_coords[:, 1])
        max_v = np.max(uv_coords[:, 1])
        
        coverage = (max_u - min_u) * (max_v - min_v)
        
        return min(1.0, coverage)
    
    def compute_accuracy(
        self,
        source_uv,
        target_uv,
        vertex_mapping: np.ndarray,
        source_vertices: np.ndarray,
        target_vertices: np.ndarray
    ) -> float:
        """
        Compute transfer accuracy between source and target UVs.
        
        Args:
            source_uv: Source UVChannel
            target_uv: Target UVChannel
            vertex_mapping: Vertex mapping array
            source_vertices: Source vertex positions
            target_vertices: Target vertex positions
        
        Returns:
            Accuracy value (lower is better)
        """
        source_uvs = source_uv.uv_coordinates
        target_uvs = target_uv.uv_coordinates
        
        matched_mask = vertex_mapping >= 0
        matched_count = np.sum(matched_mask)
        
        if matched_count == 0:
            return float('inf')
        
        total_error = 0.0
        
        for i in range(len(target_vertices)):
            if not matched_mask[i]:
                continue
            
            source_idx = vertex_mapping[i]
            
            pos_distance = np.linalg.norm(
                target_vertices[i] - source_vertices[source_idx]
            )
            
            uv_distance = calculate_uv_distance(
                target_uvs[i:i+1],
                source_uvs[source_idx:source_idx+1]
            )[0]
            
            if pos_distance > 0:
                total_error += uv_distance / pos_distance
            else:
                total_error += uv_distance
        
        return total_error / matched_count
    
    def compare_uv_channels(
        self,
        uv1,
        uv2,
        tolerance: float = 0.001
    ) -> Dict[str, Any]:
        """
        Compare two UV channels.
        
        Args:
            uv1: First UVChannel
            uv2: Second UVChannel
            tolerance: Comparison tolerance
        
        Returns:
            Dictionary with comparison results
        """
        coords1 = uv1.uv_coordinates
        coords2 = uv2.uv_coordinates
        
        result = {
            'count_match': len(coords1) == len(coords2),
            'count1': len(coords1),
            'count2': len(coords2),
        }
        
        if len(coords1) == len(coords2):
            distances = calculate_uv_distance(coords1, coords2)
            
            result['mean_distance'] = float(np.mean(distances))
            result['max_distance'] = float(np.max(distances))
            result['min_distance'] = float(np.min(distances))
            result['within_tolerance'] = float(np.sum(distances < tolerance) / len(distances))
        
        return result
    
    def detect_uv_issues(
        self,
        uv_coords: np.ndarray,
        faces: np.ndarray
    ) -> Dict[str, Any]:
        """
        Detect various UV issues.
        
        Args:
            uv_coords: UV coordinates (N, 2)
            faces: Face indices (F, 3)
        
        Returns:
            Dictionary with detected issues
        """
        issues = {
            'nan_values': 0,
            'inf_values': 0,
            'out_of_range': 0,
            'zero_area_faces': 0,
            'overlapping_uvs': 0,
        }
        
        issues['nan_values'] = int(np.sum(np.isnan(uv_coords)))
        issues['inf_values'] = int(np.sum(np.isinf(uv_coords)))
        issues['out_of_range'] = int(np.sum(
            (uv_coords < 0) | (uv_coords > 1)
        ))
        
        for face in faces:
            uv_face = uv_coords[face]
            area = calculate_uv_area(uv_face)
            
            if area < 1e-10:
                issues['zero_area_faces'] += 1
        
        unique_uvs = np.unique(uv_coords, axis=0)
        issues['overlapping_uvs'] = len(uv_coords) - len(unique_uvs)
        
        return issues
    
    def generate_validation_report(
        self,
        mesh_data,
        uv_channel_index: int = 0
    ) -> str:
        """
        Generate a detailed validation report.
        
        Args:
            mesh_data: MeshData to validate
            uv_channel_index: UV channel to validate
        
        Returns:
            Formatted report string
        """
        uv_channel = mesh_data.get_uv_channel(uv_channel_index)
        
        if uv_channel is None:
            return f"UV channel {uv_channel_index} not found in mesh {mesh_data.name}"
        
        result = self.validate_uv_channel(uv_channel)
        issues = self.detect_uv_issues(
            uv_channel.uv_coordinates,
            mesh_data.faces
        )
        
        report_lines = [
            f"UV Validation Report for {mesh_data.name}",
            "=" * 50,
            f"UV Channel: {uv_channel.name} (index {uv_channel_index})",
            f"UV Count: {result.uv_count}",
            f"Vertex Count: {mesh_data.vertex_count}",
            f"Coverage: {result.coverage:.2%}",
            "",
            "Issues:",
            f"  NaN values: {issues['nan_values']}",
            f"  Inf values: {issues['inf_values']}",
            f"  Out of range: {issues['out_of_range']}",
            f"  Zero area faces: {issues['zero_area_faces']}",
            f"  Overlapping UVs: {issues['overlapping_uvs']}",
            "",
            f"Status: {'VALID' if result.is_valid else 'INVALID'}",
        ]
        
        if result.errors:
            report_lines.append("")
            report_lines.append("Errors:")
            for error in result.errors:
                report_lines.append(f"  - {error}")
        
        if result.warnings:
            report_lines.append("")
            report_lines.append("Warnings:")
            for warning in result.warnings:
                report_lines.append(f"  - {warning}")
        
        return "\n".join(report_lines)
