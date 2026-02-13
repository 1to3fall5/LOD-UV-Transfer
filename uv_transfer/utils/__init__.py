"""
Utility modules for logging, error handling and math operations.
"""

from .logger import setup_logger, get_logger
from .error_handler import UVTransferError, FBXError, ValidationError
from .math_utils import (
    normalize_uv,
    calculate_uv_distance,
    barycentric_interpolation,
    find_nearest_vertices,
)

__all__ = [
    "setup_logger",
    "get_logger",
    "UVTransferError",
    "FBXError", 
    "ValidationError",
    "normalize_uv",
    "calculate_uv_distance",
    "barycentric_interpolation",
    "find_nearest_vertices",
]
