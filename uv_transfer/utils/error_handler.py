"""
Error handling module for UV Transfer Tool.
Provides comprehensive error types and handling mechanisms.
"""

from typing import Optional, Any
import traceback


class UVTransferError(Exception):
    """Base exception for UV Transfer Tool."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[int] = None,
        details: Optional[dict] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or 0
        self.details = details or {}
        self.stack_trace = traceback.format_exc()
    
    def to_dict(self) -> dict:
        """Convert error to dictionary."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
            "stack_trace": self.stack_trace,
        }
    
    def __str__(self):
        return f"[{self.error_code}] {self.message}"


class FBXError(UVTransferError):
    """FBX file related errors."""
    
    ERROR_CODES = {
        1001: "FBX file not found",
        1002: "FBX file read error",
        1003: "FBX file write error",
        1004: "Invalid FBX format",
        1005: "FBX SDK initialization failed",
        1006: "FBX scene creation failed",
        1007: "FBX mesh not found",
        1008: "FBX UV channel not found",
        1009: "FBX UV channel creation failed",
        1010: "FBX dependency check failed",
        1011: "FBX node not found",
        1012: "FBX geometry error",
    }
    
    def __init__(
        self,
        message: str,
        error_code: Optional[int] = None,
        file_path: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, error_code, kwargs)
        self.file_path = file_path
        if file_path:
            self.details["file_path"] = file_path


class ValidationError(UVTransferError):
    """UV data validation errors."""
    
    ERROR_CODES = {
        2001: "UV data incomplete",
        2002: "UV coordinate out of range",
        2003: "UV index mismatch",
        2004: "Vertex count mismatch",
        2005: "UV channel missing",
        2006: "UV data corrupted",
        2007: "UV transfer accuracy below threshold",
        2008: "Vertex mapping failed",
        2009: "Topology mismatch",
        2010: "Non-manifold geometry detected",
    }
    
    def __init__(
        self,
        message: str,
        error_code: Optional[int] = None,
        source_model: Optional[str] = None,
        target_model: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, error_code, kwargs)
        self.source_model = source_model
        self.target_model = target_model
        if source_model:
            self.details["source_model"] = source_model
        if target_model:
            self.details["target_model"] = target_model


class TransferError(UVTransferError):
    """UV transfer operation errors."""
    
    ERROR_CODES = {
        3001: "Transfer initialization failed",
        3002: "Source model load failed",
        3003: "Target model load failed",
        3004: "UV mapping computation failed",
        3005: "UV interpolation failed",
        3006: "UV write failed",
        3007: "Batch transfer failed",
        3008: "Configuration error",
    }
    
    def __init__(
        self,
        message: str,
        error_code: Optional[int] = None,
        source_uv_channel: Optional[int] = None,
        target_uv_channel: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, error_code, kwargs)
        self.source_uv_channel = source_uv_channel
        self.target_uv_channel = target_uv_channel
        if source_uv_channel is not None:
            self.details["source_uv_channel"] = source_uv_channel
        if target_uv_channel is not None:
            self.details["target_uv_channel"] = target_uv_channel


class ConfigError(UVTransferError):
    """Configuration related errors."""
    
    ERROR_CODES = {
        4001: "Config file not found",
        4002: "Config parse error",
        4003: "Invalid config value",
        4004: "Missing required config",
    }


class ErrorHandler:
    """Centralized error handling manager."""
    
    def __init__(self, logger=None):
        self.logger = logger
        self.error_history: list = []
        self.max_history = 100
    
    def handle(
        self,
        error: Exception,
        operation: str = "unknown",
        reraise: bool = True,
        recovery: Optional[callable] = None
    ) -> bool:
        """
        Handle an error with logging and optional recovery.
        
        Args:
            error: The exception to handle
            operation: Operation name for context
            reraise: Whether to reraise the error
            recovery: Optional recovery function
        
        Returns:
            True if error was recovered, False otherwise
        """
        error_info = {
            "operation": operation,
            "error": str(error),
            "type": type(error).__name__,
        }
        
        if isinstance(error, UVTransferError):
            error_info.update(error.to_dict())
        
        self.error_history.append(error_info)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
        
        if self.logger:
            self.logger.error(
                f"Error in {operation}: {error}",
                extra={"operation": operation}
            )
        
        if recovery:
            try:
                recovery()
                return True
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Recovery failed: {e}")
        
        if reraise:
            raise error
        
        return False
    
    def get_last_error(self) -> Optional[dict]:
        """Get the last error from history."""
        return self.error_history[-1] if self.error_history else None
    
    def clear_history(self):
        """Clear error history."""
        self.error_history.clear()
