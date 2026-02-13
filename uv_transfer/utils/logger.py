"""
Logging system for UV Transfer Tool.
Provides comprehensive logging with timestamps, operation types, and error levels.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import os


class UVTransferFormatter(logging.Formatter):
    """Custom formatter with detailed information."""
    
    FORMATS = {
        logging.DEBUG: "[%(asctime)s] [DEBUG] [%(module)s] %(message)s",
        logging.INFO: "[%(asctime)s] [INFO] [%(operation)s] %(message)s",
        logging.WARNING: "[%(asctime)s] [WARNING] [%(operation)s] %(message)s",
        logging.ERROR: "[%(asctime)s] [ERROR] [%(operation)s] %(message)s\nStack: %(exc_info)s",
        logging.CRITICAL: "[%(asctime)s] [CRITICAL] [%(operation)s] %(message)s\nFile: %(filename)s:%(lineno)d",
    }
    
    def format(self, record):
        if not hasattr(record, 'operation'):
            record.operation = 'general'
        if not hasattr(record, 'exc_info'):
            record.exc_info = 'N/A'
        
        fmt = self.FORMATS.get(record.levelno, self.FORMATS[logging.INFO])
        formatter = logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)


class UVTransferLogger(logging.Logger):
    """Custom logger with operation tracking."""
    
    def __init__(self, name: str, level: int = logging.DEBUG):
        super().__init__(name, level)
        self.current_operation = "general"
    
    def set_operation(self, operation: str):
        """Set current operation context."""
        self.current_operation = operation
    
    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False):
        if extra is None:
            extra = {}
        extra['operation'] = self.current_operation
        super()._log(level, msg, args, exc_info, extra, stack_info)


_loggers: dict = {}
_log_dir: Optional[Path] = None


def setup_logger(
    name: str = "uv_transfer",
    log_dir: Optional[str] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> UVTransferLogger:
    """
    Setup logger with console and file handlers.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        console_level: Logging level for console output
        file_level: Logging level for file output
    
    Returns:
        Configured logger instance
    """
    global _log_dir
    
    logging.setLoggerClass(UVTransferLogger)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    if logger.handlers:
        return logger
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(UVTransferFormatter())
    logger.addHandler(console_handler)
    
    if log_dir:
        _log_dir = Path(log_dir)
        _log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = _log_dir / f"uv_transfer_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(file_level)
        file_handler.setFormatter(UVTransferFormatter())
        logger.addHandler(file_handler)
    
    _loggers[name] = logger
    return logger


def get_logger(name: str = "uv_transfer") -> UVTransferLogger:
    """
    Get existing logger or create default one.
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    if name in _loggers:
        return _loggers[name]
    return setup_logger(name)


class OperationContext:
    """Context manager for operation logging."""
    
    def __init__(self, logger: UVTransferLogger, operation: str, description: str = ""):
        self.logger = logger
        self.operation = operation
        self.description = description
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.set_operation(self.operation)
        self.logger.info(f"Starting: {self.description}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        if exc_type:
            self.logger.error(f"Failed: {self.description} (Duration: {duration:.2f}s)")
        else:
            self.logger.info(f"Completed: {self.description} (Duration: {duration:.2f}s)")
        self.logger.set_operation("general")
        return False
