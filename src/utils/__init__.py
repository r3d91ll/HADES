"""
Utility modules for HADES-PathRAG.

This package contains various utility functions and tools used throughout the system:
- batching: File batching utilities for parallel processing
- device_utils: GPU/CPU device configuration and management
- git_operations: Git repository operations
"""

from .device_utils import (
    is_gpu_available,
    get_device_info,
    get_gpu_diagnostics,
    set_device_mode,
    scoped_device_mode,
)
from .git_operations import GitOperations
from .batching import FileBatcher, collect_and_batch_files

__all__ = [
    # Device utilities
    "is_gpu_available",
    "get_device_info", 
    "get_gpu_diagnostics",
    "set_device_mode",
    "scoped_device_mode",
    # Git operations
    "GitOperations",
    # Batching utilities
    "FileBatcher",
    "collect_and_batch_files",
]