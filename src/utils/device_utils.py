"""
Utilities for managing device (CPU/GPU) configurations.

This module provides functions to detect and configure device usage (CPU/GPU) 
in a way that works consistently across different ML frameworks.

The primary method of controlling GPU visibility is through the CUDA_VISIBLE_DEVICES
environment variable, which is an industry standard approach supported by:
- PyTorch
- TensorFlow
- JAX
- Most CUDA-based libraries

Key features:
- Check if GPUs are available with is_gpu_available()
- Get detailed device information with get_device_info()
- Configure device settings with set_device_mode()

Usage example:
    from src.utils.device_utils import is_gpu_available, get_device_info

    # Check if GPU is available
    if is_gpu_available():
        print("GPU is available")
    else:
        print("Running in CPU-only mode")
        
    # Get detailed device information
    device_info = get_device_info()
    print(f"Using device: {device_info['device_name']}")
"""
import os
import logging
import contextlib
from typing import Tuple, Dict, Any, Optional, Iterator

import torch

logger = logging.getLogger(__name__)


def is_gpu_available() -> bool:
    """
    Check if GPU is available based on CUDA_VISIBLE_DEVICES and PyTorch detection.
    
    Returns:
        bool: True if GPU is available, False otherwise
    """
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    
    # If CUDA_VISIBLE_DEVICES is empty or set to a negative number, no GPUs are available
    if cuda_devices == "" or cuda_devices == "-1":
        return False
    
    # Check if CUDA is available according to PyTorch
    return torch.cuda.is_available()


def get_device_info() -> Dict[str, Any]:
    """
    Get information about the available compute devices.
    
    Returns:
        Dict with device information
    """
    gpu_available = is_gpu_available()
    device_type = "gpu" if gpu_available else "cpu"
    device_name = f"cuda:0" if gpu_available else "cpu"
    
    # Get additional details if GPU is available
    gpu_details = {}
    if gpu_available:
        gpu_details = {
            "gpu_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(0),
        }
    
    return {
        "device_type": device_type,
        "device_name": device_name,
        "gpu_available": gpu_available,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"),
        "gpu_details": gpu_details,
    }


def get_gpu_diagnostics() -> Dict[str, Any]:
    """
    Get detailed diagnostic information about GPU configuration and mapping.
    
    This function is primarily for debugging GPU issues, particularly when dealing with
    CUDA_VISIBLE_DEVICES mapping and ordinal problems. It provides extensive details about
    the GPU environment including memory status, driver version, and how PyTorch sees the devices.
    
    Returns:
        Dict with comprehensive GPU diagnostic information
    """
    diagnostics = {
        "environment": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"),
            "pytorch_cuda_device": os.environ.get("PYTORCH_CUDA_DEVICE", "Not set"),
        },
        "cuda_available": torch.cuda.is_available(),
        "devices": [],
        "cuda_initialized": torch.cuda.is_initialized(),
    }
    
    if torch.cuda.is_available():
        try:
            # Add CUDA version information
            diagnostics["cuda_version"] = torch.version.cuda
            diagnostics["device_count"] = torch.cuda.device_count()
            
            # Add information for each visible device
            for i in range(torch.cuda.device_count()):
                device_info = {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "logical_device": f"cuda:{i}",
                    "memory": {
                        "total": torch.cuda.get_device_properties(i).total_memory / (1024**3),  # GB
                        "allocated": torch.cuda.memory_allocated(i) / (1024**3),  # GB
                        "reserved": torch.cuda.memory_reserved(i) / (1024**3),  # GB
                    },
                    "compute_capability": f"{torch.cuda.get_device_capability(i)[0]}.{torch.cuda.get_device_capability(i)[1]}",
                }
                diagnostics["devices"].append(device_info)
                
            # Add current device info
            diagnostics["current_device"] = {
                "index": torch.cuda.current_device(),
                "name": torch.cuda.get_device_name(torch.cuda.current_device()),
            }
            
        except Exception as e:
            diagnostics["error"] = str(e)
    
    # Add mapping explanation based on CUDA_VISIBLE_DEVICES
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_visible_devices:
        try:
            # Parse device indices from CUDA_VISIBLE_DEVICES
            mapped_indices = [int(idx.strip()) for idx in cuda_visible_devices.split(",") if idx.strip().isdigit()]
            
            # Create mapping explanation
            mapping = []
            for logical_idx, physical_idx in enumerate(mapped_indices):
                mapping.append({
                    "logical_index": logical_idx,  # What PyTorch sees
                    "logical_device": f"cuda:{logical_idx}",  # What PyTorch uses
                    "physical_index": physical_idx,  # Actual hardware GPU index
                    "physical_device": f"GPU {physical_idx}",  # Actual hardware device
                })
                
            diagnostics["device_mapping"] = mapping
            diagnostics["mapping_explanation"] = (
                f"With CUDA_VISIBLE_DEVICES={cuda_visible_devices}, PyTorch indexing is remapped. "
                f"When you specify 'cuda:0', you're actually using physical GPU {mapped_indices[0] if mapped_indices else 'None'}."
            )
        except Exception as e:
            diagnostics["mapping_error"] = str(e)
    else:
        diagnostics["mapping_explanation"] = "No CUDA_VISIBLE_DEVICES set, so logical device indexing matches physical indexing."
        
    return diagnostics


@contextlib.contextmanager
def scoped_device_mode(force_cpu: bool = False, device_id: Optional[str] = None) -> Iterator[Dict[str, Any]]:
    """
    Context manager that temporarily sets device mode for PyTorch and other frameworks
    within its scope. Original environment variables are restored when exiting the scope.
    
    Args:
        force_cpu: If True, force CPU usage regardless of GPU availability
        device_id: Specific GPU device ID to use (e.g., "0", "1"). Only used if force_cpu is False.
        
    Yields:
        Dict with device settings that were applied
    """
    # Save original environment variables
    original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    
    try:
        # Apply device settings
        if force_cpu:
            # Force CPU mode by hiding all CUDA devices
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            logger.info("Temporarily forcing CPU mode by setting CUDA_VISIBLE_DEVICES to empty string")
        elif device_id is not None:
            # Use a specific GPU device
            os.environ["CUDA_VISIBLE_DEVICES"] = device_id
            logger.info(f"Temporarily setting specific GPU device: CUDA_VISIBLE_DEVICES={device_id}")
        
        # Get device info after applying settings
        device_info = get_device_info()
        
        # Log configured device
        device_type = "CPU" if force_cpu or not device_info["gpu_available"] else "GPU"
        logger.info(f"Device temporarily configured: {device_type} ({device_info['device_name']})")
        
        # Yield device info to the caller
        yield device_info
        
    finally:
        # Restore original environment variables
        if original_cuda_visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible_devices
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        logger.info("Restored original CUDA_VISIBLE_DEVICES setting")


def set_device_mode(force_cpu: bool = False, device_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Set the device mode for PyTorch and other frameworks by configuring environment variables.
    This should be called before importing any ML libraries.
    
    Args:
        force_cpu: If True, force CPU usage regardless of GPU availability
        device_id: Specific GPU device ID to use (e.g., "0", "1"). Only used if force_cpu is False.
        
    Returns:
        Dict with device settings that were applied
    """
    if force_cpu:
        # Force CPU mode by hiding all CUDA devices
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        logger.info("Forcing CPU mode by setting CUDA_VISIBLE_DEVICES to empty string")
    elif device_id is not None:
        # Use a specific GPU device
        os.environ["CUDA_VISIBLE_DEVICES"] = device_id
        logger.info(f"Setting specific GPU device: CUDA_VISIBLE_DEVICES={device_id}")
    
    # Return current device information after setting environment variables
    device_info = get_device_info()
    
    # Log configured device
    device_type = "CPU" if force_cpu or not device_info["gpu_available"] else "GPU"
    logger.info(f"Device configured: {device_type} ({device_info['device_name']})")
    
    return device_info
