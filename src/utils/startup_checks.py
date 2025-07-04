"""
Startup checks for HADES.

This module performs essential checks when the system starts up,
including version verification, CUDA availability, and system resources.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any

try:
    import psutil
except ImportError:
    psutil = None

from .version_checker import check_all_versions, get_cuda_info

logger = logging.getLogger(__name__)


def check_system_resources() -> Dict[str, Any]:
    """Check available system resources."""
    if psutil is None:
        return {
            'error': 'psutil not installed',
            'cpu_count': 'unknown',
            'memory_gb': 'unknown',
            'available_memory_gb': 'unknown',
            'disk_usage_percent': 'unknown'
        }
    
    return {
        'cpu_count': psutil.cpu_count(),
        'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'available_memory_gb': round(psutil.virtual_memory().available / (1024**3), 2),
        'disk_usage_percent': psutil.disk_usage('/').percent
    }


def check_model_cache() -> Dict[str, Any]:
    """Check if model cache directory exists and has models."""
    cache_dirs = [
        Path.home() / '.cache' / 'huggingface',
        Path('/models'),  # Common Docker mount point
        Path('./models')  # Local models directory
    ]
    
    results = {}
    for cache_dir in cache_dirs:
        if cache_dir.exists():
            model_files = list(cache_dir.rglob('*.bin')) + list(cache_dir.rglob('*.safetensors'))
            results[str(cache_dir)] = {
                'exists': True,
                'model_count': len(model_files),
                'size_gb': round(sum(f.stat().st_size for f in model_files) / (1024**3), 2) if model_files else 0
            }
        else:
            results[str(cache_dir)] = {'exists': False}
    
    return results


def run_startup_checks(verbose: bool = True) -> bool:
    """
    Run all startup checks for HADES.
    
    Args:
        verbose: Whether to print detailed output
        
    Returns:
        True if all checks pass, False otherwise
    """
    all_good = True
    
    if verbose:
        print("=" * 60)
        print("HADES Startup Checks")
        print("=" * 60)
    
    # Check versions
    logger.info("Checking dependency versions...")
    try:
        version_results = check_all_versions(raise_on_error=True)
        if verbose:
            print("\n✓ Dependency versions OK")
    except Exception as e:
        logger.error(f"Version check failed: {e}")
        if verbose:
            print(f"\n✗ Version check failed: {e}")
        all_good = False
    
    # Check CUDA
    logger.info("Checking CUDA availability...")
    cuda_info = get_cuda_info()
    if verbose:
        print(f"\n{'✓' if cuda_info.get('cuda_available') else '⚠'} CUDA: {'Available' if cuda_info.get('cuda_available') else 'Not available'}")
        if cuda_info.get('cuda_available'):
            print(f"  - CUDA Version: {cuda_info.get('cuda_version')}")
            print(f"  - GPU Count: {cuda_info.get('gpu_count')}")
            for i, gpu_name in enumerate(cuda_info.get('gpu_names', [])):
                print(f"  - GPU {i}: {gpu_name}")
    
    # Check system resources
    logger.info("Checking system resources...")
    resources = check_system_resources()
    if verbose:
        print(f"\n✓ System Resources:")
        print(f"  - CPUs: {resources['cpu_count']}")
        print(f"  - Total Memory: {resources['memory_gb']} GB")
        print(f"  - Available Memory: {resources['available_memory_gb']} GB")
        print(f"  - Disk Usage: {resources['disk_usage_percent']}%")
    
    if isinstance(resources.get('available_memory_gb'), (int, float)) and resources['available_memory_gb'] < 8:
        logger.warning("Less than 8GB of memory available - may affect performance")
        if verbose:
            print("\n⚠ Warning: Less than 8GB of memory available")
    
    # Check model cache
    logger.info("Checking model cache directories...")
    cache_info = check_model_cache()
    if verbose:
        print("\n✓ Model Cache Directories:")
        for path, info in cache_info.items():
            if info['exists']:
                print(f"  - {path}: {info['model_count']} models ({info['size_gb']} GB)")
            else:
                print(f"  - {path}: Not found")
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"Startup checks {'PASSED' if all_good else 'FAILED'}")
        print("=" * 60 + "\n")
    
    return all_good


if __name__ == '__main__':
    # Run checks when module is executed directly
    success = run_startup_checks(verbose=True)
    sys.exit(0 if success else 1)