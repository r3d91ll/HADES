"""
Version checking utility for critical dependencies.

This module provides centralized version checking for torch and transformers
to ensure compatibility across the HADES codebase.
"""

import logging
import sys
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


# Required versions for HADES
REQUIRED_VERSIONS = {
    'torch': (2, 6, 0),  # >= 2.6.0
    'transformers': (4, 52, 0),  # >= 4.52.0
}


def parse_version(version_str: str) -> Tuple[int, ...]:
    """Parse version string into tuple of integers."""
    try:
        # Handle versions like '2.6.0+cu118'
        base_version = version_str.split('+')[0]
        return tuple(int(x) for x in base_version.split('.')[:3])
    except Exception:
        return (0, 0, 0)


def check_version(package_name: str, required_version: Tuple[int, ...]) -> Tuple[bool, str]:
    """
    Check if installed package version meets requirements.
    
    Args:
        package_name: Name of the package to check
        required_version: Required minimum version as tuple (major, minor, patch)
        
    Returns:
        Tuple of (is_valid, message)
    """
    try:
        if package_name == 'torch':
            import torch
            current_version = parse_version(torch.__version__)
        elif package_name == 'transformers':
            import transformers
            current_version = parse_version(transformers.__version__)
        else:
            return False, f"Unknown package: {package_name}"
            
        required_str = '.'.join(str(x) for x in required_version)
        current_str = '.'.join(str(x) for x in current_version)
        
        if current_version >= required_version:
            return True, f"{package_name} {current_str} meets requirement >={required_str}"
        else:
            return False, f"{package_name} {current_str} does not meet requirement >={required_str}"
            
    except ImportError:
        return False, f"{package_name} is not installed"
    except Exception as e:
        return False, f"Error checking {package_name}: {str(e)}"


def check_all_versions(raise_on_error: bool = True) -> Dict[str, Tuple[bool, str]]:
    """
    Check all required package versions.
    
    Args:
        raise_on_error: Whether to raise exception on version mismatch
        
    Returns:
        Dictionary mapping package names to (is_valid, message) tuples
    """
    results = {}
    all_valid = True
    
    for package, required_version in REQUIRED_VERSIONS.items():
        is_valid, message = check_version(package, required_version)
        results[package] = (is_valid, message)
        
        if is_valid:
            logger.info(f"✓ {message}")
        else:
            logger.error(f"✗ {message}")
            all_valid = False
    
    if not all_valid and raise_on_error:
        error_msg = "Version requirements not met. Please update dependencies:\n"
        error_msg += "pip install torch>=2.6.0 transformers>=4.52.0"
        raise RuntimeError(error_msg)
    
    return results


def ensure_torch_available() -> None:
    """Ensure torch is available and meets version requirements."""
    is_valid, message = check_version('torch', REQUIRED_VERSIONS['torch'])
    if not is_valid:
        logger.error(message)
        logger.error("Please install torch>=2.6.0")
        sys.exit(1)


def ensure_transformers_available() -> None:
    """Ensure transformers is available and meets version requirements."""
    is_valid, message = check_version('transformers', REQUIRED_VERSIONS['transformers'])
    if not is_valid:
        logger.error(message)
        logger.error("Please install transformers>=4.52.0")
        sys.exit(1)


def get_cuda_info() -> Dict[str, Any]:
    """Get CUDA availability and version information."""
    try:
        import torch
        return {
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
        }
    except Exception as e:
        return {
            'cuda_available': False,
            'error': str(e)
        }


if __name__ == '__main__':
    # Test version checking
    logging.basicConfig(level=logging.INFO)
    
    print("Checking HADES dependency versions...\n")
    results = check_all_versions(raise_on_error=False)
    
    print("\nCUDA Information:")
    cuda_info = get_cuda_info()
    for key, value in cuda_info.items():
        print(f"  {key}: {value}")