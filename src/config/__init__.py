"""
Configuration module for HADES.

This module mirrors the src/ structure exactly, providing configuration
files for each module. This pattern ensures consistency for both human
maintainability and ISNE pattern learning.

Directory Structure:
- api/ - API server and endpoint configurations
- concepts/ - Concept processing configurations
- isne/ - ISNE model and training configurations
- jina_v4/ - Jina v4 processor configuration
- pathrag/ - PathRAG strategy configurations
- storage/ - Storage backend configurations
- utils/ - Utility configurations
- validation/ - Validation configurations
"""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import os


def load_config(config_path: str, use_env: bool = True) -> Dict[str, Any]:
    """
    Load configuration from YAML file with optional environment variable override.
    
    Args:
        config_path: Path to configuration file (relative to src/config/)
        use_env: Whether to allow environment variable overrides
        
    Returns:
        Dictionary containing configuration
    """
    config_dir = Path(__file__).parent
    full_path = config_dir / config_path
    
    if not full_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {full_path}")
    
    with open(full_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if use_env:
        # Override with environment variables
        config = _apply_env_overrides(config)
    
    return config


def _apply_env_overrides(config: Dict[str, Any], prefix: str = "HADES_") -> Dict[str, Any]:
    """Apply environment variable overrides to configuration."""
    for key, value in config.items():
        env_key = f"{prefix}{key.upper()}"
        if env_key in os.environ:
            config[key] = os.environ[env_key]
        elif isinstance(value, dict):
            config[key] = _apply_env_overrides(value, f"{prefix}{key.upper()}_")
    return config