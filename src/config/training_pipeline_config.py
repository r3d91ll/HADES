"""Training pipeline configuration loader for orchestration system.

This module provides utilities for loading and validating training pipeline configuration
from YAML files, with support for performance profiles.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union, List

logger = logging.getLogger(__name__)


def get_config_path() -> Path:
    """Get the path to the training pipeline configuration file.
    
    Returns:
        Path to the training pipeline configuration file
    """
    config_dir = Path(__file__).parent.absolute()
    return config_dir / "training_pipeline_config.yaml"


def load_pipeline_config(profile: Optional[str] = None) -> Dict[str, Any]:
    """Load training pipeline configuration with optional profile selection.
    
    This function loads the training pipeline configuration from the training_pipeline_config.yaml file.
    It supports loading different performance profiles to adjust settings for different environments.
    
    Args:
        profile: Name of predefined profile to use ('high_throughput', 'balanced', 'low_memory')
        
    Returns:
        Configuration dictionary with training pipeline settings
    """
    config_path = get_config_path()
    
    if not config_path.exists():
        logger.warning(f"Pipeline configuration file not found at {config_path}")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Apply profile if specified
        if profile and profile in config.get("profiles", {}):
            profile_config = config["profiles"][profile]
            logger.info(f"Applying {profile} performance profile")
            
            # Merge profile settings into main config
            for section, settings in profile_config.items():
                if section in config:
                    # Deep merge dictionaries
                    _deep_merge(config[section], settings)
        
        return config
    except Exception as e:
        logger.error(f"Error loading pipeline configuration: {e}")
        return {}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries.
    
    Args:
        base: Base dictionary to merge into
        override: Dictionary with override values
        
    Returns:
        Merged dictionary (base is modified in-place)
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    
    return base


def get_queue_config(config: Dict[str, Any], queue_name: str) -> Dict[str, Any]:
    """Get configuration for a specific queue from the pipeline config.
    
    Args:
        config: Pipeline configuration dictionary
        queue_name: Name of the queue
        
    Returns:
        Queue configuration dictionary
    """
    queues = config.get("queues", {})
    
    if queue_name in queues:
        return queues[queue_name]
    
    # Return default/global settings if specific queue not found
    return config.get("global_settings", {})


# Export the functions
__all__ = ["load_pipeline_config", "get_queue_config"]
