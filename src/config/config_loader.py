"""
Configuration loading utilities for HADES-PathRAG.

This module provides functions to load various configuration files,
including the pipeline configuration and component-specific configs.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

# Configure logger
logger = logging.getLogger(__name__)

# Define the paths to the configuration files
CONFIG_DIR = Path(__file__).parent
TRAINING_PIPELINE_CONFIG_PATH = CONFIG_DIR / 'training_pipeline_config.yaml'
PIPELINE_CONFIG_PATH = TRAINING_PIPELINE_CONFIG_PATH  # For backward compatibility

# Set CUDA_VISIBLE_DEVICES at module import time, before any PyTorch imports
# This ensures GPU settings are applied early enough to affect all components
def _apply_cuda_environment_variables() -> None:
    """Apply CUDA environment variables from pipeline configuration at module import time.
    
    This function is called immediately when the module is imported to ensure
    CUDA_VISIBLE_DEVICES is set before any PyTorch imports.
    """
    try:
        # Load the training pipeline config to check device settings
        path = TRAINING_PIPELINE_CONFIG_PATH
        if not path.exists():
            return
            
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check if there's a device_config section
        if config and 'pipeline' in config and 'device_config' in config['pipeline']:
            device_config = config['pipeline']['device_config']
            
            # Get CUDA_VISIBLE_DEVICES from pipeline config if it exists
            if 'CUDA_VISIBLE_DEVICES' in device_config:
                cuda_devices = device_config['CUDA_VISIBLE_DEVICES']
                
                # Apply this setting - explicitly overwrites any existing environment variable
                # to ensure the config file takes precedence
                if cuda_devices is not None:  # None means use system default
                    # Ensure it's set in the proper uppercase format
                    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_devices)
                    # Print instead of log since logging may not be configured yet
                    print(f"Setting CUDA_VISIBLE_DEVICES to '{cuda_devices}' at module import time")
    except Exception as e:
        # Print to stderr since logging may not be configured yet
        import sys
        print(f"Error setting CUDA_VISIBLE_DEVICES from config: {e}", file=sys.stderr)

# Execute immediately at import time
_apply_cuda_environment_variables()


def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary with configuration values
        
    Raises:
        FileNotFoundError: If the configuration file does not exist
        yaml.YAMLError: If the YAML file is invalid
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
        
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        
    return config or {}


def load_pipeline_config(config_path: Optional[Union[str, Path]] = None, pipeline_type: str = 'training') -> Dict[str, Any]:
    """
    Load the pipeline configuration.
    
    Args:
        config_path: Path to the pipeline configuration file (optional, uses default if not provided)
        pipeline_type: Type of pipeline configuration to load ('training', 'ingestion', etc.)
        
    Returns:
        Dictionary with pipeline configuration values
    """
    # If a specific path is provided, use it
    if config_path:
        path = Path(config_path)
    # Otherwise, select the appropriate config file based on the pipeline type
    else:
        if pipeline_type == 'training':
            path = TRAINING_PIPELINE_CONFIG_PATH
        elif pipeline_type == 'ingestion':
            # Will be implemented in the future
            path = CONFIG_DIR / 'ingestion_pipeline_config.yaml'
            if not path.exists():
                # Fall back to training config if specific config doesn't exist yet
                path = TRAINING_PIPELINE_CONFIG_PATH
        else:
            # Default to training pipeline for unknown types
            path = TRAINING_PIPELINE_CONFIG_PATH
    
    try:
        config = load_yaml_config(path)
        return config
    except Exception as e:
        # Log the error but don't crash
        import logging
        logging.getLogger(__name__).warning(f"Error loading {pipeline_type} pipeline config: {e}")
        return {}


def get_device_config(pipeline_type: str = 'training') -> Dict[str, Any]:
    """
    Get the device configuration settings from the pipeline configuration.
    
    Args:
        pipeline_type: Type of pipeline configuration to load ('training', 'ingestion', etc.)
        
    Returns:
        Dictionary with device configuration values
    """
    config = load_pipeline_config(pipeline_type=pipeline_type)
    
    # Check if GPU execution is enabled
    if 'gpu_execution' in config and config['gpu_execution'].get('enabled', False):
        return {
            'mode': 'gpu',
            'config': config['gpu_execution']
        }
    
    # Check if CPU execution is enabled
    if 'cpu_execution' in config and config['cpu_execution'].get('enabled', False):
        return {
            'mode': 'cpu',
            'config': config['cpu_execution']
        }
    
    # Default to GPU if both are enabled or neither is enabled
    if 'gpu_execution' in config:
        return {
            'mode': 'gpu',
            'config': config['gpu_execution']
        }
    
    # Fall back to empty config if nothing is specified
    return {
        'mode': 'auto',
        'config': {}
    }


def get_component_device(component_name: str, pipeline_type: str = 'training') -> Optional[str]:
    """
    Get the device configuration for a specific component.
    
    Args:
        component_name: Name of the component (e.g., 'docproc', 'chunking', 'embedding')
        pipeline_type: Type of pipeline configuration to load ('training', 'ingestion', etc.)
        
    Returns:
        Device string (e.g., 'cuda:1') or None if not configured
    """
    device_config = get_device_config(pipeline_type=pipeline_type)
    
    if device_config['mode'] == 'gpu':
        component_config = device_config['config'].get(component_name, {})
        return component_config.get('device')
    elif device_config['mode'] == 'cpu':
        component_config = device_config['config'].get(component_name, {})
        return component_config.get('device')
    
    return None
