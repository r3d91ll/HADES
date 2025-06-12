"""
Configuration loading utilities for HADES.

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
TRAINING_PIPELINE_CONFIG_PATH = CONFIG_DIR / 'pipelines' / 'training' / 'legacy_config.yaml'
PIPELINE_CONFIG_PATH = TRAINING_PIPELINE_CONFIG_PATH  # For backward compatibility

# Fallback to old location if new structure doesn't exist yet
if not TRAINING_PIPELINE_CONFIG_PATH.exists():
    TRAINING_PIPELINE_CONFIG_PATH = CONFIG_DIR / 'training_pipeline_config.yaml'

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


def load_config(config_name: str) -> Dict[str, Any]:
    """
    Load a configuration file by name.
    
    Args:
        config_name: Name of the config file (without .yaml extension)
                    Can be a simple name like 'chunker_config' for backward compatibility,
                    or a component path like 'chunking/core/config' for new structure
        
    Returns:
        Dictionary containing configuration data
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file has invalid YAML
    """
    # Try new component-based structure first
    if '/' in config_name:
        # Component path format: 'chunking/core/config'
        config_file = CONFIG_DIR / f"{config_name}.yaml"
    else:
        # Legacy format: try both old and new locations
        legacy_file = CONFIG_DIR / f"{config_name}.yaml"
        
        # Map legacy names to new component paths
        legacy_mapping = {
            'chunker_config': 'components/chunking/core/config',
            'embedding_config': 'components/embedding/core/config', 
            'vllm_config': 'components/model_engine/vllm/config',
            'model_config': 'components/model_engine/core/config',
            'engine_config': 'components/model_engine/core/config',
            'database_config': 'components/database/arango/config',
            'isne_config': 'components/isne/core/config',
            'preprocessor_config': 'components/docproc/core/config'
        }
        
        if legacy_file.exists():
            # Use legacy file if it exists
            config_file = legacy_file
        elif config_name in legacy_mapping:
            # Try new component structure
            config_file = CONFIG_DIR / f"{legacy_mapping[config_name]}.yaml"
        else:
            # Default to legacy location
            config_file = legacy_file
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    return load_yaml_config(config_file)


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
        device_value = component_config.get('device')
        # Explicit cast to help mypy understand the return type
        return str(device_value) if device_value is not None else None
    elif device_config['mode'] == 'cpu':
        component_config = device_config['config'].get(component_name, {})
        device_value = component_config.get('device')
        # Explicit cast to help mypy understand the return type
        return str(device_value) if device_value is not None else None
    
    return None


def load_component_config(component: str, subcomponent: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration for a specific component.
    
    Args:
        component: Component name (e.g., 'chunking', 'embedding', 'isne')
        subcomponent: Optional subcomponent (e.g., 'core', 'text', 'vllm')
        
    Returns:
        Dictionary with component configuration
        
    Raises:
        FileNotFoundError: If component config doesn't exist
    """
    if subcomponent:
        config_path = f"{component}/{subcomponent}/config"
    else:
        config_path = f"{component}/core/config"
        
    return load_config(config_path)


def get_available_components() -> Dict[str, list]:
    """
    Get list of available components and their subcomponents.
    
    Returns:
        Dictionary mapping component names to lists of subcomponents
    """
    components = {}
    config_root = CONFIG_DIR
    
    for component_dir in config_root.iterdir():
        if component_dir.is_dir() and not component_dir.name.startswith('.'):
            component_name = component_dir.name
            subcomponents = []
            
            for sub_dir in component_dir.iterdir():
                if sub_dir.is_dir() and (sub_dir / 'config.yaml').exists():
                    subcomponents.append(sub_dir.name)
                    
            if subcomponents:
                components[component_name] = subcomponents
                
    return components


def validate_component_config(component: str, subcomponent: Optional[str] = None) -> bool:
    """
    Validate that a component configuration exists and is valid.
    
    Args:
        component: Component name
        subcomponent: Optional subcomponent name
        
    Returns:
        True if config is valid, False otherwise
    """
    try:
        config = load_component_config(component, subcomponent)
        return 'version' in config  # Basic validation
    except (FileNotFoundError, yaml.YAMLError):
        return False


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    Later configs override earlier ones.
    
    Args:
        *configs: Configuration dictionaries to merge
        
    Returns:
        Merged configuration dictionary
    """
    merged: Dict[str, Any] = {}
    
    for config in configs:
        if isinstance(config, dict):
            for key, value in config.items():
                if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                    merged[key] = merge_configs(merged[key], value)
                else:
                    merged[key] = value
                    
    return merged


def load_pipeline_config_by_type(pipeline_type: str = 'training') -> Dict[str, Any]:
    """
    Load pipeline configuration by type using new component structure.
    
    Args:
        pipeline_type: Type of pipeline ('training', 'data_ingestion', 'bootstrap')
        
    Returns:
        Dictionary with pipeline configuration
    """
    try:
        # Try new component structure first
        return load_config(f"pipelines/{pipeline_type}/config")
    except FileNotFoundError:
        # Fall back to legacy pipeline loading
        return load_pipeline_config(pipeline_type=pipeline_type)


def get_component_config_path(component: str, subcomponent: Optional[str] = None) -> Path:
    """
    Get the path to a component configuration file.
    
    Args:
        component: Component name
        subcomponent: Optional subcomponent name
        
    Returns:
        Path to the configuration file
    """
    if subcomponent:
        return CONFIG_DIR / component / subcomponent / 'config.yaml'
    else:
        return CONFIG_DIR / component / 'core' / 'config.yaml'


def list_component_configs() -> Dict[str, Dict[str, Path]]:
    """
    List all available component configurations.
    
    Returns:
        Nested dictionary of component -> subcomponent -> config_path
    """
    configs: Dict[str, Dict[str, Path]] = {}
    
    for component_dir in CONFIG_DIR.iterdir():
        if component_dir.is_dir() and not component_dir.name.startswith('.'):
            component_name = component_dir.name
            configs[component_name] = {}
            
            for sub_dir in component_dir.iterdir():
                if sub_dir.is_dir():
                    config_file = sub_dir / 'config.yaml'
                    if config_file.exists():
                        configs[component_name][sub_dir.name] = config_file
                        
    return configs
