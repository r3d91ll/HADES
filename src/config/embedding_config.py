"""Embedding configuration loader for HADES-PathRAG (YAML-based).

Loads and validates configuration for embedding models including:
- ModernBERT (default for academic texts)
- CPU-based lightweight models
- Other future embedding models

This allows flexible switching between CPU and GPU implementations through configuration.
"""
import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union, List, TypedDict, cast

# TypedDict for the embedding configuration
class EmbeddingConfig(TypedDict, total=False):
    """Type-safe configuration for embedding models."""
    version: int
    default_adapter: str
    adapters: Dict[str, Dict[str, Any]]
    cpu: Dict[str, Any]
    modernbert: Dict[str, Any]

# Default configuration settings
DEFAULT_CPU_CONFIG: Dict[str, Any] = {
    'model_name': 'all-MiniLM-L6-v2',
    'max_length': 512,
    'pooling_strategy': 'mean',
    'normalize_embeddings': True,
    'batch_size': 32,
}

DEFAULT_MODERNBERT_CONFIG: Dict[str, Any] = {
    'model_name': 'answerdotai/ModernBERT-base',
    'max_length': 8192,
    'pooling_strategy': 'cls',
    'normalize_embeddings': True,
    'batch_size': 8,
    'device': 'cpu',  # Can be 'cpu' or 'cuda:0', etc.
    
    # Model engine settings
    'use_model_engine': True,
    'engine_type': 'haystack',
    'early_availability_check': True,
    'auto_start_engine': True,
    'max_startup_retries': 3,
}

# Default adapter mapping
DEFAULT_ADAPTERS: Dict[str, Dict[str, Any]] = {
    'cpu': {'type': 'cpu', 'config': DEFAULT_CPU_CONFIG},
    'modernbert': {'type': 'modernbert', 'config': DEFAULT_MODERNBERT_CONFIG}
}

DEFAULTS: Dict[str, Any] = {
    'version': 1,
    'default_adapter': 'modernbert',  # Make ModernBERT the default
    'adapters': DEFAULT_ADAPTERS,
    'cpu': DEFAULT_CPU_CONFIG,
    'modernbert': DEFAULT_MODERNBERT_CONFIG,
}

CONFIG_PATH = Path(__file__).parent / 'embedding_config.yaml'


def load_config(config_path: Optional[Union[str, Path]] = None) -> EmbeddingConfig:
    """
    Load embedding configuration from YAML file, merging with defaults.
    
    Args:
        config_path: Path to configuration YAML file (if None, uses default)
        
    Returns:
        Merged configuration dictionary with proper types
    """
    path = Path(config_path) if config_path else CONFIG_PATH
    
    # Start with defaults
    version = DEFAULTS['version']
    default_adapter = DEFAULTS['default_adapter']
    adapters = dict(DEFAULT_ADAPTERS)
    cpu_config = dict(DEFAULT_CPU_CONFIG)
    modernbert_config = dict(DEFAULT_MODERNBERT_CONFIG)
    
    # Load from YAML if it exists
    if path.exists():
        try:
            with open(path, 'r') as f:
                user_config = yaml.safe_load(f) or {}
                
            # Update version if specified
            if 'version' in user_config:
                version = user_config['version']
                
            # Update default adapter if specified
            if 'default_adapter' in user_config:
                default_adapter = user_config['default_adapter']
            
            # Update CPU config if specified
            if 'cpu' in user_config:
                cpu_config.update(user_config['cpu'])
                
            # Update ModernBERT config if specified
            if 'modernbert' in user_config:
                modernbert_config.update(user_config['modernbert'])
                
            # Update adapter mapping
            if 'adapters' in user_config:
                for adapter_name, adapter_config in user_config['adapters'].items():
                    adapters[adapter_name] = adapter_config
                    
        except Exception as e:
            print(f"Error loading embedding config from {path}: {e}")
            # Fall back to defaults
    
    # Construct the final config dictionary
    config: EmbeddingConfig = {
        'version': version,
        'default_adapter': default_adapter,
        'adapters': adapters,
        'cpu': cpu_config,
        'modernbert': modernbert_config,
    }
    
    return config


def get_adapter_config(adapter_name: Optional[str] = None, config: Optional[EmbeddingConfig] = None) -> Dict[str, Any]:
    """
    Get configuration for a specific adapter.
    
    Args:
        adapter_name: Name of the adapter (if None, uses default)
        config: Configuration to use (if None, loads from default path)
        
    Returns:
        Configuration for the specified adapter
    """
    if config is None:
        config = load_config()
        
    if adapter_name is None:
        adapter_name = config['default_adapter']
        
    # Try to get adapter from the adapters mapping
    if adapter_name in config['adapters']:
        adapter_type = config['adapters'][adapter_name]['type']
        adapter_config = config['adapters'][adapter_name].get('config', {})
        
        # Get the base config for the adapter type
        base_config = config.get(adapter_type, {})
        
        # Merge with adapter-specific config
        merged_config = dict(base_config)
        merged_config.update(adapter_config)
        
        return merged_config
    
    # Fall back to the named config directly
    if adapter_name in config:
        return dict(config[adapter_name])
        
    # Fall back to default adapter
    return get_adapter_config(config['default_adapter'], config)
