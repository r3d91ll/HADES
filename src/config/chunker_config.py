"""Chunker configuration loader for HADES-PathRAG (YAML-based).

Loads and validates configuration for content chunkers including:
- AST-based code chunking
- Chonky (semantic text chunking)
"""
import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union, List, TypedDict, cast

# TypedDict for the chunker configuration
class ChunkerConfig(TypedDict, total=False):
    """Type-safe configuration for content chunkers."""
    version: int
    chunker_mapping: Dict[str, str]
    ast: Dict[str, Any]
    chonky: Dict[str, Any]

# Defaults (should match those in chunker_config.yaml)
DEFAULT_CHUNKER_MAPPING: Dict[str, str] = {
    'python': 'ast',
    'javascript': 'ast',
    'java': 'ast',
    'cpp': 'ast',
    'markdown': 'chonky',
    'text': 'chonky',
    'html': 'chonky',
    'pdf': 'chonky',
    'default': 'chonky',
}

DEFAULT_AST_CONFIG: Dict[str, Any] = {
    'max_tokens': 2048,
    'use_class_boundaries': True,
    'use_function_boundaries': True,
    'extract_imports': True,
    'preserve_docstrings': True,
}

DEFAULT_CHONKY_CONFIG: Dict[str, Any] = {
    'max_tokens': 2048,
    'min_tokens': 64,
    'overlap_tokens': 200,
    'semantic_chunking': True,
    'preserve_structure': True,
    'batch_size': 8,
    
    # Overlap context settings
    'overlap_context': {
        'enabled': True,
        'store_pre_context': True,
        'store_post_context': True,
        'max_pre_context_chars': 1000,
        'max_post_context_chars': 1000,
        'store_position_info': True,
    },
    
    # Cache settings
    'cache_with_device': True,
    'cache_size': 4,
    
    # Model engine settings
    'early_availability_check': True,
    'auto_start_engine': True,
    'max_startup_retries': 3,
}

DEFAULTS: Dict[str, Any] = {
    'version': 1,
    'chunker_mapping': DEFAULT_CHUNKER_MAPPING,
    'ast': DEFAULT_AST_CONFIG,
    'chonky': DEFAULT_CHONKY_CONFIG,
}

CONFIG_PATH = Path(__file__).parent / 'chunker_config.yaml'


def load_config(config_path: Optional[Union[str, Path]] = None) -> ChunkerConfig:
    """
    Load chunker configuration from YAML file, merging with defaults.
    
    Args:
        config_path: Path to configuration YAML file (if None, uses default)
        
    Returns:
        Merged configuration dictionary with proper types
    """
    path = Path(config_path) if config_path else CONFIG_PATH
    
    # Start with defaults
    version = DEFAULTS['version']
    chunker_mapping = dict(DEFAULT_CHUNKER_MAPPING)
    ast_config = dict(DEFAULT_AST_CONFIG)
    chonky_config = dict(DEFAULT_CHONKY_CONFIG)
    
    # Load from YAML if it exists
    if path.exists():
        try:
            with open(path, 'r') as f:
                user_config = yaml.safe_load(f) or {}
                
            # Update version if specified
            if 'version' in user_config:
                version = int(user_config['version'])
                
            # Update chunker mapping if specified
            if 'chunker_mapping' in user_config and isinstance(user_config['chunker_mapping'], dict):
                for key, value in user_config['chunker_mapping'].items():
                    if isinstance(value, str):
                        chunker_mapping[key] = value
            
            # Update AST config if specified
            if 'ast' in user_config and isinstance(user_config['ast'], dict):
                for key, value in user_config['ast'].items():
                    ast_config[key] = value
            
            # Update Chonky config if specified
            if 'chonky' in user_config and isinstance(user_config['chonky'], dict):
                for key, value in user_config['chonky'].items():
                    chonky_config[key] = value
                
        except Exception as e:
            print(f"Error loading chunker config from {path}: {e}")
    else:
        print(f"Chunker config file not found at {path}, using defaults")
    
    # Build type-safe config
    result: ChunkerConfig = {
        'version': version,
        'chunker_mapping': chunker_mapping,
        'ast': ast_config,
        'chonky': chonky_config,
    }
    
    return result


def get_chunker_for_language(language: str, config: Optional[ChunkerConfig] = None) -> str:
    """
    Get the appropriate chunker type for a given language.
    
    Args:
        language: Language identifier (e.g., 'python', 'markdown')
        config: Optional chunker configuration
        
    Returns:
        Chunker type ('ast' or 'chonky')
    """
    if config is None:
        config = load_config()
    
    language_lower = language.lower()
    chunker_mapping = config.get('chunker_mapping', DEFAULT_CHUNKER_MAPPING)
    
    # Check if language exists in mapping
    if language_lower in chunker_mapping:
        return chunker_mapping[language_lower]
    
    # Fall back to default
    return chunker_mapping.get('default', 'chonky')


def get_chunker_config(chunker_type: str, config: Optional[ChunkerConfig] = None) -> Dict[str, Any]:
    """
    Get configuration for a specific chunker type.
    
    Args:
        chunker_type: Type of chunker ('ast' or 'chonky')
        config: Optional chunker configuration
        
    Returns:
        Chunker-specific configuration dictionary
    """
    if config is None:
        config = load_config()
    
    if chunker_type == 'ast':
        return config.get('ast', DEFAULT_AST_CONFIG)
    elif chunker_type == 'chonky':
        return config.get('chonky', DEFAULT_CHONKY_CONFIG)
    else:
        # Return empty dict for unknown chunker types
        return {}
