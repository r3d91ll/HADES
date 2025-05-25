"""
Configuration loader for document processing.

This module loads and provides access to document processing configuration
from the central configuration system.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, cast, TypeVar, Union

from src.config.preprocessor_config import load_config
from src.types.common import PreProcessorConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global configuration cache
_cached_config: Optional[PreProcessorConfig] = None


def get_config(reload: bool = False) -> PreProcessorConfig:
    """
    Get preprocessor configuration, using cached version if available.
    
    Args:
        reload: Force reload configuration from disk
        
    Returns:
        Preprocessor configuration dictionary
    """
    global _cached_config
    
    if _cached_config is None or reload:
        try:
            logger.info("Loading document processor configuration")
            _cached_config = load_config()
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            # In case of error, create a minimal valid config to avoid crashes
            from src.types.common import PreProcessorConfig
            # Create a minimal default configuration
            _cached_config = PreProcessorConfig(
                input_dir=Path("."),
                output_dir=Path("./.symbol_table"),
                exclude_patterns=["__pycache__", ".git"],
                recursive=True,
                max_workers=4,
                file_type_map={},
                preprocessor_config={},
                options={},
            )
            logger.warning("Using minimal default configuration due to loading error")
            
    return _cached_config


def get_file_type_map() -> Dict[str, List[str]]:
    """
    Get the file type map from configuration.
    
    Returns:
        Dictionary mapping document types to file extensions
    """
    config = get_config()
    return config["file_type_map"]


def get_extension_to_format_map() -> Dict[str, str]:
    """
    Get a mapping from file extensions to format types.
    
    This inverts the file_type_map from the configuration for
    easier lookup of format types by extension.
    
    Returns:
        Dictionary mapping extensions to format types
    """
    file_type_map = get_file_type_map()
    extension_map: Dict[str, str] = {}
    
    for format_type, extensions in file_type_map.items():
        for ext in extensions:
            extension_map[ext] = format_type
            
    return extension_map


def get_format_config(format_type: str) -> Dict[str, Any]:
    """
    Get format-specific configuration.
    
    Args:
        format_type: Document format type (e.g., "python", "markdown")
        
    Returns:
        Configuration dictionary for the specified format
    """
    config = get_config()
    preprocessor_config = config["preprocessor_config"]
    
    # Return empty dict if format not found to avoid KeyError
    return preprocessor_config.get(format_type, {})


def get_option(format_type: str, option_name: str, default: Any = None) -> Any:
    """
    Get a specific configuration option for a format.
    
    Args:
        format_type: Document format type
        option_name: Name of the option to retrieve
        default: Default value if option not found
        
    Returns:
        Option value or default if not found
    """
    format_config = get_format_config(format_type)
    return format_config.get(option_name, default)

