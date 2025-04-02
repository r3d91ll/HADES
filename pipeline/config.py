"""
Configuration system for the HADES modular pipeline architecture.

This module provides utilities for loading, validating, and accessing
configuration settings for the HADES system.
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Set


class ConfigurationError(Exception):
    """Exception raised for configuration errors."""
    pass


class Configuration:
    """Configuration manager for the HADES system."""
    
    def __init__(self):
        """Initialize the configuration manager."""
        self._config: Dict[str, Any] = {}
        self._logger = logging.getLogger(__name__)
    
    def load_file(self, file_path: Union[str, Path]) -> None:
        """Load configuration from a file.
        
        Args:
            file_path: Path to the configuration file (YAML or JSON)
            
        Raises:
            ConfigurationError: If the file cannot be loaded
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise ConfigurationError(f"Configuration file not found: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                if file_path.suffix.lower() in ('.yaml', '.yml'):
                    config = yaml.safe_load(f)
                elif file_path.suffix.lower() == '.json':
                    config = json.load(f)
                else:
                    raise ConfigurationError(f"Unsupported configuration file format: {file_path.suffix}")
            
            self.update(config)
            self._logger.info(f"Loaded configuration from {file_path}")
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration file {file_path}: {e}")
    
    def load_directory(self, directory_path: Union[str, Path], recursive: bool = False) -> None:
        """Load all configuration files from a directory.
        
        Args:
            directory_path: Path to the directory containing configuration files
            recursive: Whether to search subdirectories recursively
            
        Raises:
            ConfigurationError: If the directory cannot be loaded
        """
        directory_path = Path(directory_path)
        if not directory_path.is_dir():
            raise ConfigurationError(f"Configuration directory not found: {directory_path}")
        
        # Get all YAML and JSON files in the directory
        pattern = '**/*.y*ml' if recursive else '*.y*ml'
        yaml_files = list(directory_path.glob(pattern))
        
        pattern = '**/*.json' if recursive else '*.json'
        json_files = list(directory_path.glob(pattern))
        
        # Sort files to ensure consistent loading order
        config_files = sorted(yaml_files + json_files)
        
        for file_path in config_files:
            try:
                self.load_file(file_path)
            except ConfigurationError as e:
                self._logger.warning(str(e))
    
    def update(self, config: Dict[str, Any]) -> None:
        """Update the configuration with new values.
        
        Args:
            config: Dictionary containing configuration values to update
        """
        self._deep_update(self._config, config)
    
    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Recursively update a dictionary with values from another dictionary.
        
        Args:
            target: Dictionary to update
            source: Dictionary containing values to update with
        """
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get a configuration value by its key path.
        
        Args:
            key_path: Dot-separated path to the configuration value
            default: Default value to return if the key is not found
            
        Returns:
            Configuration value or default if not found
        """
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """Set a configuration value by its key path.
        
        Args:
            key_path: Dot-separated path to the configuration value
            value: Value to set
        """
        keys = key_path.split('.')
        target = self._config
        
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            elif not isinstance(target[key], dict):
                target[key] = {}
            
            target = target[key]
        
        target[keys[-1]] = value
    
    def load_environment_variables(self, prefix: str = 'HADES_') -> None:
        """Load configuration from environment variables.
        
        Environment variables should be in the format PREFIX_KEY_SUBKEY.
        For example, HADES_STORAGE_REDIS_HOST will set config['storage']['redis']['host'].
        
        Args:
            prefix: Prefix for environment variables to load
        """
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and split by underscore
                config_path = key[len(prefix):].lower().replace('_', '.')
                
                # Try to parse value as JSON, fall back to string
                try:
                    parsed_value = json.loads(value)
                except json.JSONDecodeError:
                    parsed_value = value
                
                self.set(config_path, parsed_value)
                self._logger.debug(f"Loaded config from environment: {config_path}")
    
    def as_dict(self) -> Dict[str, Any]:
        """Return the entire configuration as a dictionary.
        
        Returns:
            Dictionary containing all configuration values
        """
        return self._config.copy()
    
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save the current configuration to a file.
        
        Args:
            file_path: Path to save the configuration to (YAML or JSON)
            
        Raises:
            ConfigurationError: If the file cannot be saved
        """
        file_path = Path(file_path)
        
        try:
            with open(file_path, 'w') as f:
                if file_path.suffix.lower() in ('.yaml', '.yml'):
                    yaml.dump(self._config, f, default_flow_style=False)
                elif file_path.suffix.lower() == '.json':
                    json.dump(self._config, f, indent=2)
                else:
                    raise ConfigurationError(f"Unsupported configuration file format: {file_path.suffix}")
            
            self._logger.info(f"Saved configuration to {file_path}")
        except Exception as e:
            raise ConfigurationError(f"Error saving configuration to {file_path}: {e}")
    
    def validate_required_keys(self, required_keys: List[str]) -> Tuple[bool, List[str]]:
        """Validate that all required keys are present in the configuration.
        
        Args:
            required_keys: List of dot-separated key paths that must be present
            
        Returns:
            Tuple of (is_valid, missing_keys)
        """
        missing_keys = []
        
        for key_path in required_keys:
            if self.get(key_path) is None:
                missing_keys.append(key_path)
        
        return len(missing_keys) == 0, missing_keys


# Singleton instance
_config = Configuration()


def get_config() -> Configuration:
    """Get the global configuration instance.
    
    Returns:
        Global configuration instance
    """
    return _config


def load_config(config_path: Union[str, Path, List[Union[str, Path]]]) -> None:
    """Load configuration from a file or directory into the global configuration.
    
    Args:
        config_path: Path to a configuration file, directory, or list of paths
    """
    if isinstance(config_path, list):
        for path in config_path:
            load_config(path)
        return
    
    config_path = Path(config_path)
    
    if config_path.is_dir():
        _config.load_directory(config_path, recursive=True)
    else:
        _config.load_file(config_path)


def get(key_path: str, default: Any = None) -> Any:
    """Get a configuration value from the global configuration.
    
    Args:
        key_path: Dot-separated path to the configuration value
        default: Default value to return if the key is not found
        
    Returns:
        Configuration value or default if not found
    """
    return _config.get(key_path, default)


def set(key_path: str, value: Any) -> None:
    """Set a configuration value in the global configuration.
    
    Args:
        key_path: Dot-separated path to the configuration value
        value: Value to set
    """
    _config.set(key_path, value)