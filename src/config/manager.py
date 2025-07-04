"""
Configuration manager for HADES.

This module provides centralized configuration management with
environment variable overrides and validation.
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration loading and access."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize configuration manager."""
        self.config_dir = config_dir or Path(__file__).parent
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._loaded = False
    
    def load_config(self, config_name: str, config_path: Optional[Path] = None) -> Dict[str, Any]:
        """Load a configuration file."""
        if config_path is None:
            config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return {}
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        
        # Apply environment variable overrides
        config = self._apply_env_overrides(config, prefix=f"HADES_{config_name.upper()}")
        
        self._configs[config_name] = config
        return config
    
    def get_config(self, config_name: str) -> Dict[str, Any]:
        """Get a loaded configuration."""
        if config_name not in self._configs:
            self.load_config(config_name)
        return self._configs.get(config_name, {})
    
    def _apply_env_overrides(self, config: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        """Apply environment variable overrides to config."""
        for key, value in config.items():
            env_var = f"{prefix}_{key.upper()}"
            if env_var in os.environ:
                env_value = os.environ[env_var]
                # Try to parse as appropriate type
                if isinstance(value, bool):
                    config[key] = env_value.lower() in ('true', 'yes', '1')
                elif isinstance(value, int):
                    try:
                        config[key] = int(env_value)
                    except ValueError:
                        logger.warning(f"Invalid int value for {env_var}: {env_value}")
                elif isinstance(value, float):
                    try:
                        config[key] = float(env_value)
                    except ValueError:
                        logger.warning(f"Invalid float value for {env_var}: {env_value}")
                else:
                    config[key] = env_value
        
        return config
    
    def load_all(self) -> None:
        """Load all configuration files in config directory."""
        for config_file in self.config_dir.glob("*.yaml"):
            config_name = config_file.stem
            self.load_config(config_name, config_file)
        self._loaded = True
    
    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """Get all loaded configurations."""
        if not self._loaded:
            self.load_all()
        return self._configs.copy()


# Global config manager instance
_config_manager = ConfigManager()


def get_config_manager() -> ConfigManager:
    """Get the global config manager instance."""
    return _config_manager


def load_config(config_name: str, config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load a configuration file."""
    return _config_manager.load_config(config_name, config_path)


def get_config(config_name: str) -> Dict[str, Any]:
    """Get a configuration."""
    return _config_manager.get_config(config_name)


__all__ = [
    "ConfigManager",
    "load_config",
    "get_config"
]