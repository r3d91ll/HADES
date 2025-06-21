"""
Centralized Configuration Manager for HADES

This module provides a comprehensive configuration management system that supports:
- Hierarchical configuration loading (base -> environment -> overrides)
- Environment-specific configurations (development, testing, production)
- Runtime configuration updates
- Configuration validation
- Dot notation access to nested values
- Configuration merging and overrides
"""

import os
import yaml
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .config_loader import CONFIG_DIR, load_yaml_config, merge_configs

logger = logging.getLogger(__name__)


@dataclass
class ConfigurationSource:
    """Represents a configuration source with metadata."""
    name: str
    path: Path
    loaded_at: datetime
    priority: int = 0  # Higher priority overrides lower priority
    source_type: str = 'file'  # 'file', 'environment', 'override'


@dataclass  
class ConfigurationState:
    """Tracks the current state of configuration loading."""
    environment: str = 'development'
    sources: List[ConfigurationSource] = field(default_factory=list)
    last_loaded: Optional[datetime] = None
    validation_errors: List[str] = field(default_factory=list)
    config_data: Dict[str, Any] = field(default_factory=dict)


class ConfigurationManager:
    """
    Centralized configuration manager for HADES.
    
    Provides hierarchical configuration loading with environment-specific overrides,
    runtime updates, validation, and easy access to configuration values.
    
    Configuration Loading Order (higher priority overrides lower):
    1. Base configuration (base.yaml)
    2. Environment-specific configuration (environments/{env}.yaml)
    3. Component-specific configurations (components/**/config.yaml)
    4. Environment variables (prefixed with HADES_)
    5. Runtime overrides
    
    Usage:
        config_manager = ConfigurationManager(environment='production')
        
        # Access configuration values
        host = config_manager.get('api.server.host', default='localhost')
        
        # Get entire sections
        db_config = config_manager.get_section('database')
        
        # Runtime overrides
        config_manager.set_override('api.server.port', 9000)
        
        # Validate configuration
        if not config_manager.validate():
            logger.error("Configuration validation failed")
    """
    
    def __init__(
        self, 
        environment: Optional[str] = None,
        config_dir: Optional[Path] = None,
        auto_load: bool = True,
        watch_for_changes: bool = False,
        strict_validation: bool = False
    ):
        """
        Initialize the Configuration Manager.
        
        Args:
            environment: Environment name ('development', 'testing', 'production')
            config_dir: Custom configuration directory (defaults to src/config)
            auto_load: Whether to automatically load configuration on initialization
            watch_for_changes: Whether to watch for configuration file changes (future feature)
            strict_validation: Whether to raise exceptions on critical validation errors
        """
        self.config_dir = config_dir or CONFIG_DIR
        self.environment = environment or os.getenv('HADES_ENVIRONMENT') or 'development'
        self.watch_for_changes = watch_for_changes
        self.strict_validation = strict_validation
        
        # Configuration state
        self.state = ConfigurationState(environment=self.environment)
        self._overrides: Dict[str, Any] = {}
        self._cache: Dict[str, Any] = {}
        self._cache_valid = False
        
        # Validation rules (will be expanded)
        self._validation_rules: Dict[str, Any] = {}
        
        logger.info(f"Initialized ConfigurationManager for environment: {self.environment}")
        
        if auto_load:
            self.load_configuration()
    
    def load_configuration(self) -> None:
        """
        Load configuration from all sources in priority order.
        
        This method orchestrates the complete configuration loading process:
        1. Base configuration
        2. Environment-specific configuration  
        3. Component configurations
        4. Environment variables
        5. Apply any existing overrides
        """
        logger.info(f"Loading configuration for environment: {self.environment}")
        start_time = datetime.now(timezone.utc)
        
        # Clear existing state
        self.state.sources.clear()
        self.state.validation_errors.clear()
        self.state.config_data.clear()
        self._cache_valid = False
        
        try:
            # 1. Load base configuration
            self._load_base_configuration()
            
            # 2. Load environment-specific configuration
            self._load_environment_configuration()
            
            # 3. Load component configurations
            self._load_component_configurations()
            
            # 4. Apply environment variables
            self._apply_environment_variables()
            
            # 5. Apply runtime overrides
            self._apply_overrides()
            
            # 6. Validate configuration
            self.validate(strict=self.strict_validation)
            
            self.state.last_loaded = datetime.now(timezone.utc)
            loading_time = (self.state.last_loaded - start_time).total_seconds()
            
            logger.info(f"Configuration loaded successfully in {loading_time:.3f}s")
            logger.info(f"Loaded {len(self.state.sources)} configuration sources")
            
            if self.state.validation_errors:
                logger.warning(f"Configuration validation found {len(self.state.validation_errors)} issues")
                for error in self.state.validation_errors:
                    logger.warning(f"  - {error}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise RuntimeError(f"Configuration loading failed: {e}") from e
    
    def _load_base_configuration(self) -> None:
        """Load base configuration from base.yaml."""
        base_config_path = self.config_dir / 'base.yaml'
        
        if base_config_path.exists():
            try:
                base_config = load_yaml_config(base_config_path)
                self.state.config_data = merge_configs(self.state.config_data, base_config)
                
                source = ConfigurationSource(
                    name='base',
                    path=base_config_path,
                    loaded_at=datetime.now(timezone.utc),
                    priority=1,
                    source_type='file'
                )
                self.state.sources.append(source)
                
                logger.debug(f"Loaded base configuration from {base_config_path}")
                
            except Exception as e:
                logger.error(f"Failed to load base configuration: {e}")
        else:
            logger.info("No base configuration found (base.yaml), using component configs only")
    
    def _load_environment_configuration(self) -> None:
        """Load environment-specific configuration."""
        env_config_path = self.config_dir / 'environments' / f'{self.environment}.yaml'
        
        if env_config_path.exists():
            try:
                env_config = load_yaml_config(env_config_path)
                self.state.config_data = merge_configs(self.state.config_data, env_config)
                
                source = ConfigurationSource(
                    name=f'environment-{self.environment}',
                    path=env_config_path,
                    loaded_at=datetime.now(timezone.utc),
                    priority=2,
                    source_type='file'
                )
                self.state.sources.append(source)
                
                logger.debug(f"Loaded environment configuration for {self.environment}")
                
            except Exception as e:
                logger.error(f"Failed to load environment configuration: {e}")
        else:
            logger.info(f"No environment-specific configuration found for {self.environment}")
    
    def _load_component_configurations(self) -> None:
        """Load all component configurations."""
        components_dir = self.config_dir / 'components'
        
        if not components_dir.exists():
            logger.warning("Components configuration directory not found")
            return
        
        component_configs: Dict[str, Any] = {}
        
        # Walk through all component directories
        for component_path in components_dir.rglob('config.yaml'):
            try:
                # Get relative path from components directory
                rel_path = component_path.relative_to(components_dir)
                component_key = str(rel_path.parent).replace('/', '.')
                
                component_config = load_yaml_config(component_path)
                
                # Store in nested structure
                keys = component_key.split('.')
                current = component_configs
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[keys[-1]] = component_config
                
                source = ConfigurationSource(
                    name=f'component-{component_key}',
                    path=component_path,
                    loaded_at=datetime.now(timezone.utc),
                    priority=3,
                    source_type='file'
                )
                self.state.sources.append(source)
                
                logger.debug(f"Loaded component configuration: {component_key}")
                
            except Exception as e:
                logger.error(f"Failed to load component config {component_path}: {e}")
        
        # Merge component configurations
        if component_configs:
            if 'components' not in self.state.config_data:
                self.state.config_data['components'] = {}
            self.state.config_data['components'] = merge_configs(
                self.state.config_data['components'], 
                component_configs
            )
    
    def _apply_environment_variables(self) -> None:
        """Apply environment variables with HADES_ prefix using robust parsing."""
        env_overrides: Dict[str, Any] = {}
        
        # Collect all HADES environment variables
        hades_env_vars = {k: v for k, v in os.environ.items() if k.startswith('HADES_')}
        
        # Sort by length (longest first) to avoid naming conflicts
        # This ensures HADES_API_SERVER_HOST is processed before HADES_API_SERVER
        sorted_env_vars = sorted(hades_env_vars.items(), key=lambda x: len(x[0]), reverse=True)
        
        processed_keys: set[str] = set()
        
        for key, value in sorted_env_vars:
            # Remove HADES_ prefix and convert to lowercase
            config_key_parts = key[6:].lower().split('_')
            config_key = '.'.join(config_key_parts)
            
            # Check for conflicts with already processed keys
            if self._has_config_key_conflict(config_key, processed_keys):
                logger.warning(f"Skipping environment variable {key}: conflicts with existing configuration path")
                continue
            
            # Try to parse as number or boolean
            parsed_value = self._parse_env_value(value)
            
            # Set in nested structure
            self._set_nested_value(env_overrides, config_key, parsed_value)
            processed_keys.add(config_key)
            
            logger.debug(f"Applied environment variable: {key} -> {config_key} = {parsed_value}")
        
        if env_overrides:
            self.state.config_data = merge_configs(self.state.config_data, env_overrides)
            
            source = ConfigurationSource(
                name='environment-variables',
                path=Path('env://'),
                loaded_at=datetime.now(timezone.utc),
                priority=4,
                source_type='environment'
            )
            self.state.sources.append(source)
    
    def _apply_overrides(self) -> None:
        """Apply runtime configuration overrides."""
        if self._overrides:
            self.state.config_data = merge_configs(self.state.config_data, self._overrides)
            
            source = ConfigurationSource(
                name='runtime-overrides',
                path=Path('runtime://'),
                loaded_at=datetime.now(timezone.utc),
                priority=5,
                source_type='override'
            )
            self.state.sources.append(source)
            
            logger.debug(f"Applied {len(self._overrides)} runtime overrides")
    
    def _validate_configuration(self) -> None:
        """Validate the complete configuration."""
        # Clear previous validation errors
        self.state.validation_errors.clear()
        
        # Critical validation - ensure required sections exist
        required_sections = ['api', 'database']
        for section in required_sections:
            if section not in self.state.config_data:
                self.state.validation_errors.append(f"Missing required section: {section}")
        
        # Validate critical API configuration
        self._validate_api_config()
        
        # Validate critical database configuration  
        self._validate_database_config()
        
        # Validate component configurations have required fields
        if 'components' in self.state.config_data:
            components = self.state.config_data['components']
            for component_name, component_config in components.items():
                if isinstance(component_config, dict):
                    self._validate_component_config(component_name, component_config)
    
    def _validate_component_config(self, component_name: str, config: Dict[str, Any]) -> None:
        """Validate a specific component configuration."""
        # Check for version field in component configs
        def check_versions(cfg: Any, path: str = "") -> None:
            if isinstance(cfg, dict):
                for key, value in cfg.items():
                    current_path = f"{path}.{key}" if path else key
                    if isinstance(value, dict):
                        if 'version' not in value and any(k in value for k in ['enabled', 'config']):
                            self.state.validation_errors.append(
                                f"Component {component_name}{current_path} missing version field"
                            )
                        check_versions(value, current_path)
        
        check_versions(config)
    
    def _is_critical_error(self, error: str) -> bool:
        """Determine if a validation error is critical and should prevent startup."""
        critical_patterns = [
            "Missing required section",
            "Missing api.server.host",
            "Missing api.server.port", 
            "Missing database.arango.host",
            "Missing database.arango.port",
            "Invalid api.server.host",
            "Invalid api.server.port",
            "Invalid database.arango.host",
            "Invalid database.arango.port",
            "SECURITY:",
            "connection failed",
            "authentication failed"
        ]
        
        return any(pattern.lower() in error.lower() for pattern in critical_patterns)
    
    def _validate_api_config(self) -> None:
        """Validate critical API configuration."""
        if 'api' not in self.state.config_data:
            return
            
        api_config = self.state.config_data['api']
        
        # Validate server configuration
        if 'server' in api_config:
            server_config = api_config['server']
            
            # Check required server fields
            if 'host' not in server_config:
                self.state.validation_errors.append("Missing api.server.host configuration")
            elif not isinstance(server_config['host'], str) or not server_config['host'].strip():
                self.state.validation_errors.append("Invalid api.server.host: must be non-empty string")
                
            if 'port' not in server_config:
                self.state.validation_errors.append("Missing api.server.port configuration")
            elif not isinstance(server_config['port'], int) or not (1 <= server_config['port'] <= 65535):
                self.state.validation_errors.append("Invalid api.server.port: must be integer between 1-65535")
        else:
            self.state.validation_errors.append("Missing api.server configuration section")
    
    def _validate_database_config(self) -> None:
        """Validate critical database configuration."""
        if 'database' not in self.state.config_data:
            return
            
        db_config = self.state.config_data['database']
        
        # Validate ArangoDB configuration
        if 'arango' in db_config:
            arango_config = db_config['arango']
            
            # Check required database fields
            if 'host' not in arango_config:
                self.state.validation_errors.append("Missing database.arango.host configuration")
            elif not isinstance(arango_config['host'], str) or not arango_config['host'].strip():
                self.state.validation_errors.append("Invalid database.arango.host: must be non-empty string")
                
            if 'port' not in arango_config:
                self.state.validation_errors.append("Missing database.arango.port configuration")
            elif not isinstance(arango_config['port'], int) or not (1 <= arango_config['port'] <= 65535):
                self.state.validation_errors.append("Invalid database.arango.port: must be integer between 1-65535")
                
            if 'username' not in arango_config:
                self.state.validation_errors.append("Missing database.arango.username configuration")
            elif not isinstance(arango_config['username'], str):
                self.state.validation_errors.append("Invalid database.arango.username: must be string")
                
            # Security validation for production environment
            if self.environment == 'production':
                if 'password' in arango_config and arango_config['password'] == "":
                    self.state.validation_errors.append("SECURITY: Empty database password not allowed in production environment")
                if 'use_ssl' in arango_config and not arango_config['use_ssl']:
                    self.state.validation_errors.append("SECURITY: SSL should be enabled in production environment")
        else:
            self.state.validation_errors.append("Missing database.arango configuration section")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'api.server.host')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Examples:
            host = config.get('api.server.host', 'localhost')
            batch_size = config.get('components.embedding.core.batch_size', 32)
        """
        return self._get_nested_value(self.state.config_data, key, default)
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Args:
            section: Section name (e.g., 'database', 'api')
            
        Returns:
            Dictionary containing the section configuration
        """
        result = self.get(section, {})
        return result if isinstance(result, dict) else {}
    
    def set_override(self, key: str, value: Any) -> None:
        """
        Set runtime configuration override.
        
        Args:
            key: Configuration key in dot notation
            value: Value to set
        """
        self._set_nested_value(self._overrides, key, value)
        self._cache_valid = False
        
        logger.debug(f"Set configuration override: {key} = {value}")
    
    def remove_override(self, key: str) -> bool:
        """
        Remove runtime configuration override.
        
        Args:
            key: Configuration key to remove
            
        Returns:
            True if override was removed, False if it didn't exist
        """
        if self._remove_nested_value(self._overrides, key):
            self._cache_valid = False
            logger.debug(f"Removed configuration override: {key}")
            return True
        return False
    
    def clear_overrides(self) -> None:
        """Clear all runtime configuration overrides."""
        self._overrides.clear()
        self._cache_valid = False
        logger.debug("Cleared all configuration overrides")
    
    def reload(self) -> None:
        """Reload configuration from all sources."""
        logger.info("Reloading configuration")
        self.load_configuration()
    
    def validate(self, strict: bool = False) -> bool:
        """
        Validate current configuration.
        
        Args:
            strict: If True, raise exception on validation errors
            
        Returns:
            True if configuration is valid, False otherwise
            
        Raises:
            ValueError: If strict=True and validation errors exist
        """
        self._validate_configuration()
        is_valid = len(self.state.validation_errors) == 0
        
        if not is_valid and strict:
            critical_errors = [err for err in self.state.validation_errors if self._is_critical_error(err)]
            if critical_errors:
                error_msg = f"Critical configuration validation failed:\n" + "\n".join(f"  - {err}" for err in critical_errors)
                raise ValueError(error_msg)
        
        return is_valid
    
    def get_validation_errors(self) -> List[str]:
        """Get list of configuration validation errors."""
        return self.state.validation_errors.copy()
    
    def export_config(self, format: str = 'yaml') -> str:
        """
        Export current configuration in specified format.
        
        Args:
            format: Export format ('yaml', 'json')
            
        Returns:
            Configuration as string in requested format
        """
        if format.lower() == 'yaml':
            return yaml.dump(self.state.config_data, default_flow_style=False, sort_keys=True)
        elif format.lower() == 'json':
            import json
            return json.dumps(self.state.config_data, indent=2, sort_keys=True)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_config_info(self) -> Dict[str, Any]:
        """
        Get information about the current configuration state.
        
        Returns:
            Dictionary containing configuration metadata
        """
        return {
            'environment': self.environment,
            'last_loaded': self.state.last_loaded.isoformat() if self.state.last_loaded else None,
            'sources_count': len(self.state.sources),
            'sources': [
                {
                    'name': source.name,
                    'path': str(source.path),
                    'priority': source.priority,
                    'type': source.source_type,
                    'loaded_at': source.loaded_at.isoformat()
                }
                for source in self.state.sources
            ],
            'overrides_count': len(self._overrides),
            'validation_errors_count': len(self.state.validation_errors),
            'validation_errors': self.state.validation_errors
        }
    
    def _has_config_key_conflict(self, config_key: str, processed_keys: set) -> bool:
        """Check if a config key would conflict with already processed keys."""
        # Check if this key is a prefix of any processed key
        # or if any processed key is a prefix of this key
        for processed_key in processed_keys:
            if (config_key.startswith(processed_key + '.') or 
                processed_key.startswith(config_key + '.') or
                config_key == processed_key):
                return True
        return False
    
    # Helper methods
    
    def _get_nested_value(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Get value from nested dictionary using dot notation."""
        keys = key.split('.')
        current = data
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        
        return current
    
    def _set_nested_value(self, data: Dict[str, Any], key: str, value: Any) -> None:
        """Set value in nested dictionary using dot notation."""
        keys = key.split('.')
        current = data
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def _remove_nested_value(self, data: Dict[str, Any], key: str) -> bool:
        """Remove value from nested dictionary using dot notation."""
        keys = key.split('.')
        current = data
        
        # Navigate to parent
        for k in keys[:-1]:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return False  # Path doesn't exist
        
        # Remove the final key
        if isinstance(current, dict) and keys[-1] in current:
            del current[keys[-1]]
            return True
        
        return False
    
    def _parse_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Parse environment variable value to appropriate type."""
        # Boolean values
        if value.lower() in ('true', 'yes', '1', 'on'):
            return True
        elif value.lower() in ('false', 'no', '0', 'off'):
            return False
        
        # Numeric values
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # String value
        return value


# Global configuration manager instance
_config_manager: Optional[ConfigurationManager] = None
_config_lock = threading.Lock()


def get_config_manager(environment: Optional[str] = None, reload: bool = False) -> ConfigurationManager:
    """
    Get the global configuration manager instance.
    
    Args:
        environment: Environment name (only used if creating new instance)
        reload: Whether to reload configuration if manager already exists
        
    Returns:
        ConfigurationManager instance
    """
    global _config_manager
    
    if _config_manager is None:
        with _config_lock:
            # Double-check locking pattern to ensure thread safety
            if _config_manager is None:
                _config_manager = ConfigurationManager(environment=environment)
    elif reload:
        with _config_lock:
            _config_manager.reload()
    
    return _config_manager


def get_config(key: str, default: Any = None) -> Any:
    """
    Convenience function to get configuration value.
    
    Args:
        key: Configuration key in dot notation
        default: Default value if key not found
        
    Returns:
        Configuration value
    """
    return get_config_manager().get(key, default)


def set_config_override(key: str, value: Any) -> None:
    """
    Convenience function to set configuration override.
    
    Args:
        key: Configuration key in dot notation
        value: Value to set
    """
    get_config_manager().set_override(key, value)