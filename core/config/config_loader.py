"""
Configuration Loader with Schema Validation
===========================================

Theory Connection - Information Reconstructionism:
The configuration loading system implements hierarchical WHERE positioning
through source priority (environment > local > base). Schema validation
ensures shared-context (`C_ext`) coherence by preventing configuration drift that would
degrade the exponential amplification factor (`C_ext^α`).

From Conveyance Framework: Loading efficiency represents the TIME dimension,
while validation quality represents WHAT. The loader acts as a translation
mechanism in Actor-Network Theory, converting external representations
into internal semantic structures.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import yaml

from .config_base import BaseConfig, ConfigError, ConfigValidationError

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseConfig)


class ConfigFormat(Enum):
    """Supported configuration formats."""
    YAML = "yaml"
    JSON = "json"


@dataclass
class ConfigSource:
    """
    Configuration source metadata.

    Theory Connection: Represents positioning in the WHERE dimension,
    with priority determining hierarchical influence on the resulting `C_ext`.
    """
    path: Path
    format: ConfigFormat
    priority: int  # Higher numbers = higher priority
    exists: bool = field(default=False, init=False)
    readable: bool = field(default=False, init=False)

    def __post_init__(self):
        """Validate source accessibility."""
        self.exists = self.path.exists()
        if self.exists:
            try:
                # Test readability
                with open(self.path, 'r', encoding='utf-8') as f:
                    f.read(1)
                self.readable = True
            except (IOError, OSError, PermissionError):
                self.readable = False


class ConfigSchema:
    """
    Configuration schema validator.

    Theory Connection: Ensures shared-context components maintain semantic
    relationships. Schema violations trigger zero propagation where `C_pair = 0`.
    """

    def __init__(self, schema_dict: Dict[str, Any]):
        """
        Initialize schema validator.

        Args:
            schema_dict: JSON schema dictionary
        """
        self.schema = schema_dict

    def validate(self, config_data: Dict[str, Any]) -> List[str]:
        """
        Validate configuration data against schema.

        Args:
            config_data: Configuration data to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Basic type validation
        for field, field_schema in self.schema.get('properties', {}).items():
            if field in config_data:
                value = config_data[field]
                expected_type = field_schema.get('type')

                if expected_type and not self._validate_type(value, expected_type):
                    errors.append(f"Field '{field}': expected {expected_type}, got {type(value).__name__}")

        # Required fields validation
        required_fields = self.schema.get('required', [])
        for field in required_fields:
            if field not in config_data:
                errors.append(f"Required field missing: '{field}'")

        return errors

    @staticmethod
    def _validate_type(value: Any, expected_type: str) -> bool:
        """
        Validate value type.

        Args:
            value: Value to validate
            expected_type: Expected type string

        Returns:
            True if type matches
        """
        type_map: Dict[str, Union[type, Tuple[type, ...]]] = {
            'string': str,
            'integer': int,
            'number': (int, float),
            'boolean': bool,
            'array': list,
            'object': dict
        }

        expected_python_type = type_map.get(expected_type)
        if expected_python_type is None:
            return True  # Unknown type, skip validation

        return isinstance(value, expected_python_type)


class ConfigLoader:
    """
    Hierarchical configuration loader with schema validation.

    Theory Connection - Conveyance Framework:
    Implements efficient loading (TIME dimension) with hierarchical
    source resolution (WHERE dimension). Schema validation ensures
    WHAT quality, while access controls represent WHO dimension.

    The loader optimizes for `C_pair = Hmean(C_out, C_in)·C_ext^α·P_ij` by:
    - Minimizing load time (T) through caching and lazy evaluation
    - Maximizing shared-context coherence (`C_ext`) through validation
    - Supporting hierarchical positioning (R) through source priorities
    """

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize configuration loader.

        Args:
            base_dir: Base directory for configuration files
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._schemas: Dict[str, ConfigSchema] = {}

    def register_schema(self, name: str, schema_dict: Dict[str, Any]) -> None:
        """
        Register a configuration schema.

        Args:
            name: Schema name
            schema_dict: JSON schema dictionary
        """
        self._schemas[name] = ConfigSchema(schema_dict)
        logger.debug(f"Registered configuration schema: {name}")

    def discover_sources(self, config_name: str) -> List[ConfigSource]:
        """
        Discover configuration sources in priority order.

        Theory Connection: Implements WHERE dimension through hierarchical
        source discovery. Priority determines influence on final Context.

        Args:
            config_name: Configuration name (without extension)

        Returns:
            List of configuration sources, highest priority first
        """
        sources = []

        # Source priority (highest to lowest):
        # 1. Current directory specific config
        # 2. Home directory specific config
        # 3. Base directory specific config
        # 4. Current directory base config
        # 5. Home directory base config
        # 6. Base directory base config

        search_paths = [
            (Path.cwd(), 100),              # Current directory (highest priority)
            (Path.home() / ".hades", 90),    # User home directory
            (self.base_dir, 80),             # Base directory
        ]

        config_variants = [
            (f"{config_name}.yaml", ConfigFormat.YAML),
            (f"{config_name}.json", ConfigFormat.JSON),
            ("config.yaml", ConfigFormat.YAML),
            ("config.json", ConfigFormat.JSON),
        ]

        for search_path, base_priority in search_paths:
            for i, (filename, format_type) in enumerate(config_variants):
                file_path = search_path / filename
                # Specific configs get higher priority than generic configs
                priority = base_priority - i
                sources.append(ConfigSource(
                    path=file_path,
                    format=format_type,
                    priority=priority
                ))

        # Sort by priority (highest first)
        sources.sort(key=lambda s: s.priority, reverse=True)

        return sources

    def load_file(self, file_path: Path, format_type: ConfigFormat) -> Dict[str, Any]:
        """
        Load configuration from file.

        Args:
            file_path: Path to configuration file
            format_type: Configuration format

        Returns:
            Configuration data dictionary

        Raises:
            ConfigError: If file cannot be loaded
        """
        cache_key = str(file_path)
        if cache_key in self._cache:
            logger.debug(f"Using cached configuration: {file_path}")
            return self._cache[cache_key]

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if format_type == ConfigFormat.YAML:
                data = yaml.safe_load(content) or {}
            elif format_type == ConfigFormat.JSON:
                data = json.loads(content)
            else:
                raise ConfigError(f"Unsupported format: {format_type}")

            # Cache the result
            self._cache[cache_key] = data
            logger.debug(f"Loaded configuration from {file_path}")
            return data

        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in {file_path}: {e}")
        except json.JSONDecodeError as e:
            raise ConfigError(f"Invalid JSON in {file_path}: {e}")
        except (IOError, OSError) as e:
            raise ConfigError(f"Cannot read {file_path}: {e}")

    def load_environment(self, prefix: str = "HADES_") -> Dict[str, Any]:
        """
        Load configuration from environment variables.

        Args:
            prefix: Environment variable prefix

        Returns:
            Configuration data from environment
        """
        config: Dict[str, Any] = {}
        prefix_lower = prefix.lower()

        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Convert HADES_FOO_BAR to foo.bar
                config_key = key[len(prefix):].lower()
                config_path = config_key.split('_')

                # Build nested dictionary
                current = config
                for part in config_path[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]

                # Convert value to appropriate type
                final_key = config_path[-1]
                current[final_key] = self._convert_env_value(value)

        return config

    @staticmethod
    def _convert_env_value(value: str) -> Union[str, int, float, bool]:
        """
        Convert environment variable string to appropriate type.

        Args:
            value: String value from environment

        Returns:
            Converted value
        """
        # Boolean conversion
        if value.lower() in ('true', 'yes', '1', 'on'):
            return True
        elif value.lower() in ('false', 'no', '0', 'off'):
            return False

        # Numeric conversion
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass

        # String (default)
        return value

    def load_hierarchical(self,
                         config_name: str,
                         config_class: Type[T],
                         schema_name: Optional[str] = None,
                         env_prefix: str = "HADES_") -> T:
        """
        Load configuration using hierarchical source resolution.

        Theory Connection: Implements full Conveyance Framework optimization:
        - WHERE (R): Hierarchical source positioning
        - WHAT (W): Schema validation for content quality
        - WHO (H): Access control through file permissions
        - TIME (T): Efficient loading with caching
        - Shared context (C_ext): Exponential amplification through validation

        Args:
            config_name: Configuration name
            config_class: Configuration class to instantiate
            schema_name: Schema name for validation
            env_prefix: Environment variable prefix

        Returns:
            Loaded and validated configuration instance

        Raises:
            ConfigError: If configuration cannot be loaded
            ConfigValidationError: If validation fails
        """
        logger.info(f"Loading hierarchical configuration: {config_name}")

        # Start with empty base configuration
        merged_config: Dict[str, Any] = {}

        # 1. Load from files (lowest to highest priority)
        sources = self.discover_sources(config_name)
        loaded_sources = []

        for source in reversed(sources):  # Process lowest priority first
            if not source.exists or not source.readable:
                continue

            try:
                file_config = self.load_file(source.path, source.format)
                # Deep merge with existing configuration
                merged_config = self._deep_merge(merged_config, file_config)
                loaded_sources.append(source)
                logger.debug(f"Merged config from {source.path} (priority {source.priority})")

            except ConfigError as e:
                logger.warning(f"Failed to load {source.path}: {e}")
                continue

        # 2. Load from environment (highest priority)
        env_config = self.load_environment(env_prefix)
        if env_config:
            merged_config = self._deep_merge(merged_config, env_config)
            logger.debug(f"Merged environment config with prefix {env_prefix}")

        # 3. Schema validation (if specified)
        if schema_name and schema_name in self._schemas:
            validation_errors = self._schemas[schema_name].validate(merged_config)
            if validation_errors:
                raise ConfigValidationError(
                    f"Schema validation failed for {config_name}",
                    validation_errors
                )
            logger.debug(f"Schema validation passed for {schema_name}")

        # 4. Create configuration instance
        try:
            config_instance = config_class.from_dict(merged_config)
            logger.info(f"Successfully loaded {config_name} configuration from {len(loaded_sources)} sources")
            return config_instance

        except ConfigValidationError:
            raise
        except Exception as e:
            raise ConfigError(f"Failed to create {config_name} configuration: {e}") from e

    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two configuration dictionaries.

        Args:
            base: Base configuration
            override: Override configuration (higher priority)

        Returns:
            Merged configuration
        """
        result = base.copy()

        for key, value in override.items():
            if (key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)):
                result[key] = ConfigLoader._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def clear_cache(self) -> None:
        """Clear configuration cache."""
        self._cache.clear()
        logger.debug("Configuration cache cleared")

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get cache information.

        Returns:
            Cache statistics
        """
        return {
            'cached_files': len(self._cache),
            'registered_schemas': len(self._schemas),
            'cache_keys': list(self._cache.keys())
        }
