"""
Centralized Configuration Management
===================================

Theory Connection - Information Reconstructionism:
The configuration manager serves as the central "obligatory passage point"
(Actor-Network Theory) through which all system components access configuration.
It optimizes the Conveyance Framework `C_pair = Hmean(C_out, C_in)·C_ext^α·P_ij`
by maintaining configuration coherence across distributed processing components.

The manager implements shared-context (`C_ext`) amplification through:
- L = Local coherence via centralized validation
- I = Instruction fit via component-specific configurations
- A = Actionability via factory patterns and ready-to-use instances
- G = Grounding via persistent configuration state and versioning
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

from .config_base import BaseConfig, ConfigError, ProcessingConfig, StorageConfig
from .config_loader import ConfigLoader

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseConfig)


class ConfigScope(Enum):
    """Configuration scope levels."""
    GLOBAL = "global"       # System-wide configuration
    MODULE = "module"       # Module-specific configuration
    COMPONENT = "component" # Component-specific configuration
    INSTANCE = "instance"   # Instance-specific configuration


@dataclass
class ConfigRegistration:
    """
    Configuration registration metadata.

    Theory Connection: Represents positioning in the WHERE dimension
    of the configuration hierarchy, with scope determining influence range.
    """
    name: str
    config_class: Type[BaseConfig]
    scope: ConfigScope
    schema: Optional[Dict[str, Any]] = None
    factory_func: Optional[callable] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: Optional[datetime] = None
    access_count: int = 0

    def update_access(self):
        """Update access statistics."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1


class ConfigCache:
    """
    Thread-safe configuration cache with TTL support.

    Theory Connection: Optimizes TIME dimension by reducing configuration
    loading overhead while maintaining Context coherence through TTL-based
    invalidation that prevents stale configuration drift.
    """

    def __init__(self, default_ttl: timedelta = timedelta(minutes=30)):
        """
        Initialize configuration cache.

        Args:
            default_ttl: Default time-to-live for cached configurations
        """
        self._cache: Dict[str, Any] = {}
        self._timestamps: Dict[str, datetime] = {}
        self._ttl: Dict[str, timedelta] = {}
        self._lock = threading.RLock()
        self.default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        """
        Get cached configuration.

        Args:
            key: Cache key

        Returns:
            Cached configuration or None if expired/missing
        """
        with self._lock:
            if key not in self._cache:
                return None

            # Check TTL
            if self._is_expired(key):
                self._remove_key(key)
                return None

            return self._cache[key]

    def set(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> None:
        """
        Set cached configuration.

        Args:
            key: Cache key
            value: Configuration to cache
            ttl: Time-to-live (uses default if None)
        """
        with self._lock:
            self._cache[key] = value
            self._timestamps[key] = datetime.utcnow()
            self._ttl[key] = ttl or self.default_ttl

    def invalidate(self, key: str) -> bool:
        """
        Invalidate cached configuration.

        Args:
            key: Cache key to invalidate

        Returns:
            True if key was cached
        """
        with self._lock:
            if key in self._cache:
                self._remove_key(key)
                return True
            return False

    def clear(self) -> None:
        """Clear entire cache."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._ttl.clear()

    def cleanup_expired(self) -> int:
        """
        Remove expired entries from cache.

        Returns:
            Number of expired entries removed
        """
        with self._lock:
            expired_keys = [key for key in self._cache.keys() if self._is_expired(key)]
            for key in expired_keys:
                self._remove_key(key)
            return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Cache statistics dictionary
        """
        with self._lock:
            return {
                'total_entries': len(self._cache),
                'expired_entries': len([k for k in self._cache.keys() if self._is_expired(k)]),
                'cache_keys': list(self._cache.keys()),
                'default_ttl_seconds': self.default_ttl.total_seconds()
            }

    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self._timestamps:
            return True

        timestamp = self._timestamps[key]
        ttl = self._ttl.get(key, self.default_ttl)
        return datetime.utcnow() - timestamp > ttl

    def _remove_key(self, key: str) -> None:
        """Remove key from all cache structures."""
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)
        self._ttl.pop(key, None)


class ConfigManager:
    """
    Centralized configuration management with factory patterns and caching.

    Theory Connection - Conveyance Framework Optimization:
    The manager maximizes `C_pair = Hmean(C_out, C_in)·C_ext^α·P_ij` through:

    1. WHERE (R): Hierarchical scope management (global > module > component)
    2. WHAT (W): Validated configuration quality through registration system
    3. WHO (H): Access control and component isolation
    4. TIME (T): Efficient loading with caching and factory patterns
    5. Shared context (`C_ext^α`): Exponential amplification via centralized coherence

    From Actor-Network Theory: Acts as an "immutable mobile" that maintains
    stable configuration relationships across distributed system components.
    """

    _instance: Optional['ConfigManager'] = None
    _lock = threading.Lock()

    def __new__(cls) -> 'ConfigManager':
        """Singleton pattern for global configuration management."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize configuration manager (only once due to singleton)."""
        if hasattr(self, '_initialized'):
            return

        self._initialized = True
        self._registrations: Dict[str, ConfigRegistration] = {}
        self._loader = ConfigLoader()
        self._cache = ConfigCache()
        self._scope_hierarchy = {
            ConfigScope.GLOBAL: 0,
            ConfigScope.MODULE: 1,
            ConfigScope.COMPONENT: 2,
            ConfigScope.INSTANCE: 3
        }

        # Register built-in configuration types
        self._register_builtin_configs()

        logger.info("Configuration manager initialized")

    def _register_builtin_configs(self) -> None:
        """Register built-in configuration types."""
        self.register('processing', ProcessingConfig, ConfigScope.COMPONENT)
        self.register('storage', StorageConfig, ConfigScope.COMPONENT)

    def register(self,
                name: str,
                config_class: Type[T],
                scope: ConfigScope = ConfigScope.COMPONENT,
                schema: Optional[Dict[str, Any]] = None,
                factory_func: Optional[callable] = None) -> None:
        """
        Register a configuration type.

        Theory Connection: Establishes WHERE positioning in configuration
        hierarchy. Higher scopes have broader influence on Context amplification.

        Args:
            name: Configuration name
            config_class: Configuration class
            scope: Configuration scope
            schema: Optional JSON schema for validation
            factory_func: Optional factory function for complex initialization

        Raises:
            ConfigError: If registration fails
        """
        if name in self._registrations:
            existing = self._registrations[name]
            if existing.config_class != config_class:
                raise ConfigError(f"Configuration '{name}' already registered with different class")
            logger.warning(f"Re-registering configuration '{name}'")

        registration = ConfigRegistration(
            name=name,
            config_class=config_class,
            scope=scope,
            schema=schema,
            factory_func=factory_func
        )

        self._registrations[name] = registration

        # Register schema with loader if provided
        if schema:
            self._loader.register_schema(name, schema)

        logger.debug(f"Registered configuration '{name}' with scope {scope.value}")

    def get(self,
           name: str,
           instance_id: Optional[str] = None,
           force_reload: bool = False,
           **overrides) -> BaseConfig:
        """
        Get configuration instance.

        Theory Connection: Optimizes Conveyance through efficient Context
        retrieval with caching (TIME optimization) while maintaining
        semantic coherence (WHAT quality).

        Args:
            name: Configuration name
            instance_id: Optional instance identifier for instance-scoped configs
            force_reload: Force reload from sources
            **overrides: Runtime configuration overrides

        Returns:
            Configuration instance

        Raises:
            ConfigError: If configuration cannot be loaded
        """
        if name not in self._registrations:
            raise ConfigError(f"Configuration '{name}' not registered")

        registration = self._registrations[name]
        registration.update_access()

        # Build cache key
        cache_key = name
        if instance_id:
            cache_key = f"{name}:{instance_id}"
        if overrides:
            override_hash = hash(frozenset(overrides.items()))
            cache_key = f"{cache_key}:{override_hash}"

        # Check cache
        if not force_reload:
            cached_config = self._cache.get(cache_key)
            if cached_config is not None:
                logger.debug(f"Using cached configuration: {cache_key}")
                return cached_config

        # Load configuration
        try:
            if registration.factory_func:
                # Use custom factory function
                config_instance = registration.factory_func(
                    name, instance_id, overrides
                )
            else:
                # Use hierarchical loader
                config_instance = self._loader.load_hierarchical(
                    config_name=name,
                    config_class=registration.config_class,
                    schema_name=name if registration.schema else None
                )

                # Apply runtime overrides
                if overrides:
                    override_config = registration.config_class.from_dict(overrides)
                    config_instance = config_instance.merge(override_config)

            # Cache the result
            self._cache.set(cache_key, config_instance)

            logger.debug(f"Loaded configuration: {cache_key}")
            return config_instance

        except Exception as e:
            raise ConfigError(f"Failed to load configuration '{name}': {e}")

    def get_or_create(self,
                     name: str,
                     config_class: Type[T],
                     instance_id: Optional[str] = None,
                     **kwargs) -> T:
        """
        Get existing configuration or create with default values.

        Args:
            name: Configuration name
            config_class: Configuration class
            instance_id: Optional instance identifier
            **kwargs: Default configuration values

        Returns:
            Configuration instance
        """
        try:
            return self.get(name, instance_id, **kwargs)
        except ConfigError:
            # Create with defaults
            if name not in self._registrations:
                self.register(name, config_class)

            config_instance = config_class.from_dict(kwargs)

            # Cache the result
            cache_key = name
            if instance_id:
                cache_key = f"{name}:{instance_id}"
            self._cache.set(cache_key, config_instance)

            return config_instance

    def invalidate(self, name: str, instance_id: Optional[str] = None) -> None:
        """
        Invalidate cached configuration.

        Args:
            name: Configuration name
            instance_id: Optional instance identifier
        """
        cache_key = name
        if instance_id:
            cache_key = f"{name}:{instance_id}"

        invalidated = self._cache.invalidate(cache_key)
        if invalidated:
            logger.debug(f"Invalidated cached configuration: {cache_key}")

    def invalidate_all(self) -> None:
        """Invalidate all cached configurations."""
        self._cache.clear()
        logger.debug("Invalidated all cached configurations")

    def list_registrations(self) -> List[str]:
        """
        List all registered configuration names.

        Returns:
            List of configuration names
        """
        return list(self._registrations.keys())

    def get_registration_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get registration information.

        Args:
            name: Configuration name

        Returns:
            Registration information or None if not found
        """
        if name not in self._registrations:
            return None

        registration = self._registrations[name]
        return {
            'name': registration.name,
            'class': registration.config_class.__name__,
            'scope': registration.scope.value,
            'has_schema': registration.schema is not None,
            'has_factory': registration.factory_func is not None,
            'created_at': registration.created_at.isoformat(),
            'last_accessed': registration.last_accessed.isoformat() if registration.last_accessed else None,
            'access_count': registration.access_count
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get configuration manager statistics.

        Returns:
            Statistics dictionary
        """
        return {
            'total_registrations': len(self._registrations),
            'registrations_by_scope': {
                scope.value: len([r for r in self._registrations.values() if r.scope == scope])
                for scope in ConfigScope
            },
            'cache_stats': self._cache.get_stats(),
            'most_accessed': [
                {
                    'name': reg.name,
                    'access_count': reg.access_count,
                    'last_accessed': reg.last_accessed.isoformat() if reg.last_accessed else None
                }
                for reg in sorted(self._registrations.values(),
                                key=lambda r: r.access_count, reverse=True)[:5]
            ]
        }

    def cleanup(self) -> None:
        """Cleanup expired cache entries and reset statistics."""
        expired_count = self._cache.cleanup_expired()
        if expired_count > 0:
            logger.debug(f"Cleaned up {expired_count} expired cache entries")

    def export_config(self, name: str, output_path: Path) -> None:
        """
        Export configuration to file.

        Args:
            name: Configuration name
            output_path: Output file path

        Raises:
            ConfigError: If export fails
        """
        try:
            config = self.get(name)
            config.save_to_file(output_path)
            logger.info(f"Exported configuration '{name}' to {output_path}")
        except Exception as e:
            raise ConfigError(f"Failed to export configuration '{name}': {e}")


# Global configuration manager instance
config_manager = ConfigManager()
