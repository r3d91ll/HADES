"""
Core Storage Component

This module provides the core storage component that implements the
Storage protocol. It acts as a factory and coordinator for different 
storage backends (ArangoDB, Memory, NetworkX).
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import storage implementations
from src.components.storage.arangodb.storage import ArangoStorage
from src.components.storage.memory.storage import MemoryStorage
from src.components.storage.networkx.processor import NetworkXStorage

# Import component contracts and protocols
from src.types.components.contracts import (
    ComponentType,
    ComponentMetadata,
    StorageInput,
    StorageOutput,
    QueryInput,
    QueryOutput,
    ProcessingStatus
)
from src.types.components.protocols import Storage


class CoreStorage(Storage):
    """
    Core storage component implementing Storage protocol.
    
    This component acts as a factory and coordinator for different storage
    backends, automatically selecting the appropriate storage implementation
    based on configuration and requirements.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize core storage component.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self._config = config or {}
        
        # Component metadata
        self._metadata = ComponentMetadata(
            component_type=ComponentType.STORAGE,
            component_name="core",
            component_version="1.0.0",
            config=self._config
        )
        
        # Storage backend configuration
        self._storage_type = self._config.get('storage_type', 'memory')
        self._fallback_storage = self._config.get('fallback_storage', 'memory')
        
        # Initialize storage backend
        self._storage: Optional[Storage] = None
        self._fallback: Optional[Storage] = None
        
        self.logger.info(f"Initialized core storage with backend: {self._storage_type}")
    
    @property
    def name(self) -> str:
        """Component name for identification."""
        return "core"
    
    @property
    def version(self) -> str:
        """Component version string."""
        return "1.0.0"
    
    @property
    def component_type(self) -> ComponentType:
        """Type of component."""
        return ComponentType.STORAGE
    
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure component with parameters.
        
        Args:
            config: Configuration dictionary containing component parameters
            
        Raises:
            ValueError: If configuration is invalid
        """
        if not self.validate_config(config):
            raise ValueError("Invalid configuration provided")
        
        self._config.update(config)
        self._metadata.config = self._config
        self._metadata.processed_at = datetime.utcnow()
        
        # Update storage type if specified
        if 'storage_type' in config:
            self._storage_type = config['storage_type']
        
        if 'fallback_storage' in config:
            self._fallback_storage = config['fallback_storage']
        
        # Reset storage to force re-initialization
        self._storage = None
        self._fallback = None
        
        self.logger.info(f"Updated core storage configuration")
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        if not isinstance(config, dict):
            return False
        
        # Validate storage type
        if 'storage_type' in config:
            valid_types = ['memory', 'arangodb', 'networkx']
            if config['storage_type'] not in valid_types:
                return False
        
        # Validate fallback storage
        if 'fallback_storage' in config:
            valid_types = ['memory', 'arangodb', 'networkx']
            if config['fallback_storage'] not in valid_types:
                return False
        
        return True
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get the JSON schema for component configuration.
        
        Returns:
            JSON schema dictionary describing valid configuration
        """
        return {
            "type": "object",
            "properties": {
                "storage_type": {
                    "type": "string",
                    "enum": ["memory", "arangodb", "networkx"],
                    "description": "Primary storage backend to use",
                    "default": "memory"
                },
                "fallback_storage": {
                    "type": "string",
                    "enum": ["memory", "arangodb", "networkx"],
                    "description": "Fallback storage backend",
                    "default": "memory"
                },
                "storage_config": {
                    "type": "object",
                    "description": "Configuration for the selected storage backend",
                    "properties": {
                        "host": {"type": "string", "default": "localhost"},
                        "port": {"type": "integer", "default": 8529},
                        "database": {"type": "string", "default": "hades"},
                        "username": {"type": "string", "default": "root"},
                        "password": {"type": "string", "default": ""},
                        "max_items": {"type": "integer", "default": 100000}
                    }
                }
            }
        }
    
    def health_check(self) -> bool:
        """
        Check if component is healthy and ready to process data.
        
        Returns:
            True if component is healthy, False otherwise
        """
        try:
            # Initialize storage if needed
            if not self._storage:
                self._initialize_storage()
            
            # Check primary storage health
            if self._storage and self._storage.health_check():
                return True
            
            # Check fallback storage health
            if not self._fallback:
                self._initialize_fallback()
            
            return self._fallback.health_check() if self._fallback else False
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get component performance metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        metrics = {
            "component_name": self.name,
            "component_version": self.version,
            "storage_type": self._storage_type,
            "fallback_storage": self._fallback_storage,
            "primary_storage_initialized": self._storage is not None,
            "fallback_storage_initialized": self._fallback is not None,
            "last_health_check": datetime.utcnow().isoformat()
        }
        
        # Add primary storage metrics
        if self._storage:
            try:
                primary_metrics = self._storage.get_metrics()
                metrics["primary_storage_metrics"] = primary_metrics
            except Exception as e:
                self.logger.warning(f"Could not get primary storage metrics: {e}")
        
        # Add fallback storage metrics
        if self._fallback:
            try:
                fallback_metrics = self._fallback.get_metrics()
                metrics["fallback_storage_metrics"] = fallback_metrics
            except Exception as e:
                self.logger.warning(f"Could not get fallback storage metrics: {e}")
        
        return metrics
    
    def store(self, input_data: StorageInput) -> StorageOutput:
        """
        Store data according to the contract.
        
        Args:
            input_data: Input data conforming to StorageInput contract
            
        Returns:
            Output data conforming to StorageOutput contract
        """
        try:
            # Initialize storage if needed
            if not self._storage:
                self._initialize_storage()
            
            # Try primary storage
            if self._storage:
                try:
                    return self._storage.store(input_data)
                except Exception as e:
                    self.logger.error(f"Primary storage failed: {e}")
            
            # Fallback to secondary storage
            if not self._fallback:
                self._initialize_fallback()
            
            if self._fallback:
                self.logger.info("Using fallback storage")
                return self._fallback.store(input_data)
            
            raise RuntimeError("No storage backend available")
            
        except Exception as e:
            self.logger.error(f"Storage operation failed: {e}")
            
            metadata = ComponentMetadata(
                component_type=self.component_type,
                component_name=self.name,
                component_version=self.version,
                processed_at=datetime.utcnow(),
                config=self._config,
                status=ProcessingStatus.ERROR
            )
            
            return StorageOutput(
                stored_items=[],
                metadata=metadata,
                errors=[f"Storage failed: {str(e)}"]
            )
    
    def query(self, query_data: QueryInput) -> QueryOutput:
        """
        Query stored data according to the contract.
        
        Args:
            query_data: Query data conforming to QueryInput contract
            
        Returns:
            Output data conforming to QueryOutput contract
        """
        try:
            # Initialize storage if needed
            if not self._storage:
                self._initialize_storage()
            
            # Try primary storage
            if self._storage:
                try:
                    return self._storage.query(query_data)
                except Exception as e:
                    self.logger.error(f"Primary storage query failed: {e}")
            
            # Fallback to secondary storage
            if not self._fallback:
                self._initialize_fallback()
            
            if self._fallback:
                self.logger.info("Using fallback storage for query")
                return self._fallback.query(query_data)
            
            raise RuntimeError("No storage backend available")
            
        except Exception as e:
            self.logger.error(f"Query operation failed: {e}")
            
            metadata = ComponentMetadata(
                component_type=self.component_type,
                component_name=self.name,
                component_version=self.version,
                processed_at=datetime.utcnow(),
                config=self._config,
                status=ProcessingStatus.ERROR
            )
            
            return QueryOutput(
                results=[],
                metadata=metadata,
                errors=[f"Query failed: {str(e)}"]
            )
    
    def delete(self, item_ids: List[str]) -> bool:
        """
        Delete items by their IDs.
        
        Args:
            item_ids: List of item IDs to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            # Try primary storage
            if self._storage and self._storage.delete(item_ids):
                return True
            
            # Try fallback storage
            if self._fallback and self._fallback.delete(item_ids):
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Delete operation failed: {e}")
            return False
    
    def update(self, item_id: str, data: Dict[str, Any]) -> bool:
        """
        Update an existing item.
        
        Args:
            item_id: ID of the item to update
            data: New data for the item
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            # Try primary storage
            if self._storage and self._storage.update(item_id, data):
                return True
            
            # Try fallback storage
            if self._fallback and self._fallback.update(item_id, data):
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Update operation failed: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary containing storage statistics
        """
        stats = {
            "storage_type": self._storage_type,
            "fallback_storage": self._fallback_storage
        }
        
        if self._storage:
            try:
                primary_stats = self._storage.get_statistics()
                stats["primary_storage"] = primary_stats
            except Exception:
                pass
        
        if self._fallback:
            try:
                fallback_stats = self._fallback.get_statistics()
                stats["fallback_storage_stats"] = fallback_stats
            except Exception:
                pass
        
        return stats
    
    def get_capacity_info(self) -> Dict[str, Any]:
        """
        Get storage capacity information.
        
        Returns:
            Dictionary containing capacity information
        """
        capacity = {"storage_type": self._storage_type}
        
        if self._storage:
            try:
                primary_capacity = self._storage.get_capacity_info()
                capacity["primary_storage"] = primary_capacity
            except Exception:
                pass
        
        return capacity
    
    def supports_transactions(self) -> bool:
        """
        Check if storage supports transactions.
        
        Returns:
            True if storage supports transactions, False otherwise
        """
        try:
            if self._storage:
                return self._storage.supports_transactions()
            return False
        except Exception:
            return False
    
    def get_supported_storage_types(self) -> List[str]:
        """
        Get list of supported storage types.
        
        Returns:
            List of supported storage type names
        """
        return ["memory", "arangodb", "networkx"]
    
    def switch_storage_backend(self, storage_type: str) -> bool:
        """
        Switch to a different storage backend.
        
        Args:
            storage_type: Type of storage backend to switch to
            
        Returns:
            True if switch was successful, False otherwise
        """
        if storage_type not in self.get_supported_storage_types():
            return False
        
        try:
            self._storage_type = storage_type
            self._storage = None  # Force re-initialization
            self._initialize_storage()
            return self._storage is not None
        except Exception as e:
            self.logger.error(f"Failed to switch storage backend: {e}")
            return False
    
    def _initialize_storage(self) -> None:
        """Initialize the primary storage backend."""
        try:
            storage_config = self._config.get('storage_config', {})
            
            if self._storage_type == 'memory':
                self._storage = MemoryStorage(config=storage_config)
            elif self._storage_type == 'arangodb':
                self._storage = ArangoStorage(config=storage_config)
            elif self._storage_type == 'networkx':
                self._storage = NetworkXStorage(config=storage_config)
            else:
                raise ValueError(f"Unsupported storage type: {self._storage_type}")
            
            self.logger.info(f"Initialized {self._storage_type} storage successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize {self._storage_type} storage: {e}")
            self._storage = None
    
    def _initialize_fallback(self) -> None:
        """Initialize the fallback storage backend."""
        try:
            if self._fallback_storage == self._storage_type:
                # Don't use the same storage as fallback
                self._fallback = None
                return
            
            storage_config = self._config.get('storage_config', {})
            
            if self._fallback_storage == 'memory':
                self._fallback = MemoryStorage(config=storage_config)
            elif self._fallback_storage == 'arangodb':
                self._fallback = ArangoStorage(config=storage_config)
            elif self._fallback_storage == 'networkx':
                self._fallback = NetworkXStorage(config=storage_config)
            else:
                self._fallback = MemoryStorage(config={})  # Default fallback
            
            self.logger.info(f"Initialized {self._fallback_storage} fallback storage")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize fallback storage: {e}")
            # Use memory as ultimate fallback
            try:
                self._fallback = MemoryStorage(config={})
            except Exception:
                self._fallback = None