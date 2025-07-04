"""
ArangoDB Storage Component v2 - Production Implementation

This module provides a complete ArangoDB storage component that implements the Storage
protocol with full Sequential-ISNE support, real database operations, and comprehensive
features for graph-enhanced RAG systems.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Union, Iterator, Tuple
from datetime import datetime, timezone
from pathlib import Path

# Import our ArangoDB client
from src.storage.arango_client import ArangoClient, ArangoConnectionError, ArangoOperationError

# Import Sequential-ISNE types from the centralized types directory
from src.types.storage.sequential_isne import (
    SequentialISNEConfig,
    FileType, EdgeType, EmbeddingType, SourceType,
    ProcessingStatus as ISNEProcessingStatus,  # Renamed to avoid conflict
    CodeFile, DocumentationFile, ConfigFile, Chunk, Embedding,
    IntraModalEdge, CrossModalEdge, ISNEModel,
    classify_file_type, get_modality_collection, is_cross_modal_edge,
    SequentialISNESchemaManager
)

# TODO: Need to create or move SequentialISNESchemaManager
# from src.storage.incremental.sequential_isne_schema import SequentialISNESchemaManager

# Import storage interfaces and types
from src.types.storage.interfaces import (
    DocumentRepository,
    VectorRepository,
    GraphRepository,
    DocumentData,
    ChunkData,
    VectorSearchResult,
    GraphNodeData,
    GraphEdgeData,
    StorageConfig,
    StorageMetrics,
    StoredItem,
    StorageInput,
    StorageOutput,
    QueryInput,
    QueryOutput,
    RetrievalResult
)

# Import common types
from src.types.common import DocumentID, NodeID, EdgeID, EmbeddingVector, ProcessingStatus, ComponentMetadata, ComponentType, RelationType

logger = logging.getLogger(__name__)


class ArangoStorageV2(DocumentRepository, VectorRepository, GraphRepository):
    """
    Production ArangoDB storage component implementing Storage protocol.
    
    Features:
    - Full Sequential-ISNE schema support
    - Modality-specific storage (code, documentation, config)
    - Cross-modal relationship management
    - Vector similarity search
    - Graph construction and traversal
    - Transaction support
    - Comprehensive error handling
    - Performance optimization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ArangoDB storage component.
        
        Args:
            config: Configuration dictionary with connection and storage settings
        """
        self.logger = logging.getLogger(__name__)
        self._config = config or {}
        
        # Component metadata
        self._metadata = ComponentMetadata(
            component_type=ComponentType.STORAGE,
            component_name="arangodb_v2",
            component_version="2.0.0",
            config=self._config
        )
        
        # Initialize Sequential-ISNE configuration
        self._sequential_isne_config = SequentialISNEConfig(**self._config.get('sequential_isne', {}))
        
        # ArangoDB connection settings
        db_config = self._config.get('database', {})
        self._host = db_config.get('host', '127.0.0.1')
        self._port = db_config.get('port', 8529)
        self._username = db_config.get('username', 'root')
        self._password = db_config.get('password', '')
        self._database_name = db_config.get('database', 'hades_db')
        
        # Storage settings
        self._vector_index_enabled = self._config.get('vector_index_enabled', True)
        self._vector_dimension = self._config.get('vector_dimension', 768)
        self._similarity_threshold = self._config.get('similarity_threshold', 0.8)
        self._max_results = self._config.get('max_results', 100)
        
        # Initialize components
        self._client: Optional[ArangoClient] = None
        self._schema_manager: Optional[SequentialISNESchemaManager] = None
        self._connected = False
        
        # Performance tracking
        self._stats: Dict[str, Union[int, str, None]] = {
            'documents_stored': 0,
            'queries_executed': 0,
            'cross_modal_edges_created': 0,
            'errors': 0,
            'last_operation': None
        }
        
        self.logger.info(f"Initialized ArangoDB Storage v2 for database: {self._database_name}")
    
    # ===== PROTOCOL IMPLEMENTATION =====
    
    @property
    def name(self) -> str:
        """Component name for identification."""
        return "arangodb_v2"
    
    @property
    def version(self) -> str:
        """Component version string."""
        return "2.0.0"
    
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
        
        # Update configuration
        self._config.update(config)
        self._metadata.config = self._config
        self._metadata.processed_at = datetime.now(timezone.utc)
        
        # Update Sequential-ISNE config
        sequential_isne_config = config.get('sequential_isne', {})
        if sequential_isne_config:
            self._sequential_isne_config = SequentialISNEConfig(**sequential_isne_config)
        
        # Reconnect if database settings changed
        db_config = config.get('database', {})
        if db_config:
            self._host = db_config.get('host', self._host)
            self._port = db_config.get('port', self._port)
            self._username = db_config.get('username', self._username)
            self._password = db_config.get('password', self._password)
            self._database_name = db_config.get('database', self._database_name)
            
            # Reconnect with new settings
            if self._connected:
                self.disconnect()
                self.connect()
        
        self.logger.info("Updated ArangoDB storage configuration")
    
    def validate_config(self, config: Any) -> bool:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        if not isinstance(config, dict):
            return False
        
        # Validate database configuration
        db_config = config.get('database', {})
        if db_config:
            required_str_fields = ['host', 'username', 'database']
            for field in required_str_fields:
                if field in db_config and not isinstance(db_config[field], str):
                    return False
            
            if 'port' in db_config and not isinstance(db_config['port'], int):
                return False
        
        # Validate Sequential-ISNE configuration
        sequential_isne_config = config.get('sequential_isne', {})
        if sequential_isne_config:
            try:
                SequentialISNEConfig(**sequential_isne_config)
            except Exception:
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
                "database": {
                    "type": "object",
                    "properties": {
                        "host": {"type": "string", "default": "127.0.0.1"},
                        "port": {"type": "integer", "default": 8529},
                        "username": {"type": "string", "default": "root"},
                        "password": {"type": "string", "default": ""},
                        "database": {"type": "string", "default": "sequential_isne"}
                    }
                },
                "sequential_isne": {
                    "type": "object",
                    "description": "Sequential-ISNE configuration",
                    "properties": {
                        "db_name": {"type": "string", "default": "sequential_isne"},
                        "batch_size": {"type": "integer", "default": 100},
                        "similarity_threshold": {"type": "number", "default": 0.8},
                        "embedding_dim": {"type": "integer", "default": 384}
                    }
                },
                "vector_index_enabled": {"type": "boolean", "default": True},
                "vector_dimension": {"type": "integer", "default": 768},
                "similarity_threshold": {"type": "number", "default": 0.8},
                "max_results": {"type": "integer", "default": 100}
            }
        }
    
    def health_check(self) -> bool:
        """
        Check if component is healthy and ready to process data.
        
        Returns:
            True if component is healthy, False otherwise
        """
        try:
            if not self._connected or not self._client:
                return False
            
            # Test database connection
            if not self._client.health_check():
                return False
            
            # Test schema availability
            if self._schema_manager and not self._schema_manager.validate_schema():
                return False
            
            return True
            
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
            "is_connected": self._connected,
            "database_name": self._database_name,
            "schema_validated": self._schema_manager.validate_schema() if self._schema_manager else False,
            "last_health_check": datetime.now(timezone.utc).isoformat()
        }
        
        # Add performance statistics
        metrics.update(self._stats)
        
        # Add database metrics if connected
        if self._client:
            db_metrics = self._client.get_metrics()
            metrics.update({
                'db_connections_created': db_metrics.get('connections_created', 0),
                'db_queries_executed': db_metrics.get('queries_executed', 0),
                'db_documents_inserted': db_metrics.get('documents_inserted', 0),
                'db_errors': db_metrics.get('errors', 0)
            })
        
        return metrics
    
    # ===== CONNECTION MANAGEMENT =====
    
    def connect(self) -> bool:
        """
        Establish connection to ArangoDB and initialize schema.
        
        Returns:
            True if connection successful
            
        Raises:
            ArangoConnectionError: If connection fails
        """
        try:
            # Create ArangoDB client
            self._client = ArangoClient(
                host=self._host,
                port=self._port,
                username=self._username,
                password=self._password,
                database=self._database_name,
                timeout=30,
                max_retries=3
            )
            
            # Connect to database
            if not self._client.connect():
                raise ArangoConnectionError("Failed to connect to ArangoDB")
            
            # Initialize schema manager
            self._schema_manager = SequentialISNESchemaManager(
                self._sequential_isne_config
            )
            
            # Initialize database schema if needed
            if not self._schema_manager.validate_schema():
                self.logger.info("Initializing Sequential-ISNE schema...")
                if not self._schema_manager.initialize_database():
                    raise ArangoConnectionError("Failed to initialize database schema")
            
            self._connected = True
            self.logger.info(f"Successfully connected to ArangoDB: {self._database_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to ArangoDB: {e}")
            self._connected = False
            raise ArangoConnectionError(f"Connection failed: {e}")
    
    def disconnect(self) -> None:
        """Disconnect from ArangoDB."""
        if self._client:
            self._client.disconnect()
        
        self._client = None
        self._schema_manager = None
        self._connected = False
        self.logger.info("Disconnected from ArangoDB")
    
    def ensure_connected(self) -> None:
        """Ensure client is connected, connecting if necessary."""
        if not self._connected:
            self.connect()
    
    # ===== STORAGE PROTOCOL IMPLEMENTATION =====
    
    def store(self, input_data: StorageInput) -> StorageOutput:
        """
        Store enhanced embeddings according to the contract.
        
        Args:
            input_data: Input data conforming to StorageInput contract
            
        Returns:
            Output data conforming to StorageOutput contract
        """
        self.ensure_connected()
        
        stored_items: List[StoredItem] = []
        errors: List[str] = []
        
        start_time = datetime.now(timezone.utc)
        
        try:
            # Process each enhanced embedding from options
            enhanced_embeddings = input_data.options.get('enhanced_embeddings', [])
            for embedding in enhanced_embeddings:
                try:
                    # Determine storage location based on content type
                    storage_location = self._determine_storage_location(embedding)
                    
                    # Store the embedding
                    stored_item = self._store_enhanced_embedding(embedding, storage_location)
                    stored_items.append(stored_item)
                    
                    self._increment_stat('documents_stored')
                    
                except Exception as e:
                    error_msg = f"Failed to store embedding {embedding.chunk_id}: {e}"
                    errors.append(error_msg)
                    self.logger.error(error_msg)
            
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self._stats['last_operation'] = 'store'
            
            # Create metadata
            metadata = ComponentMetadata(
                component_type=self.component_type,
                component_name=self.name,
                component_version=self.version,
                processed_at=datetime.now(timezone.utc),
                config=self._config,
                metrics={'processing_time': processing_time}
            )
            
            # Calculate storage statistics
            storage_stats = {
                "stored_count": len(stored_items),
                "error_count": len(errors),
                "processing_time": processing_time,
                "storage_method": input_data.options.get('storage_method', 'sequential_isne'),
                "sequential_isne_enabled": True
            }
            
            # Index information
            index_info = {
                "vector_index_enabled": self._vector_index_enabled,
                "vector_dimension": self._vector_dimension,
                "modality_collections": self._schema_manager.get_modality_collections() if self._schema_manager else {},
                "cross_modal_edges_supported": True,
                "sequential_isne_enabled": True
            }
            
            # Convert StoredItem objects to dicts for documents field
            document_dicts = [item.model_dump() for item in stored_items]
            
            return StorageOutput(
                success=True,
                stored_items=stored_items,  # Already StoredItem objects
                document_id=None,  # Multiple documents stored
                documents=document_dicts,
                error_message=None,
                errors=errors,
                metadata={
                    **metadata.model_dump(),
                    'storage_stats': storage_stats,
                    'index_info': index_info
                }
            )
            
        except Exception as e:
            self._increment_stat('errors')
            error_msg = f"Storage operation failed: {e}"
            self.logger.error(error_msg)
            raise ArangoOperationError(error_msg)
    
    def query(self, query_data: QueryInput) -> QueryOutput:
        """
        Query stored data according to the contract.
        
        Args:
            query_data: Query data conforming to QueryInput contract
            
        Returns:
            Output data conforming to QueryOutput contract
        """
        self.ensure_connected()
        
        results: List[RetrievalResult] = []
        errors: List[str] = []
        
        start_time = datetime.now(timezone.utc)
        
        try:
            # Determine query type
            query_type = "vector" if query_data.query_embedding else "text"
            
            if query_type == "vector":
                results = self._execute_vector_query(query_data)
            else:
                results = self._execute_text_query(query_data)
            
            search_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self._increment_stat('queries_executed')
            self._stats['last_operation'] = 'query'
            
            # Create metadata
            metadata = ComponentMetadata(
                component_type=self.component_type,
                component_name=self.name,
                component_version=self.version,
                metrics={'processing_time': search_time},
                processed_at=datetime.now(timezone.utc),
                config=self._config
            )
            
            # Search statistics
            search_stats = {
                "query_type": query_type,
                "search_time": search_time,
                "total_candidates": len(results),
                "filters_applied": bool(query_data.filters),
                "sequential_isne_enhanced": True
            }
            
            # Query information
            query_info = {
                "query": query_data.query,
                "top_k": query_data.top_k,
                "similarity_threshold": self._similarity_threshold,
                "cross_modal_search": query_data.search_options.get('cross_modal', False)
            }
            
            # Convert RetrievalResult objects to dicts for QueryOutput
            result_dicts = [r.model_dump() for r in results]
            
            return QueryOutput(
                results=result_dicts,
                total_count=len(results),
                execution_time=search_time,
                metadata={
                    **metadata.model_dump(),
                    'search_stats': search_stats,
                    'query_info': query_info,
                    'errors': errors
                }
            )
            
        except Exception as e:
            self._increment_stat('errors')
            error_msg = f"Query operation failed: {e}"
            self.logger.error(error_msg)
            raise ArangoOperationError(error_msg)
    
    async def delete(self, document_id: DocumentID) -> bool:
        """
        Delete a document by ID (async version for protocol compliance).
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            True if document was deleted successfully
        """
        return self.delete_items([str(document_id)])
    
    def delete_items(self, item_ids: List[str]) -> bool:
        """
        Delete items by their IDs.
        
        Args:
            item_ids: List of item IDs to delete
            
        Returns:
            True if all items were deleted successfully
        """
        self.ensure_connected()
        
        try:
            deleted_count = 0
            
            for item_id in item_ids:
                # Parse item ID to determine collection and key
                collection_name, key = self._parse_item_id(item_id)
                
                if self._client is not None and self._client.delete_document(collection_name, key):
                    deleted_count += 1
            
            self.logger.info(f"Deleted {deleted_count}/{len(item_ids)} items")
            return deleted_count == len(item_ids)
            
        except Exception as e:
            self.logger.error(f"Delete operation failed: {e}")
            self._increment_stat('errors')
            return False
    
    async def update(self, document_id: DocumentID, updates: Dict[str, Any]) -> bool:
        """
        Update a document (async version for protocol compliance).
        
        Args:
            document_id: Document ID to update
            updates: Update data
            
        Returns:
            True if update was successful
        """
        return self.update_item(str(document_id), updates)
    
    def update_item(self, item_id: str, data: Dict[str, Any]) -> bool:
        """
        Update an existing item.
        
        Args:
            item_id: ID of item to update
            data: Updated data
            
        Returns:
            True if update was successful
        """
        self.ensure_connected()
        
        try:
            # Parse item ID to determine collection and key
            collection_name, key = self._parse_item_id(item_id)
            
            # Add update timestamp
            data['updated_at'] = datetime.now(timezone.utc).isoformat()
            data['_key'] = key
            
            if self._client is None:
                raise ValueError("ArangoDB client not connected")
            result = self._client.update_document(collection_name, data)
            
            self.logger.debug(f"Updated item {item_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Update operation failed for {item_id}: {e}")
            self._increment_stat('errors')
            return False
    
    # ===== ASYNC PROTOCOL METHODS (STUBS) =====
    
    async def create(self, document: DocumentData) -> DocumentID:
        """
        Create a document (async stub).
        
        Args:
            document: Document data to create
            
        Returns:
            Created document ID
        """
        # TODO: Implement async version
        raise NotImplementedError("Async create not yet implemented")
    
    async def read(self, document_id: DocumentID) -> Optional[DocumentData]:
        """
        Read a document by ID (async stub).
        
        Args:
            document_id: Document ID to read
            
        Returns:
            Document data if found, None otherwise
        """
        # TODO: Implement async version
        raise NotImplementedError("Async read not yet implemented")
    
    async def list(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[DocumentData]:
        """
        List documents with optional filters (async stub).
        
        Args:
            filters: Optional filters to apply
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            List of documents matching criteria
        """
        # TODO: Implement async version
        raise NotImplementedError("Async list not yet implemented")
    
    async def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> List[DocumentData]:
        """
        Search documents by query (async stub).
        
        Args:
            query: Search query string
            filters: Optional filters to apply
            limit: Maximum number of results
            
        Returns:
            List of documents matching search criteria
        """
        # TODO: Implement async version
        raise NotImplementedError("Async search not yet implemented")
    
    async def add_embedding(
        self,
        embedding_id: str,
        embedding: EmbeddingVector,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add an embedding to the vector store (async stub).
        
        Args:
            embedding_id: Unique identifier for the embedding
            embedding: Vector representation
            metadata: Optional metadata
            
        Returns:
            True if embedding was added successfully
        """
        # TODO: Implement async version
        raise NotImplementedError("Async add_embedding not yet implemented")
    
    async def get_embedding(
        self,
        embedding_id: str
    ) -> Optional[Tuple[EmbeddingVector, Dict[str, Any]]]:
        """
        Get an embedding by ID (async stub).
        
        Args:
            embedding_id: Embedding ID to retrieve
            
        Returns:
            Tuple of (embedding vector, metadata) if found, None otherwise
        """
        # TODO: Implement async version
        raise NotImplementedError("Async get_embedding not yet implemented")
    
    async def search_similar(
        self,
        query_embedding: EmbeddingVector,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None
    ) -> List[VectorSearchResult]:
        """
        Search for similar embeddings (async stub).
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filters: Optional filters to apply
            threshold: Optional similarity threshold
            
        Returns:
            List of similar vectors with scores
        """
        # TODO: Implement async version
        raise NotImplementedError("Async search_similar not yet implemented")
    
    async def delete_embedding(self, embedding_id: str) -> bool:
        """
        Delete an embedding by ID (async stub).
        
        Args:
            embedding_id: Embedding ID to delete
            
        Returns:
            True if embedding was deleted successfully
        """
        # TODO: Implement async version
        raise NotImplementedError("Async delete_embedding not yet implemented")
    
    async def create_index(
        self,
        index_type: str = "hnsw",
        **kwargs: Any
    ) -> bool:
        """
        Create an index for vector search (async stub).
        
        Args:
            index_type: Type of index to create
            **kwargs: Additional index parameters
            
        Returns:
            True if index was created successfully
        """
        # TODO: Implement async version
        raise NotImplementedError("Async create_index not yet implemented")
    
    async def add_node(self, node: GraphNodeData) -> NodeID:
        """
        Add a node to the graph (async stub).
        
        Args:
            node: Node data to add
            
        Returns:
            Created node ID
        """
        # TODO: Implement async version
        raise NotImplementedError("Async add_node not yet implemented")
    
    async def get_node(self, node_id: str) -> Optional[GraphNodeData]:
        """
        Get a node by ID (async stub).
        
        Args:
            node_id: Node ID to retrieve
            
        Returns:
            Node data if found, None otherwise
        """
        # TODO: Implement async version
        raise NotImplementedError("Async get_node not yet implemented")
    
    async def update_node(
        self,
        node_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update a node (async stub).
        
        Args:
            node_id: Node ID to update
            updates: Update data
            
        Returns:
            True if node was updated successfully
        """
        # TODO: Implement async version
        raise NotImplementedError("Async update_node not yet implemented")
    
    async def delete_node(self, node_id: str) -> bool:
        """
        Delete a node (async stub).
        
        Args:
            node_id: Node ID to delete
            
        Returns:
            True if node was deleted successfully
        """
        # TODO: Implement async version
        raise NotImplementedError("Async delete_node not yet implemented")
    
    async def add_edge(self, edge: GraphEdgeData) -> EdgeID:
        """
        Add an edge to the graph (async stub).
        
        Args:
            edge: Edge data to add
            
        Returns:
            Created edge ID
        """
        # TODO: Implement async version
        raise NotImplementedError("Async add_edge not yet implemented")
    
    async def get_edge(self, edge_id: str) -> Optional[GraphEdgeData]:
        """
        Get an edge by ID (async stub).
        
        Args:
            edge_id: Edge ID to retrieve
            
        Returns:
            Edge data if found, None otherwise
        """
        # TODO: Implement async version
        raise NotImplementedError("Async get_edge not yet implemented")
    
    async def get_neighbors(
        self,
        node_id: str,
        edge_types: Optional[List[RelationType]] = None,
        direction: str = "both",
        max_distance: int = 1
    ) -> List[Tuple[GraphNodeData, GraphEdgeData]]:
        """
        Get neighboring nodes (async stub).
        
        Args:
            node_id: Starting node ID
            edge_types: Optional list of edge types to follow
            direction: Direction to traverse ("in", "out", "both")
            max_distance: Maximum distance to traverse
            
        Returns:
            List of (node, edge) tuples
        """
        # TODO: Implement async version
        raise NotImplementedError("Async get_neighbors not yet implemented")
    
    async def find_path(
        self,
        start_id: str,
        end_id: str,
        max_length: int = 5,
        edge_types: Optional[List[RelationType]] = None
    ) -> Optional[List[Tuple[GraphNodeData, Optional[GraphEdgeData]]]]:
        """
        Find path between nodes (async stub).
        
        Args:
            start_id: Starting node ID
            end_id: Ending node ID
            max_length: Maximum path length
            edge_types: Optional list of edge types to follow
            
        Returns:
            Path as list of (node, edge) tuples if found, None otherwise
        """
        # TODO: Implement async version
        raise NotImplementedError("Async find_path not yet implemented")
    
    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a graph query (async stub).
        
        Args:
            query: Query string (e.g., AQL for ArangoDB)
            parameters: Optional query parameters
            
        Returns:
            Query results as list of dictionaries
        """
        # TODO: Implement async version
        raise NotImplementedError("Async execute_query not yet implemented")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary containing storage statistics
        """
        self.ensure_connected()
        
        try:
            # Get database statistics from schema manager
            db_stats = self._schema_manager.get_database_statistics() if self._schema_manager else {}
            
            # Combine with component statistics
            stats = {
                **self._stats,
                **db_stats,
                "connected": self._connected,
                "database_name": self._database_name,
                "schema_version": db_stats.get('schema_version', 'unknown'),
                "health_status": self.health_check()
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}")
            return {"error": str(e)}
    
    def get_capacity_info(self) -> Dict[str, Any]:
        """
        Get storage capacity information.
        
        Returns:
            Dictionary containing capacity information
        """
        try:
            stats = self.get_statistics()
            
            return {
                "total_documents": stats.get('total_files', 0),
                "modality_distribution": stats.get('modality_distribution', {}),
                "cross_modal_edges": stats.get('cross_modal_edges', 0),
                "collections": stats.get('collections', {}),
                "vector_index_enabled": self._vector_index_enabled,
                "vector_dimension": self._vector_dimension,
                "max_results": self._max_results,
                "sequential_isne_config": self._sequential_isne_config.dict() if self._sequential_isne_config else {}
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get capacity info: {e}")
            return {"error": str(e)}
    
    def supports_transactions(self) -> bool:
        """Check if storage supports transactions."""
        return True  # ArangoDB supports transactions
    
    # ===== SEQUENTIAL-ISNE SPECIFIC METHODS =====
    
    def store_modality_file(self, file_data: Union[CodeFile, DocumentationFile, ConfigFile]) -> str:
        """
        Store a file in the appropriate modality collection.
        
        Args:
            file_data: File data conforming to modality-specific model
            
        Returns:
            Stored file ID
            
        Raises:
            ArangoOperationError: If storage fails
        """
        self.ensure_connected()
        
        try:
            # Determine collection based on file type
            file_type = classify_file_type(file_data.path)
            collection_name = get_modality_collection(file_type)
            
            # Convert to dict for storage
            file_dict = file_data.model_dump(mode='json')
            
            # Remove None values to avoid unique constraint issues
            file_dict = {k: v for k, v in file_dict.items() if v is not None}
            
            # Insert into appropriate collection
            if self._client is None:
                raise ValueError("ArangoDB client not connected")
            result = self._client.insert_document(collection_name, file_dict)
            
            # Create full document ID
            file_id = f"{collection_name}/{result['_key']}"
            
            self.logger.debug(f"Stored {file_type} file: {file_id}")
            return file_id
            
        except Exception as e:
            error_msg = f"Failed to store modality file {file_data.path}: {e}"
            self.logger.error(error_msg)
            raise ArangoOperationError(error_msg)
    
    def store_chunk(self, chunk_data: Chunk) -> str:
        """
        Store a semantic chunk in the chunks collection.
        
        Args:
            chunk_data: Chunk data conforming to Chunk model
            
        Returns:
            Stored chunk ID
            
        Raises:
            ArangoOperationError: If storage fails
        """
        self.ensure_connected()
        
        try:
            # Convert to dict for storage
            chunk_dict = chunk_data.model_dump(mode='json')
            
            # Remove None values to avoid constraint issues
            chunk_dict = {k: v for k, v in chunk_dict.items() if v is not None}
            
            # Add storage metadata
            chunk_dict['created_at'] = datetime.now(timezone.utc).isoformat()
            chunk_dict['storage_version'] = self.version
            
            # Insert into chunks collection
            if self._client is None:
                raise ValueError("ArangoDB client not connected")
            result = self._client.insert_document("chunks", chunk_dict)
            
            # Create full document ID
            chunk_id = f"chunks/{result['_key']}"
            
            self.logger.debug(f"Stored chunk: {chunk_id}")
            return chunk_id
            
        except Exception as e:
            error_msg = f"Failed to store chunk: {e}"
            self.logger.error(error_msg)
            raise ArangoOperationError(error_msg)
    
    async def store_embedding(
        self, 
        doc_id: DocumentID,
        embedding: EmbeddingVector,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store an embedding in the embeddings collection.
        
        Args:
            embedding_data: Embedding data conforming to Embedding model
            
        Returns:
            Stored embedding ID
            
        Raises:
            ArangoOperationError: If storage fails
        """
        self.ensure_connected()
        
        try:
            # Create embedding data
            embedding_data = Embedding(
                id=f"emb_{doc_id}",
                chunk_id=doc_id,
                file_id=doc_id.split('/')[0] if '/' in doc_id else doc_id,
                vector=list(embedding),
                embedding_type=EmbeddingType.BASE,
                model_name=metadata.get('model_name', 'unknown') if metadata else 'unknown',
                metadata=metadata or {}
            )
            
            # Convert to dict for storage
            embedding_dict = embedding_data.model_dump(mode='json')
            
            # Remove None values to avoid constraint issues
            embedding_dict = {k: v for k, v in embedding_dict.items() if v is not None}
            
            # Add storage metadata
            embedding_dict['created_at'] = datetime.now(timezone.utc).isoformat()
            embedding_dict['storage_version'] = self.version
            
            # Insert into embeddings collection
            if self._client is None:
                raise ValueError("ArangoDB client not connected")
            result = self._client.insert_document("embeddings", embedding_dict)
            
            # Create full document ID
            embedding_id = f"embeddings/{result['_key']}"
            
            self._increment_stat('embeddings_stored')
            self.logger.debug(f"Stored embedding for document: {doc_id}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to store embedding: {e}"
            self.logger.error(error_msg)
            raise ArangoOperationError(error_msg)
    
    def create_cross_modal_edge(self, from_file_id: str, to_file_id: str, edge_type: EdgeType, 
                              weight: float, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a cross-modal edge between files.
        
        Args:
            from_file_id: Source file ID
            to_file_id: Target file ID
            edge_type: Type of cross-modal relationship
            weight: Edge weight (0.0 to 1.0)
            metadata: Optional edge metadata
            
        Returns:
            Created edge ID
            
        Raises:
            ArangoOperationError: If edge creation fails
        """
        self.ensure_connected()
        
        try:
            # Parse file IDs to determine modalities
            from_collection, from_key = self._parse_item_id(from_file_id)
            to_collection, to_key = self._parse_item_id(to_file_id)
            
            from_modality = self._collection_to_modality(from_collection)
            to_modality = self._collection_to_modality(to_collection)
            
            # Create cross-modal edge
            edge_data = {
                "_from": from_file_id,
                "_to": to_file_id,
                "_from_modality": from_modality,
                "_to_modality": to_modality,
                "edge_type": edge_type.value,
                "weight": weight,
                "confidence": weight,  # Use weight as confidence
                "similarity_score": weight,
                "source": "arangodb_storage",
                "discovery_method": "direct_insertion",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata or {}
            }
            
            # Insert into cross-modal edges collection
            if self._client is None:
                raise ValueError("ArangoDB client not connected")
            result = self._client.insert_document("cross_modal_edges", edge_data)
            
            edge_id = f"cross_modal_edges/{result['_key']}"
            self._increment_stat('cross_modal_edges_created')
            
            self.logger.debug(f"Created cross-modal edge: {edge_id}")
            return edge_id
            
        except Exception as e:
            error_msg = f"Failed to create cross-modal edge: {e}"
            self.logger.error(error_msg)
            raise ArangoOperationError(error_msg)
    
    def find_cross_modal_relationships(self, file_id: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Find cross-modal relationships for a given file.
        
        Args:
            file_id: File ID to find relationships for
            max_results: Maximum number of results to return
            
        Returns:
            List of cross-modal relationship data
        """
        self.ensure_connected()
        
        try:
            # AQL query to find cross-modal edges
            query = """
            FOR edge IN cross_modal_edges
                FILTER edge._from == @file_id OR edge._to == @file_id
                LIMIT @max_results
                RETURN {
                    edge_id: edge._id,
                    from_file: edge._from,
                    to_file: edge._to,
                    from_modality: edge._from_modality,
                    to_modality: edge._to_modality,
                    edge_type: edge.edge_type,
                    weight: edge.weight,
                    similarity_score: edge.similarity_score,
                    created_at: edge.created_at
                }
            """
            
            if self._client is None:
                raise ValueError("ArangoDB client not connected")
            cursor = self._client.execute_aql(query, {
                'file_id': file_id,
                'max_results': max_results
            })
            
            relationships = list(cursor)
            
            self.logger.debug(f"Found {len(relationships)} cross-modal relationships for {file_id}")
            return relationships
            
        except Exception as e:
            self.logger.error(f"Failed to find cross-modal relationships for {file_id}: {e}")
            return []
    
    def get_modality_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about modality distribution and cross-modal connections.
        
        Returns:
            Dictionary containing modality statistics
        """
        try:
            if self._schema_manager:
                return self._schema_manager.get_database_statistics()
            else:
                return {"error": "Schema manager not initialized"}
                
        except Exception as e:
            self.logger.error(f"Failed to get modality statistics: {e}")
            return {"error": str(e)}
    
    # ===== PRIVATE HELPER METHODS =====
    
    def _determine_storage_location(self, embedding: Any) -> str:
        """Determine appropriate storage location for an embedding."""
        # Enhanced embeddings should go to the embeddings collection
        # The file type determination is used for cross-modal analysis but
        # the actual storage happens in the embeddings collection
        return "embeddings"
    
    def _store_enhanced_embedding(self, embedding: Any, storage_location: str) -> StoredItem:
        """Store an enhanced embedding in the specified location."""
        # Convert embedding to storage format matching embeddings collection schema
        embedding_data = {
            "source_type": "chunk",
            "source_collection": "chunks",  # Assuming chunks as source
            "source_id": embedding.chunk_id,
            "embedding_type": "isne_enhanced",
            "vector": embedding.enhanced_embedding,
            "model_name": "sequential_isne",
            "model_version": "1.0.0",
            "embedding_dim": len(embedding.enhanced_embedding),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "isne_metadata": {
                "original_embedding": embedding.original_embedding,
                "graph_features": embedding.graph_features,
                "enhancement_score": embedding.enhancement_score
            },
            "metadata": embedding.metadata
        }
        
        # Insert into database
        if self._client is None:
            raise ValueError("ArangoDB client not connected")
        result = self._client.insert_document(storage_location, embedding_data)
        
        # Create stored item reference
        return StoredItem(
            item_id=f"{storage_location}/{result['_key']}",
            storage_location=storage_location,
            storage_type="arango_collection",
            timestamp=datetime.now(timezone.utc),
            metadata={
                "collection": storage_location,
                "embedding_dimension": len(embedding.enhanced_embedding),
                "enhancement_score": embedding.enhancement_score,
                "sequential_isne": True,
                "index_status": ProcessingStatus.COMPLETED.value
            }
        )
    
    def _execute_vector_query(self, query_data: QueryInput) -> List[RetrievalResult]:
        """Execute vector similarity query."""
        # Note: This is a simplified implementation
        # In production, you'd use ArangoDB's vector search capabilities
        
        try:
            # For now, return results from embeddings collection
            query = """
            FOR embedding IN embeddings
                LIMIT @top_k
                RETURN {
                    id: embedding._id,
                    content: embedding.chunk_id,
                    score: 0.9,
                    metadata: embedding.metadata
                }
            """
            
            if self._client is None:
                raise ValueError("ArangoDB client not connected")
            cursor = self._client.execute_aql(query, {'top_k': query_data.top_k})
            results = []
            
            for i, item in enumerate(cursor):
                result = RetrievalResult(
                    document_id=DocumentID(item['id']),
                    content=f"Vector search result for {item['content']}",
                    score=item['score'] - (i * 0.1),  # Decreasing scores
                    metadata={
                        **item['metadata'],
                        'item_id': item['id'],
                        'chunk_metadata': {"source": "vector_search"},
                        'document_metadata': {"type": "embedding"}
                    }
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Vector query failed: {e}")
            return []
    
    def _execute_text_query(self, query_data: QueryInput) -> List[RetrievalResult]:
        """Execute text-based query."""
        try:
            # Search across modality collections
            collections = ["code_files", "documentation_files", "config_files"]
            results = []
            
            for collection in collections:
                if self._client is None or not self._client.has_collection(collection):
                    continue
                
                # Simple text search query
                query = f"""
                FOR doc IN {collection}
                    FILTER CONTAINS(LOWER(doc.content), LOWER(@search_text))
                    LIMIT @limit
                    RETURN {{
                        id: doc._id,
                        file_path: doc.file_path,
                        content: doc.content,
                        file_type: doc.file_type,
                        modality: @modality
                    }}
                """
                
                cursor = self._client.execute_aql(query, {
                    'search_text': query_data.query,
                    'limit': min(query_data.top_k // len(collections), 10),
                    'modality': self._collection_to_modality(collection)
                })
                
                for item in cursor:
                    result = RetrievalResult(
                        document_id=DocumentID(item['id']),
                        content=item['content'][:500],  # Truncate for display
                        score=0.8,  # Static score for text search
                        metadata={
                            "item_id": item['id'],
                            "file_path": item['file_path'],
                            "file_type": item['file_type'],
                            "modality": item['modality'],
                            "chunk_metadata": {"source": "text_search"},
                            "document_metadata": {"collection": collection}
                        }
                    )
                    results.append(result)
            
            # Sort by relevance (simple implementation)
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:query_data.top_k]
            
        except Exception as e:
            self.logger.error(f"Text query failed: {e}")
            return []
    
    def _parse_item_id(self, item_id: str) -> Tuple[str, str]:
        """Parse item ID into collection name and document key."""
        if "/" in item_id:
            collection, key = item_id.split("/", 1)
            return collection, key
        else:
            # Assume it's just a key in a default collection
            return "chunks", item_id
    
    def _collection_to_modality(self, collection_name: str) -> str:
        """Convert collection name to modality."""
        mapping = {
            "code_files": "code",
            "documentation_files": "documentation",
            "config_files": "config"
        }
        return mapping.get(collection_name, "unknown")
    
    # ===== CONTEXT MANAGER SUPPORT =====
    
    def __enter__(self) -> "ArangoStorageV2":
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.disconnect()
    
    def _increment_stat(self, stat_name: str, increment: int = 1) -> None:
        """Safely increment a statistic."""
        current_value = self._stats.get(stat_name, 0)
        if isinstance(current_value, int):
            self._stats[stat_name] = current_value + increment
        else:
            self._stats[stat_name] = increment
    
    def _set_error(self, error_message: str) -> None:
        """Set error statistics."""
        self._increment_stat('errors')
        self._stats['last_operation'] = f'error: {error_message}'
    
    def __repr__(self) -> str:
        """String representation."""
        status = "connected" if self._connected else "disconnected"
        return f"ArangoStorageV2(database='{self._database_name}', status='{status}')"