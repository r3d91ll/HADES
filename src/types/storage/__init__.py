"""Storage type definitions.

This module provides centralized type definitions for the storage system,
including repository types, storage metadata, and data access patterns.
"""

# Import repository types
from src.types.storage.repository import (
    OperationResult,
    ConnectionConfigType,
    DatabaseCredentials,
    ConnectionInterface,
    RepositoryInterface,
    DocumentRepository,
    GraphRepository,
    VectorRepository
)

# Import document types
from src.types.storage.document import (
    DocumentFormat,
    DocumentStatus,
    DocumentMetadata,
    DocumentStorageRequest,
    DocumentQueryFilter,
    DocumentStorageResponse,
    DocumentRetrievalResponse,
    DocumentIndexConfig,
    BulkOperationResult
)

# Import connection types
from src.types.storage.connection import (
    ConnectionState,
    DatabaseType,
    ConnectionConfig,
    ConnectionCredentials,
    ConnectionError,
    TransactionOptions,
    TransactionContext,
    ConnectionInterface,
    ConnectionPoolInterface,
    ConnectionFactory,
    TransactionCallback,
    ConnectionResult
)

# Import query types
from src.types.storage.query import (
    SortDirection,
    FilterOperator,
    FilterCondition,
    CompoundFilter,
    QueryFilter,
    SortOption,
    PaginationOptions,
    QueryOptions,
    VectorQueryOptions,
    HybridQueryOptions,
    TraversalQueryOptions,
    ShortestPathOptions,
    QueryResult,
    VectorSearchResult,
    HybridSearchResult,
    TraversalResult,
    PathResult
)

__all__ = [
    # Repository types
    "OperationResult",
    "ConnectionConfigType",
    "DatabaseCredentials",
    "ConnectionInterface",
    "RepositoryInterface",
    "DocumentRepository",
    "GraphRepository",
    "VectorRepository",
    
    # Document types
    "DocumentFormat",
    "DocumentStatus",
    "DocumentMetadata",
    "DocumentStorageRequest",
    "DocumentQueryFilter",
    "DocumentStorageResponse",
    "DocumentRetrievalResponse",
    "DocumentIndexConfig",
    "BulkOperationResult",
    
    # Connection types
    "ConnectionState",
    "DatabaseType",
    "ConnectionConfig",
    "ConnectionCredentials",
    "ConnectionError",
    "TransactionOptions",
    "TransactionContext",
    "ConnectionInterface",
    "ConnectionPoolInterface",
    "ConnectionFactory",
    "TransactionCallback",
    "ConnectionResult",
    
    # Query types
    "SortDirection",
    "FilterOperator",
    "FilterCondition",
    "CompoundFilter",
    "QueryFilter",
    "SortOption",
    "PaginationOptions",
    "QueryOptions",
    "VectorQueryOptions",
    "HybridQueryOptions",
    "TraversalQueryOptions",
    "ShortestPathOptions",
    "QueryResult",
    "VectorSearchResult",
    "HybridSearchResult",
    "TraversalResult",
    "PathResult"
]
