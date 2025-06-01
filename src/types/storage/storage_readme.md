# Storage Module Type Definitions

This directory contains centralized type definitions for the HADES storage system, providing a comprehensive type system for storage-related operations.

## Overview

The storage types are organized into four main categories:

1. **Repository Types** - Interfaces and types for document, graph, and vector repositories
2. **Document Types** - Type definitions for document metadata, storage requests/responses
3. **Connection Types** - Types for database connections, pools, and transactions
4. **Query Types** - Types for query filters, sorting, pagination, and results

## Files

- `__init__.py` - Exports all type definitions
- `connection.py` - Database connection related types
- `document.py` - Document storage related types
- `repository.py` - Repository interface definitions
- `query.py` - Query related type definitions

## Type Definitions

### Repository Types

- `RepositoryInterface` - Base protocol for all repositories
- `DocumentRepository` - Protocol for document storage and retrieval
- `GraphRepository` - Protocol for graph operations
- `VectorRepository` - Protocol for vector operations
- `OperationResult` - Result type for operations that may fail

### Document Types

- `DocumentMetadata` - Metadata for documents
- `DocumentStorageRequest` - Request for storing documents
- `DocumentQueryFilter` - Filters for document queries
- `DocumentStorageResponse` - Response from document storage operations
- `DocumentRetrievalResponse` - Response from document retrieval operations
- `DocumentIndexConfig` - Configuration for document indexing
- `BulkOperationResult` - Results of bulk operations

### Connection Types

- `ConnectionState` - Enum for connection states
- `DatabaseType` - Enum for database types
- `ConnectionConfig` - Configuration for database connections
- `ConnectionCredentials` - Credentials for database connections
- `ConnectionInterface` - Protocol for database connections
- `ConnectionPoolInterface` - Protocol for connection pools
- `TransactionContext` - Protocol for transaction contexts

### Query Types

- `QueryFilter` - Type for query filters
- `SortOption` - Type for sort options
- `PaginationOptions` - Type for pagination options
- `QueryOptions` - Type for general query options
- `VectorQueryOptions` - Type for vector queries
- `HybridQueryOptions` - Type for hybrid queries
- `QueryResult` - Generic type for query results
- `VectorSearchResult` - Type for vector search results
- `HybridSearchResult` - Type for hybrid search results

## Usage Examples

### Defining a Repository Implementation

```python
from src.types.storage import DocumentRepository, DocumentMetadata, DocumentStorageResponse

class MyDocumentRepository(DocumentRepository):
    def store_document(self, document: NodeData) -> NodeID:
        # Implementation here
        return document_id
        
    def get_document(self, document_id: NodeID) -> Optional[NodeData]:
        # Implementation here
        return document_data
```

### Using Query Types

```python
from src.types.storage import QueryFilter, FilterOperator, SortDirection, SortOption

# Create a filter
filter_condition: QueryFilter = {
    "field": "document_type",
    "operator": FilterOperator.EQUALS,
    "value": "article"
}

# Create sort options
sort_option: SortOption = {
    "field": "created_at",
    "direction": SortDirection.DESC
}
```

## Testing

All types in this module are designed to be compatible with mypy static type checking. Ensure all implementations conform to the defined protocols.
