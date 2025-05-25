# Text Storage Module

This module provides integration between text processing pipelines and ArangoDB storage.

## Overview

The Text Storage module is responsible for:

1. Storing processed text documents in ArangoDB
2. Managing document chunks and their embeddings
3. Creating and maintaining relationships between documents
4. Supporting various search capabilities including vector search, full-text search, and hybrid search

## Architecture

The module follows a layered architecture:

1. **Text Storage Service** - High-level interface for storing processed documents
2. **Text Arango Repository** - Implementation of the repository interface for ArangoDB
3. **Storage Models** - Type-safe data models for document storage

## Components

### TextStorageService

The main service class that:

- Takes document processing pipeline output and stores it in ArangoDB
- Handles document, chunk, and embedding storage with full type safety
- Creates relationships between documents based on similarity
- Supports different embedding types including ISNE embeddings

### DocumentMapper

Translates between document processing pipeline output and ArangoDB storage models:

- Maps documents to nodes
- Maps chunks to nodes
- Maps embeddings to vector fields
- Creates appropriate edges between nodes

## Storage Schema

### Collections

- **nodes** - Stores documents and chunks
- **edges** - Stores relationships between nodes

### Node Types

- **document** - A complete PDF document
- **chunk** - A section/chunk of a document

### Edge Types

- **contains** - Document to chunk relationship
- **similar_to** - Chunk to chunk relationship based on embedding similarity
- **references** - Cross-document reference relationship

## Usage Example

```python
from src.storage.text_storage import TextStorageService
from src.storage.arango.connection import ArangoConnection

# Initialize connection
connection = ArangoConnection(db_name="hades", host="http://localhost:8529")

# Create storage service
storage_service = TextStorageService(connection)

# Store processed document
async def store_document(document_json):
    await storage_service.store_processed_document(document_json)
```

## Performance Considerations

- Bulk operations are used for efficient storage of embeddings
- Dedicated ArangoDB indexes for text search and vector search
- Pre-computed similarity relationships to speed up retrieval
- Parameter tuning for optimal vector search performance

## Testing

The module includes:
- Unit tests for all components with 85% code coverage for the TextStorageService
- Mock repositories for dependency isolation in testing
- Edge case handling for empty vectors, missing chunks, etc.
- Integration tests with ArangoDB

## Type Safety

The module implements comprehensive type safety:

- TypedDict classes for all data structures
- Consistent type annotations across all methods
- Repository interfaces with well-defined method signatures
- Runtime type checking for critical operations

## Search Capabilities

The module provides three primary search methods:

1. **Vector Search** - Find documents by embedding similarity
   - Supports different embedding types
   - Configurable similarity thresholds
   - Optional filtering by document type

2. **Full-text Search** - Find documents by text content
   - Leverages ArangoDB's full-text indexing
   - Supports exact phrase matching
   - Optional filtering by metadata

3. **Hybrid Search** - Combines vector and text search
   - Weighted combination of text and vector results
   - Configurable weights for each component
   - Improved relevance by leveraging both semantic and lexical matching
