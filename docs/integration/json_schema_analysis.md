# JSON Schema Analysis for HADES Pipeline

## Overview

This document analyzes the actual JSON output structure from our document processing and chunking pipeline stages, based on testing with 2 PDF documents (ISNE and PathRAG research papers). This analysis provides the foundation for designing the storage component.

## Pipeline Stage Outputs

### Stage 1: Document Processing (DocProc)

**Output**: List of `DocumentSchema` objects (as string representations in JSON)

**Structure Analysis**:
```json
[
  "file_id='69c49d59-cd70-4a3a-b6f7-3b2fff9d0e2c' file_path='/path/to/file.pdf' file_name='filename.pdf' file_type='pdf' content_type='application/pdf' chunks=[] metadata={'size_bytes': 509475, 'last_modified': 1749268221.7250228, 'processor': 'document_processor', 'content_length': 61043, 'format_detected': 'pdf', 'content': '...'} processed_at=datetime(...) validation=ValidationResult(...)"
]
```

**Key Fields Extracted**:
- `file_id`: UUID for document identification  
- `file_path`: Absolute path to source document
- `file_name`: Original filename
- `file_type`: File extension (pdf, py, etc.)
- `content_type`: MIME type
- `metadata.size_bytes`: File size in bytes
- `metadata.content_length`: Extracted text length
- `metadata.content`: Full extracted text content
- `processed_at`: Processing timestamp
- `validation`: Validation results

**Issues Observed**:
- Missing `source` field (validation warning)
- Missing `metadata.format` field (validation warning)
- Documents saved as Pydantic object string representations, not pure JSON

### Stage 2: Chunking

**Output**: List of chunk dictionaries (proper JSON)

**Structure Analysis**:
```json
[
  {
    "id": "69c49d59-cd70-4a3a-b6f7-3b2fff9d0e2c_chunk_0",
    "text": "## ORIGINAL ARTICLE\n\n## Unsupervised Graph Representation Learning...",
    "source_document": "Unsupervised Graph Representation Learning with Inductive Shallow Node Embedding.pdf",
    "metadata": {
      "index": 0,
      "document_id": "69c49d59-cd70-4a3a-b6f7-3b2fff9d0e2c",
      "document_name": "filename.pdf",
      "chunking_strategy": "paragraph",
      "chunker_type": "cpu",
      "symbol_type": "paragraph",
      "name": "paragraph_0",
      "path": "/path/to/source/file.pdf",
      "token_count": 1496,
      "content_hash": "769cfc6d744782abbb4a1a28eed208c5"
    }
  }
]
```

**Key Fields**:
- `id`: Unique chunk identifier (format: `{document_id}_chunk_{index}`)
- `text`: Extracted chunk content
- `source_document`: Original filename for traceability
- `metadata.index`: Chunk position within document
- `metadata.document_id`: Parent document UUID
- `metadata.chunking_strategy`: Strategy used (paragraph, ast, etc.)
- `metadata.chunker_type`: Chunker implementation (cpu, gpu, etc.)
- `metadata.token_count`: Token count for the chunk
- `metadata.content_hash`: Hash for deduplication

## Test Results Summary

### Corpus Statistics
- **Total Documents**: 2 PDF files
- **Total Chunks Created**: 50 chunks
- **Average Chunk Length**: ~1,400 characters  
- **Processing Success**: 100% (both documents processed successfully)

### Chunk Distribution
- **ISNE Paper**: 26 chunks
- **PathRAG Paper**: 24 chunks  
- **Chunk Size Range**: 500-2,000 characters typically
- **Token Count Range**: 100-1,500 tokens per chunk

### Performance Metrics
- **Document Processing Time**: ~13-15 seconds per PDF (GPU-accelerated Docling)
- **Chunking Time**: ~15-20 seconds per document (CPU chunking with ModernBERT)
- **Total Pipeline Time**: ~60 seconds for 2 documents

## Schema Design Recommendations

### 1. Document Schema for Storage

```json
{
  "id": "uuid",
  "file_path": "string",
  "file_name": "string", 
  "file_type": "string",
  "content_type": "string",
  "source": "string",  // Missing in current output
  "content": "string",
  "metadata": {
    "size_bytes": "integer",
    "last_modified": "timestamp",
    "content_length": "integer",
    "format": "string",  // Missing in current output
    "processor": "string",
    "format_detected": "string"
  },
  "processed_at": "timestamp",
  "validation": {
    "is_valid": "boolean",
    "issues": ["array"],
    "timestamp": "timestamp",
    "stage_name": "string"
  }
}
```

### 2. Chunk Schema for Storage

```json
{
  "id": "string",
  "text": "string",
  "source_document": "string",
  "embedding": "array<float>",  // To be added by embedding stage
  "isne_embedding": "array<float>",  // To be added by ISNE stage
  "metadata": {
    "index": "integer",
    "document_id": "uuid",
    "document_name": "string", 
    "chunking_strategy": "string",
    "chunker_type": "string",
    "symbol_type": "string",
    "name": "string",
    "path": "string",
    "token_count": "integer",
    "content_hash": "string"
  },
  "relationships": {  // To be added for graph construction
    "similar_chunks": ["array<chunk_id>"],
    "sequential_chunks": ["array<chunk_id>"],
    "cross_document_links": ["array<chunk_id>"]
  }
}
```

## Next Pipeline Stages

### Stage 3: Embedding (Missing Implementation)

**Expected Output**: Chunks with base embeddings added
```json
{
  // ... existing chunk fields
  "embedding": [0.1, 0.2, 0.3, ...],  // 768-dim ModernBERT embedding
  "embedding_metadata": {
    "model": "modernbert",
    "dimension": 768,
    "timestamp": "2025-06-07T11:01:32Z"
  }
}
```

### Stage 4: ISNE Enhancement  

**Expected Output**: Chunks with ISNE-enhanced embeddings
```json
{
  // ... existing chunk fields + base embedding
  "isne_embedding": [0.15, 0.25, 0.35, ...],  // Enhanced embedding
  "isne_metadata": {
    "model_version": "v1",
    "training_timestamp": "2025-06-07T11:01:32Z",
    "neighborhood_size": 5,
    "enhancement_quality": 0.85
  },
  "graph_relationships": {
    "neighbors": ["chunk_id_1", "chunk_id_2"],
    "similarity_scores": [0.8, 0.75],
    "edge_types": ["semantic", "sequential"]
  }
}
```

### Stage 5: Storage

**Expected Operations**:
- Insert documents into ArangoDB document collection
- Insert chunks into ArangoDB document collection  
- Create edges between related chunks in ArangoDB edge collection
- Index embeddings for vector similarity search
- Create metadata indices for efficient querying

## Storage Component Design

### ArangoDB Collections

1. **Documents Collection**: `documents`
   - Stores full document metadata and content
   - Primary key: `file_id`
   - Indices: `file_name`, `file_type`, `processed_at`

2. **Chunks Collection**: `chunks`
   - Stores individual chunk data with embeddings
   - Primary key: `id` 
   - Indices: `document_id`, `content_hash`, `token_count`

3. **Chunk Relationships Collection**: `chunk_relationships` (Edge Collection)
   - Stores relationships between chunks
   - Schema: `{_from: chunk_id, _to: chunk_id, type: string, weight: float}`
   - Types: `sequential`, `semantic`, `cross_document`

4. **Embeddings Index**: Vector search index on `embedding` and `isne_embedding` fields

### JSON Schema Validation

The storage component should validate input JSON against the schemas above and handle:
- Missing required fields gracefully
- Type validation and conversion
- Embedding dimension validation
- Relationship integrity checks

## Recommendations for Storage Implementation

1. **Fix Document Processing Issues**:
   - Add missing `source` field in document schema
   - Add missing `metadata.format` field
   - Ensure proper JSON serialization instead of Pydantic object strings

2. **Implement Embedding Stage**:
   - Add ModernBERT embedding generation
   - Include embedding metadata for traceability
   - Validate embedding dimensions (768 for ModernBERT)

3. **Storage Component Features**:
   - Batch insertion for efficiency
   - Duplicate detection using content hashes
   - Incremental updates for dynamic ingestion
   - Vector similarity search capabilities
   - Graph traversal queries for PathRAG

4. **Quality Assurance**:
   - Schema validation at ingestion time
   - Embedding quality metrics
   - Relationship consistency checks
   - Performance monitoring and optimization

This analysis provides the foundation for implementing the storage component with proper JSON schema handling and ArangoDB integration.