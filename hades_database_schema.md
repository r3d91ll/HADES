# HADES Database Schema

**Heuristic Adaptive Data Extrapolation System**

## Overview

HADES implements a simple yet powerful database schema designed for cross-domain knowledge discovery and adaptive performance optimization. The system starts with unified collections for maximum discovery capability and evolves into performance-optimized structures based on actual usage patterns.

## Core Philosophy

- **Start unified, evolve intelligently** - Begin with mixed collections, migrate to specialized collections based on real usage data
- **Anti-Windows performance model** - System gets faster over time as data self-organizes through usage patterns
- **LLM-driven interactions** - Schema designed for natural language querying and prompt-engineering
- **Project inheritance** - New projects can selectively inherit relevant knowledge from the main corpus

## Schema Design

### Source Collections (Metadata)

#### documents

```javascript
{
  _id: "documents/doc123",
  file_path: "/path/to/document.pdf",
  document_type: "pdf|md|txt|json|yaml",
  metadata: {
    title: "Research Paper Title",
    authors: ["Author 1", "Author 2"],
    created_at: "2025-01-15T10:00:00Z",
    page_count: 25,
    processing_version: "1.0"
  }
}
```

#### code_files

```javascript
{
  _id: "code_files/file456", 
  file_path: "/src/authentication/jwt_handler.py",
  language: "python",
  git_info: {
    hash: "abc123",
    last_modified: "2025-01-15T10:00:00Z",
    author: "developer@company.com"
  },
  metadata: {
    functions_count: 5,
    classes_count: 2,
    lines_of_code: 150,
    file_type: "source|config|test"
  }
}
```

### Main Working Collection

#### mixed_chunks

The primary collection where all content lives for optimal cross-domain discovery.

```javascript
{
  _id: "mixed_chunks/chunk789",
  content: "def validate_token(token): ...",
  embeddings: [0.1, 0.2, ...], // Vector embeddings
  
  // Source reference
  source_id: "code_files/file456",
  source_type: "document|code",
  
  // Chunk metadata
  chunk_type: "paragraph|function|class|section|block|comment",
  chunk_index: 0,
  position_info: {
    start_line: 10,
    end_line: 25,
    start_char: 250,
    end_char: 500
  },
  
  // Performance tracking
  access_count: 15,
  last_accessed: "2025-01-15T10:00:00Z",
  
  // Source-specific metadata
  document_metadata: {
    section_title: "Authentication Methods",
    page_num: 5
  },
  code_metadata: {
    function_name: "validate_token", 
    parameters: ["token"],
    return_type: "boolean",
    ast_data: {...}
  }
}
```

### Relationships Collection

#### relationships

ISNE-discovered connections between chunks across all domains.

```javascript
{
  _id: "relationships/rel123",
  _from: "mixed_chunks/chunk789",
  _to: "mixed_chunks/chunk456", 
  
  type: "similarity|conceptual|references|calls|imports",
  confidence: 0.87,
  context: "Both implement JWT validation patterns",
  
  // Discovery metadata
  isne_discovered: true,
  cross_domain: true, // code ↔ document relationship
  discovered_at: "2025-01-15T10:00:00Z"
}
```

### Project Collections (Dynamic)

#### project_{name}_chunks

Project-specific collections for isolated development with knowledge inheritance.

```javascript
{
  _id: "project_alpha_chunks/chunk999",
  content: "class AuthenticationService: ...",
  embeddings: [0.1, 0.2, ...],
  
  // Inheritance tracking
  inherited_from: "mixed_chunks/chunk789", // null if original
  project_context: "alpha",
  adaptation_notes: "Modified for microservices architecture",
  
  // Standard chunk fields
  source_type: "code",
  chunk_type: "class",
  // ... other mixed_chunks fields
}
```

## Usage Patterns

### Data Flow

```text
Documents → DocProc → Chunking → Embedding → ISNE → mixed_chunks → ArangoDB
```

### Performance Evolution

1. **Development Phase**: All chunks in mixed_chunks for maximum discovery
2. **Usage Pattern Emergence**: System tracks access patterns and query performance
3. **Adaptive Optimization**: Cold data migrates to dedicated collections (document_chunks, code_chunks)
4. **Mature State**: Self-organized hot/cold data distribution

### Cross-Domain Discovery

```javascript
// Find research papers related to current code implementation
FOR chunk IN mixed_chunks
  FILTER chunk.source_type == "document" 
  FOR related IN 1..2 ANY chunk relationships
    FILTER related.source_type == "code"
    FILTER related.content LIKE "%authentication%"
    RETURN {research: chunk.content, code: related.content}
```

### Project Knowledge Inheritance

```javascript
// Bootstrap new project with relevant authentication patterns
FOR chunk IN mixed_chunks
  FILTER chunk.chunk_type == "function"
  AND chunk.content LIKE "%auth%"
  AND chunk.access_count > 10
  INSERT {
    ...chunk,
    inherited_from: chunk._id,
    project_context: "new_microservice"
  } INTO project_new_microservice_chunks
```

## Performance Optimization Strategy

### Cold Storage Migration

```javascript
// Migrate rarely accessed chunks to dedicated collections
FOR chunk IN mixed_chunks
  FILTER chunk.access_count < 5 
  AND chunk.last_accessed < DATE_SUB(NOW(), 90, 'days')
  LET target_collection = chunk.source_type == "code" ? "code_chunks" : "document_chunks"
  // Move to appropriate cold storage collection
```

### Hot Data Optimization

- Frequently accessed chunks remain in mixed_chunks
- High-value cross-domain relationships stay in primary collection
- Cache optimization focused on mixed_chunks

## LLM Integration

### Natural Language Queries

The schema is designed for LLM-driven interactions:

```text
Human: "Find code patterns similar to our JWT implementation"
LLM: → ArangoDB query on mixed_chunks with ISNE similarity filtering
```

### Prompt Engineering Integration

Schema designed for natural language interface:

- Simple, consistent field names for LLM understanding
- Explicit relationship types for context building
- Metadata structure optimized for LLM consumption

## Implementation Notes

### ArangoDB Specific Features

- **Views**: Create virtual collections for cross-collection queries
- **Indexes**: Composite indexes on (source_type, chunk_type, access_count)
- **Graph traversal**: Native support for relationship following
- **Caching**: Leverage ArangoDB's memory caching for hot data

### ISNE Integration Points

- **Relationship Discovery**: ISNE populates relationships collection
- **Similarity Scoring**: ISNE confidence scores guide relationship strength
- **Cross-Domain Connections**: ISNE enables document ↔ code relationships

### Monitoring and Analytics

Track key metrics:

- Query response times by collection
- Cross-domain relationship usage
- Project inheritance patterns
- Cold storage migration effectiveness

## Future Considerations

### Schema Evolution

- Add new relationship types as discovered by ISNE
- Extend metadata fields based on domain requirements
- Create specialized project collection templates

### Performance Scaling

- Horizontal partitioning of mixed_chunks by domain
- Dedicated read replicas for analytical queries
- Archive collections for historical data

### AI/ML Integration

- Store model training metadata alongside chunks
- Track feature importance for relationship scoring
- Version control for embedding model updates

---

**Key Principle**: Start simple, evolve intelligently based on real usage patterns. The schema serves the discovery process, not the other way around.
