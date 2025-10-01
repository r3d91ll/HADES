# HADES Experiential Memory Schema v1.0

**Author:** HADES team
**Date:** 2025-09-29
**Status:** Design specification

## Executive Summary

This document specifies the schema for HADES experiential memory system, which stores agent observations, reflections, entities, and their relationships.

**Prompt engineering patterns inspired by:**
<https://github.com/doobidoo/mcp-memory-service>

**Infrastructure powered by HADES stack:**

- ArangoDB graph database (Unix socket connections)
- PathRAG retrieval with graph traversal
- Jina v4 embeddings (32k context, 2048 dimensions)

**Implementation:** Local Python library - no network services, no MCP protocol.

## Design Principles

1. **Graph-native**: Use ArangoDB's native graph capabilities for entity relationships
2. **Embedding-first**: Every memory has Jina v4 embeddings for semantic search
3. **Time-aware**: All memories timestamped for temporal queries
4. **Type-safe**: Strict memory_type field for filtering and routing
5. **PathRAG-ready**: Schema designed for efficient graph traversal with caps

## Collection Specifications

### 1. observations (document collection)

Raw experiences/events witnessed by the agent.

```javascript
{
  "_key": "obs_2025-09-29T12:34:56_abc123",  // timestamp + uuid suffix
  "_id": "observations/obs_2025-09-29T12:34:56_abc123",

  // Content
  "content": "Fixed race condition in authentication by adding mutex locks",
  "content_hash": "sha256:...",  // for deduplication

  // Classification
  "memory_type": "observation",  // observation|decision|learning|event
  "tags": ["debugging", "concurrency", "authentication"],

  // Vector embedding (Jina v4, 2048 dimensions)
  "embedding": [0.123, -0.456, ...],  // 2048 floats

  // Temporal context
  "created_at": "2025-09-29T12:34:56.789Z",
  "created_at_unix": 1727612096.789,

  // Session/conversation context
  "session_id": "chat_abc123",
  "turn_index": 42,

  // Metadata
  "metadata": {
    "project": "HADES",
    "component": "auth_system",
    "confidence": 0.9,
    "source": "user_statement",  // user_statement|agent_inference|code_analysis
    "context_window": 5000  // tokens of context when stored
  },

  // Lifecycle
  "updated_at": "2025-09-29T13:00:00.000Z",
  "revision": 1,
  "status": "active"  // active|archived|superseded
}
```

**Indexes:**

- `content_hash` (unique, sparse)
- `created_at_unix` (persistent)
- `session_id` (persistent)
- `tags[*]` (persistent)
- `memory_type` (persistent)
- `embedding` (vector index, dimensions=2048, metric=cosine)

### 2. reflections (document collection)

Higher-order insights synthesized from multiple observations.

```javascript
{
  "_key": "refl_2025-09-29T15:00:00_xyz789",
  "_id": "reflections/refl_2025-09-29T15:00:00_xyz789",

  // Content
  "content": "Authentication bugs in multi-threaded systems often involve race conditions that can be mitigated with proper mutex usage",
  "content_hash": "sha256:...",

  // Classification
  "memory_type": "reflection",  // reflection|pattern|principle|insight
  "tags": ["pattern", "concurrency", "architecture", "authentication"],

  // Vector embedding
  "embedding": [0.234, -0.567, ...],  // 2048 floats

  // Temporal context
  "created_at": "2025-09-29T15:00:00.000Z",
  "created_at_unix": 1727620800.000,

  // Source observations
  "consolidates": [
    "observations/obs_2025-09-29T12:34:56_abc123",
    "observations/obs_2025-09-28T10:20:30_def456"
  ],
  "source_count": 2,

  // Quality metrics
  "metadata": {
    "strength": 0.85,  // how well-supported by evidence
    "revision": 2,     // updated as more evidence accrues
    "method": "llm_synthesis",  // llm_synthesis|rule_based|manual
    "model": "Qwen3-30B-A3B-Thinking-2507"
  },

  // Lifecycle
  "updated_at": "2025-09-29T16:00:00.000Z",
  "status": "active"
}
```

**Indexes:**

- `content_hash` (unique, sparse)
- `created_at_unix` (persistent)
- `tags[*]` (persistent)
- `memory_type` (persistent)
- `consolidates[*]` (persistent)
- `embedding` (vector index, dimensions=2048, metric=cosine)

### 3. entities (document collection)

People, components, concepts, projects extracted from experiences.

```javascript
{
  "_key": "ent_component_arango_client",
  "_id": "entities/ent_component_arango_client",

  // Identity
  "name": "ArangoDB HTTP/2 Client",
  "aliases": ["arango client", "database client", "optimized client"],

  // Classification
  "entity_type": "component",  // person|component|concept|project|tool
  "tags": ["infrastructure", "database", "performance"],

  // Description
  "description": "Optimized ArangoDB client using HTTP/2 over Unix sockets for sub-millisecond latency",
  "embedding": [0.345, -0.678, ...],  // 2048 floats

  // Temporal context
  "first_seen": "2025-09-20T10:00:00.000Z",
  "last_updated": "2025-09-29T16:30:00.000Z",
  "mention_count": 45,  // how many times referenced

  // Metadata
  "metadata": {
    "importance": 0.95,  // derived from mention frequency + centrality
    "certainty": 0.9,    // confidence in entity extraction
    "source": "codebase_analysis"
  },

  // Lifecycle
  "status": "active"
}
```

**Indexes:**

- `name` (fulltext)
- `entity_type` (persistent)
- `tags[*]` (persistent)
- `mention_count` (persistent)
- `embedding` (vector index, dimensions=2048, metric=cosine)

### 4. relationships (edge collection)

Graph edges connecting entities and memories.

```javascript
{
  "_key": "rel_abc123",
  "_id": "relationships/rel_abc123",
  "_from": "entities/ent_component_auth",
  "_to": "entities/ent_component_arango_client",

  // Relationship semantics
  "relationship_type": "uses",  // uses|causes|relates_to|improves|conflicts_with
  "strength": 0.9,  // 0.0-1.0, derived from evidence

  // Evidence trail
  "evidence": [
    "observations/obs_2025-09-29T12:34:56_abc123",
    "reflections/refl_2025-09-29T15:00:00_xyz789"
  ],
  "evidence_count": 2,

  // Temporal context
  "created_at": "2025-09-29T16:45:00.000Z",
  "last_reinforced": "2025-09-29T17:00:00.000Z",

  // Metadata
  "metadata": {
    "confidence": 0.85,
    "method": "co_occurrence",  // co_occurrence|causal_inference|explicit_statement
    "decay_rate": 0.05  // how fast strength degrades without reinforcement
  },

  // Lifecycle
  "status": "active"
}
```

**Indexes:**

- `_from` (edge index, automatic)
- `_to` (edge index, automatic)
- `relationship_type` (persistent)
- `strength` (persistent)
- `created_at` (persistent)

### 5. sessions (document collection)

Conversation boundaries for contextual grouping.

```javascript
{
  "_key": "chat_abc123",
  "_id": "sessions/chat_abc123",

  // Identity
  "session_name": "Bug fix in auth system - 2025-09-29",
  "session_type": "debugging",  // debugging|design|learning|exploration

  // Temporal context
  "started_at": "2025-09-29T12:00:00.000Z",
  "ended_at": null,  // null if ongoing
  "duration_seconds": null,

  // Memory statistics
  "stats": {
    "observations_count": 15,
    "reflections_count": 2,
    "entities_extracted": 8,
    "turns_count": 50
  },

  // Metadata
  "metadata": {
    "project": "HADES",
    "user_id": "todd",
    "agent_version": "v0.1.0"
  },

  // Lifecycle
  "status": "active"  // active|completed|archived
}
```

**Indexes:**

- `started_at` (persistent)
- `session_type` (persistent)
- `status` (persistent)

## Graph Traversal Patterns

### Pattern 1: Observation → Entity → Related Observations

```text
Query: "What have we learned about authentication?"

1. Vector search: observations matching "authentication" (K=6)
2. Extract entities: observations -> entities (via relationships)
3. Expand: entities -> related observations (H=1, F=3, B=8)
4. Include: observations -> reflections (consolidates edge)
```

### Pattern 2: Entity-Centric Memory Retrieval

```text
Query: "Tell me about the ArangoDB client"

1. Entity search: find entity "ArangoDB HTTP/2 Client"
2. Inbound edges: observations/reflections -> entity (mentions)
3. Outbound edges: entity -> related entities (uses, improves)
4. PathRAG: traverse with caps (H=2, F=3, B=8)
```

### Pattern 3: Temporal Context Reconstruction

```text
Query: "What did we do last week?"

1. Time filter: observations WHERE created_at >= last_week
2. Session grouping: GROUP BY session_id
3. Entity extraction: linked entities per session
4. Reflection lookup: consolidating reflections
```

## PathRAG Integration

### Query Plan for Memory Retrieval

```json
{
  "q": "authentication race conditions",
  "retriever": {
    "collection": "observations",
    "k": 6,
    "mmr": 0.4,
    "time_filter": "last_month",
    "tags_filter": ["authentication", "debugging"],
    "timeout_ms": 200
  },
  "graph": {
    "collections": ["observations", "reflections", "entities"],
    "edge_collections": ["relationships", "consolidates"],
    "hops": 2,
    "fanout": 3,
    "beam": 8,
    "timeout_ms": 500
  },
  "reranker": {
    "m": 20,
    "timeout_ms": 500
  },
  "final_context": 8,
  "budget": {
    "docs_max": 160,
    "latency_ms": 2000
  }
}
```

### Edge Collection for Consolidation

```javascript
// Implicit edge: observation -> reflection (via "consolidates" array)
// Can be queried with:
FOR obs IN observations
  FOR refl IN reflections
    FILTER obs._id IN refl.consolidates
    RETURN {observation: obs, reflection: refl}
```

## Migration Path

### Phase 1: Schema Creation (current task)

1. Create new collections with indexes
2. Ensure vector indexes on embedding fields
3. Create graph definition linking collections

### Phase 2: Dual-Write Period

1. Write to both old (messages/recall_log) and new schemas
2. Validate data consistency
3. Test PathRAG on new schema

### Phase 3: Migration Script

1. Convert existing messages → observations
2. Generate initial entities from observations
3. Infer relationships from co-occurrence
4. Deprecate old collections

### Phase 4: Consolidation Engine

1. Batch process observations → reflections
2. Entity extraction and linking
3. Relationship inference and scoring

## Performance Targets

- **Store operation**: p99 < 50ms (write + embedding)
- **Retrieve operation**: p99 < 250ms (vector search + PathRAG)
- **Entity extraction**: p99 < 100ms per observation
- **Consolidation**: ~1-5s per reflection (model inference)
- **Graph traversal**: Meet existing PathRAG SLOs (2s global)

## Security & Access Control

- **RO socket**: Vector search, graph traversal (read-only)
- **RW socket**: Store observations, update entities, create relationships
- **Admin socket**: Schema changes, index management
- **Permissions**: All operations require session_id validation

## Comparison with mcp-memory-service

This implementation borrows prompt engineering patterns from mcp-memory-service
but uses completely different infrastructure and capabilities.

### Operation Naming (inspired by mcp-memory-service)

| Their Operation | Our Operation | Notes |
|----------------|---------------|-------|
| `store_memory` | `store_observation()` | Same concept, our terminology |
| `retrieve_memory` | `retrieve_memories()` | PathRAG-powered graph traversal |
| `search_by_tag` | `search_by_tag()` | Similar implementation |
| `delete_memory` | `archive_observation()` | We use soft delete |
| `list_memories` | `list_observations()` | Same concept |

### Our Enhanced Capabilities (vs. mcp-memory-service)

1. **Local library**: In-process operations, no network/MCP protocol
2. **Graph database**: ArangoDB vs SQLite-vec
3. **Graph traversal**: PathRAG (H≤2, F≤3, B≤8) vs simple vector search
4. **Advanced embeddings**: Jina v4 (32k context, 2048d) vs generic
5. **Entity extraction**: Automatic NER + knowledge graph linking
6. **Reflection synthesis**: LLM-powered (Qwen3) consolidation
7. **Relationship inference**: Graph-based pattern detection
8. **Performance**: Unix socket connections, sub-millisecond queries

## Next Steps

1. Implement models in `core/runtime/memory/models.py`
2. Create schema provisioning in `memory_store.py`
3. Implement operations in `core/runtime/memory/operations.py`
4. Adapt PathRAG for memory graph in `pathrag.py`
5. Write consolidation engine in `consolidation.py`
