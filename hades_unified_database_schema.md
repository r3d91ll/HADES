# HADES Unified Database Schema
**Heuristic Adaptive Data Extrapolation System**

## Executive Summary

HADES implements a unified, adaptive database schema that combines the simplicity of mixed-collection discovery with the operational intelligence of incremental processing. The system starts with unified collections for maximum cross-domain discovery capability and evolves into performance-optimized structures based on actual usage patterns, while maintaining full incremental update capabilities and model version tracking.

## Core Philosophy

- **Start unified, evolve intelligently** - Begin with mixed collections, migrate to specialized collections based on real usage data
- **Anti-Windows performance model** - System gets faster over time as data self-organizes through usage patterns
- **Incremental by design** - Only process new or changed content via content hashing
- **Model evolution tracking** - Full lineage tracking for ISNE model versions
- **LLM-driven interactions** - Schema designed for natural language querying and prompt-engineering
- **Project inheritance** - New projects can selectively inherit relevant knowledge from the main corpus
- **Operational intelligence** - Built-in rollback, batch processing, and consistency guarantees

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                   HADES Unified System                     │
├─────────────────────────────────────────────────────────────┤
│  Document Ingestion Layer                                   │
│  ├── Incremental Detector (content hashing)                │
│  ├── Change Analyzer                                        │
│  └── Batch Processor                                        │
├─────────────────────────────────────────────────────────────┤
│  ArangoDB Unified Storage Layer                             │
│  ├── Source Collections (metadata)                         │
│  ├── Mixed Chunks Collection (primary working data)        │
│  ├── Relationships Collection (ISNE-discovered)            │
│  ├── Model Versions Collection (ISNE evolution)            │
│  ├── Processing Batches Collection (operational)           │
│  └── Project Collections (dynamic inheritance)             │
├─────────────────────────────────────────────────────────────┤
│  ISNE Training Layer                                        │
│  ├── Incremental Trainer                                    │
│  ├── Model Version Manager                                  │
│  ├── Graph Population Engine                               │
│  └── Graph Consistency Engine                              │
├─────────────────────────────────────────────────────────────┤
│  Query & Retrieval Layer                                   │
│  ├── PathRAG Query Engine                                   │
│  ├── Cross-Domain Discovery                                │
│  └── Adaptive Performance Optimization                     │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow Pipeline

```text
Documents → Content Hash → Change Detection → Chunking → Embedding → ISNE Enhancement → mixed_chunks → ArangoDB
                ↓                                    ↓                      ↓                    ↓
         Batch Tracking ←←←←←←←←← ISNE Training ←←←←←←←←←←←←←←←←←←←←←← Graph Population ←←←←←←←←
                                      ↓
                              Model Versioning
                                      ↓
                            Relationship Discovery
                                      ↓
                              ArangoDB Relationships
```

## Database Schema Design

### 1. Source Collections (Metadata)

#### 1.1 documents

Stores document-level metadata with incremental processing support.

```javascript
{
  _id: "documents/doc_<hash>",
  file_path: "/path/to/document.pdf",
  document_type: "pdf|md|txt|json|yaml",
  
  // Content tracking
  content_hash: "sha256_hash_of_content",
  previous_hash: null, // For change detection
  
  // Processing status
  processing_status: "pending|processing|completed|failed",
  ingestion_batch_id: "batch_12345",
  last_processed: "2025-01-15T10:00:00Z",
  
  // Metadata
  metadata: {
    title: "Research Paper Title",
    authors: ["Author 1", "Author 2"],
    created_at: "2025-01-15T10:00:00Z",
    page_count: 25,
    processing_version: "1.0"
  },
  
  // Versioning
  version: 1,
  change_type: "new|updated|deleted",
  parent_document_key: null
}
```

#### 1.2 code_files

Stores code file metadata with Git integration and incremental tracking.

```javascript
{
  _id: "code_files/file_<hash>", 
  file_path: "/src/authentication/jwt_handler.py",
  language: "python",
  
  // Content tracking
  content_hash: "sha256_hash_of_content",
  previous_hash: null,
  
  // Git integration
  git_info: {
    hash: "abc123",
    last_modified: "2025-01-15T10:00:00Z",
    author: "developer@company.com",
    branch: "main"
  },
  
  // Processing status
  processing_status: "pending|processing|completed|failed",
  ingestion_batch_id: "batch_12345",
  
  // Code metadata
  metadata: {
    functions_count: 5,
    classes_count: 2,
    lines_of_code: 150,
    file_type: "source|config|test",
    ast_complexity: 45
  }
}
```

### 2. Main Working Collection

#### 2.1 mixed_chunks

The primary collection where all content lives for optimal cross-domain discovery, enhanced with incremental processing capabilities.

```javascript
{
  _id: "mixed_chunks/chunk_<hash>_<index>",
  content: "def validate_token(token): ...",
  embeddings: [0.1, 0.2, ...], // Vector embeddings
  
  // Source reference
  source_id: "documents/doc_<hash>",
  source_type: "document|code",
  
  // Content tracking
  content_hash: "sha256_hash_of_chunk",
  previous_version_id: null, // For version tracking
  
  // Chunk metadata
  chunk_type: "paragraph|function|class|section|block|comment",
  chunk_index: 0,
  position_info: {
    start_line: 10,
    end_line: 25,
    start_char: 250,
    end_char: 500
  },
  
  // Processing pipeline status
  pipeline_status: {
    embedding: "completed",
    isne_enhancement: "completed",
    embedding_model: "ModernBERT",
    isne_model_version: "1.0.0",
    last_updated: "2025-01-15T10:00:00Z"
  },
  
  // Incremental processing
  processing_batch_id: "batch_123",
  requires_reprocessing: false,
  
  // Graph readiness
  graph_ready: true,
  graph_update_needed: false,
  
  // Performance tracking
  access_count: 15,
  last_accessed: "2025-01-15T10:00:00Z",
  performance_tier: "hot|warm|cold",
  
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

### 3. Graph and Relationships

#### 3.1 relationships

ISNE-discovered connections between chunks across all domains with versioning support.

```javascript
{
  _id: "relationships/rel_<hash>",
  _from: "mixed_chunks/chunk_<hash1>",
  _to: "mixed_chunks/chunk_<hash2>", 
  
  // Relationship metadata
  type: "similarity|conceptual|references|calls|imports|inheritance",
  confidence: 0.87,
  context: "Both implement JWT validation patterns",
  
  // Discovery metadata
  isne_discovered: true,
  cross_domain: true, // code ↔ document relationship
  discovered_at: "2025-01-15T10:00:00Z",
  
  // Version tracking
  discovered_in_version: "1.0.0",
  last_validated: "2025-01-15T10:00:00Z",
  model_version: "1.0.0",
  
  // Incremental updates
  update_batch_id: "batch_123",
  preserved_from_version: "0.9.0", // Track relationship lineage
  update_history: [
    {
      version: "0.9.0",
      confidence: 0.82,
      updated_at: "2025-01-10T10:00:00Z"
    }
  ],
  
  // Performance tracking
  usage_count: 25,
  last_used: "2025-01-15T10:00:00Z"
}
```

### 4. Operational Collections

#### 4.1 model_versions

Tracks ISNE model evolution and training history.

```javascript
{
  _id: "model_versions/v1.0.0",
  version: "1.0.0",
  parent_version: "0.9.0",
  created_at: "2025-01-15T10:00:00Z",
  
  // Model artifacts
  model_path: "/models/isne_v1.0.0.pth",
  config_snapshot: {...}, // Training configuration used
  
  // Training statistics
  training_stats: {
    chunks_processed: 10000,
    graph_nodes: 8500,
    graph_edges: 15000,
    training_duration_seconds: 7200,
    final_loss: 0.0245,
    validation_score: 0.89
  },
  
  // Model metadata
  active: true,
  description: "Incremental update with new research papers",
  training_data_hash: "sha256_...",
  
  // Performance metrics
  benchmark_results: {
    inductive_performance: 0.87,
    cross_domain_accuracy: 0.82,
    inference_speed_ms: 45
  },
  
  // Incremental update info
  incremental_update: true,
  new_chunks_count: 1500,
  updated_relationships: 3200
}
```

#### 4.2 processing_batches

Tracks processing batches for rollback capability and operational monitoring.

```javascript
{
  _id: "processing_batches/batch_12345",
  batch_id: "batch_12345",
  started_at: "2025-01-15T10:00:00Z",
  completed_at: "2025-01-15T11:30:00Z",
  
  // Processing statistics
  status: "pending|processing|completed|failed|rolled_back",
  chunks_processed: 150,
  documents_processed: 25,
  relationships_discovered: 85,
  
  // Model information
  model_version: "1.0.0",
  embedding_model: "ModernBERT",
  
  // Change tracking
  change_summary: {
    new_documents: 10,
    updated_documents: 15,
    new_chunks: 150,
    updated_chunks: 45,
    new_relationships: 85
  },
  
  // Error handling
  errors: [],
  warnings: [],
  
  // Rollback information
  rollback_point: "batch_12344",
  can_rollback: true
}
```

### 5. Project Collections (Dynamic)

#### 5.1 project_{name}_chunks

Project-specific collections for isolated development with knowledge inheritance and incremental capabilities.

```javascript
{
  _id: "project_alpha_chunks/chunk_<hash>",
  content: "class AuthenticationService: ...",
  embeddings: [0.1, 0.2, ...],
  
  // Inheritance tracking
  inherited_from: "mixed_chunks/chunk_<parent_hash>", // null if original
  project_context: "alpha",
  adaptation_notes: "Modified for microservices architecture",
  inheritance_confidence: 0.95,
  
  // Project-specific metadata
  project_version: "1.0.0",
  local_modifications: true,
  sync_with_parent: true,
  
  // Standard chunk fields (same as mixed_chunks)
  source_type: "code",
  chunk_type: "class",
  pipeline_status: {...},
  // ... other mixed_chunks fields
}
```

## Usage Patterns and Queries

### 1. Cross-Domain Discovery

Find research papers related to current code implementation:

```javascript
FOR chunk IN mixed_chunks
  FILTER chunk.source_type == "document" 
  FOR related IN 1..2 ANY chunk relationships
    FILTER related.source_type == "code"
    FILTER related.content LIKE "%authentication%"
    AND related.pipeline_status.isne_enhancement == "completed"
    RETURN {
      research: chunk.content,
      code: related.content,
      confidence: related.confidence,
      discovered_in: related.discovered_in_version
    }
```

### 2. Incremental Processing

Process only changed documents:

```javascript
FOR doc IN documents
  FILTER doc.processing_status == "pending"
  OR doc.content_hash != doc.previous_hash
  RETURN doc._id
```

### 3. Project Knowledge Inheritance

Bootstrap new project with relevant patterns:

```javascript
FOR chunk IN mixed_chunks
  FILTER chunk.chunk_type == "function"
  AND chunk.content LIKE "%auth%"
  AND chunk.access_count > 10
  AND chunk.performance_tier IN ["hot", "warm"]
  AND chunk.pipeline_status.isne_enhancement == "completed"
  INSERT {
    ...chunk,
    inherited_from: chunk._id,
    project_context: "new_microservice",
    inheritance_confidence: chunk.confidence || 0.8
  } INTO project_new_microservice_chunks
```

### 4. Model Version Rollback

Rollback to previous model version:

```javascript
// Find chunks processed in failed batch
FOR chunk IN mixed_chunks
  FILTER chunk.processing_batch_id == "batch_failed_123"
  UPDATE chunk WITH {
    pipeline_status: {
      ...chunk.pipeline_status,
      isne_enhancement: "requires_reprocessing",
      isne_model_version: "0.9.0"
    },
    requires_reprocessing: true
  } IN mixed_chunks
```

## Performance Optimization Strategy

### 1. Adaptive Cold Storage Migration

```javascript
// Migrate rarely accessed chunks to dedicated collections
FOR chunk IN mixed_chunks
  FILTER chunk.access_count < 5 
  AND chunk.last_accessed < DATE_SUB(NOW(), 90, 'days')
  AND chunk.performance_tier == "cold"
  LET target_collection = chunk.source_type == "code" ? "code_chunks_cold" : "document_chunks_cold"
  // Move to appropriate cold storage collection with full metadata preserved
```

### 2. Hot Data Optimization

- Frequently accessed chunks remain in mixed_chunks
- High-value cross-domain relationships stay in primary collection
- Cache optimization focused on mixed_chunks and active relationships
- Index optimization based on access patterns

### 3. Incremental Processing Optimization

- Content hashing prevents redundant processing
- Pipeline status tracking avoids duplicate work
- Batch processing enables efficient rollbacks
- Model version tracking supports selective updates

## LLM Integration

### Natural Language Queries

The schema is designed for LLM-driven interactions:

```text
Human: "Find code patterns similar to our JWT implementation from the latest model version"
LLM: → ArangoDB query on mixed_chunks with ISNE similarity filtering + model version constraints
```

### Prompt Engineering Integration

Schema designed for natural language interface:

- Simple, consistent field names for LLM understanding
- Explicit relationship types for context building
- Metadata structure optimized for LLM consumption
- Version tracking for temporal reasoning
- Confidence scores for uncertainty handling

## ISNE Graph Population Integration

### Core Concept: ISNE as Graph Database Creator

ISNE doesn't just enhance embeddings - it **creates the knowledge graph** by discovering relationships between chunks that wouldn't be found through traditional similarity measures.

### Current Training Test → Graph Creation Flow

Our running ISNE training test (`test_training_simple.py`) demonstrates this pipeline:

```python
# Current test produces:
# output/training_output_*/isne_model_final.pth ← Graph relationship creator
# Training logs with metrics ← Relationship quality validation
```

### ISNE Graph Population Pipeline

```python
class ISNEGraphPopulator:
    """
    Converts ISNE-enhanced embeddings into ArangoDB relationship graph.
    
    This is the critical bridge between ISNE training and graph database creation.
    """
    
    def __init__(self, arango_client, confidence_threshold=0.75):
        self.arango_client = arango_client
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
    
    def populate_from_trained_model(self, model_path: str, chunks_collection: str, 
                                   batch_id: str, model_version: str):
        """
        Main entry point: Convert trained ISNE model to relationship graph.
        
        Args:
            model_path: Path to trained ISNE model (e.g., from our current test)
            chunks_collection: ArangoDB collection containing chunks
            batch_id: Processing batch ID for tracking
            model_version: Model version (e.g., "1.0.0")
        """
        
        # 1. Load trained ISNE model
        isne_model = self._load_isne_model(model_path)
        
        # 2. Get all chunks from ArangoDB
        chunks = self._get_chunks_from_arango(chunks_collection)
        
        # 3. Generate ISNE-enhanced embeddings
        # This is where ISNE adds graph-aware context
        enhanced_embeddings = isne_model.enhance_embeddings([c.content for c in chunks])
        
        # 4. Discover relationships using enhanced embeddings
        relationships = self._discover_relationships(chunks, enhanced_embeddings, 
                                                   batch_id, model_version)
        
        # 5. Populate ArangoDB relationships collection
        self._bulk_insert_relationships(relationships)
        
        # 6. Update chunks with graph readiness status
        self._mark_chunks_graph_ready(chunks, batch_id, model_version)
        
        return {
            "relationships_created": len(relationships),
            "chunks_processed": len(chunks),
            "model_version": model_version,
            "batch_id": batch_id
        }
    
    def _discover_relationships(self, chunks, enhanced_embeddings, 
                               batch_id: str, model_version: str):
        """
        Core relationship discovery using ISNE-enhanced embeddings.
        
        ISNE enhancement means these aren't just similarity scores - they're
        graph-aware relationships that understand multi-hop connections.
        """
        relationships = []
        
        for i, chunk_a in enumerate(chunks):
            for j, chunk_b in enumerate(chunks[i+1:], i+1):
                
                # ISNE-enhanced similarity (not just cosine similarity)
                # This captures graph context and multi-hop relationships
                similarity = self._compute_isne_similarity(
                    enhanced_embeddings[i], 
                    enhanced_embeddings[j],
                    chunk_a.metadata,
                    chunk_b.metadata
                )
                
                if similarity > self.confidence_threshold:
                    # Determine relationship type based on content analysis
                    rel_type = self._classify_relationship_type(chunk_a, chunk_b, similarity)
                    
                    relationships.append({
                        "_from": f"mixed_chunks/{chunk_a.id}",
                        "_to": f"mixed_chunks/{chunk_b.id}",
                        "type": rel_type,
                        "confidence": similarity,
                        "isne_discovered": True,
                        "cross_domain": chunk_a.source_type != chunk_b.source_type,
                        "discovered_in_version": model_version,
                        "update_batch_id": batch_id,
                        "context": self._generate_relationship_context(chunk_a, chunk_b),
                        "discovered_at": datetime.utcnow().isoformat()
                    })
        
        return relationships
    
    def _classify_relationship_type(self, chunk_a, chunk_b, similarity):
        """
        Classify the type of relationship based on content and similarity.
        
        ISNE helps here because it understands semantic relationships
        beyond just text similarity.
        """
        # Cross-domain relationships (code ↔ document)
        if chunk_a.source_type != chunk_b.source_type:
            if chunk_a.source_type == "code" and chunk_b.source_type == "document":
                return "implements" if similarity > 0.85 else "conceptual"
            elif chunk_a.source_type == "document" and chunk_b.source_type == "code":
                return "documented_by" if similarity > 0.85 else "conceptual"
        
        # Same-domain relationships
        if chunk_a.source_type == "code" and chunk_b.source_type == "code":
            if similarity > 0.9:
                return "calls" if self._detect_function_call(chunk_a, chunk_b) else "similarity"
            elif similarity > 0.8:
                return "imports" if self._detect_import_relationship(chunk_a, chunk_b) else "similarity"
            else:
                return "similarity"
        
        # Document relationships
        if chunk_a.source_type == "document" and chunk_b.source_type == "document":
            if similarity > 0.85:
                return "references" if self._detect_citation(chunk_a, chunk_b) else "conceptual"
            else:
                return "conceptual"
        
        return "similarity"  # Default
```

### Integration with Current Training Pipeline

Add to `src/config/isne_training_config.yaml`:

```yaml
# Existing training config...

post_training:
  graph_population:
    enabled: true
    arango_config:
      database: "hades_unified"
      chunks_collection: "mixed_chunks"
      relationships_collection: "relationships"
    
    # Relationship discovery settings
    confidence_threshold: 0.75
    cross_domain_bonus: 0.1  # Boost cross-domain relationships
    batch_size: 1000
    
    # Relationship type classification
    high_confidence_threshold: 0.85  # For specific relationship types
    medium_confidence_threshold: 0.75  # For general similarity
    
    # Performance settings
    parallel_processing: true
    max_workers: 4
```

### Post-Training Graph Population

```python
# Add to ISNETrainingPipeline.train() method
def train(self) -> ISNETrainingResult:
    # ... existing training logic ...
    
    if result.success and self.config.post_training.graph_population.enabled:
        self.logger.info("Starting graph population from trained ISNE model...")
        
        graph_populator = ISNEGraphPopulator(
            arango_client=self._get_arango_client(),
            confidence_threshold=self.config.post_training.graph_population.confidence_threshold
        )
        
        population_result = graph_populator.populate_from_trained_model(
            model_path=result.model_path,
            chunks_collection="mixed_chunks",
            batch_id=f"training_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            model_version=result.version
        )
        
        # Add graph population results to training result
        result.graph_population_stats = population_result
        
        self.logger.info(f"Graph population completed: {population_result['relationships_created']} relationships created")
    
    return result
```

### Graph Database Creation Flow

```text
Current Test Flow:
test_training_simple.py → ISNE Training → Model Saved → [MANUAL STEP]

Enhanced Flow:
test_training_simple.py → ISNE Training → Model Saved → Graph Population → ArangoDB Relationships → Ready for PathRAG
```

### Connection to Current Running Test

When our current training completes, we'll have:

1. **Trained Model**: `output/training_output_*/isne_model_final.pth`
2. **Training Stats**: Performance metrics, validation scores
3. **Ready for Graph Creation**: Model can now discover relationships

Next steps after training completes:
1. Implement `ISNEGraphPopulator` class
2. Create ArangoDB collections (mixed_chunks, relationships)
3. Run graph population on test-data3 chunks
4. Validate cross-domain relationship discovery

### Relationship Discovery Examples

**Code ↔ Document Discovery**:
```python
# Document chunk: "JWT tokens should be validated on every request..."
# Code chunk: "def validate_token(token): ..."
# ISNE discovers: type="implements", confidence=0.89, cross_domain=True
```

**Code ↔ Code Discovery**:
```python
# Function A: "def authenticate_user(username, password): ..."
# Function B: "def validate_token(token): ..."  
# ISNE discovers: type="calls", confidence=0.82, cross_domain=False
```

**Document ↔ Document Discovery**:
```python
# Research paper section: "Authentication mechanisms in distributed systems..."
# Technical doc section: "Our microservice authentication architecture..."
# ISNE discovers: type="conceptual", confidence=0.77, cross_domain=False
```

## Implementation Roadmap

### Phase 1: Core Schema Setup (In Progress)
1. ✅ Create base collections with unified schema
2. ✅ Implement content hashing system  
3. ✅ Set up basic incremental detection
4. ✅ Create model version tracking
5. 🔄 **Current**: ISNE training test running

### Phase 2: ISNE Graph Integration (Next)
1. 📋 **Implement ISNEGraphPopulator class**
2. 📋 **Create ArangoDB collections setup**
3. 📋 **Integrate graph population with training pipeline**
4. 📋 **Test relationship discovery on current training output**
5. 📋 **Validate cross-domain relationship quality**

### Phase 3: Relationship Classification & Quality
1. Implement relationship type classification
2. Add confidence scoring and validation
3. Create relationship quality metrics
4. Set up relationship validation feedback loop

### Phase 4: Performance Optimization
1. Implement adaptive cold storage migration
2. Add access pattern tracking
3. Optimize indexes based on usage
4. Set up performance monitoring

### Phase 5: Project Inheritance
1. Create dynamic project collection system
2. Implement inheritance confidence scoring
3. Add project-specific adaptation tracking
4. Set up cross-project knowledge sharing

## Monitoring and Analytics

### Key Metrics to Track

- **Processing Performance**: Batch processing times, incremental vs. full processing ratios
- **Model Evolution**: Version performance comparisons, incremental update success rates
- **Cross-Domain Discovery**: Relationship discovery rates, cross-domain query success
- **Storage Optimization**: Hot/warm/cold distribution, migration effectiveness
- **Project Inheritance**: Inheritance success rates, adaptation effectiveness

### Operational Dashboards

1. **Ingestion Dashboard**: Batch status, processing queues, error rates
2. **Model Performance**: Version comparisons, training metrics, accuracy trends
3. **Storage Efficiency**: Collection sizes, access patterns, migration statistics
4. **Discovery Analytics**: Relationship networks, cross-domain connections, query patterns

## Security and Data Governance

### Access Control
- Collection-level permissions for different user roles
- Project-specific access controls for isolated development
- Model version access based on stability and approval status

### Data Lineage
- Full traceability from source documents to final relationships
- Version history for all chunks and relationships
- Audit trails for all processing batches and model updates

### Compliance
- Data retention policies based on performance tiers
- Privacy controls for sensitive content
- Export capabilities for data portability

## Future Considerations

### Advanced Features
- Multi-tenant project isolation
- Federated learning across project collections
- Advanced relationship types (temporal, causal, hierarchical)
- Integration with external knowledge graphs

### Scaling Strategies
- Horizontal partitioning of mixed_chunks by domain or time
- Dedicated read replicas for analytical queries
- Archive collections for historical data
- Distributed ISNE training across multiple models

### AI/ML Integration
- Store model training metadata alongside chunks
- Track feature importance for relationship scoring
- Version control for embedding model updates
- A/B testing framework for model improvements

---

**Key Principle**: Start simple with unified collections, evolve intelligently based on real usage patterns, maintain full operational intelligence for incremental updates and model evolution. The schema serves both discovery and operational needs without compromise.