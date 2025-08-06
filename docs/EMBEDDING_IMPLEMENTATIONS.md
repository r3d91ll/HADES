# Embedding Implementations

## Current State (To Be Consolidated)

We have multiple Jina v4 embedding implementations that should be unified into a single module.

### 1. Jina V4 Embedder (Core)
**Location**: `processors/arxiv/core/jina_v4_embedder.py`
- Clean Jina v4 implementation
- Supports late chunking
- FP16 optimization
- Code-specific embedding support
- 2048-dimensional vectors

### 2. Batch Embed Jina
**Location**: `processors/arxiv/core/batch_embed_jina.py`
- Batch processing optimized
- GPU memory management
- Progress tracking
- Database integration

### 3. Enhanced Docling Processor V2
**Location**: `processors/arxiv/core/enhanced_docling_processor_v2.py`
- Document-specific embedding
- Integrates with Docling extraction
- Handles both abstracts and full-text

### 4. Daily ArXiv Update (Embedded)
**Location**: `processors/arxiv/scripts/daily_arxiv_update.py`
- Contains inline embedding logic
- Real-time processing

### 5. Rebuild Dual GPU Script
**Location**: `processors/arxiv/scripts/rebuild_dual_gpu.py`
- Dual GPU coordination
- Memory-efficient batching
- NVLink optimization

## Key Features Across Implementations

### Common Patterns
- Model: `jinaai/jina-embeddings-v4`
- Dimensions: 2048
- Late chunking support
- GPU acceleration
- Batch processing

### Unique Features to Preserve
1. **Late Chunking** (jina_v4_embedder.py)
   - Context preservation across chunks
   - 32k token context window

2. **Dual GPU** (rebuild_dual_gpu.py)
   - NVLink coordination
   - Load balancing

3. **Code Embeddings** (jina_v4_embedder.py)
   - Task-specific: "code" vs "retrieval"
   - LoRA adapter support

4. **Memory Management** (batch_embed_jina.py)
   - Dynamic batch sizing
   - OOM prevention

## Consolidation Plan (After Database Rebuild)

### Target Architecture
```
embeddings/
├── __init__.py
├── jina_v4.py              # Core Jina v4 implementation
├── batch_processor.py      # Batch processing utilities
├── gpu_manager.py          # GPU memory and multi-GPU
├── late_chunking.py        # Late chunking strategies
└── task_adapters.py        # Code, retrieval, etc.
```

### Unified Interface
```python
class UnifiedEmbedder:
    """Single entry point for all embedding operations."""
    
    def __init__(self, config: EmbeddingConfig):
        # Auto-detect GPU, memory, optimal batch size
        pass
    
    def embed(self, 
              texts: List[str], 
              task: str = "retrieval",
              use_late_chunking: bool = True) -> np.ndarray:
        # Unified embedding interface
        pass
    
    def embed_documents(self,
                        documents: List[Document],
                        batch_size: Optional[int] = None) -> List[np.ndarray]:
        # Document-specific processing
        pass
```

### Migration Benefits
1. **Single source of truth** for embedding logic
2. **Consistent API** across all processors
3. **Easier testing** and maintenance
4. **Performance optimization** in one place
5. **Configuration management** simplified

## Implementation Priority

1. **Phase 1**: Extract and unify core embedding logic
2. **Phase 2**: Add batch processing and GPU management
3. **Phase 3**: Implement late chunking as plugin
4. **Phase 4**: Add task-specific adapters
5. **Phase 5**: Migrate existing code to use unified module

## Testing Requirements

- Unit tests for each component
- Integration tests with real documents
- Performance benchmarks
- Memory usage profiling
- Multi-GPU coordination tests
- Late chunking validation

## Timeline
- **Now**: Document current implementations
- **After DB Rebuild**: Create unified module
- **Testing**: Validate against existing implementations
- **Migration**: Update all processors
- **Cleanup**: Move old implementations to Acheron