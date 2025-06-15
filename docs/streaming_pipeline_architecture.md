# Hybrid Streaming-Batch ISNE Pipeline Architecture

## Executive Summary

The current ISNE bootstrap pipeline processes documents in sequential batches, creating inefficiencies and underutilizing available hardware resources. This document outlines a **hybrid streaming-batch architecture** that streams the I/O-heavy front-end stages (document processing → chunking → embedding) while maintaining batch processing for the memory-intensive back-end stages (graph construction → ISNE training). This approach delivers 80% of full streaming benefits with 20% of the complexity.

## Current State Analysis

### Sequential Batch Pipeline Limitations
```
┌─────────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐
│ Document Proc   │───▶│ Chunking    │───▶│ Embedding   │───▶│ Graph Construct │
│ ALL 4,056 docs  │    │ ALL chunks  │    │ ALL embeds  │    │ ALL at once     │
└─────────────────┘    └─────────────┘    └─────────────┘    └─────────────────┘
     100% CPU               IDLE               IDLE               IDLE
```

**Problems:**
- **Resource Underutilization**: Only one stage active at a time
- **Memory Peaks**: All data held in memory between stages
- **Latency**: Later stages wait for complete upstream processing
- **No Parallelization**: Single-threaded stage execution
- **Poor Scalability**: Cannot leverage multiple CPUs/GPUs effectively

### Performance Bottlenecks
1. **Document Processing**: CPU-bound, can parallelize across cores
2. **Chunking**: CPU-bound, depends on document size
3. **Embedding**: GPU-bound, benefits from batching
4. **Graph Construction**: Memory-bound, can stream updates

## Proposed Hybrid Architecture

### High-Level Design
```
STREAMING FRONT-END (I/O Heavy, Parallelizable)
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Doc Workers │───▶│ Chunk Queue │───▶│ Embed Batch │───▶│ Embedding   │
│ (4 threads) │    │ (async)     │    │ (GPU)       │    │ Storage     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                              │
                                                              ▼
                                                         [CHECKPOINT]
                                                       HDF5/Parquet File
                                                              │
                                                              ▼
BATCH BACK-END (Memory Intensive, Complex)
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Load        │───▶│ Graph       │───▶│ ISNE        │
│ Embeddings  │    │ Construction│    │ Training    │
└─────────────┘    └─────────────┘    └─────────────┘
```

### Architectural Decision: Hybrid Approach

**Stream Where It Helps Most:**
- Document Processing (I/O bound, parallelizes well)
- Chunking (CPU bound, benefits from parallel workers)  
- Embedding Generation (GPU bound, benefits from batching)

**Batch Where Complexity Is High:**
- Graph Construction (avoids incremental FAISS complexity)
- ISNE Training (existing optimized implementation)

**Natural Checkpoint:** Embeddings persisted to disk provide perfect recovery boundary.

### Component Architecture

#### 1. **File Discovery & Queuing Stage**
```python
class FileDiscoveryStage:
    """Asynchronously discover and queue input files."""
    
    async def scan_directory(self, path: Path) -> AsyncIterator[Path]:
        """Stream file discovery with filtering."""
        
    async def queue_files(self, file_queue: asyncio.Queue):
        """Feed files into processing queue with backpressure."""
```

**Features:**
- Asynchronous file system scanning
- Intelligent file filtering (size, type, format)
- Backpressure handling to prevent memory overflow
- Priority queuing for important file types

#### 2. **Document Processing Stage**
```python
class DocumentProcessingStage:
    """Parallel document processing with hybrid routing."""
    
    def __init__(self, num_workers: int = 4):
        self.core_processor = CoreDocumentProcessor()
        self.docling_processor = DoclingDocumentProcessor()
        self.workers = []
        
    async def process_stream(self, 
                           file_queue: asyncio.Queue,
                           document_queue: asyncio.Queue):
        """Process files with multiple workers."""
        
    def route_processor(self, file_path: Path) -> DocumentProcessor:
        """Route to appropriate processor (core vs docling)."""
```

**Features:**
- **Parallel Workers**: 4 concurrent document processors
- **Hybrid Routing**: Core for Python/text, Docling for PDFs
- **Load Balancing**: Distribute work across available processors
- **Error Isolation**: Failed documents don't block pipeline
- **Progress Tracking**: Real-time processing metrics

#### 3. **Chunking Stage**
```python
class ChunkingStage:
    """Streaming document chunking with smart batching."""
    
    async def chunk_stream(self,
                          document_queue: asyncio.Queue,
                          chunk_queue: asyncio.Queue):
        """Stream chunking with adaptive batching."""
        
    def adaptive_batching(self, documents: List[Document]) -> List[Chunk]:
        """Batch documents by type for optimal chunking."""
```

**Features:**
- **Streaming Chunking**: Process documents as they arrive
- **Type-Aware Batching**: Group Python files for AST processing
- **Memory Management**: Bounded chunk queue prevents overflow
- **Chunk Validation**: Real-time validation during processing

#### 4. **Embedding Stage**
```python
class EmbeddingStage:
    """GPU-accelerated streaming embedding generation."""
    
    def __init__(self, batch_size: int = 64, gpu_device: str = "cuda:0"):
        self.embedder = ModernBERTEmbedder(device=gpu_device)
        self.batch_size = batch_size
        
    async def embed_stream(self,
                          chunk_queue: asyncio.Queue,
                          embedding_queue: asyncio.Queue):
        """Stream embedding with optimal GPU batching."""
        
    async def adaptive_batching(self) -> List[Chunk]:
        """Dynamic batch sizing based on chunk length."""
```

**Features:**
- **GPU Optimization**: Maximize GPU utilization with smart batching
- **Adaptive Batching**: Variable batch sizes based on chunk characteristics
- **Memory Streaming**: Process embeddings without full memory load
- **Queue Management**: Backpressure to prevent GPU memory overflow

#### 5. **Embedding Storage Stage**
```python
class EmbeddingStorageStage:
    """Persist embeddings to disk for batch graph construction."""
    
    def __init__(self, storage_format: str = "hdf5"):
        self.storage_format = storage_format
        
    async def store_embeddings(self,
                              embedding_queue: asyncio.Queue,
                              output_path: Path):
        """Stream embeddings to persistent storage."""
        
    def create_embedding_dataset(self, embeddings: List[Embedding]) -> Dict:
        """Create structured dataset with embeddings and metadata."""
        return {
            'embeddings': np.array,           # Shape: (N, 384)
            'chunk_ids': List[str],
            'document_sources': List[str], 
            'chunk_text': List[str],          # For debugging
            'processing_timestamp': datetime,
            'model_info': dict                # Embedding model details
        }
```

**Features:**
- **Efficient Storage**: HDF5 or Parquet for fast batch loading
- **Rich Metadata**: Preserve chunk context and lineage
- **Checkpoint Recovery**: Perfect boundary for pipeline restart
- **Reusability**: Cached embeddings for multiple experiments

#### 6. **Batch Graph Construction Stage**
```python
class BatchGraphStage:
    """Traditional batch graph construction from stored embeddings."""
    
    def __init__(self, similarity_threshold: float = 0.5):
        self.similarity_threshold = similarity_threshold
        
    def load_embeddings(self, embedding_file: Path) -> EmbeddingDataset:
        """Efficiently load embeddings and metadata."""
        
    def build_graph_batch(self, embeddings: EmbeddingDataset) -> Graph:
        """Build complete graph using batch similarity computation."""
        
    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Vectorized similarity computation for all pairs."""
```

**Features:**
- **Batch Efficiency**: Vectorized similarity computation
- **Memory Optimization**: Chunked processing for large datasets
- **Existing Integration**: Uses proven graph construction logic
- **Parameter Flexibility**: Easy to experiment with thresholds

## Queue Architecture & Flow Control

### Queue Specifications
```python
@dataclass
class QueueConfig:
    file_queue_size: int = 100          # Input files buffer
    document_queue_size: int = 200      # Processed documents
    chunk_queue_size: int = 500         # Chunked content
    embedding_queue_size: int = 300     # Generated embeddings
    graph_update_queue_size: int = 100  # Graph construction updates
```

### Backpressure Management
```python
class BackpressureManager:
    """Manages flow control across pipeline stages."""
    
    async def monitor_queues(self):
        """Monitor queue depths and adjust production rates."""
        
    def adjust_batch_sizes(self, queue_depths: Dict[str, int]):
        """Dynamically adjust batch sizes based on queue pressure."""
```

## Resource Optimization Strategy

### CPU Allocation
- **Document Processing**: 4 workers (file I/O + parsing)
- **Chunking**: 2 workers (text processing)
- **Queue Management**: 1 worker (async coordination)
- **Monitoring**: 1 worker (metrics collection)

### Memory Management
- **Streaming Buffers**: Fixed-size queues prevent memory bloat
- **Garbage Collection**: Aggressive cleanup of processed items
- **Memory Mapping**: Large files processed via memory mapping
- **Batch Processing**: Optimal batch sizes for memory efficiency

### GPU Utilization
- **Batch Optimization**: Dynamic batching for maximum GPU throughput
- **Memory Streaming**: Prevent GPU memory overflow
- **Pipeline Parallelism**: Overlap CPU processing with GPU computation

## Performance Projections

### Current vs Hybrid Performance
```
Stage                   Current    Hybrid       Improvement
=======================================================
Document Processing     30 min     8 min        3.8x faster
Chunking               10 min     3 min        3.3x faster  
Embedding              25 min     10 min       2.5x faster
Graph Construction     15 min     15 min       Same
ISNE Training          20 min     20 min       Same
-------------------------------------------------------
Total Pipeline Time    100 min    56 min       1.8x faster
Peak Memory Usage      8 GB       5 GB         1.6x reduction
CPU Utilization        25%        75%          3x improvement
GPU Utilization        15%        60%          4x improvement
```

### Additional Benefits
- **Checkpoint Recovery**: Resume from embeddings if training fails
- **Embedding Reusability**: Experiment with graph parameters without re-embedding
- **Reduced Complexity**: 80% of streaming benefits, 20% of complexity
- **Faster Implementation**: 3-4 weeks vs 10-12 weeks for full streaming

### Scalability Benefits
- **Parallel Scaling**: Front-end stages scale with CPU cores
- **GPU Optimization**: Optimal batching for embedding generation
- **Memory Efficiency**: Lower peak usage, persistent checkpoint

## Implementation Phases

### Phase 1: Streaming Front-End Infrastructure (2-3 weeks)
- Async queue architecture for document processing
- Parallel document workers with hybrid routing (Core/Docling)
- Streaming chunking pipeline with type-aware batching
- GPU-optimized embedding generation with adaptive batching
- Embedding persistence to HDF5/Parquet with rich metadata

**Deliverables:**
- StreamingFrontEnd class with complete document → embedding pipeline
- Persistent embedding storage with metadata
- Basic monitoring and error handling
- Performance benchmarking vs current pipeline

### Phase 2: Enhanced Batch Back-End (1 week)
- Efficient embedding loading from persistent storage
- Optimized batch graph construction using existing algorithms
- Integration with existing ISNE training pipeline
- End-to-end testing and validation

**Deliverables:**
- BatchBackEnd class with embedding → trained model pipeline
- Performance validation of hybrid approach
- Error recovery and checkpoint mechanisms
- Configuration integration with existing HADES

### Phase 3: Production Optimization (1 week)
- Error handling and recovery mechanisms
- Comprehensive monitoring and alerting
- Load testing with production-scale datasets
- Documentation and operational runbooks

**Deliverables:**
- Production-ready hybrid pipeline
- Complete monitoring and observability
- Deployment automation
- Performance validation report

## Integration with Existing HADES Architecture

### Backward Compatibility
- Maintain existing pipeline interfaces
- Support both batch and streaming modes
- Gradual migration path for existing workflows

### Configuration Integration
```yaml
pipeline:
  mode: "hybrid"  # "batch", "hybrid", or "full_streaming" (future)
  
  # Streaming front-end configuration
  streaming_frontend:
    workers:
      document_processing: 4
      chunking: 2
    queues:
      file_buffer_size: 100
      chunk_buffer_size: 500
      embedding_buffer_size: 200
    gpu:
      device: "cuda:0"
      batch_size: 64
      memory_fraction: 0.7
    storage:
      embedding_format: "hdf5"  # or "parquet"
      checkpoint_dir: "./checkpoints"
      compression: "gzip"
  
  # Batch back-end configuration  
  batch_backend:
    graph_construction:
      similarity_threshold: 0.5
      max_edges_per_node: 20
      chunk_processing: true  # Process in chunks for large datasets
    isne_training:
      # Existing ISNE configuration unchanged
      epochs: 25
      learning_rate: 0.001
      batch_size: 1024
```

### Monitoring Integration
- Real-time pipeline metrics
- Stage-specific performance tracking
- Queue depth monitoring
- Resource utilization dashboards

## Risk Assessment & Mitigation

### Technical Risks (Reduced vs Full Streaming)
1. **Queue Overflow**: Lower risk - smaller queue depths, checkpoint boundary
2. **GPU Memory Issues**: Mitigated by adaptive batching and monitoring
3. **Storage I/O Bottlenecks**: Mitigated by efficient HDF5/Parquet formats
4. **Checkpoint Corruption**: Mitigated by atomic writes and validation

### Implementation Risks (Significantly Reduced)
1. **Complexity**: Much lower than full streaming - proven batch back-end
2. **Integration Issues**: Lower risk - existing ISNE pipeline unchanged
3. **Performance Regression**: Lower risk - incremental optimization approach
4. **Testing Complexity**: Reduced - clear boundaries between streaming/batch stages

### Migration Risks
1. **Embedding Format Changes**: Mitigated by backward compatibility
2. **Performance Expectations**: Mitigated by conservative estimates
3. **Resource Requirements**: Mitigated by gradual rollout

## Future Enhancements

### Distributed Processing
- Multi-machine pipeline deployment
- Shared queue infrastructure (Redis/RabbitMQ)
- Load balancing across compute nodes

### Advanced Optimizations
- ML-driven batch size optimization
- Predictive queue management
- Adaptive resource allocation
- Dynamic pipeline reconfiguration

### Cloud Integration
- Kubernetes deployment patterns
- Auto-scaling based on queue depths
- Cost optimization strategies
- Multi-cloud deployment options

## Conclusion

The hybrid streaming-batch pipeline architecture represents a pragmatic and achievable improvement to HADES processing efficiency. By streaming the I/O-heavy front-end stages while maintaining batch processing for complex back-end operations, we can achieve:

- **1.8x performance improvement** (100 min → 56 min)
- **1.6x memory reduction** (8GB → 5GB peak usage)
- **Checkpoint recovery** with embedding persistence
- **80% of streaming benefits with 20% of complexity**
- **3-4 week implementation** vs 10-12 weeks for full streaming

### Key Advantages of Hybrid Approach

1. **Pragmatic Engineering**: Delivers meaningful performance gains quickly
2. **Reduced Risk**: Proven batch algorithms for complex graph/training stages  
3. **Natural Boundary**: Embeddings provide perfect checkpoint for recovery
4. **Future Ready**: Foundation for full streaming when complexity is justified
5. **Immediate Value**: Embedding caching enables rapid experimentation

### Implementation Strategy

**Phase 1 (2-3 weeks)**: Streaming front-end with embedding persistence
**Phase 2 (1 week)**: Enhanced batch back-end integration  
**Phase 3 (1 week)**: Production optimization and validation

**Total Timeline**: 4-5 weeks to production-ready hybrid pipeline

### Next Steps

1. **Complete current ISNE training** to establish baseline performance
2. **Validate hybrid architecture assumptions** with performance profiling
3. **Begin Phase 1 implementation** of streaming front-end
4. **Consider full streaming migration** as future optimization when justified by scale

This hybrid approach positions HADES for immediate performance gains while laying the foundation for future streaming enhancements as requirements and complexity justify the investment.