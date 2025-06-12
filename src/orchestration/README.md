# Orchestration Module

The orchestration module provides comprehensive pipeline orchestration and parallel processing capabilities for the HADES system. It coordinates complex multi-stage workflows, manages resource allocation, and ensures reliable execution of document processing and training pipelines.

## 📋 Overview

This module implements:

- **Pipeline orchestration** for coordinating multi-stage document processing workflows
- **Parallel processing** with worker management and load balancing
- **Queue management** for task distribution and execution coordination
- **Monitoring and alerting** for pipeline health and performance tracking
- **Error recovery** with checkpointing and resume capabilities
- **Resource management** for optimal CPU and GPU utilization

## 🏗️ Architecture

### Pipeline Orchestration Design

```text
Input Data → Pipeline Stages → Parallel Workers → Output Processing → Result Aggregation
     ↓           ↓               ↓                 ↓                  ↓
Documents → Stage Queue → Worker Pool → Stage Results → Combined Output
```

### Multi-Stage Pipeline Flow

```text
Document Processing → Chunking → Embedding → ISNE → Storage
        ↓               ↓         ↓          ↓       ↓
    DocProc Stage → Chunk Stage → Embed Stage → ISNE Stage → Store Stage
        ↓               ↓         ↓          ↓       ↓
   Parallel Docs → Parallel Chunks → Batch Embed → Graph Process → DB Write
```

## 📁 Module Contents

### Core Orchestration

- **`__init__.py`** - Module exports and orchestration interfaces

### Pipeline Framework

**`pipelines/`** - Pipeline implementations and management:

- **`orchestrator.py`** - Main pipeline orchestrator for workflow coordination
- **`parallel_pipeline.py`** - Parallel pipeline execution with worker management
- **`text_pipeline.py`** - Text processing pipeline for document workflows
- **`training_pipeline.py`** - Training pipeline for model training workflows
- **`data_ingestion_pipeline.py`** - Data ingestion pipeline for batch processing
- **`schema.py`** - Pipeline schema definitions and validation

### Pipeline Stages

**`pipelines/stages/`** - Individual pipeline stage implementations:

- **`base.py`** - Base stage interface and common functionality
- **`document_processor.py`** - Document processing stage
- **`chunking.py`** - Document chunking stage
- **`embedding.py`** - Embedding generation stage
- **`isne.py`** - ISNE enhancement stage
- **`storage.py`** - Data storage stage

### Processing Infrastructure

**`core/`** - Core orchestration infrastructure:

- **`monitoring.py`** - Pipeline monitoring and metrics collection
- **`parallel_worker.py`** - Parallel worker implementation and management

**`core/queue/`** - Queue management system:

- **`queue_manager.py`** - Task queue management and distribution

### Legacy and Development

**`pipelines/ingest/`** - Legacy ingestion pipeline components
**`pipelines/training/`** - Training-specific pipeline components:

- **`isne_training_legacy.py`** - Legacy ISNE training pipeline

### Pipeline Development Tools

- **`debug_pipeline_runner.py`** - Pipeline debugging and testing utilities
- **`enhanced_e2e_test.py`** - End-to-end pipeline testing

### Documentation

- **`orchestration_readme.md`** - Legacy documentation (superseded by this README)
- **`pipelines_readme.md`** - Pipeline-specific documentation

## 🚀 Key Features

### Multi-Stage Pipeline Orchestration

**Complete document processing pipeline**:

```python
from src.orchestration.pipelines.text_pipeline import TextProcessingPipeline

# Initialize comprehensive text processing pipeline
pipeline = TextProcessingPipeline(
    config_file="./config/text_pipeline_config.yaml",
    output_dir="./processed/",
    enable_monitoring=True
)

# Process documents through all stages
results = pipeline.process_documents(
    input_files=["./docs/paper1.pdf", "./docs/code.py", "./docs/readme.md"],
    stages=["docproc", "chunking", "embedding", "isne", "storage"],
    parallel=True,
    max_workers=4
)

print(f"Processed {results['total_documents']} documents")
print(f"Generated {results['total_chunks']} chunks")
print(f"Pipeline time: {results['processing_time']} seconds")
```

**Configurable stage execution**:

```python
from src.orchestration.pipelines.orchestrator import PipelineOrchestrator

# Create custom pipeline with specific stages
orchestrator = PipelineOrchestrator()

# Define pipeline stages
pipeline_config = {
    "stages": [
        {
            "name": "document_processing",
            "class": "DocumentProcessorStage",
            "config": {"preserve_structure": True}
        },
        {
            "name": "chunking", 
            "class": "ChunkingStage",
            "config": {"strategy": "adaptive", "chunk_size": 512}
        },
        {
            "name": "embedding",
            "class": "EmbeddingStage", 
            "config": {"model": "modernbert", "batch_size": 32}
        }
    ],
    "execution": {
        "parallel": True,
        "max_workers": 8,
        "checkpoint_interval": 100
    }
}

# Execute pipeline
results = orchestrator.execute_pipeline(
    pipeline_config,
    input_data=input_documents
)
```

### Parallel Processing Framework

**Efficient parallel execution**:

```python
from src.orchestration.pipelines.parallel_pipeline import ParallelPipeline

# Initialize parallel pipeline
pipeline = ParallelPipeline(
    worker_config={
        "num_workers": 8,
        "worker_type": "process",  # process, thread, async
        "max_queue_size": 1000,
        "timeout": 300
    },
    monitoring_config={
        "enable_metrics": True,
        "checkpoint_interval": 50,
        "progress_reporting": True
    }
)

# Process large document collection
large_document_batch = get_document_collection()  # 1000+ documents

results = pipeline.process_parallel(
    input_data=large_document_batch,
    batch_size=50,
    stage_configs={
        "chunking": {"parallel_chunks": True},
        "embedding": {"batch_size": 64, "gpu_acceleration": True},
        "isne": {"batch_processing": True}
    }
)
```

### Advanced Queue Management

**Distributed task processing**:

```python
from src.orchestration.core.queue.queue_manager import QueueManager

# Initialize queue manager
queue_manager = QueueManager(
    backend="redis",  # redis, memory, database
    config={
        "host": "localhost",
        "port": 6379,
        "max_queue_size": 10000,
        "worker_timeout": 600
    }
)

# Distribute processing tasks
documents = get_documents_to_process()

# Enqueue processing tasks
for doc in documents:
    task = {
        "task_type": "document_processing",
        "input_data": doc,
        "stage_configs": pipeline_configs,
        "priority": doc.get("priority", 1)
    }
    queue_manager.enqueue_task("processing_queue", task)

# Process tasks with multiple workers
async def worker_process():
    while True:
        task = await queue_manager.dequeue_task("processing_queue")
        if task:
            result = await process_document_task(task)
            await queue_manager.mark_task_complete(task["id"], result)

# Start worker pool
worker_pool = [worker_process() for _ in range(8)]
await asyncio.gather(*worker_pool)
```

## 🔧 Pipeline Stages

### Document Processing Stage

```python
from src.orchestration.pipelines.stages.document_processor import DocumentProcessorStage

# Initialize document processing stage
doc_stage = DocumentProcessorStage(
    config={
        "adapters": ["pdf", "python", "markdown"],
        "output_format": "json",
        "preserve_metadata": True,
        "parallel_processing": True
    }
)

# Process documents
input_files = ["./paper.pdf", "./code.py", "./readme.md"]
processed_docs = doc_stage.run(input_files)

print(f"Processed {len(processed_docs)} documents")
for doc in processed_docs:
    print(f"- {doc.file_name}: {doc.content_type}")
```

### Chunking Stage

```python
from src.orchestration.pipelines.stages.chunking import ChunkingStage

# Initialize chunking stage
chunking_stage = ChunkingStage(
    config={
        "strategy": "adaptive",
        "chunk_size": 512,
        "overlap": 64,
        "preserve_structure": True,
        "cross_document_chunking": True
    }
)

# Chunk processed documents
documents = get_processed_documents()
chunks = chunking_stage.run(documents)

print(f"Generated {len(chunks)} chunks from {len(documents)} documents")
```

### Embedding Stage

```python
from src.orchestration.pipelines.stages.embedding import EmbeddingStage

# Initialize embedding stage
embedding_stage = EmbeddingStage(
    config={
        "model": "modernbert",
        "batch_size": 32,
        "normalize": True,
        "cache_embeddings": True,
        "gpu_acceleration": True
    }
)

# Generate embeddings
chunks = get_document_chunks()
embeddings_result = embedding_stage.run(chunks)

chunks_with_embeddings = embeddings_result.processed_chunks
print(f"Generated embeddings for {len(chunks_with_embeddings)} chunks")
```

### ISNE Enhancement Stage

```python
from src.orchestration.pipelines.stages.isne import ISNEStage

# Initialize ISNE stage
isne_stage = ISNEStage(
    config={
        "model_path": "./models/refined_isne_model.pt",
        "enhance_embeddings": True,
        "build_relationships": True,
        "batch_processing": True
    }
)

# Enhance embeddings with graph information
chunks_with_embeddings = get_embedded_chunks()
enhanced_result = isne_stage.run(chunks_with_embeddings)

enhanced_chunks = enhanced_result.enhanced_chunks
print(f"Enhanced {len(enhanced_chunks)} chunks with ISNE")
```

### Storage Stage

```python
from src.orchestration.pipelines.stages.storage import StorageStage

# Initialize storage stage
storage_stage = StorageStage(
    config={
        "backend": "arango",
        "batch_size": 100,
        "create_indexes": True,
        "validate_schema": True
    }
)

# Store processed data
enhanced_chunks = get_enhanced_chunks()
storage_result = storage_stage.run(enhanced_chunks)

print(f"Stored {storage_result.stored_count} chunks")
print(f"Created {storage_result.relationship_count} relationships")
```

## 🔍 Monitoring and Performance

### Pipeline Monitoring

```python
from src.orchestration.core.monitoring import PipelineMonitor

# Initialize monitoring
monitor = PipelineMonitor(
    config={
        "metrics_backend": "prometheus",  # prometheus, json, database
        "alert_thresholds": {
            "error_rate": 0.05,           # 5% error rate
            "processing_time": 300,       # 5 minutes per document
            "queue_size": 1000            # Max queue size
        }
    }
)

# Monitor pipeline execution
@monitor.track_stage("document_processing")
def process_documents_with_monitoring(documents):
    results = []
    for doc in documents:
        try:
            with monitor.track_operation("process_single_doc"):
                result = process_document(doc)
                monitor.record_success("document_processed")
                results.append(result)
        except Exception as e:
            monitor.record_error("document_processing_failed", str(e))
            continue
    return results

# Get monitoring data
metrics = monitor.get_metrics()
print(f"Success rate: {metrics['success_rate']:.2%}")
print(f"Average processing time: {metrics['avg_processing_time']:.2f}s")
print(f"Current queue size: {metrics['queue_size']}")
```

### Performance Optimization

```python
from src.orchestration.core.parallel_worker import OptimizedWorkerPool

class HighPerformancePipeline:
    """High-performance pipeline with advanced optimizations."""
    
    def __init__(self):
        self.worker_pool = OptimizedWorkerPool(
            num_workers=16,
            worker_type="process",
            optimization_config={
                "cpu_affinity": True,
                "memory_limit": "4GB",
                "gpu_allocation": "round_robin"
            }
        )
        
        self.setup_optimization()
    
    def setup_optimization(self):
        """Configure performance optimizations."""
        # Enable memory-mapped file processing
        self.enable_memory_mapping = True
        
        # Configure batching optimization
        self.adaptive_batching = True
        self.batch_size_range = (16, 128)
        
        # Enable result caching
        self.result_cache = LRUCache(maxsize=10000)
    
    async def process_optimized_batch(self, documents):
        """Process documents with advanced optimizations."""
        
        # Pre-filter and deduplicate
        documents = self.deduplicate_documents(documents)
        
        # Adaptive batch sizing
        optimal_batch_size = self.calculate_optimal_batch_size(documents)
        
        # Parallel processing with load balancing
        tasks = []
        for batch in self.create_batches(documents, optimal_batch_size):
            worker = self.worker_pool.get_least_loaded_worker()
            task = worker.process_batch_async(batch)
            tasks.append(task)
        
        # Gather results with timeout handling
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        successful_results = []
        failed_results = []
        
        for result in results:
            if isinstance(result, Exception):
                failed_results.append(result)
            else:
                successful_results.extend(result)
        
        return {
            "successful": successful_results,
            "failed": failed_results,
            "processing_stats": self.get_processing_stats()
        }
```

## 🔧 Configuration and Customization

### Pipeline Configuration

Configure pipeline behavior in `training_pipeline_config.yaml`:

```yaml
pipeline:
  # Global settings
  name: "hades_processing_pipeline"
  version: "1.0"
  
  # Execution settings
  execution:
    parallel: true
    max_workers: 8
    worker_type: "process"  # process, thread, async
    timeout: 3600          # 1 hour
    
    # Checkpointing
    enable_checkpoints: true
    checkpoint_interval: 100
    checkpoint_dir: "./checkpoints/"
    
    # Error handling
    max_retries: 3
    retry_delay: 5
    continue_on_error: true
    
  # Stage configurations
  stages:
    document_processing:
      enabled: true
      config:
        adapters: ["pdf", "python", "markdown", "json", "yaml"]
        batch_size: 10
        timeout_per_document: 300
        
    chunking:
      enabled: true
      config:
        strategy: "adaptive"
        chunk_size: 512
        overlap: 64
        min_chunk_size: 50
        
    embedding:
      enabled: true
      config:
        model: "modernbert"
        batch_size: 32
        normalize_embeddings: true
        cache_embeddings: true
        
    isne:
      enabled: true
      config:
        model_path: "./models/refined_isne_model.pt"
        batch_processing: true
        build_relationships: true
        
    storage:
      enabled: true
      config:
        backend: "arango"
        batch_size: 100
        create_indexes: true

# Monitoring configuration
monitoring:
  enabled: true
  metrics_interval: 30
  alert_thresholds:
    error_rate: 0.05
    processing_time: 600
    memory_usage: 0.85
    
# Resource management
resources:
  cpu_limit: 16
  memory_limit: "32GB"
  gpu_limit: 4
  temp_storage: "./temp/"
  max_temp_size: "10GB"
```

### Custom Stage Development

```python
from src.orchestration.pipelines.stages.base import BasePipelineStage
from typing import List, Any

class CustomProcessingStage(BasePipelineStage):
    """Custom pipeline stage for specialized processing."""
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.custom_processor = self.initialize_processor()
    
    def initialize_processor(self):
        """Initialize custom processing logic."""
        # Implement custom initialization
        pass
    
    def run(self, input_data: List[Any]) -> List[Any]:
        """Execute custom processing stage."""
        try:
            self.logger.info(f"Processing {len(input_data)} items in custom stage")
            
            results = []
            for item in input_data:
                # Custom processing logic
                processed_item = self.process_single_item(item)
                results.append(processed_item)
                
                # Update progress
                self.update_progress(len(results), len(input_data))
            
            self.logger.info(f"Custom stage completed: {len(results)} items processed")
            return results
            
        except Exception as e:
            self.logger.error(f"Custom stage failed: {e}")
            raise
    
    def process_single_item(self, item: Any) -> Any:
        """Process a single item."""
        # Implement custom processing logic
        return self.custom_processor.process(item)
    
    def validate_input(self, input_data: List[Any]) -> bool:
        """Validate input data for custom stage."""
        return all(self.is_valid_item(item) for item in input_data)
    
    def get_output_schema(self) -> dict:
        """Define output schema for validation."""
        return {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "processed_data": {"type": "object"},
                    "metadata": {"type": "object"}
                }
            }
        }

# Register custom stage
from src.orchestration.pipelines.registry import register_stage
register_stage("custom_processing", CustomProcessingStage)
```

## 🛠️ Development and Testing

### Pipeline Testing Framework

```python
import pytest
from src.orchestration.pipelines.text_pipeline import TextProcessingPipeline

def test_text_pipeline_basic():
    """Test basic text pipeline functionality."""
    pipeline = TextProcessingPipeline(
        config={
            "stages": ["docproc", "chunking"],
            "parallel": False,  # Disable for testing
            "output_dir": "./test_output/"
        }
    )
    
    # Test with sample documents
    test_files = ["./test_data/sample.pdf", "./test_data/sample.py"]
    results = pipeline.process_documents(test_files)
    
    assert results["success"] == True
    assert results["total_documents"] == 2
    assert results["total_chunks"] > 0

def test_parallel_processing():
    """Test parallel processing functionality."""
    from src.orchestration.pipelines.parallel_pipeline import ParallelPipeline
    
    pipeline = ParallelPipeline(worker_config={"num_workers": 2})
    
    # Create test data
    test_data = [f"test_document_{i}" for i in range(10)]
    
    results = pipeline.process_parallel(
        input_data=test_data,
        batch_size=2
    )
    
    assert len(results) == len(test_data)

def test_stage_isolation():
    """Test stage error isolation."""
    from src.orchestration.pipelines.stages.chunking import ChunkingStage
    
    stage = ChunkingStage()
    
    # Test with invalid input
    invalid_input = [{"invalid": "data"}]
    
    with pytest.raises(Exception):
        stage.run(invalid_input)
    
    # Test recovery
    valid_input = create_valid_test_documents()
    results = stage.run(valid_input)
    
    assert len(results) > 0

def test_monitoring_integration():
    """Test pipeline monitoring."""
    from src.orchestration.core.monitoring import PipelineMonitor
    
    monitor = PipelineMonitor()
    
    # Test metric collection
    with monitor.track_operation("test_operation"):
        time.sleep(0.1)  # Simulate work
    
    metrics = monitor.get_metrics()
    assert "test_operation" in metrics
    assert metrics["test_operation"]["count"] > 0
```

### Performance Benchmarking

```python
from src.orchestration.pipelines.orchestrator import PipelineOrchestrator
import time

class PipelineBenchmark:
    """Benchmark pipeline performance."""
    
    def __init__(self):
        self.orchestrator = PipelineOrchestrator()
        self.results = []
    
    def benchmark_document_processing(self, document_sizes: List[int]):
        """Benchmark processing speed vs document size."""
        for size in document_sizes:
            test_docs = self.create_test_documents(size)
            
            start_time = time.time()
            results = self.orchestrator.process_documents(test_docs)
            processing_time = time.time() - start_time
            
            self.results.append({
                "document_count": len(test_docs),
                "processing_time": processing_time,
                "throughput": len(test_docs) / processing_time,
                "memory_usage": self.get_memory_usage()
            })
    
    def benchmark_parallel_scaling(self, worker_counts: List[int]):
        """Benchmark scaling with different worker counts."""
        test_docs = self.create_test_documents(100)
        
        for worker_count in worker_counts:
            pipeline = ParallelPipeline(
                worker_config={"num_workers": worker_count}
            )
            
            start_time = time.time()
            results = pipeline.process_parallel(test_docs)
            processing_time = time.time() - start_time
            
            efficiency = (len(test_docs) / processing_time) / worker_count
            
            self.results.append({
                "worker_count": worker_count,
                "processing_time": processing_time,
                "efficiency": efficiency,
                "scaling_factor": processing_time / self.results[0]["processing_time"]
            })
    
    def generate_report(self):
        """Generate performance report."""
        return {
            "benchmark_results": self.results,
            "recommendations": self.get_optimization_recommendations(),
            "system_info": self.get_system_info()
        }
```

## 📚 Related Documentation

- **Pipeline Stages**: See individual stage documentation in `pipelines/stages/`
- **Configuration**: See `config/README.md` for pipeline configuration
- **Types**: See `types/orchestration/` for orchestration type definitions
- **Monitoring**: See `alerts/README.md` for monitoring and alerting
- **Performance**: See performance tuning guides for optimization strategies

## 🎯 Best Practices

1. **Design modular stages** - Create reusable, composable pipeline stages
2. **Handle errors gracefully** - Implement proper error handling and recovery
3. **Monitor performance** - Track pipeline metrics and resource usage
4. **Use appropriate parallelism** - Choose between process, thread, or async workers
5. **Implement checkpointing** - Enable recovery from pipeline failures
6. **Validate data flow** - Ensure data consistency between pipeline stages
7. **Optimize resource usage** - Tune worker counts and batch sizes for your hardware
8. **Test thoroughly** - Validate pipeline behavior with comprehensive tests
