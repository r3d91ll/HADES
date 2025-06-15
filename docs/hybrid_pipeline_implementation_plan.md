# Hybrid Streaming-Batch Pipeline Implementation Plan
## Production-Ready Design with Scalability and Reliability

### Executive Summary

This document outlines a production-ready implementation plan for the hybrid streaming-batch ISNE pipeline, incorporating critical scalability, reliability, and operational concerns identified through architectural review. The plan prioritizes robust data management, memory safety, and error recovery over speed of implementation.

**Revised Timeline: 6-8 weeks** (vs original 4-5 weeks)
**Focus: Production reliability over rapid deployment**

---

## Phase 1: Foundation Infrastructure (3-4 weeks)

### 1.1 Sharded Embedding Storage Architecture (Week 1-2)

#### **Core Storage Design**
```python
class ShardedEmbeddingStorage:
    """Production-grade embedding storage with sharding and versioning."""
    
    def __init__(self, 
                 shard_size: int = 50000,  # 50K chunks per shard (~76MB)
                 storage_format: str = "parquet",  # Better for analytics
                 compression: str = "snappy"):
        self.shard_size = shard_size
        self.storage_format = storage_format
        self.compression = compression
        
    async def store_embeddings_stream(self, 
                                    embedding_stream: AsyncIterator[Embedding],
                                    run_id: str) -> List[Path]:
        """Stream embeddings to multiple shards with atomic writes."""
        
    def create_shard_metadata(self, shard_id: int, embedding_count: int) -> Dict:
        """Create comprehensive shard metadata."""
        return {
            'shard_id': shard_id,
            'embedding_count': embedding_count,
            'file_size_bytes': int,
            'checksum_sha256': str,
            'creation_timestamp': datetime,
            'model_info': {
                'model_name': 'ModernBERT',
                'embedding_dim': 384,
                'chunking_strategy': 'ast_aware_v1',
                'version': '1.0'
            },
            'chunk_metadata': {
                'chunk_ids': List[str],
                'document_sources': List[str],
                'processing_stats': Dict
            }
        }
```

#### **Storage Layout**
```
./checkpoints/{run_id}/
├── embeddings/
│   ├── shard_000.parquet          # 50K embeddings + metadata
│   ├── shard_001.parquet
│   └── shard_N.parquet
├── metadata/
│   ├── run_manifest.json          # Overall run metadata
│   ├── shard_index.json           # Shard location index
│   └── validation_checksums.json  # Integrity validation
└── logs/
    ├── processing.log
    └── error_recovery.log
```

#### **Versioning and Compatibility**
```python
class EmbeddingVersionManager:
    """Handle embedding format evolution and compatibility."""
    
    SUPPORTED_VERSIONS = ["1.0", "1.1"]
    CURRENT_VERSION = "1.1"
    
    def validate_compatibility(self, embedding_metadata: Dict) -> bool:
        """Check if stored embeddings are compatible with current system."""
        
    def migrate_format(self, old_path: Path, new_path: Path) -> bool:
        """Migrate embeddings between versions."""
        
    def get_format_schema(self, version: str) -> Dict:
        """Get schema definition for embedding format version."""
```

### 1.2 Memory-Bounded Graph Construction (Week 2-3)

#### **Chunked Similarity Computation**
```python
class ChunkedGraphConstruction:
    """Memory-safe graph construction for large embedding sets."""
    
    def __init__(self, 
                 max_memory_gb: float = 4.0,
                 similarity_threshold: float = 0.5,
                 max_edges_per_node: int = 20):
        self.max_memory_gb = max_memory_gb
        self.similarity_threshold = similarity_threshold
        self.max_edges_per_node = max_edges_per_node
        
    def calculate_chunk_size(self, embedding_dim: int, total_embeddings: int) -> int:
        """Calculate safe chunk size based on memory constraints."""
        # Similarity matrix memory: chunk_size^2 * 4 bytes
        # Target: 25% of available memory for similarity matrix
        target_memory_bytes = self.max_memory_gb * 0.25 * 1024**3
        max_chunk_size = int(np.sqrt(target_memory_bytes / 4))
        
        # Ensure we can process all embeddings
        min_chunk_size = min(10000, total_embeddings)
        return max(min_chunk_size, min(max_chunk_size, 50000))
    
    async def build_graph_chunked(self, 
                                embedding_shards: List[Path],
                                output_path: Path) -> GraphConstructionResult:
        """Build graph using memory-bounded chunked processing."""
        
        chunk_size = self.calculate_chunk_size(384, total_embeddings)
        
        # Process embeddings in overlapping chunks
        graph_builder = IncrementalGraphBuilder()
        cross_chunk_handler = CrossChunkSimilarityHandler(
            overlap_size=5000, 
            top_k=100,
            similarity_threshold=self.similarity_threshold
        )
        
        prev_chunk_overlap = None
        prev_indices_overlap = None
        
        for i in range(0, total_embeddings, chunk_size):
            chunk_embeddings = await self.load_embedding_chunk(
                embedding_shards, i, chunk_size
            )
            chunk_indices = list(range(i, min(i + chunk_size, total_embeddings)))
            
            # Compute similarities within chunk
            similarity_matrix = self.compute_chunk_similarities(chunk_embeddings)
            
            # Add edges to graph
            edges = self.extract_edges(similarity_matrix, i)
            graph_builder.add_edges(edges)
            
            # Cross-chunk similarities with previous chunk
            if prev_chunk_overlap is not None:
                cross_chunk_edges = await cross_chunk_handler.compute_cross_chunk_similarities(
                    prev_chunk_overlap, chunk_embeddings,
                    prev_indices_overlap, chunk_indices
                )
                graph_builder.add_edges(cross_chunk_edges)
            
            # Prepare overlap for next iteration
            prev_chunk_overlap, prev_indices_overlap = cross_chunk_handler.prepare_overlap(
                chunk_embeddings, chunk_indices
            )
                
            # Memory cleanup
            del similarity_matrix, chunk_embeddings
            
        return graph_builder.finalize()
```

#### **Cross-Chunk Similarity Handler**
```python
class CrossChunkSimilarityHandler:
    """Handle similarity computation between adjacent chunks efficiently."""
    
    def __init__(self, 
                 overlap_size: int = 5000,
                 top_k: int = 100, 
                 similarity_threshold: float = 0.5,
                 use_lsh: bool = True):
        """
        overlap_size: Number of embeddings from previous chunk to keep
        top_k: Keep only top-k most similar candidates from each chunk  
        similarity_threshold: Minimum similarity for edge creation
        use_lsh: Use Locality Sensitive Hashing for large chunks
        """
        self.overlap_size = overlap_size
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.use_lsh = use_lsh
        
        if use_lsh:
            from datasketch import MinHashLSH
            self.lsh_index = MinHashLSH(threshold=similarity_threshold, num_perm=128)
    
    def prepare_overlap(self, 
                       chunk_embeddings: np.ndarray, 
                       chunk_indices: List[int]) -> Tuple[np.ndarray, List[int]]:
        """Prepare overlap embeddings for next cross-chunk computation."""
        if len(chunk_embeddings) <= self.overlap_size:
            return chunk_embeddings, chunk_indices
        
        # Take the last overlap_size embeddings
        overlap_embeddings = chunk_embeddings[-self.overlap_size:]
        overlap_indices = chunk_indices[-self.overlap_size:]
        
        return overlap_embeddings, overlap_indices
    
    async def compute_cross_chunk_similarities(self,
                                             prev_chunk: np.ndarray,
                                             curr_chunk: np.ndarray,
                                             prev_indices: List[int],
                                             curr_indices: List[int]) -> List[Tuple[int, int, float]]:
        """Compute similarities between chunks with memory efficiency."""
        
        if self.use_lsh and len(prev_chunk) > 1000:
            return await self._compute_lsh_similarities(
                prev_chunk, curr_chunk, prev_indices, curr_indices
            )
        else:
            return await self._compute_direct_similarities(
                prev_chunk, curr_chunk, prev_indices, curr_indices
            )
    
    async def _compute_direct_similarities(self,
                                         prev_chunk: np.ndarray,
                                         curr_chunk: np.ndarray,
                                         prev_indices: List[int],
                                         curr_indices: List[int]) -> List[Tuple[int, int, float]]:
        """Direct similarity computation for smaller chunks."""
        
        # Compute similarity matrix between chunks
        similarities = np.dot(prev_chunk, curr_chunk.T)
        
        # Find edges above threshold
        edges = []
        prev_rows, curr_cols = np.where(similarities >= self.similarity_threshold)
        
        for prev_idx, curr_idx in zip(prev_rows, curr_cols):
            similarity = similarities[prev_idx, curr_idx]
            edge = (prev_indices[prev_idx], curr_indices[curr_idx], similarity)
            edges.append(edge)
        
        # Limit edges per node to prevent explosion
        edges = self._limit_edges_per_node(edges)
        
        return edges
    
    async def _compute_lsh_similarities(self,
                                      prev_chunk: np.ndarray,
                                      curr_chunk: np.ndarray,
                                      prev_indices: List[int],
                                      curr_indices: List[int]) -> List[Tuple[int, int, float]]:
        """LSH-based similarity computation for large chunks."""
        
        # Convert embeddings to LSH signatures
        prev_signatures = self._embeddings_to_lsh(prev_chunk)
        curr_signatures = self._embeddings_to_lsh(curr_chunk)
        
        # Find candidate pairs using LSH
        candidate_pairs = []
        for i, curr_sig in enumerate(curr_signatures):
            # Query LSH index for similar items
            similar_items = self.lsh_index.query(curr_sig)
            for similar_idx in similar_items:
                if similar_idx < len(prev_signatures):
                    candidate_pairs.append((similar_idx, i))
        
        # Compute exact similarities for candidates only
        edges = []
        for prev_idx, curr_idx in candidate_pairs:
            similarity = np.dot(prev_chunk[prev_idx], curr_chunk[curr_idx])
            if similarity >= self.similarity_threshold:
                edge = (prev_indices[prev_idx], curr_indices[curr_idx], similarity)
                edges.append(edge)
        
        return self._limit_edges_per_node(edges)
    
    def _limit_edges_per_node(self, edges: List[Tuple[int, int, float]], 
                             max_edges: int = 20) -> List[Tuple[int, int, float]]:
        """Limit edges per node to prevent graph explosion."""
        from collections import defaultdict
        
        edges_by_node = defaultdict(list)
        for edge in edges:
            edges_by_node[edge[0]].append(edge)
            edges_by_node[edge[1]].append(edge)
        
        # Keep only top-k edges per node
        filtered_edges = set()
        for node_edges in edges_by_node.values():
            # Sort by similarity (descending)
            sorted_edges = sorted(node_edges, key=lambda x: x[2], reverse=True)
            for edge in sorted_edges[:max_edges]:
                filtered_edges.add(edge)
        
        return list(filtered_edges)
```

#### **Incremental Graph Builder**
```python
class IncrementalGraphBuilder:
    """Build graph incrementally to avoid memory spikes."""
    
    def __init__(self):
        self.edge_buffer = []
        self.edge_buffer_size = 100000  # Buffer before writing
        self.total_edges = 0
        
    def add_edges(self, edges: List[Tuple[int, int, float]]):
        """Add edges with buffered writing."""
        self.edge_buffer.extend(edges)
        
        if len(self.edge_buffer) >= self.edge_buffer_size:
            self.flush_edges()
            
    def flush_edges(self):
        """Write buffered edges to storage."""
        # Sort and deduplicate edges
        # Write to temporary edge files
        # Clear buffer
        
    def finalize(self) -> Graph:
        """Create final graph from all edge files."""
        # Merge all edge files
        # Create final graph structure
        # Clean up temporary files
```

### 1.3 Robust Error Recovery System (Week 3-4)

#### **Checkpoint Validation and Recovery**
```python
class CheckpointManager:
    """Handle checkpoint validation and recovery."""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        
    def validate_checkpoint_integrity(self, run_id: str) -> ValidationResult:
        """Comprehensive checkpoint validation."""
        validation_result = ValidationResult()
        
        # Check run manifest exists and is valid
        manifest_path = self.checkpoint_dir / run_id / "metadata" / "run_manifest.json"
        if not self.validate_manifest(manifest_path):
            validation_result.add_error("Invalid or missing run manifest")
            
        # Validate all embedding shards
        shard_results = self.validate_embedding_shards(run_id)
        validation_result.merge(shard_results)
        
        # Check shard completeness
        expected_shards = self.get_expected_shard_count(run_id)
        actual_shards = len(self.get_shard_paths(run_id))
        if actual_shards != expected_shards:
            validation_result.add_error(
                f"Incomplete shards: {actual_shards}/{expected_shards}"
            )
            
        return validation_result
    
    def recover_partial_checkpoint(self, run_id: str) -> RecoveryResult:
        """Recover from partial checkpoint generation."""
        # Identify last complete shard
        complete_shards = self.find_complete_shards(run_id)
        
        # Calculate resumption point
        total_processed = sum(shard.embedding_count for shard in complete_shards)
        
        # Create recovery manifest
        recovery_manifest = {
            'recovery_timestamp': datetime.now(),
            'last_complete_shard': max(shard.shard_id for shard in complete_shards),
            'embeddings_processed': total_processed,
            'resumption_strategy': 'continue_from_last_shard'
        }
        
        return RecoveryResult(
            can_recover=True,
            resumption_point=total_processed,
            recovery_manifest=recovery_manifest
        )
    
    def validate_embedding_shards(self, run_id: str) -> ValidationResult:
        """Validate individual embedding shards."""
        result = ValidationResult()
        shard_paths = self.get_shard_paths(run_id)
        
        for shard_path in shard_paths:
            try:
                # Check file integrity
                if not self.verify_shard_checksum(shard_path):
                    result.add_error(f"Checksum mismatch: {shard_path}")
                    continue
                    
                # Validate embedding dimensions
                embeddings = self.load_shard_embeddings(shard_path)
                if embeddings.shape[1] != 384:
                    result.add_error(f"Invalid embedding dim: {shard_path}")
                    
                # Check for NaN/inf values
                if not np.isfinite(embeddings).all():
                    result.add_error(f"Invalid embedding values: {shard_path}")
                    
            except Exception as e:
                result.add_error(f"Shard validation failed {shard_path}: {e}")
                
        return result
```

#### **Atomic Operations and Consistency**
```python
class AtomicWriter:
    """Ensure atomic writes for checkpoint files."""
    
    @staticmethod
    async def write_atomic(data: Any, target_path: Path):
        """Write data atomically with checksum validation."""
        temp_path = target_path.with_suffix('.tmp')
        
        try:
            # Write to temporary file
            await asyncio.to_thread(joblib.dump, data, temp_path)
            
            # Calculate and verify checksum
            checksum = hashlib.sha256(temp_path.read_bytes()).hexdigest()
            
            # Write checksum file
            checksum_path = target_path.with_suffix('.checksum')
            checksum_path.write_text(checksum)
            
            # Atomic move
            temp_path.rename(target_path)
            
        except Exception as e:
            # Cleanup on failure
            if temp_path.exists():
                temp_path.unlink()
            raise e
```

#### **Storage Failure Handler**
```python
class StorageFailureHandler:
    """Handle storage write and read failures gracefully."""
    
    def __init__(self, max_retries: int = 3, backup_enabled: bool = True):
        self.max_retries = max_retries
        self.backup_enabled = backup_enabled
        self.alternate_locations = [
            "/tmp/hades_backup",
            "/var/tmp/hades_backup"
        ]
    
    async def handle_write_failure(self, 
                                 shard_path: Path, 
                                 data: Any, 
                                 error: Exception,
                                 retry_count: int = 0) -> bool:
        """Handle storage write failures with retries and fallbacks."""
        
        logger.error(f"Storage write failed for {shard_path}: {error}")
        
        if retry_count < self.max_retries:
            # Exponential backoff retry
            wait_time = 2 ** retry_count
            await asyncio.sleep(wait_time)
            
            try:
                await AtomicWriter.write_atomic(data, shard_path)
                logger.info(f"Retry {retry_count + 1} successful for {shard_path}")
                return True
            except Exception as retry_error:
                return await self.handle_write_failure(
                    shard_path, data, retry_error, retry_count + 1
                )
        
        # Try alternate storage locations
        for alt_location in self.alternate_locations:
            try:
                alt_path = Path(alt_location) / shard_path.name
                alt_path.parent.mkdir(parents=True, exist_ok=True)
                await AtomicWriter.write_atomic(data, alt_path)
                
                # Update manifest with alternate location
                await self._update_manifest_alternate_location(shard_path, alt_path)
                logger.warning(f"Stored {shard_path.name} in alternate location: {alt_path}")
                return True
                
            except Exception as alt_error:
                logger.error(f"Alternate storage failed for {alt_path}: {alt_error}")
                continue
        
        # All storage attempts failed
        await self._mark_shard_failed(shard_path, error)
        return False
    
    async def handle_corrupted_read(self, 
                                  shard_path: Path, 
                                  error: Exception) -> Optional[Any]:
        """Handle corrupted shard reads with recovery attempts."""
        
        logger.error(f"Corrupted shard detected: {shard_path}: {error}")
        
        # Try to read from backup if available
        if self.backup_enabled:
            backup_path = self._get_backup_path(shard_path)
            if backup_path.exists():
                try:
                    data = await asyncio.to_thread(joblib.load, backup_path)
                    logger.info(f"Recovered data from backup: {backup_path}")
                    return data
                except Exception as backup_error:
                    logger.error(f"Backup read failed: {backup_error}")
        
        # Check alternate locations
        for alt_location in self.alternate_locations:
            alt_path = Path(alt_location) / shard_path.name
            if alt_path.exists():
                try:
                    data = await asyncio.to_thread(joblib.load, alt_path)
                    logger.info(f"Recovered data from alternate location: {alt_path}")
                    return data
                except Exception as alt_error:
                    logger.error(f"Alternate read failed: {alt_error}")
                    continue
        
        # Mark shard for regeneration
        await self._mark_shard_for_regeneration(shard_path)
        return None
    
    async def _update_manifest_alternate_location(self, 
                                                original_path: Path, 
                                                alternate_path: Path):
        """Update run manifest with alternate storage location."""
        # Implementation to update manifest with alternate paths
        pass
    
    async def _mark_shard_failed(self, shard_path: Path, error: Exception):
        """Mark shard as failed in recovery manifest."""
        failed_shards_log = shard_path.parent / "failed_shards.json"
        failure_record = {
            'shard_path': str(shard_path),
            'failure_timestamp': datetime.now().isoformat(),
            'error_message': str(error),
            'recovery_needed': True
        }
        
        # Append to failed shards log
        existing_failures = []
        if failed_shards_log.exists():
            with open(failed_shards_log) as f:
                existing_failures = json.load(f)
        
        existing_failures.append(failure_record)
        
        with open(failed_shards_log, 'w') as f:
            json.dump(existing_failures, f, indent=2)
```

---

## Phase 2: Streaming Front-End Implementation (2-3 weeks)

### 2.1 Adaptive Queue Management (Week 5)

#### **Dynamic Queue Sizing**
```python
class AdaptiveQueueManager:
    """Dynamically size queues based on processing rates."""
    
    def __init__(self):
        self.rate_tracker = ProcessingRateTracker()
        self.queue_configs = {}
        
    def calculate_queue_size(self, 
                           stage_name: str,
                           upstream_rate: float, 
                           downstream_rate: float,
                           target_latency_seconds: float = 30.0) -> int:
        """Calculate optimal queue size based on processing rates."""
        
        if downstream_rate <= 0:
            return 1000  # Default for startup
            
        # Account for rate difference and target latency
        rate_difference = max(0, upstream_rate - downstream_rate)
        buffer_size = int(rate_difference * target_latency_seconds)
        
        # Add safety factor and bounds
        safety_factor = 1.5
        min_size = 50
        max_size = 5000
        
        optimal_size = int(buffer_size * safety_factor)
        return max(min_size, min(optimal_size, max_size))
    
    async def adjust_queue_sizes(self, pipeline_metrics: Dict[str, float]):
        """Dynamically adjust queue sizes based on current metrics."""
        
        # Calculate new sizes based on current rates
        new_configs = {}
        for stage in ['document_processing', 'chunking', 'embedding']:
            upstream_rate = pipeline_metrics.get(f'{stage}_input_rate', 0)
            downstream_rate = pipeline_metrics.get(f'{stage}_output_rate', 0)
            
            new_size = self.calculate_queue_size(stage, upstream_rate, downstream_rate)
            new_configs[stage] = new_size
        
        # Apply changes if significant difference
        await self.apply_queue_config_changes(new_configs)
```

#### **GPU Memory-Aware Batching**
```python
class GPUMemoryProfiler:
    """Profile actual GPU memory usage for embeddings."""
    
    def __init__(self):
        self.memory_per_token_cache = None
        self.last_profile_time = 0
        self.profile_interval = 3600  # Re-profile every hour
        
    def profile_memory_per_token(self, 
                                model: Any = None, 
                                sample_size: int = 100) -> float:
        """Profile actual memory usage per token with statistical rigor."""
        
        # Check if we need to re-profile
        current_time = time.time()
        if (self.memory_per_token_cache is not None and 
            current_time - self.last_profile_time < self.profile_interval):
            return self.memory_per_token_cache
        
        logger.info("Profiling GPU memory usage per token...")
        
        # Generate sample texts of varying lengths
        sample_texts = self._generate_sample_texts(sample_size)
        memory_measurements = []
        
        try:
            for text_length in [100, 500, 1000, 2000, 4000]:
                # Filter texts by length
                texts = [t for t in sample_texts if len(t.split()) <= text_length]
                if len(texts) < 5:  # Need sufficient samples
                    continue
                
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                
                # Measure baseline memory
                baseline = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                
                # Process batch
                with torch.no_grad():
                    if model:
                        embeddings = model.encode(texts)
                    else:
                        # Simulate embedding computation
                        total_tokens = sum(len(t.split()) for t in texts)
                        embeddings = torch.randn(len(texts), 384, device='cuda' if torch.cuda.is_available() else 'cpu')
                
                # Measure peak memory
                peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else baseline
                
                # Calculate per-token usage
                total_tokens = sum(len(t.split()) for t in texts)
                if total_tokens > 0:
                    memory_per_token = (peak - baseline) / total_tokens
                    memory_measurements.append(memory_per_token)
                    
                    logger.debug(f"Length {text_length}: {memory_per_token:.2f} bytes/token")
        
        except Exception as e:
            logger.error(f"Memory profiling failed: {e}")
            # Return conservative default
            return 1024.0  # 1KB per token default
        
        if memory_measurements:
            # Return conservative estimate (75th percentile for safety)
            result = float(np.percentile(memory_measurements, 75))
            self.memory_per_token_cache = result
            self.last_profile_time = current_time
            
            logger.info(f"Memory profiling complete: {result:.2f} bytes/token (p75)")
            return result
        else:
            logger.warning("No memory measurements obtained, using default")
            return 1024.0
    
    def _generate_sample_texts(self, sample_size: int) -> List[str]:
        """Generate diverse sample texts for profiling."""
        import random
        import string
        
        samples = []
        
        # Short texts (technical terms, keywords)
        for _ in range(sample_size // 4):
            words = random.randint(5, 20)
            text = ' '.join(random.choices([
                'machine', 'learning', 'neural', 'network', 'embedding',
                'transformer', 'attention', 'gradient', 'optimization'
            ], k=words))
            samples.append(text)
        
        # Medium texts (sentences, code snippets)  
        for _ in range(sample_size // 4):
            words = random.randint(50, 200)
            text = ' '.join(random.choices(string.ascii_lowercase, k=words))
            samples.append(text)
        
        # Long texts (paragraphs, documentation)
        for _ in range(sample_size // 2):
            words = random.randint(200, 1000)
            text = ' '.join(random.choices(string.ascii_lowercase, k=words))
            samples.append(text)
        
        return samples
    
    def get_available_memory_mb(self) -> int:
        """Get currently available GPU memory in MB."""
        if not torch.cuda.is_available():
            return 0
        
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated()
        available_memory = total_memory - allocated_memory
        
        return int(available_memory / (1024 ** 2))  # Convert to MB


class GPUBatchOptimizer:
    """Optimize GPU batching based on actual memory usage."""
    
    def __init__(self, 
                 model_memory_mb: int = 2048,
                 target_utilization: float = 0.75):
        self.model_memory_mb = model_memory_mb
        self.target_utilization = target_utilization
        self.memory_profiler = GPUMemoryProfiler()
        
    def calculate_optimal_batch_size(self, 
                                   chunk_lengths: List[int],
                                   available_memory_mb: int) -> int:
        """Calculate batch size based on actual GPU memory constraints."""
        
        # Profile memory usage per token (if not cached)
        if not hasattr(self, '_memory_per_token'):
            self._memory_per_token = self.memory_profiler.profile_memory_per_token()
        
        # Calculate memory for attention matrices (quadratic in sequence length)
        def estimate_batch_memory(batch_size: int) -> int:
            total_tokens = sum(chunk_lengths[:batch_size])
            
            # Model memory + attention memory + gradient memory
            attention_memory = (total_tokens ** 2) * 4 / (1024 ** 2)  # MB
            gradient_memory = self.model_memory_mb * 0.3  # Rough estimate
            
            return self.model_memory_mb + attention_memory + gradient_memory
        
        # Binary search for optimal batch size
        left, right = 1, len(chunk_lengths)
        optimal_batch = 1
        
        target_memory = available_memory_mb * self.target_utilization
        
        while left <= right:
            mid = (left + right) // 2
            estimated_memory = estimate_batch_memory(mid)
            
            if estimated_memory <= target_memory:
                optimal_batch = mid
                left = mid + 1
            else:
                right = mid - 1
        
        # Enforce minimum efficient batch size for GPU processing
        MIN_EFFICIENT_BATCH = 8
        
        if optimal_batch < MIN_EFFICIENT_BATCH and available_memory_mb > 1000:
            # Try using shorter sequences to maintain batch efficiency
            sorted_chunks = sorted(enumerate(chunk_lengths), key=lambda x: x[1])
            
            # Find minimum batch that fits in memory with shorter sequences
            for batch_size in range(MIN_EFFICIENT_BATCH, min(len(sorted_chunks), 64)):
                subset_indices = [idx for idx, _ in sorted_chunks[:batch_size]]
                subset_lengths = [chunk_lengths[i] for i in subset_indices]
                
                estimated_memory = estimate_batch_memory_from_lengths(subset_lengths)
                if estimated_memory <= target_memory:
                    logger.info(f"Using shorter sequences: batch_size={batch_size}, "
                              f"avg_length={np.mean(subset_lengths):.0f}")
                    return subset_indices  # Return indices instead of count
                    
        return max(MIN_EFFICIENT_BATCH, optimal_batch)
        
    def estimate_batch_memory_from_lengths(self, lengths: List[int]) -> int:
        """Estimate memory usage for specific sequence lengths."""
        total_tokens = sum(lengths)
        
        # Model memory + attention memory (quadratic) + gradient memory
        attention_memory = (total_tokens ** 2) * 4 / (1024 ** 2)  # MB
        gradient_memory = self.model_memory_mb * 0.3  # Rough estimate
        
        return self.model_memory_mb + attention_memory + gradient_memory
    
    async def get_adaptive_batch(self, 
                               pending_chunks: List[Chunk]) -> List[Chunk]:
        """Get optimal batch based on current GPU state."""
        
        # Check current GPU memory
        available_memory = self.memory_profiler.get_available_memory_mb()
        
        # Sort chunks by length for better batching
        chunks_by_length = sorted(pending_chunks, key=lambda c: len(c.text))
        
        # Calculate optimal batch size
        chunk_lengths = [len(c.text) for c in chunks_by_length]
        batch_size = self.calculate_optimal_batch_size(chunk_lengths, available_memory)
        
        return chunks_by_length[:batch_size]
```

#### **Queue Starvation Detection and Recovery**
```python
class QueueStarvationDetector:
    """Detect and handle queue starvation events."""
    
    def __init__(self, starvation_threshold_seconds: float = 5.0):
        self.starvation_threshold = starvation_threshold_seconds
        self.last_item_timestamps = {}
        self.starvation_events = {}
        self.recovery_strategies = {}
        
    async def monitor_queue(self, 
                          queue_name: str, 
                          queue: asyncio.Queue,
                          upstream_stage: str,
                          downstream_stage: str):
        """Continuously monitor queue for starvation events."""
        
        logger.info(f"Starting starvation monitoring for {queue_name}")
        
        while True:
            try:
                current_time = time.time()
                queue_size = queue.qsize()
                
                if queue_size == 0:
                    # Queue is empty - check if this is starvation
                    last_item_time = self.last_item_timestamps.get(queue_name, current_time)
                    time_since_last = current_time - last_item_time
                    
                    if time_since_last > self.starvation_threshold:
                        await self._handle_starvation_event(
                            queue_name, time_since_last, upstream_stage, downstream_stage
                        )
                else:
                    # Queue has items - update timestamp and clear starvation
                    self.last_item_timestamps[queue_name] = current_time
                    if queue_name in self.starvation_events:
                        await self._handle_starvation_recovery(queue_name)
                
                # Log queue metrics
                if queue_size > 0:
                    logger.debug(f"Queue {queue_name}: {queue_size} items, healthy")
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Queue monitoring error for {queue_name}: {e}")
                await asyncio.sleep(5)  # Back off on errors
    
    async def _handle_starvation_event(self, 
                                     queue_name: str, 
                                     starvation_duration: float,
                                     upstream_stage: str,
                                     downstream_stage: str):
        """Handle detected queue starvation."""
        
        # Record starvation event
        if queue_name not in self.starvation_events:
            self.starvation_events[queue_name] = {
                'start_time': time.time() - starvation_duration,
                'upstream_stage': upstream_stage,
                'downstream_stage': downstream_stage,
                'duration': starvation_duration,
                'recovery_attempts': 0
            }
            
            logger.warning(f"Queue starvation detected: {queue_name} "
                         f"(empty for {starvation_duration:.1f}s)")
            
            # Alert monitoring system
            await self._send_starvation_alert(queue_name, starvation_duration)
        
        # Update starvation duration
        self.starvation_events[queue_name]['duration'] = starvation_duration
        
        # Attempt recovery if duration exceeds critical threshold
        if starvation_duration > 30.0:  # 30 seconds is critical
            await self._attempt_starvation_recovery(queue_name)
    
    async def _attempt_starvation_recovery(self, queue_name: str):
        """Attempt to recover from queue starvation."""
        
        event = self.starvation_events[queue_name]
        upstream_stage = event['upstream_stage']
        downstream_stage = event['downstream_stage']
        
        event['recovery_attempts'] += 1
        
        logger.warning(f"Attempting starvation recovery for {queue_name} "
                      f"(attempt {event['recovery_attempts']})")
        
        # Recovery strategies based on queue type
        if 'document' in queue_name.lower():
            await self._recover_document_queue_starvation(queue_name, upstream_stage)
        elif 'chunk' in queue_name.lower():
            await self._recover_chunk_queue_starvation(queue_name, upstream_stage)
        elif 'embedding' in queue_name.lower():
            await self._recover_embedding_queue_starvation(queue_name, upstream_stage)
        
        # If multiple recovery attempts fail, escalate
        if event['recovery_attempts'] >= 3:
            await self._escalate_starvation_failure(queue_name)
    
    async def _recover_document_queue_starvation(self, queue_name: str, upstream_stage: str):
        """Recover from document processing queue starvation."""
        
        # Check if upstream workers are stuck
        logger.info(f"Checking {upstream_stage} worker health...")
        
        # Strategies:
        # 1. Check file system access
        # 2. Restart stuck workers
        # 3. Reduce processing complexity temporarily
        # 4. Skip problematic files
        
        recovery_actions = [
            "Checked file system access",
            "Restarted processing workers", 
            "Reduced processing complexity",
            "Enabled problematic file skipping"
        ]
        
        for action in recovery_actions:
            logger.info(f"Recovery action: {action}")
            await asyncio.sleep(1)  # Simulate recovery action
    
    async def _recover_chunk_queue_starvation(self, queue_name: str, upstream_stage: str):
        """Recover from chunking queue starvation."""
        
        # Chunking starvation usually means:
        # 1. Document processing is slow
        # 2. Chunking workers are overwhelmed
        # 3. Complex documents causing delays
        
        logger.info(f"Recovering chunking starvation: {queue_name}")
        
        # Recovery strategies:
        # 1. Increase chunking worker pool
        # 2. Simplify chunking strategy temporarily
        # 3. Break large documents into smaller pieces
        
    async def _recover_embedding_queue_starvation(self, queue_name: str, upstream_stage: str):
        """Recover from embedding queue starvation."""
        
        # Embedding starvation usually means:
        # 1. GPU is overloaded or crashed
        # 2. Batch sizes are too large
        # 3. Memory issues
        
        logger.info(f"Recovering embedding starvation: {queue_name}")
        
        # Recovery strategies:
        # 1. Reduce batch sizes
        # 2. Clear GPU cache
        # 3. Restart embedding service
        # 4. Fall back to CPU processing temporarily
        
    async def _handle_starvation_recovery(self, queue_name: str):
        """Handle successful recovery from starvation."""
        
        if queue_name in self.starvation_events:
            event = self.starvation_events[queue_name]
            total_duration = time.time() - event['start_time']
            
            logger.info(f"Queue starvation recovered: {queue_name} "
                       f"(total duration: {total_duration:.1f}s)")
            
            # Record recovery metrics
            await self._record_recovery_metrics(queue_name, total_duration, event)
            
            # Clear starvation state
            del self.starvation_events[queue_name]
    
    async def _send_starvation_alert(self, queue_name: str, duration: float):
        """Send alert for queue starvation event."""
        
        alert_message = f"Queue starvation: {queue_name} empty for {duration:.1f}s"
        
        # Send to monitoring system
        logger.error(f"ALERT: {alert_message}")
        
        # Could integrate with alerting systems:
        # - Slack/Discord notifications
        # - PagerDuty alerts
        # - Email notifications
        
    async def _escalate_starvation_failure(self, queue_name: str):
        """Escalate starvation that couldn't be recovered."""
        
        event = self.starvation_events[queue_name]
        
        logger.critical(f"CRITICAL: Queue starvation escalation for {queue_name} "
                       f"after {event['recovery_attempts']} failed recovery attempts")
        
        # Critical escalation actions:
        # 1. Page on-call engineer
        # 2. Trigger emergency procedures
        # 3. Consider pipeline shutdown
        
    async def _record_recovery_metrics(self, 
                                     queue_name: str, 
                                     duration: float, 
                                     event: Dict):
        """Record starvation and recovery metrics."""
        
        metrics = {
            'queue_name': queue_name,
            'starvation_duration': duration,
            'recovery_attempts': event['recovery_attempts'],
            'upstream_stage': event['upstream_stage'],
            'downstream_stage': event['downstream_stage'],
            'timestamp': time.time()
        }
        
        # Store metrics for analysis and alerting
        logger.info(f"Starvation metrics recorded: {metrics}")
```

### 2.2 Parallel Document Processing (Week 5-6)

#### **Enhanced Hybrid Processing**
```python
class ProductionHybridProcessor:
    """Production-grade hybrid document processor with improved error handling."""
    
    def __init__(self, num_workers: int = 4):
        self.core_processor = CoreDocumentProcessor()
        self.docling_processor = DoclingDocumentProcessor()
        self.num_workers = num_workers
        self.error_handler = ProcessingErrorHandler()
        
    async def process_document_stream(self, 
                                    file_queue: asyncio.Queue,
                                    document_queue: asyncio.Queue) -> ProcessingStats:
        """Process documents with robust error handling and monitoring."""
        
        # Create worker tasks
        workers = [
            asyncio.create_task(
                self._process_worker(file_queue, document_queue, worker_id)
            )
            for worker_id in range(self.num_workers)
        ]
        
        # Monitor and collect results
        processing_stats = ProcessingStats()
        
        try:
            await asyncio.gather(*workers, return_exceptions=True)
        finally:
            # Collect final statistics
            for worker in workers:
                if not worker.cancelled():
                    worker_stats = await worker
                    processing_stats.merge(worker_stats)
                    
        return processing_stats
    
    async def _process_worker(self, 
                            file_queue: asyncio.Queue,
                            document_queue: asyncio.Queue,
                            worker_id: int) -> ProcessingStats:
        """Individual worker process with error isolation."""
        
        stats = ProcessingStats()
        
        while True:
            try:
                # Get file with timeout
                file_path = await asyncio.wait_for(
                    file_queue.get(), timeout=30.0
                )
                
                if file_path is None:  # Sentinel for shutdown
                    break
                    
                # Route to appropriate processor
                processor = self._route_processor(file_path)
                
                # Process with timeout and retries
                document = await self._process_with_retries(
                    processor, file_path, max_retries=3
                )
                
                if document:
                    await document_queue.put(document)
                    stats.documents_processed += 1
                else:
                    stats.documents_failed += 1
                    
            except asyncio.TimeoutError:
                stats.timeouts += 1
                continue
            except Exception as e:
                stats.errors += 1
                self.error_handler.handle_processing_error(file_path, e, worker_id)
                continue
                
        return stats
```

---

## Phase 3: Integration and Production Hardening (1-2 weeks)

### 3.1 Comprehensive Monitoring (Week 7)

#### **Hybrid-Specific Metrics**
```python
class HybridPipelineMonitor:
    """Comprehensive monitoring for hybrid pipeline."""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        
    def collect_streaming_metrics(self) -> Dict[str, float]:
        """Collect streaming front-end metrics."""
        return {
            # Throughput metrics
            'documents_per_second': self.get_processing_rate('document_processing'),
            'chunks_per_second': self.get_processing_rate('chunking'),
            'embeddings_per_second': self.get_processing_rate('embedding'),
            
            # Queue health
            'document_queue_depth': self.get_queue_depth('document_queue'),
            'chunk_queue_depth': self.get_queue_depth('chunk_queue'),
            'embedding_queue_depth': self.get_queue_depth('embedding_queue'),
            
            # Queue starvation events
            'queue_starvation_events': self.get_starvation_count(),
            
            # Storage metrics
            'storage_write_rate_mbps': self.get_storage_write_rate(),
            'checkpoint_validation_success_rate': self.get_validation_success_rate(),
            
            # GPU metrics
            'gpu_utilization_percent': self.get_gpu_utilization(),
            'gpu_memory_usage_percent': self.get_gpu_memory_usage(),
            'average_batch_size': self.get_average_batch_size(),
            
            # Error rates
            'processing_error_rate': self.get_error_rate('processing'),
            'storage_error_rate': self.get_error_rate('storage'),
            'recovery_success_rate': self.get_recovery_success_rate()
        }
    
    def collect_batch_metrics(self) -> Dict[str, float]:
        """Collect batch back-end metrics."""
        return {
            # Graph construction
            'graph_construction_time_seconds': self.get_stage_duration('graph_construction'),
            'similarity_computation_rate': self.get_similarity_computation_rate(),
            'memory_peak_usage_mb': self.get_peak_memory_usage(),
            
            # ISNE training
            'training_time_seconds': self.get_stage_duration('isne_training'),
            'training_convergence_epochs': self.get_convergence_epochs(),
            'final_loss_value': self.get_final_loss(),
            
            # Overall pipeline
            'end_to_end_latency_seconds': self.get_total_pipeline_time(),
            'checkpoint_load_time_seconds': self.get_checkpoint_load_time()
        }
```

### 3.2 Production Testing and Validation (Week 7-8)

#### **Comprehensive Test Suite**
```python
class HybridPipelineTestSuite:
    """Production testing for hybrid pipeline."""
    
    async def test_streaming_batch_boundary(self):
        """Test handoff between streaming and batch stages."""
        # Create test embeddings
        test_embeddings = self.create_test_embeddings(1000)
        
        # Stream to checkpoint
        checkpoint_path = await self.stream_to_checkpoint(test_embeddings)
        
        # Validate checkpoint
        validation_result = self.checkpoint_manager.validate_checkpoint_integrity(
            checkpoint_path
        )
        assert validation_result.is_valid
        
        # Load from checkpoint and compare
        loaded_embeddings = await self.load_from_checkpoint(checkpoint_path)
        np.testing.assert_array_almost_equal(test_embeddings, loaded_embeddings)
    
    async def test_partial_completion_recovery(self):
        """Test recovery from failures at different stages."""
        # Simulate failure after 70% completion
        partial_checkpoint = await self.simulate_partial_completion(0.7)
        
        # Attempt recovery
        recovery_result = self.checkpoint_manager.recover_partial_checkpoint(
            partial_checkpoint.run_id
        )
        
        assert recovery_result.can_recover
        assert recovery_result.resumption_point > 0
        
        # Resume and complete
        completed_checkpoint = await self.resume_from_checkpoint(recovery_result)
        assert completed_checkpoint.is_complete
    
    async def test_memory_bounds_compliance(self):
        """Test that memory usage stays within bounds."""
        memory_monitor = MemoryMonitor()
        
        # Start monitoring
        memory_monitor.start()
        
        try:
            # Run pipeline with large dataset
            result = await self.run_pipeline_with_memory_limits(
                max_memory_gb=6.0,
                test_embeddings=100000
            )
            
            # Check memory compliance
            peak_memory = memory_monitor.get_peak_usage_gb()
            assert peak_memory <= 6.0, f"Memory exceeded limit: {peak_memory}GB"
            
        finally:
            memory_monitor.stop()
    
    async def test_concurrent_pipeline_isolation(self):
        """Test multiple concurrent pipeline runs."""
        # Start 3 concurrent pipelines
        run_ids = [str(uuid.uuid4()) for _ in range(3)]
        
        pipeline_tasks = [
            asyncio.create_task(self.run_isolated_pipeline(run_id))
            for run_id in run_ids
        ]
        
        # Wait for completion
        results = await asyncio.gather(*pipeline_tasks, return_exceptions=True)
        
        # Verify all succeeded and are isolated
        for i, result in enumerate(results):
            assert not isinstance(result, Exception)
            assert result.run_id == run_ids[i]
            assert result.checkpoint_path.name == run_ids[i]
```

#### **Performance Regression Testing**
```python
class PerformanceBenchmark:
    """Automated performance regression testing."""
    
    def __init__(self, baseline_path: Path):
        self.baseline_metrics = self.load_baseline(baseline_path)
        
    async def run_performance_comparison(self, 
                                       test_dataset: Path) -> BenchmarkResult:
        """Run performance test and compare against baseline."""
        
        # Run current pipeline
        start_time = time.time()
        result = await self.run_pipeline_benchmark(test_dataset)
        total_time = time.time() - start_time
        
        # Collect metrics
        current_metrics = {
            'total_time_seconds': total_time,
            'throughput_docs_per_second': result.documents_processed / total_time,
            'peak_memory_mb': result.peak_memory_usage,
            'gpu_utilization_average': result.average_gpu_utilization,
            'error_rate': result.errors / result.total_operations
        }
        
        # Compare with baseline
        comparison = self.compare_metrics(current_metrics, self.baseline_metrics)
        
        return BenchmarkResult(
            current_metrics=current_metrics,
            baseline_metrics=self.baseline_metrics,
            comparison=comparison,
            regression_detected=comparison.has_regressions()
        )
    
    def compare_metrics(self, current: Dict, baseline: Dict) -> MetricsComparison:
        """Compare current metrics against baseline."""
        comparison = MetricsComparison()
        
        for metric, current_value in current.items():
            baseline_value = baseline.get(metric)
            if baseline_value is None:
                continue
                
            # Calculate percentage change
            change_percent = ((current_value - baseline_value) / baseline_value) * 100
            
            # Define regression thresholds
            regression_thresholds = {
                'total_time_seconds': 10,      # 10% slower is regression
                'throughput_docs_per_second': -10,  # 10% slower throughput
                'peak_memory_mb': 20,          # 20% more memory usage
                'error_rate': 50               # 50% more errors
            }
            
            threshold = regression_thresholds.get(metric, 15)  # Default 15%
            
            if abs(change_percent) > threshold:
                comparison.add_regression(metric, baseline_value, current_value, change_percent)
            else:
                comparison.add_acceptable_change(metric, change_percent)
                
        return comparison
```

---

## Implementation Configuration

### Comprehensive Configuration Schema
```yaml
hybrid_pipeline:
  # Global settings
  run_id: null  # Auto-generated if not provided
  mode: "hybrid"
  
  # Streaming front-end configuration
  streaming_frontend:
    # Worker configuration
    workers:
      document_processing: 4
      chunking: 2
      embedding_gpu_batching: 1
    
    # Adaptive queue management
    queues:
      initial_size: 100
      max_size: 5000
      resize_interval_seconds: 30
      target_latency_seconds: 30
      safety_factor: 1.5
    
    # GPU optimization
    gpu:
      device: "cuda:0"
      target_utilization: 0.75
      memory_profiling_enabled: true
      adaptive_batching: true
      batch_size_range: [8, 128]
    
    # Storage configuration
    storage:
      format: "parquet"  # or "hdf5"
      compression: "snappy"
      shard_size: 50000  # embeddings per shard
      checkpoint_dir: "./checkpoints"
      atomic_writes: true
      checksum_validation: true
      backup_shards: true
  
  # Batch back-end configuration
  batch_backend:
    # Memory management
    memory:
      max_memory_gb: 6.0
      chunk_processing: true
      similarity_batch_size: 10000
      
    # Graph construction
    graph_construction:
      similarity_threshold: 0.5
      max_edges_per_node: 20
      chunked_processing: true
      edge_buffer_size: 100000
      
    # ISNE training (existing config)
    isne_training:
      device: "cuda"
      epochs: 25
      learning_rate: 0.001
      batch_size: 1024
      
  # Error handling and recovery
  error_handling:
    max_retries: 3
    retry_backoff_seconds: [1, 5, 15]
    dead_letter_queue: true
    checkpoint_validation: true
    partial_recovery: true
    
  # Monitoring and alerting
  monitoring:
    enabled: true
    metrics_interval_seconds: 10
    alert_thresholds:
      memory_usage_percent: 85
      queue_depth_ratio: 0.8
      error_rate_percent: 5
      processing_rate_drop_percent: 30
    
  # Testing and validation
  testing:
    performance_regression_detection: true
    baseline_metrics_path: "./baselines/performance_baseline.json"
    memory_compliance_testing: true
    concurrent_run_testing: false
```

---

## Risk Mitigation Strategy

### Technical Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Embedding Storage Corruption** | Medium | High | Atomic writes, checksums, backup shards |
| **Graph Construction OOM** | High | High | Chunked processing, memory monitoring |
| **GPU Memory Overflow** | Medium | Medium | Adaptive batching, memory profiling |
| **Checkpoint Recovery Failure** | Low | High | Comprehensive validation, partial recovery |
| **Performance Regression** | Medium | Medium | Automated benchmarking, baseline tracking |
| **Queue Deadlock** | Low | High | Adaptive sizing, timeout mechanisms |
| **Concurrent Run Conflicts** | Medium | Medium | Run isolation, UUID-based directories |

### Operational Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Disk Space Exhaustion** | Medium | High | Storage monitoring, cleanup automation |
| **Configuration Errors** | High | Medium | Schema validation, safe defaults |
| **Monitoring Blind Spots** | Medium | Medium | Comprehensive metrics, health checks |
| **Recovery Procedure Failure** | Low | High | Automated testing, documented procedures |

---

## Success Metrics and Validation Criteria

### Performance Metrics
- **Overall Pipeline Speedup**: 1.6-2.0x vs current implementation
- **Memory Usage**: Peak < 6GB (vs 8GB current)
- **GPU Utilization**: 60-80% average during embedding generation
- **Error Rate**: < 1% of processed documents
- **Recovery Success Rate**: > 95% for partial failures

### Reliability Metrics
- **Checkpoint Validation**: 100% success rate
- **Storage Corruption**: 0 incidents per 1000 runs
- **Memory Bounds Compliance**: 100% compliance with configured limits
- **Concurrent Run Isolation**: 0 cross-contamination incidents

### Operational Metrics
- **Deployment Success Rate**: > 95% successful deployments
- **Performance Regression Detection**: < 5% false positives
- **Documentation Coverage**: 100% of operational procedures documented
- **Monitoring Coverage**: All critical paths monitored

---

## Additional Production Infrastructure Components

### Configuration Validation System
```python
class PipelineConfigValidator:
    """Comprehensive configuration validation for production deployment."""
    
    def __init__(self):
        self.schema_validator = ConfigSchemaValidator()
        self.resource_validator = ResourceAvailabilityValidator()
        self.compatibility_validator = ComponentCompatibilityValidator()
    
    def validate_full_configuration(self, config: Dict) -> ValidationResult:
        """Perform comprehensive configuration validation."""
        
        result = ValidationResult()
        
        # Schema validation
        schema_result = self.schema_validator.validate_schema(config)
        result.merge(schema_result)
        
        # Resource validation
        resource_result = self.resource_validator.validate_resources(config)
        result.merge(resource_result)
        
        # Component compatibility
        compat_result = self.compatibility_validator.validate_compatibility(config)
        result.merge(compat_result)
        
        # Configuration consistency checks
        consistency_result = self._validate_configuration_consistency(config)
        result.merge(consistency_result)
        
        return result
    
    def _validate_configuration_consistency(self, config: Dict) -> ValidationResult:
        """Validate configuration internal consistency."""
        result = ValidationResult()
        
        # Check GPU availability vs configuration
        if config.get('streaming_frontend', {}).get('gpu', {}).get('device', '').startswith('cuda'):
            if not torch.cuda.is_available():
                result.add_error("CUDA device specified but not available")
        
        # Check queue sizes are compatible
        queue_config = config.get('streaming_frontend', {}).get('queues', {})
        max_size = queue_config.get('max_size', 5000)
        initial_size = queue_config.get('initial_size', 100)
        
        if initial_size > max_size:
            result.add_error(f"Initial queue size ({initial_size}) exceeds maximum ({max_size})")
        
        # Check memory configuration
        max_memory = config.get('batch_backend', {}).get('memory', {}).get('max_memory_gb', 6.0)
        available_memory = psutil.virtual_memory().available / (1024**3)
        
        if max_memory > available_memory * 0.8:
            result.add_warning(f"Configured memory ({max_memory}GB) may exceed available ({available_memory:.1f}GB)")
        
        # Check storage path accessibility
        checkpoint_dir = Path(config.get('streaming_frontend', {}).get('storage', {}).get('checkpoint_dir', './checkpoints'))
        try:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            test_file = checkpoint_dir / '.write_test'
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            result.add_error(f"Checkpoint directory not writable: {e}")
        
        return result


class ResourceAvailabilityValidator:
    """Validate system resources meet pipeline requirements."""
    
    def validate_resources(self, config: Dict) -> ValidationResult:
        """Check if system has required resources."""
        result = ValidationResult()
        
        # CPU validation
        required_workers = (
            config.get('streaming_frontend', {}).get('workers', {}).get('document_processing', 4) +
            config.get('streaming_frontend', {}).get('workers', {}).get('chunking', 2) + 1
        )
        available_cores = psutil.cpu_count()
        
        if required_workers > available_cores:
            result.add_warning(f"Required workers ({required_workers}) may exceed CPU cores ({available_cores})")
        
        # Memory validation
        max_memory_gb = config.get('batch_backend', {}).get('memory', {}).get('max_memory_gb', 6.0)
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        if max_memory_gb > available_memory_gb * 0.9:
            result.add_error(f"Required memory ({max_memory_gb}GB) exceeds available ({available_memory_gb:.1f}GB)")
        
        # GPU validation
        gpu_config = config.get('streaming_frontend', {}).get('gpu', {})
        if gpu_config.get('device', '').startswith('cuda'):
            if not torch.cuda.is_available():
                result.add_error("CUDA required but not available")
            else:
                device_index = int(gpu_config.get('device', 'cuda:0').split(':')[1])
                if device_index >= torch.cuda.device_count():
                    result.add_error(f"GPU device {device_index} not available")
        
        # Disk space validation
        checkpoint_dir = Path(config.get('streaming_frontend', {}).get('storage', {}).get('checkpoint_dir', './checkpoints'))
        try:
            available_space_gb = psutil.disk_usage(checkpoint_dir.parent).free / (1024**3)
            estimated_usage_gb = 10.0  # Conservative estimate
            
            if available_space_gb < estimated_usage_gb * 2:
                result.add_warning(f"Low disk space: {available_space_gb:.1f}GB available, ~{estimated_usage_gb}GB needed")
        except Exception as e:
            result.add_warning(f"Could not check disk space: {e}")
        
        return result


class ComponentCompatibilityValidator:
    """Validate component compatibility and version alignment."""
    
    def validate_compatibility(self, config: Dict) -> ValidationResult:
        """Check component compatibility."""
        result = ValidationResult()
        
        # Check embedding model compatibility
        embedding_config = config.get('components', {}).get('embedding', {})
        if embedding_config.get('type') == 'modernbert':
            try:
                from transformers import AutoModel
                model_name = embedding_config.get('params', {}).get('model_name', 'answerdotai/ModernBERT-base')
                # Try to load model config to verify availability
                AutoModel.from_pretrained(model_name, config_only=True)
            except Exception as e:
                result.add_error(f"Embedding model not available: {e}")
        
        # Check storage format compatibility
        storage_format = config.get('streaming_frontend', {}).get('storage', {}).get('format', 'parquet')
        if storage_format == 'parquet':
            try:
                import pyarrow.parquet as pq
            except ImportError:
                result.add_error("Parquet format specified but pyarrow not available")
        elif storage_format == 'hdf5':
            try:
                import h5py
            except ImportError:
                result.add_error("HDF5 format specified but h5py not available")
        
        # Check compression compatibility
        compression = config.get('streaming_frontend', {}).get('storage', {}).get('compression', 'snappy')
        if compression == 'snappy':
            try:
                import snappy
            except ImportError:
                result.add_warning("Snappy compression not available, falling back to gzip")
        
        return result
```

### Monitoring Dashboard Specification
```python
class HybridPipelineMonitoringDashboard:
    """Comprehensive monitoring dashboard for hybrid pipeline."""
    
    def __init__(self):
        self.metrics_collector = HybridPipelineMonitor()
        self.alert_manager = AlertManager()
        
    def get_dashboard_layout(self) -> Dict[str, Any]:
        """Define comprehensive monitoring dashboard layout."""
        
        return {
            "dashboard": {
                "title": "HADES Hybrid Pipeline Monitoring",
                "refresh": "5s",
                "time": {"from": "now-1h", "to": "now"},
                "panels": [
                    {
                        "title": "Pipeline Overview",
                        "type": "stat",
                        "targets": [
                            {"expr": "pipeline_status", "legendFormat": "Status"},
                            {"expr": "pipeline_uptime_seconds", "legendFormat": "Uptime"},
                            {"expr": "rate(documents_processed_total[5m])", "legendFormat": "Doc/sec"}
                        ]
                    },
                    {
                        "title": "Queue Health",
                        "type": "graph",
                        "targets": [
                            {"expr": "queue_depth{stage=\"document_processing\"}", "legendFormat": "Document Queue"},
                            {"expr": "queue_depth{stage=\"chunking\"}", "legendFormat": "Chunk Queue"},
                            {"expr": "queue_depth{stage=\"embedding\"}", "legendFormat": "Embedding Queue"}
                        ],
                        "alert": {
                            "conditions": [
                                {"query": "queue_depth", "reducer": "avg", "threshold": 0.8}
                            ]
                        }
                    },
                    {
                        "title": "GPU Utilization",
                        "type": "graph",
                        "targets": [
                            {"expr": "gpu_utilization_percent", "legendFormat": "GPU Utilization"},
                            {"expr": "gpu_memory_usage_percent", "legendFormat": "GPU Memory"},
                            {"expr": "gpu_temperature_celsius", "legendFormat": "GPU Temperature"}
                        ]
                    },
                    {
                        "title": "Storage Metrics",
                        "type": "graph",
                        "targets": [
                            {"expr": "rate(storage_writes_total[5m])", "legendFormat": "Writes/sec"},
                            {"expr": "storage_write_latency_seconds", "legendFormat": "Write Latency"},
                            {"expr": "storage_error_rate", "legendFormat": "Error Rate"}
                        ]
                    },
                    {
                        "title": "Memory Usage",
                        "type": "graph",
                        "targets": [
                            {"expr": "process_memory_usage_bytes", "legendFormat": "Process Memory"},
                            {"expr": "system_memory_usage_percent", "legendFormat": "System Memory"},
                            {"expr": "pipeline_peak_memory_bytes", "legendFormat": "Peak Memory"}
                        ]
                    },
                    {
                        "title": "Error Tracking",
                        "type": "table",
                        "targets": [
                            {"expr": "increase(pipeline_errors_total[1h])", "legendFormat": "Errors (1h)"},
                            {"expr": "pipeline_error_rate", "legendFormat": "Error Rate"},
                            {"expr": "last_error_timestamp", "legendFormat": "Last Error"}
                        ]
                    },
                    {
                        "title": "Throughput Analysis",
                        "type": "graph",
                        "targets": [
                            {"expr": "rate(documents_processed_total[5m])", "legendFormat": "Documents/sec"},
                            {"expr": "rate(chunks_generated_total[5m])", "legendFormat": "Chunks/sec"},
                            {"expr": "rate(embeddings_generated_total[5m])", "legendFormat": "Embeddings/sec"}
                        ]
                    }
                ],
                "alerts": [
                    {
                        "name": "Queue Starvation",
                        "condition": "queue_depth == 0 for 30s",
                        "severity": "warning"
                    },
                    {
                        "name": "High Memory Usage",
                        "condition": "process_memory_usage_bytes > 6GB",
                        "severity": "critical"
                    },
                    {
                        "name": "GPU Overheating",
                        "condition": "gpu_temperature_celsius > 85",
                        "severity": "critical"
                    },
                    {
                        "name": "Storage Failure",
                        "condition": "storage_error_rate > 0.05",
                        "severity": "high"
                    }
                ]
            }
        }


class DisasterRecoveryManager:
    """Comprehensive disaster recovery procedures."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.backup_manager = BackupManager(config)
        self.health_monitor = HealthMonitor()
        
    async def execute_disaster_recovery(self, failure_type: str, context: Dict) -> RecoveryResult:
        """Execute appropriate disaster recovery procedure."""
        
        logger.critical(f"Executing disaster recovery for {failure_type}")
        
        recovery_procedures = {
            'storage_corruption': self._recover_from_storage_corruption,
            'memory_exhaustion': self._recover_from_memory_exhaustion,
            'gpu_failure': self._recover_from_gpu_failure,
            'pipeline_deadlock': self._recover_from_pipeline_deadlock,
            'checkpoint_corruption': self._recover_from_checkpoint_corruption
        }
        
        if failure_type not in recovery_procedures:
            return RecoveryResult(success=False, message=f"Unknown failure type: {failure_type}")
        
        try:
            return await recovery_procedures[failure_type](context)
        except Exception as e:
            logger.critical(f"Disaster recovery failed: {e}")
            return RecoveryResult(success=False, message=f"Recovery failed: {e}")
    
    async def _recover_from_storage_corruption(self, context: Dict) -> RecoveryResult:
        """Recover from storage corruption issues."""
        
        # 1. Stop pipeline immediately
        await self._emergency_pipeline_stop()
        
        # 2. Identify corrupted components
        corrupted_files = await self._identify_corrupted_storage()
        
        # 3. Restore from backups
        if self.backup_manager.has_recent_backup():
            backup_restored = await self.backup_manager.restore_latest_backup()
            if backup_restored:
                return RecoveryResult(success=True, message="Restored from backup")
        
        # 4. Partial recovery - salvage what we can
        salvaged_data = await self._salvage_uncorrupted_data(corrupted_files)
        
        # 5. Resume from last good checkpoint
        resumption_point = await self._find_last_good_checkpoint()
        
        return RecoveryResult(
            success=True,
            message=f"Partial recovery completed, resuming from {resumption_point}",
            resumption_point=resumption_point
        )
    
    async def _recover_from_memory_exhaustion(self, context: Dict) -> RecoveryResult:
        """Recover from memory exhaustion."""
        
        # 1. Reduce memory configuration
        new_config = self._reduce_memory_footprint(self.config)
        
        # 2. Clear caches and temporary data
        await self._clear_all_caches()
        
        # 3. Restart with reduced settings
        await self._restart_pipeline_with_config(new_config)
        
        return RecoveryResult(
            success=True,
            message="Recovered from memory exhaustion with reduced memory settings"
        )
    
    async def _recover_from_gpu_failure(self, context: Dict) -> RecoveryResult:
        """Recover from GPU failure by falling back to CPU."""
        
        # 1. Switch to CPU-only mode
        cpu_config = self._convert_to_cpu_config(self.config)
        
        # 2. Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 3. Restart with CPU configuration
        await self._restart_pipeline_with_config(cpu_config)
        
        return RecoveryResult(
            success=True,
            message="Recovered from GPU failure using CPU fallback"
        )


class EmbeddingVersionMigrator:
    """Handle embedding format migration between versions."""
    
    def __init__(self):
        self.version_registry = {
            "1.0": EmbeddingFormatV1(),
            "1.1": EmbeddingFormatV1_1(),
            "2.0": EmbeddingFormatV2()
        }
    
    async def migrate_embeddings(self, 
                               source_path: Path, 
                               target_path: Path, 
                               source_version: str, 
                               target_version: str) -> MigrationResult:
        """Migrate embeddings between format versions."""
        
        if source_version == target_version:
            return MigrationResult(success=True, message="No migration needed")
        
        source_format = self.version_registry.get(source_version)
        target_format = self.version_registry.get(target_version)
        
        if not source_format or not target_format:
            return MigrationResult(success=False, message="Unsupported version")
        
        # Load embeddings in source format
        embeddings = await source_format.load(source_path)
        
        # Convert to target format
        converted_embeddings = await self._convert_embeddings(
            embeddings, source_format, target_format
        )
        
        # Save in target format
        await target_format.save(converted_embeddings, target_path)
        
        # Validate migration
        validation_result = await self._validate_migration(
            source_path, target_path, source_format, target_format
        )
        
        return MigrationResult(
            success=validation_result.is_valid,
            message=f"Migrated from {source_version} to {target_version}",
            validation_details=validation_result
        )


class GracefulDegradationManager:
    """Manage graceful degradation when components fail."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.degradation_strategies = {
            'embedding_service_down': self._degrade_to_cached_embeddings,
            'gpu_unavailable': self._degrade_to_cpu_processing,
            'storage_slow': self._degrade_to_memory_only,
            'queue_overflow': self._degrade_to_reduced_throughput
        }
    
    async def apply_degradation(self, component: str, issue: str) -> DegradationResult:
        """Apply appropriate degradation strategy."""
        
        strategy_key = f"{component}_{issue}"
        if strategy_key in self.degradation_strategies:
            return await self.degradation_strategies[strategy_key]()
        
        # Generic degradation - reduce throughput
        return await self._generic_throughput_reduction(component)
    
    async def _degrade_to_cached_embeddings(self) -> DegradationResult:
        """Use cached embeddings when embedding service is down."""
        
        # Switch to cache-only mode
        self.config['embedding']['cache_only'] = True
        self.config['embedding']['fallback_enabled'] = True
        
        return DegradationResult(
            success=True,
            degradation_level="moderate",
            message="Using cached embeddings only",
            limitations=["No new embeddings will be generated", "Limited to cached content"]
        )
    
    async def _degrade_to_cpu_processing(self) -> DegradationResult:
        """Fall back to CPU when GPU is unavailable."""
        
        # Switch all GPU operations to CPU
        self.config['embedding']['device'] = 'cpu'
        self.config['isne_training']['device'] = 'cpu'
        
        # Reduce batch sizes for CPU efficiency
        self.config['embedding']['batch_size'] = min(8, self.config['embedding']['batch_size'])
        
        return DegradationResult(
            success=True,
            degradation_level="significant",
            message="Operating in CPU-only mode",
            limitations=["Slower processing", "Reduced throughput", "Higher latency"]
        )
```

---

## Conclusion

This implementation plan addresses the critical scalability, reliability, and operational concerns identified in the architectural review. By prioritizing robust data management, memory safety, and comprehensive error recovery, the plan ensures production-ready deployment while maintaining the performance benefits of the hybrid approach.

**Key Improvements Over Original Plan:**
1. **Sharded embedding storage** for scalability
2. **Memory-bounded graph construction** to prevent OOM
3. **Comprehensive error recovery** for reliability
4. **Production-grade monitoring** for operations
5. **Automated testing and validation** for quality assurance

**Implementation Timeline: 6-8 weeks**
- **Weeks 1-4**: Foundation infrastructure (storage, memory management, error recovery)
- **Weeks 5-6**: Streaming front-end with adaptive optimization
- **Weeks 7-8**: Integration, testing, and production hardening

This plan balances performance improvements with production reliability, ensuring the hybrid pipeline can scale to enterprise workloads while maintaining system stability and operational excellence.