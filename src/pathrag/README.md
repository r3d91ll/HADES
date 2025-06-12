# PathRAG Module

⚠️ **IMPORTANT: This module requires complete refactoring** ⚠️

The PathRAG module contains legacy implementation of the Path-based Retrieval Augmented Generation system. **All code in this module needs to be completely refactored** to integrate with the modern HADES architecture and align with current system design patterns.

## 📋 Current Status: LEGACY CODE

This module represents the original PathRAG implementation that has been superseded by the integrated HADES architecture. The code here serves as:

- **Reference implementation** for understanding original PathRAG concepts
- **Migration source** for extracting core algorithms and logic
- **Historical documentation** of the evolution from PathRAG to HADES

**⛔ DO NOT USE THIS CODE IN PRODUCTION ⛔**

## 🔄 Refactoring Requirements

### Critical Issues

1. **Architecture Mismatch** - Does not follow current HADES module patterns
2. **Integration Problems** - Not compatible with current pipeline system
3. **Dependency Conflicts** - Uses outdated interfaces and APIs
4. **Performance Issues** - Not optimized for current infrastructure
5. **Maintenance Burden** - Code patterns inconsistent with rest of system

### Required Changes

**Complete replacement needed for**:

- Integration with current embedding system (`src/embedding/`)
- Compatibility with ISNE enhancements (`src/isne/`)
- ArangoDB storage backend (`src/storage/arango/`)
- Modern pipeline orchestration (`src/orchestration/`)
- Current configuration system (`src/config/`)

## 📁 Legacy Module Contents

### Current Files (TO BE REFACTORED)

- **`PathRAG.py`** - Main PathRAG class (legacy implementation)
- **`base.py`** - Base PathRAG interfaces (outdated)
- **`llm.py`** - Language model integration (needs modernization)
- **`model_engine_adapter.py`** - Model engine adapter (incompatible)
- **`operate.py`** - Core operations (requires rewrite)
- **`prompt.py`** - Prompt management (needs update)
- **`storage.py`** - Storage abstraction (outdated)
- **`utils.py`** - Utility functions (mixed relevance)
- **`vllm_adapter.py`** - vLLM adapter (superseded)
- **`vllm_server.py`** - vLLM server (replaced by model_engine)

### What This Code Originally Did

**PathRAG Core Concepts** (to be preserved in refactor):

- **Path-based retrieval** through document relationship graphs
- **Multi-hop reasoning** across connected document chunks
- **Hybrid retrieval modes** (naive, local, global, hybrid)
- **Graph traversal algorithms** for knowledge discovery
- **Context aggregation** from multiple related sources

## 🎯 Refactoring Plan

### Phase 1: Analysis and Design

**Extract core algorithms**:

```python
# Example of what needs to be preserved (concepts, not code)
class PathRAGConcepts:
    """Core PathRAG concepts to preserve in refactor."""
    
    def path_based_retrieval(self, query, mode="hybrid"):
        """
        Core path-based retrieval algorithm.
        
        NEEDS COMPLETE REWRITE to integrate with:
        - src.embedding for vector similarity
        - src.isne for graph-aware embeddings  
        - src.storage.arango for graph traversal
        - src.orchestration for pipeline execution
        """
        pass
    
    def multi_hop_reasoning(self, start_chunks, max_hops=3):
        """
        Multi-hop reasoning through document graph.
        
        NEEDS INTEGRATION with current graph storage
        and ISNE-enhanced relationship scoring.
        """
        pass
    
    def hybrid_retrieval_modes(self, query):
        """
        Different retrieval strategies.
        
        NEEDS MODERNIZATION to use current embedding
        and graph neural network infrastructure.
        """
        pass
```

### Phase 2: New Implementation

**Create modern PathRAG integration**:

```python
# Future implementation should look like this
from src.embedding.registry import get_embedding_adapter
from src.isne.pipeline.isne_pipeline import ISNEPipeline
from src.storage.arango.repository import ArangoRepository
from src.orchestration.pipelines.orchestrator import PipelineOrchestrator

class ModernPathRAG:
    """Modern PathRAG implementation integrated with HADES architecture."""
    
    def __init__(self, config_file: str):
        # Use current configuration system
        self.config = load_config(config_file)
        
        # Integrate with current embedding system
        self.embedding_adapter = get_embedding_adapter(
            self.config['embedding']['adapter']
        )
        
        # Use ISNE for graph-aware embeddings
        self.isne_pipeline = ISNEPipeline.from_config(
            self.config['isne_config']
        )
        
        # Use current storage backend
        self.graph_storage = ArangoRepository(
            self.config['database_config']
        )
        
        # Integrate with pipeline orchestration
        self.orchestrator = PipelineOrchestrator()
    
    def query(self, question: str, mode: str = "hybrid") -> dict:
        """Modern PathRAG query implementation."""
        # Implement using current architecture
        pass
```

### Phase 3: Migration Strategy

**Recommended approach**:

1. **Preserve algorithms** - Extract core PathRAG logic patterns
2. **Modernize interfaces** - Rewrite to use current module APIs
3. **Integrate storage** - Replace with ArangoDB graph operations
4. **Update embeddings** - Use ISNE-enhanced vector representations
5. **Pipeline integration** - Make PathRAG a pipeline stage
6. **Configuration alignment** - Use current config system
7. **Testing framework** - Validate against legacy behavior

## 🚧 Interim Usage

**Until refactoring is complete**:

1. **DO NOT import from this module** in new code
2. **Reference for algorithms only** - understand concepts, don't copy code
3. **Use modern alternatives**:
   - Query processing → `src.api` interfaces
   - Embedding → `src.embedding` adapters  
   - Graph operations → `src.storage.arango`
   - Pipeline execution → `src.orchestration`

## 🔍 Code Analysis for Migration

### Salvageable Concepts

**Core algorithms to preserve**:

- Path-based graph traversal logic
- Multi-hop reasoning patterns
- Hybrid retrieval mode selection
- Context aggregation strategies
- Query processing workflows

**Integration patterns to modernize**:

- Storage operations → ArangoDB queries
- Embedding generation → Current embedding system
- Model serving → Current model engine
- Configuration → YAML-based config system

### Code to Discard

**Obsolete implementations**:

- Direct vLLM integration (use `src.model_engine`)
- Custom storage abstractions (use `src.storage.arango`)
- Legacy embedding code (use `src.embedding`)
- Old configuration patterns (use `src.config`)
- Standalone server implementations (use `src.api`)

## 📚 Refactoring References

### Current Architecture to Follow

Study these modules for modern patterns:

- **`src.embedding`** - For embedding integration patterns
- **`src.isne`** - For graph-aware processing
- **`src.storage.arango`** - For graph database operations
- **`src.orchestration`** - For pipeline integration
- **`src.api`** - For query interface patterns

### Configuration Integration

Follow current config patterns:

```yaml
# pathrag_config.yaml (future)
pathrag:
  retrieval:
    modes: ["naive", "local", "global", "hybrid"]
    default_mode: "hybrid"
    max_hops: 3
    
  graph_traversal:
    similarity_threshold: 0.7
    max_results_per_hop: 10
    path_scoring_method: "isne_weighted"
    
  integration:
    embedding_adapter: "modernbert"
    isne_pipeline: "production"
    storage_backend: "arango"
```

## ⚠️ Development Guidelines

### For Developers Working on Refactor

1. **Study legacy concepts** but don't copy implementation
2. **Follow current architectural patterns** from other modules
3. **Use existing infrastructure** (embedding, storage, orchestration)
4. **Maintain API compatibility** where possible
5. **Add comprehensive tests** comparing with legacy behavior
6. **Document design decisions** and architectural choices

### For General Development

1. **Avoid this module entirely** for new features
2. **Use modern alternatives** from other modules
3. **Reference documentation only** for understanding PathRAG concepts
4. **Contribute to refactoring effort** if possible

## 🎯 Success Criteria for Refactoring

The refactoring will be complete when:

1. **Full integration** with current HADES architecture
2. **Performance parity** or improvement over legacy implementation
3. **API compatibility** for existing users (where feasible)
4. **Comprehensive test coverage** validating correctness
5. **Documentation completeness** for new implementation
6. **Legacy code removal** - this directory can be deleted

## 📞 Getting Involved

**To contribute to the refactoring effort**:

1. **Study the current architecture** in other modules
2. **Understand PathRAG concepts** from this legacy code
3. **Design modern implementation** following current patterns
4. **Coordinate with team** to avoid duplicate work
5. **Submit incremental changes** rather than massive rewrites

---

**Remember**: This module represents valuable research and algorithmic contributions, but the implementation must be completely modernized to fit the current HADES architecture. The goal is to preserve the intelligence while rebuilding the implementation.
