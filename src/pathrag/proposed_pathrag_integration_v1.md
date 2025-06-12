# Proposed PathRAG Integration Plan (v1)

## 1. Current State Analysis

The `src/pathrag` directory contains legacy implementation of the Path-based Retrieval Augmented Generation system. The code is not integrated with the modern HADES architecture and needs to be completely refactored.

### Key Components in Current Implementation

- **PathRAG.py**: Main class implementing core PathRAG functionality
- **Base classes**: Storage abstractions, vector storage, graph storage
- **LLM integration**: Direct vLLM adapter implementation
- **Operational functions**: Entity extraction, knowledge graph querying
- **Utilities**: Various helper functions

### Issues with Current Implementation

1. **Architecture Mismatch**: Doesn't follow current HADES module patterns
2. **Integration Problems**: Not compatible with current pipeline system
3. **Dependency Conflicts**: Uses outdated interfaces and APIs
4. **Performance Issues**: Not optimized for current infrastructure
5. **Maintenance Burden**: Code patterns inconsistent with rest of system

## 2. Refactoring Goals

The goal is to create a modern, integrated PathRAG implementation that:

1. Follows HADES architectural patterns
2. Integrates seamlessly with the pipeline system
3. Uses current storage backends (ArangoDB)
4. Leverages ISNE for graph-aware embeddings
5. Maintains API compatibility where feasible

## 3. Proposed Architecture

### High-Level Design

```text
Document Input → Processing → Chunking → Embedding → Graph Construction → Storage → Retrieval
```

### Core Components

1. **PathRAG Engine**: New implementation of PathRAG core functionality
2. **Pipeline Integration**: Make PathRAG a pipeline stage in the query pipeline (not data ingestion)
3. **Storage Backend**: Use ArangoDB for graph storage
4. **Embedding System**: Integrate with current embedding system
5. **ISNE Enhancement**: Leverage ISNE for graph-aware embeddings

## 4. Detailed Refactoring Plan

### Phase 1: Analysis and Design

#### Extract Core Algorithms

Identify and preserve core PathRAG algorithms while discarding outdated implementation details.

```python
# Example of what needs to be preserved (concepts, not code)
class PathRAGConcepts:
    """Core PathRAG concepts to preserve in refactor."""

    def path_based_retrieval(self, query, mode="hybrid"):
        """
        Core path-based retrieval algorithm.
        """

    def multi_hop_reasoning(self, start_chunks, max_hops=3):
        """
        Multi-hop reasoning through document graph.
        """

    def hybrid_retrieval_modes(self, query):
        """
        Different retrieval strategies.
        """
```

### Phase 2: New Implementation

#### Create Modern PathRAG Integration

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
```

### Phase 3: Migration Strategy

#### Recommended Approach

1. **Preserve algorithms**: Extract core PathRAG logic patterns
2. **Modernize interfaces**: Rewrite to use current module APIs
3. **Integrate storage**: Replace with ArangoDB graph operations
4. **Update embeddings**: Use ISNE-enhanced vector representations
5. **Pipeline integration**: Make PathRAG a pipeline stage in the query pipeline (not data ingestion)
6. **Configuration alignment**: Use current config system
7. **Testing framework**: Validate against legacy behavior

## 5. Implementation Plan

### Step-by-Step Execution

1. **Create new module structure** in `src/pathrag_v2/`
2. **Implement core algorithms** with modern interfaces
3. **Integrate with embedding system**
4. **Connect to ArangoDB storage**
5. **Add ISNE enhancement support**
6. **Create pipeline integration points in the query pipeline**
7. **Update configuration system**
8. **Write comprehensive tests**

### Milestones

1. Basic functionality prototype (2 weeks)
2. Full integration with embedding and storage systems (3 weeks)
3. ISNE enhancement implementation (2 weeks)
4. Pipeline integration complete (2 weeks)
5. Final testing and optimization (2 weeks)

## 6. Testing Strategy

### Key Areas to Test

1. **Algorithm correctness**: Ensure core PathRAG algorithms work as expected
2. **Integration points**: Verify seamless operation with embedding, storage, ISNE
3. **Pipeline compatibility**: Test as part of the query pipeline (not data ingestion)
4. **Performance**: Benchmark against legacy implementation

### Testing Methods

1. **Unit tests** for individual components
2. **Integration tests** for system interactions
3. **End-to-end tests** with sample queries
4. **Benchmarking** against legacy performance metrics

## 7. Documentation Plan

### Required Documentation

1. **Module-level docstrings**: Explain purpose and usage
2. **Function/class docstrings**: With parameters, return types, examples
3. **README.md**: Comprehensive overview of the module
4. **API documentation**: For all public interfaces
5. **Usage examples**: In script files

### Documentation Standards

- Follow HADES documentation guidelines
- Maintain consistency with other modules
- Include diagrams where helpful

## 8. Transition Plan

### Migration Strategy

1. **Parallel development**: Work on new implementation alongside legacy code
2. **Incremental replacement**: Gradually replace legacy components
3. **Thorough testing**: At each stage of migration
4. **Final switch**: Once fully tested and validated

### Deprecation Policy

- Mark legacy code as deprecated but functional during transition
- Provide clear migration path for users
- Set timeline for complete removal (6 months post-migration)

## 9. Success Criteria

The refactoring will be complete when:

1. **Full integration** with current HADES architecture
2. **Performance parity** or improvement over legacy implementation
3. **API compatibility** for existing users (where feasible)
4. **Comprehensive test coverage** validating correctness
5. **Documentation completeness** for new implementation
6. **Legacy code removal**: Old directory can be deleted

## 10. Risks and Mitigation

### Potential Risks

1. **Algorithm drift**: Loss of core functionality during refactoring
   - *Mitigation*: Extensive testing against legacy behavior

2. **Integration issues**: Problems with new system components
   - *Mitigation*: Incremental integration and thorough testing

3. **Performance regression**: Slower than legacy implementation
   - *Mitigation*: Benchmarking and optimization throughout development

4. **Configuration complexity**: Difficulty in migrating settings
   - *Mitigation*: Clear migration path and documentation

### Risk Management

- Regular progress reviews
- Continuous integration with automated testing
- Frequent stakeholder updates
