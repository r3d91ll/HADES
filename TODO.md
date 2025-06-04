# HADES-PathRAG Development Tasks

## Type System Migration Status

This section outlines the current status of type migration to centralized `src/types` directory. These are our top priority tasks.

### 1. src/docproc

**Status**: Fully migrated ✅

- ✅ Basic document types migrated to `src/types/docproc/document.py`
- ✅ Adapter protocols migrated to `src/types/docproc/adapter.py`
- ✅ Python-specific schemas migrated to `src/types/docproc/python.py` (models from `python_document.py` including `PythonMetadata`, `PythonEntity`, etc.)
- ✅ Code model enums migrated to `src/types/docproc/enums.py` (enums from `python_code.py` including `RelationshipType` and `AccessLevel`)
- ✅ TypedDict definitions migrated to `src/types/docproc/code_elements.py` (from `python_code.py`)
- ✅ Original files updated to re-export from centralized types for backward compatibility

### 2. src/chunking

**Status**: Mostly migrated

- ✅ Core chunking types migrated to `src/types/chunking/chunk.py`
- ✅ Strategy enums and types migrated to `src/types/chunking/strategy.py`
- ✅ Text chunking types migrated to `src/types/chunking/text.py`
- ❌ **Needs Migration**: CPU chunker specific types in `src/chunking/text_chunkers/cpu_chunker.py`

### 3. src/embedding

**Status**: Partially migrated

- ✅ Vector types migrated to `src/types/embedding/vector.py`
- ❌ **Needs Migration**: `EmbeddingAdapter` protocol in `src/embedding/base.py` should be moved to `src/types/embedding/adapter.py`

### 4. src/isne

**Status**: Partially migrated

- ✅ Base models migrated to `src/types/isne/models.py`
- ❌ **Needs Migration**: Document and relation type enums in `src/isne/types/models.py` are duplicated
- ❌ **Needs Migration**: Pipeline configuration types in `src/isne/pipeline/config.py` and `src/isne/pipeline/isne_pipeline.py`

### 5. src/storage

**Status**: Mostly migrated

- ✅ Connection types migrated to `src/types/storage/connection.py`
- ✅ Document types migrated to `src/types/storage/document.py`
- ✅ Query types migrated to `src/types/storage/query.py`
- ✅ Repository interfaces migrated to `src/types/storage/repository.py`
- ❌ **Needs Migration**: Interfaces in `src/storage/interfaces.py` that might overlap with the centralized types

### 6. src/orchestration

**Status**: Not found in centralized types

- ❌ **Needs Migration**: Any type definitions in this directory should be moved to a new `src/types/orchestration` directory

### 7. src/database

**Status**: Not found in centralized types

- ❌ **Needs Migration**: Any type definitions in this directory should be moved to a new `src/types/database` directory

### 8. src/schemas

**Status**: Partially migrated

- ✅ Some pipeline types migrated to `src/types/pipeline/queue.py` and `src/types/pipeline/worker.py`
- ❌ **Needs Migration**: Multiple enums in `src/schemas/common/enums.py` should be migrated to appropriate modules
- ❌ **Needs Migration**: Pipeline enums and types in various files:
  - `src/schemas/pipeline/base.py`: `PipelineStage` and `PipelineStatus` enums
  - `src/schemas/pipeline/config.py`: `InputSourceType`, `OutputDestinationType`, and `DatabaseType` enums
  - `src/schemas/pipeline/jobs.py`: `JobStatus` enum
  - `src/schemas/pipeline/queue.py`: `TaskPriority` and `TaskStatus` enums
  - `src/schemas/pipeline/text.py`: `ChunkingStrategy` enum
  - `src/schemas/pipeline/workers.py`: `WorkerStatus` and `WorkerType` enums
- ❌ **Needs Migration**: Embedding schema types in `src/schemas/embedding/adapters.py` and `src/schemas/embedding/models.py`
- ❌ **Needs Migration**: ISNE model types in `src/schemas/isne/models.py`

## Completed Tasks

### Medium Priority

#### Model Engine Type System Migration

- [x] Fix typing issues in model_engine module
  - [x] Fix typing issues in VLLM engine
  - [x] Fix unreachable code in session availability checks
  - [x] Add proper type annotations for loaded_models
  - [x] Fix return type of _normalize_embedding
  - [x] Add proper type ignore comments for third-party imports
  - [x] Consolidate VLLMModelEngine implementations from multiple files
  - [x] Fix ModelMode enum usage to use correct enum values
  - [x] Ensure consistent constructor parameters with config_path support
  - [x] Implement transitional solution in vllm_engine.py for backward compatibility
  - [x] Fix typing issues in Haystack engine
  - [x] Add missing unload_model implementation
  - [x] Fix typing issues in factory.py
  - [x] Fix ENGINE_REGISTRY typing
  - [x] Fix parameter consistency across engines
  - [x] Implement centralized type system for model_engine
  - [x] Create centralized result types in `src/types/model_engine/results.py`
  - [x] Update adapter interfaces to use centralized types
  - [x] Migrate VLLM adapter to use centralized types
    - [x] Fix return type of _normalize_embedding
    - [x] Add proper type ignore comments for third-party imports
    - [x] Consolidate VLLMModelEngine implementations from multiple files into a single implementation in engine.py
    - [x] Fix ModelMode enum usage to use correct enum values instead of accessing .value
    - [x] Ensure consistent constructor parameters with config_path support
    - [x] Implement transitional solution in vllm_engine.py for backward compatibility
  - [x] Fix typing issues in Haystack engine
    - [x] Add missing unload_model implementation
  - [x] Fix typing issues in factory.py
    - [x] Fix ENGINE_REGISTRY typing
    - [x] Fix parameter consistency across engines
  - [ ] Fix typing issues in PathRAG module
  - [ ] Add missing return type annotations
  - [ ] Fix type inconsistencies in storage classes
  - [ ] Address indexing and attribute access errors
  - [ ] Implement centralized type system
  - [x] Create src/types/docproc/ directory (rename from src/types/documents/)
  - [x] Create src/types/chunking/ directory for chunking types
  - [x] Create src/types/storage/ directory for storage types
  - [x] Create src/types/pathrag/ directory for PathRAG types
  - [x] Create src/types/indexing/ directory for indexing types
  - [x] Create src/types/model_engine/ directory for model engine types
  - [x] Create src/types/embedding/ directory for embedding types
  - [x] Create src/types/isne/ directory for ISNE types
  - [x] Update type migration plan document in src/types/types_readme.md
  
### Type System Refactoring Strategy

Since we're in active development, we'll use a **replace-don't-migrate** approach to establish a clean type system without complex migrations. Each module will be refactored completely with proper type annotations.
  
### Git Strategy for Type Refactoring

For each module:

  1. Create a feature branch: `git checkout -b feature/types-{module_name}`
  2. Complete all type refactoring for that module
  3. Run type checks with mypy
  4. Create a pull request for review
  5. After approval, merge to main branch
  
### Module Refactoring Order and Approach

#### 1. docproc (First Priority)

- **Branch**: `feature/types-docproc`
  - **Type Definitions**:
    - `src/types/docproc/document.py`: Document type definitions
    - `src/types/docproc/metadata.py`: Metadata type definitions
    - `src/types/docproc/adapter.py`: Adapter interface types
    - `src/types/docproc/processing.py`: Processing result types
  - **Implementation Steps**:
    - Replace all internal types with centralized types
    - Fix any syntax errors in type annotations
    - Ensure consistent usage of types across the module
    - Run mypy to verify type correctness
  
#### 2. model_engine (Second Priority)

- **Branch**: `feature/types-model-engine`
  - **Type Definitions**:
    - `src/types/model_engine/engine.py`: Engine interface types
    - `src/types/model_engine/adapters.py`: Model adapter types
    - `src/types/model_engine/inference.py`: Inference request/response types
    - `src/types/model_engine/config.py`: Configuration types
  - **Implementation Steps**:
    - Refactor all engine implementations to use centralized types
    - Standardize interface for all engine types
    - Run mypy to verify type correctness
  
#### 3. chunking (Third Priority)

- **Branch**: `feature/types-chunking`
  - **Type Definitions**:
    - `src/types/chunking/chunk.py`: Chunk data types
    - `src/types/chunking/strategy.py`: Chunking strategy interfaces
    - `src/types/chunking/text.py`: Text-specific chunking types
    - `src/types/chunking/code.py`: Code-specific chunking types
  - **Implementation Steps**:
    - Replace local type definitions with centralized types
    - Ensure compatibility with document types
    - Run mypy to verify type correctness
  
#### 4. embedding (Fourth Priority)

- **Branch**: `feature/types-embedding`
  - **Type Definitions**:
    - `src/types/embedding/vector.py`: Embedding vector types
    - `src/types/embedding/model.py`: Embedding model interfaces
    - `src/types/embedding/service.py`: Embedding service types
  - **Implementation Steps**:
    - Standardize embedding vector representation
    - Ensure compatibility with both text and code embedding needs
    - Run mypy to verify type correctness
  
#### 5. isne (Fifth Priority) [COMPLETED]

- **Branch**: `feature/types-isne`
  - **Type Definitions**:
    - `src/types/isne/models.py`: Core ISNE model types
    - `src/types/isne/graph.py`: Graph structure types
    - `src/types/isne/layers.py`: Neural network layer types
    - `src/types/isne/training.py`: Training types
    - `src/types/isne/inference.py`: Inference types
  - **Implementation Steps**:
    - [x] Consolidate types from multiple locations
    - [x] Migrate DocumentType, RelationType, IngestDocument, DocumentRelation, LoaderResult, and EmbeddingVector
    - [x] Update imports across codebase to use centralized types
    - [x] Add deprecation warnings to legacy type locations
    - [x] Ensure compatibility with embedding types
    - [x] Created missing PipelineConfig type in src/isne/pipeline/config.py
    - [x] Enhanced EmbeddingConfig TypedDict with additional fields (model_dimension, use_gpu, normalize_embeddings)
    - [x] Fixed import errors in src/cli/query.py and related modules
    - [x] Run mypy to verify type correctness
  
#### 6. storage (Sixth Priority)

- **Branch**: `feature/types-storage`
  - **Type Definitions**:
    - `src/types/storage/repository.py`: Repository interface types
    - `src/types/storage/document.py`: Document storage types
    - `src/types/storage/connection.py`: Database connection types
    - `src/types/storage/query.py`: Query types
  - **Implementation Steps**:
    - Refactor storage interfaces to use centralized types
    - Ensure compatibility with document and embedding types
    - Run mypy to verify type correctness
  
#### Common Type Utilities

- **Branch**: `feature/types-common`
  - **Type Definitions**:
    - `src/types/common/result.py`: Success/failure result types
    - `src/types/common/validation.py`: Validation result types
    - `src/types/common/config.py`: Configuration types
    - `src/types/common/utils.py`: Type utilities and helpers
  - **Implementation Steps**:
    - Create shared type utilities used across modules
    - Ensure compatibility with all module-specific types
    - Run mypy to verify type correctness
  
#### Final Integration

- **Branch**: `feature/types-integration`
  - **Implementation Steps**:
    - Resolve any cross-module type compatibility issues
    - Run comprehensive mypy checks across the entire codebase
    - Create final documentation for the type system
  
After all type issues are addressed, we will circle back for unit and integration testing to ensure the refactored code meets our test coverage requirements.

- [ ] Improve test coverage to ≥85% for schema modules

#### 6. Streaming Document Processing

- **DEPENDENCY**: requires an existing database. Initial database pipeline must be run first.
- [ ] Modify pipeline to support streaming processing mode
  - [ ] Add batch processing with configurable batch size
  - [ ] Implement document streaming to storage
  - [ ] Add memory usage monitoring
- [ ] Create tests for streaming document processing

#### 7. Error Handling & Retry Logic

- [ ] Enhance `AlertManager` to handle stage-specific failures
- [ ] Implement retry mechanisms for each stage
- [ ] Add detailed error reporting with context
- [ ] Create tests for error handling and recovery

#### 8. Multiprocessing Optimization

- [ ] Refactor to use `concurrent.futures.ProcessPoolExecutor`
- [ ] Improve exception handling in multiprocessing
- [ ] Optimize resource allocation
- [ ] Create tests for multiprocessing performance

### Lower Priority

#### 9. Dry Run Mode

- [ ] Implement dry run mode for testing without storage side effects
- [ ] Add detailed validation reporting in dry run mode
- [ ] Create tests for dry run functionality

#### 10. Logging and Debugging

- [ ] Add `--debug` flag for detailed pipeline execution information
- [ ] Implement intermediate artifact saving
- [ ] Create visualization tools for pipeline execution
- [ ] Add tests for debugging functionality

#### 11. Directory Scanning Improvements

- [ ] Add `--exclude-dir` and `--include-ext` CLI options
- [ ] Implement file manifest caching
- [ ] Optimize directory traversal for large repositories
- [ ] Create tests for directory scanning

## Implementation Approach

The implementation will follow an incremental approach:

1. First, implement the `PipelineStage` interface and refactor existing code
2. Then define Pydantic models for data structures
3. Implement the storage stage with streaming support
4. Add error handling and retry logic
5. Implement debugging and dry run features
This approach allows for continuous testing and validation as new components are added, ensuring the system remains functional throughout the development process.

## Completion Criteria

Each task will be considered complete when:

- Implementation passes all unit and integration tests
- Documentation is complete and accurate
- Code meets quality standards (type hints, docstrings, etc.)
- Performance benchmarks show no degradation (or document acceptable tradeoffs)
