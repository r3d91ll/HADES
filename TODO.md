# HADES-PathRAG Development Tasks

## Pipeline Architecture Enhancements

This section outlines the planned improvements to the pipeline architecture for enhanced modularity, scalability, and maintainability. These changes are designed to prepare the ingestion pipeline for separation into dedicated services.

### High Priority

#### 1. Pipeline Stage Interface Implementation

- [x] Create `src/pipeline/stages` directory with appropriate module structure
- [x] Implement abstract `PipelineStage` base class with:
  - [x] `run(input_data: Dict) -> Dict` method
  - [x] `validate(input_data: Dict) -> bool` method
  - [x] Standardized error handling interface
  - [x] Logging mechanisms
- [x] Create comprehensive documentation for stage interface
- [x] Implement unit tests for stage interface validation

#### 2. Concrete Stage Implementations

- [x] Implement `DocumentProcessingStage` class wrapping document processing functionality
  - [x] Add support for different document adapter types
  - [x] Implement validation for document processing inputs/outputs
- [x] Implement `ChunkingStage` class for document chunking
  - [x] Support both text and code chunking
  - [x] Add chunk validation mechanisms
- [x] Implement `EmbeddingStage` for generating embeddings
  - [x] Support multiple embedding models
  - [x] Include embedding validation
- [x] Implement `ISNEStage` for graph-based embedding enhancement
  - [x] Support relationship preservation
  - [x] Include validation for ISNE embeddings and relationships
- [x] Create comprehensive tests for each stage

#### 3. Pydantic Schema Definition

- [x] Create `src/pipeline/schema.py` module with Pydantic models
- [x] Define core data models:
  - [x] `ChunkSchema` model with embeddings and relationships
  - [x] `DocumentSchema` model with chunks and metadata
  - [x] `ValidationResult` model for tracking validation status
- [x] Integrate schema validation with pipeline stages
- [x] Create tests for schema validation

#### 4. Storage Stage Implementation

- [x] Implement `StorageStage` for ArangoDB interaction
  - [x] Support different storage modes (create/append/upsert)
  - [x] Add retry logic with exponential backoff
  - [x] Implement transactional operations
- [x] Add validation for database operations
- [x] Create tests for storage stage

### Medium Priority

#### 5. Streaming Document Processing

- **DEPENDENCY**: requires an existing database. Initial database pipeline must be run first.
- [ ] Modify pipeline to support streaming processing mode
  - [ ] Add batch processing with configurable batch size
  - [ ] Implement document streaming to storage
  - [ ] Add memory usage monitoring
- [ ] Create tests for streaming document processing

#### 6. Error Handling & Retry Logic

- [ ] Enhance `AlertManager` to handle stage-specific failures
- [ ] Implement retry mechanisms for each stage
- [ ] Add detailed error reporting with context
- [ ] Create tests for error handling and recovery

#### 7. Multiprocessing Optimization

- [ ] Refactor to use `concurrent.futures.ProcessPoolExecutor`
- [ ] Improve exception handling in multiprocessing
- [ ] Optimize resource allocation
- [ ] Create tests for multiprocessing performance

### Lower Priority

#### 8. Dry Run Mode

- [ ] Implement dry run mode for testing without storage side effects
- [ ] Add detailed validation reporting in dry run mode
- [ ] Create tests for dry run functionality

#### 9. Logging and Debugging

- [ ] Add `--debug` flag for detailed pipeline execution information
- [ ] Implement intermediate artifact saving
- [ ] Create visualization tools for pipeline execution
- [ ] Add tests for debugging functionality

#### 10. Directory Scanning Improvements

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
