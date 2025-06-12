# HADES Service Architecture Design

**Document ID**: HADES-ARCH-001  
**Version**: 1.0  
**Date**: 2025-01-11  
**Author**: System Design  
**Status**: Draft

## Related TaskManager Requests

- **req-65**: Design and implement HADES CLI interface as focused RAG component
- **req-62**: Build HADES framework with swappable components
- **req-61**: Complete migration to modular component architecture

## Table of Contents

1. [Overview](#1-overview)
2. [Service Architecture](#2-service-architecture)
3. [CLI Interface Design](#3-cli-interface-design)
4. [Service Specifications](#4-service-specifications)
5. [Implementation Plan](#5-implementation-plan)
6. [Deployment Strategy](#6-deployment-strategy)

## 1. Overview

### 1.1 HADES Scope in Olympus Ecosystem

HADES (Heuristic Adaptive Data Extrapolation System) is the RAG component of the Olympus AI framework:

**HADES Responsibilities:**

- Document ingestion and processing
- Graph-enhanced embeddings (ISNE/PathRAG)
- Knowledge graph management
- Retrieval API for other Olympus components

**HADES Interfaces:**

- **FastAPI Services**: REST API endpoints for all operations
- **MCP Integration**: Same endpoints available as MCP tools (via fastapi-mcp)
- **Service Orchestration**: Pipeline coordination via service-to-service calls

**Other Olympus Components:**

- **Delphi**: Interactive chat interface (consumes HADES API)
- **Hermes**: Message bus between Olympus services
- **Apollo**: LLM inference service
- **Zeus**: System coordinator

### 1.2 Architecture Principles

- **Unified API**: Single HADES service with modular internal components
- **Single MCP Server**: One MCP connection point for all RAG functionality
- **Internal modularity**: Clean component separation within unified service
- **Config-driven**: YAML configuration for all operations
- **Hardware flexible**: CPU/GPU configurable for embedding and ISNE services
- **Component swappable**: Implementations can be changed via config (ISNE → SageGraph)

## 2. Service Architecture

### 2.1 Unified API Design Rationale

**Why a Single HADES Service?**

After careful consideration of microservices vs monolithic approaches, a unified API provides the optimal solution:

1. **Single MCP Server**: Avoiding "MCP sprawl" - one connection for all RAG functionality
   - Claude Code connects to ONE MCP server
   - All RAG tools available through single interface
   - Simpler configuration and management

2. **Natural Dependencies**: The pipeline has sequential dependencies
   - Process → Chunk → Embed → Enhance → Store
   - These operations often happen in sequence
   - Internal orchestration is cleaner than service-to-service calls

3. **Modular Internals**: Best of both worlds
   - Components are internally separated and swappable
   - Clean interfaces between components
   - Easy to test individual components
   - Can still swap implementations (ISNE ↔ SageGraph)

4. **Simplified Operations**:
   - One service to deploy and monitor
   - Single configuration file
   - Unified logging and debugging
   - Easier resource management

### 2.2 API Structure

```
HADES Unified API (Port 8000)
├── FastAPI REST Endpoints
│   ├── /api/v1/process     # Document processing
│   ├── /api/v1/chunk       # Text chunking
│   ├── /api/v1/embed       # Embedding generation
│   ├── /api/v1/enhance     # ISNE enhancement
│   ├── /api/v1/store       # Store to ArangoDB
│   ├── /api/v1/retrieve    # Query and retrieval
│   ├── /api/v1/pipeline    # Full pipeline execution
│   └── /health             # Service health check
│
└── MCP Tools (via fastapi-mcp)
    ├── process_documents()
    ├── chunk_text()
    ├── generate_embeddings()
    ├── enhance_embeddings()
    ├── store_data()
    ├── retrieve_data()
    └── run_pipeline()
```

### 2.3 Internal Component Architecture

```
src/
├── api/
│   ├── main.py           # FastAPI application
│   └── routes/           # API endpoint definitions
├── components/           # Internal modular components
│   ├── docproc/         # Document processing
│   ├── chunking/        # Text chunking
│   ├── embedding/       # Embedding generation
│   ├── graph_enhancement/  # ISNE/SageGraph
│   └── storage/         # ArangoDB integration
└── mcp/
    └── server.py        # Single MCP server integration
```

### 2.4 Hardware Resource Allocation

**Target Hardware**: ASRock TRX50 WS + Threadripper 7960x + 256GB RAM + 2x RTX A6000

```yaml
hades_service:
  port: 8000
  total_resources:
    cpu_cores: 16-24       # Plenty for all operations
    memory: 128GB          # Shared across components
    storage:
      models: RAID0        # Fast model loading
      data: RAID1          # Safe data storage
      working: RAID5       # Document processing
    
  component_allocation:
    docproc:
      cpu_cores: 4-8
      memory: 16GB
      
    chunking:
      cpu_cores: 2-4
      memory: 8GB
      
    embedding:
      device: "gpu"        # Configurable: "gpu" or "cpu"
      gpu: A6000_0         # When device=gpu
      cpu_cores: 4
      memory: 32GB
      
    isne:
      device: "gpu"        # Configurable: "gpu" or "cpu"
      gpu: [A6000_0, A6000_1]  # Multi-GPU when device=gpu
      cpu_cores: 8
      memory: 64GB
      
    storage:
      cpu_cores: 4-8
      memory: 32GB
```

### 2.5 Communication Patterns

- **Framework**: Single FastAPI application with fastapi-mcp integration
- **External API**: REST endpoints on port 8000
- **MCP Protocol**: All endpoints exposed as MCP tools automatically
- **Internal**: Direct function calls between components (no network overhead)
- **Data Format**: JSON for API, native Python objects internally
- **Deployment**: Single service on localhost:8000

## 3. FastAPI-MCP Service Design

### 3.1 Unified Interface Pattern

**Each service provides both REST and MCP interfaces simultaneously:**

```python
# Single FastAPI service with dual interfaces
from fastapi import FastAPI
from fastapi_mcp import FastApiMCP

app = FastAPI(title="HADES DocProc Service")
mcp = FastApiMCP(app)

@app.post("/process")  # REST endpoint
async def process_documents(files: List[UploadFile]) -> ProcessedDocs:
    """Process documents - available as both REST and MCP tool"""
    return await docproc.process(files)

mcp.mount()  # Enables MCP tools automatically
```

### 3.2 Service Access Patterns

```bash
# Direct service interaction (development/testing)
curl -X POST http://localhost:8001/process -F "files=@doc.pdf"

# Service orchestration (pipeline execution)
python orchestrator.py --config config/ingest.yaml

# Claude Code integration (MCP protocol)
# Automatic via fastapi-mcp - no additional setup needed
```

### 3.2 Configuration Schema

**Unified YAML structure** for all operations:

```yaml
# Base configuration template
pipeline:
  type: "ingestion|training|bootstrap|serving"
  version: "1.0"
  description: "Human readable description"

execution:
  input_dir: "${INPUT_DIR}"
  output_dir: "${OUTPUT_DIR}"
  batch_size: 10
  parallel_workers: 4
  debug_mode: false

services:
  docproc:
    enabled: true
    type: "docling"
    host: "127.0.0.1"
    port: 8001
    params:
      timeout: 300
      batch_size: 5

  chunking:
    enabled: true
    type: "chonky"
    host: "127.0.0.1"
    port: 8002
    params:
      chunk_size: 512
      overlap: 50

# Environment variable support
database:
  url: "${ARANGO_URL:-http://localhost:8529}"
  username: "${ARANGO_USER:-root}"
  password: "${ARANGO_PASS}"
```

## 4. API Endpoint Specifications

### 4.1 Document Processing Endpoints

**Endpoint**: `POST /api/v1/process`
**Purpose**: Process raw documents into structured format
**Input**: Raw files (PDF, MD, JSON, YAML, Python)
**Output**: Structured document objects
**MCP Tool**: `process_documents(files: List[str])`

### 4.2 Chunking Endpoints

**Endpoint**: `POST /api/v1/chunk`
**Purpose**: Break documents into manageable chunks
**Input**: Structured document objects
**Output**: Document chunks with metadata
**MCP Tool**: `chunk_text(documents: List[Document])`

### 4.3 Embedding Endpoints

**Endpoint**: `POST /api/v1/embed`
**Purpose**: Generate embeddings for chunks
**Input**: Document chunks
**Output**: Embedded chunks with vectors
**Hardware**: Configurable GPU/CPU via config
**MCP Tool**: `generate_embeddings(chunks: List[Chunk])`

### 4.4 ISNE Enhancement Endpoints

**Endpoint**: `POST /api/v1/enhance`
**Purpose**: Graph-based embedding enhancement
**Input**: Embedded chunks
**Output**: Graph-enhanced embeddings
**Hardware**: Configurable multi-GPU/CPU via config
**MCP Tool**: `enhance_embeddings(embeddings: List[Embedding])`

**Endpoint**: `POST /api/v1/train`
**Purpose**: Train ISNE model
**Input**: Training configuration
**Output**: Training status
**MCP Tool**: `train_isne_model(config: TrainingConfig)`

### 4.5 Storage Endpoints

**Endpoint**: `POST /api/v1/store`
**Purpose**: Persist data to ArangoDB
**Input**: Enhanced embeddings
**Output**: Storage confirmation
**MCP Tool**: `store_data(data: List[EnhancedEmbedding])`

**Endpoint**: `POST /api/v1/retrieve`
**Purpose**: Query and retrieve data
**Input**: Query parameters
**Output**: Retrieved documents/chunks
**MCP Tool**: `retrieve_data(query: str, params: QueryParams)`

### 4.6 Pipeline Endpoint

**Endpoint**: `POST /api/v1/pipeline`
**Purpose**: Execute full ingestion pipeline
**Input**: Raw files + pipeline configuration
**Output**: Pipeline execution status
**MCP Tool**: `run_pipeline(files: List[str], config: PipelineConfig)`

## 5. Implementation Plan

### 5.1 Development Phases

**Phase 1: Service Foundation** (TaskManager: req-65)

- [ ] [task-572] Design unified config schema
- [ ] [task-574] Create service config templates
- [ ] [task-579] Add config validation system
- [ ] [task-XXX] Test fastapi-mcp integration proof-of-concept

**Phase 2: Service Framework** (TaskManager: req-62)

- [ ] [task-XXX] Create HADESService base class with FastAPI+MCP
- [ ] [task-578] Implement service protocol interfaces
- [ ] [task-XXX] Build service orchestration framework
- [ ] [task-XXX] Create pipeline coordination via service calls

**Phase 3: Component Services** (TaskManager: req-61)

- [ ] [task-XXX] Implement DocProc service
- [ ] [task-XXX] Implement Chunking service  
- [ ] [task-XXX] Implement Embedding service
- [ ] [task-XXX] Implement Storage service

**Phase 4: ISNE Integration** (TaskManager: req-63)

- [ ] [task-XXX] Implement ISNE service
- [ ] [task-XXX] Multi-GPU training support
- [ ] [task-XXX] Model persistence and loading

**Phase 5: Pipeline Assembly** (TaskManager: req-64)

- [ ] [task-575] Implement PipelineRunner framework
- [ ] [task-XXX] Service orchestration
- [ ] [task-XXX] End-to-end testing

### 5.2 Testing Strategy

- **Unit Tests**: Each service component (85% coverage requirement)
- **Integration Tests**: Service-to-service communication
- **End-to-End Tests**: Full pipeline execution
- **Performance Tests**: Throughput and latency benchmarks

## 6. Deployment Strategy

### 6.1 Single-Node Deployment

**Target**: Threadripper workstation with unified HADES service

```bash
# Start HADES service
hades serve --config config/production.yaml

# Or with uvicorn directly
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Service will be available at:
# REST API: http://localhost:8000/api/v1/
# MCP Server: http://localhost:8000 (via fastapi-mcp)
# API Docs: http://localhost:8000/docs
```

### 6.2 Future Distributed Deployment

**Architecture**: Ready for multi-node deployment

```yaml
# Future distributed config
services:
  docproc:
    host: "cpu-node-1.local"
  embedding:
    host: "gpu-node-1.local"
  isne:
    host: "gpu-node-2.local"
```

### 6.3 Integration Points

**Olympus Ecosystem Integration Examples**:

- **Delphi (Chat Interface)**:
  ```python
  # Pre-process documents before sending to cloud LLM
  embeddings = await httpx.post("http://localhost:8000/api/v1/embed", json=doc)
  relevant_chunks = filter_by_similarity(embeddings, query)
  # Or retrieve from knowledge base
  results = await httpx.post("http://localhost:8000/api/v1/retrieve", 
                            json={"query": user_question})
  ```

- **Apollo (LLM Service)**:
  ```python
  # Chunk large contexts for LLM processing
  chunks = await httpx.post("http://localhost:8000/api/v1/chunk", json=document)
  ```

- **Zeus (System Coordinator)**:
  ```python
  # Full pipeline for system documentation
  result = await httpx.post("http://localhost:8000/api/v1/pipeline", 
                           files=config_files,
                           json={"config": pipeline_config})
  ```

- **Claude Code**: Single MCP connection provides all RAG tools:
  ```
  mcp_client.connect("http://localhost:8000")
  # All tools available: process_documents(), chunk_text(), 
  # generate_embeddings(), retrieve_data(), etc.
  ```

**External Dependencies**:

- **ArangoDB**: Graph database backend (pre-installed)
- **FastAPI**: Web framework for services
- **fastapi-mcp**: MCP integration library
- **vLLM**: Embedding model serving
- **PyTorch**: ISNE model training

## 7. Future Enhancements

### 7.1 MCP Integration Status

**Status**: Integrated via fastapi-mcp from Phase 1
**Purpose**: Claude Code integration for development assistance
**Implementation**: Automatic MCP tool exposure of FastAPI endpoints

### 7.2 A/B Testing Framework

**Feature**: Component swapping (ISNE ↔ SageGraph)
**Implementation**: Config-driven component selection

### 7.3 Monitoring and Observability

**Tools**: Prometheus metrics, structured logging
**Dashboards**: Service health, performance metrics

---

## Task References

**Implementation Tasks** (to be created in TaskManager):

- CLI Framework: Section 3.1, 5.1 Phase 1
- Service Contracts: Section 4.0, 5.1 Phase 2
- Component Development: Section 4.1-4.5, 5.1 Phase 3
- Pipeline Assembly: Section 5.1 Phase 5

**Related Documentation**:

- CLAUDE.md: Development protocols and standards
- CONFIG_DRIVEN_ARCHITECTURE.md: Modular design principles
- MIGRATION_STATUS.md: Current architectural transition status
