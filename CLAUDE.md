# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Python Development
- `poetry install` - Install Python dependencies
- `poetry install --with dev` - Include development dependencies
- `poetry run pytest` - Run all tests
- `poetry run pytest -k <pattern>` - Run specific tests matching pattern
- `poetry run pytest tests/smoke/` - Run smoke tests only
- `poetry run pytest tests/runtime/` - Run runtime tests only
- `poetry run pytest --cov=core` - Run tests with coverage report
- `poetry run ruff check` - Lint Python code
- `poetry run ruff format` - Auto-format Python code
- `make lint` - Run linting via Makefile
- `make format` - Format code via Makefile
- `make test` - Run tests via Makefile

### Database Operations
- `make db-bootstrap` - Bootstrap ArangoDB (runs scripts/dev_bootstrap.sh)
- `make migrate` - Apply ArangoDB migrations (currently a stub)

### Main Entry Points
- `poetry run python scripts/ingest_repo_workflow.py` - Ingest repository (normalize → embed → chunk → write)
- `poetry run python scripts/hades_chat.py` - Interactive HADES agent chat with experiential memory and PathRAG retrieval
- `poetry run python scripts/hades_chat.py --retrieve --graphsage` - Chat with GraphSAGE-enhanced retrieval
- `poetry run python scripts/clear_hades_db.py` - Clear database collections

### GraphSAGE Training & Inference
- `poetry run python scripts/export_graph_for_training.py` - Export ArangoDB graph to PyTorch Geometric format
- `poetry run python scripts/generate_training_data.py` - Generate positive/negative query-node pairs for training
- `poetry run python scripts/train_graphsage.py --epochs 5 --batch-size 16` - Train GraphSAGE model with contrastive loss
- `poetry run python scripts/test_graphsage_pathrag_integration.py` - Test GraphSAGE + PathRAG end-to-end pipeline

### Model Testing (Qwen3-30B)
- `poetry run python scripts/test_qwen_engine.py --dry-run` - Dry run model loading
- `poetry run python scripts/test_qwen_engine.py` - Load model and print limits
- `poetry run python scripts/probe_qwen_longctx.py --chars 262144 --max-new-tokens 64 --telemetry` - Test long context handling

## Architecture Overview

### Core Components

**Experiential Memory System**
Local Python library for agent memory operations (NOT MemGPT paper architecture).
Prompt engineering patterns inspired by: https://github.com/doobidoo/mcp-memory-service

Memory types:
- observations: Raw agent experiences (bug fixes, decisions, events)
- reflections: Synthesized insights from multiple observations
- entities: People/concepts/components extracted from experiences
- relationships: Knowledge graph edges connecting entities

All operations run in-process via ArangoDB Unix sockets (no network services).

**Retrieval Pipeline (GraphSAGE + PathRAG)**
The system implements GraphSAGE-enhanced PathRAG retrieval with strict performance targets:

**Pipeline Architecture (User-Corrected Order):**
```
query → GraphSAGE (find k=50 candidates) → PathRAG (prune to H≤2, F≤3, B≤8) → context
```

GraphSAGE finds many semantically relevant nodes using learned embeddings, then PathRAG enforces budget constraints through graph traversal.

**GraphSAGE Model:**
- Architecture: 2048-dim (Jina v4) → 1024-dim → 512-dim
- Multi-relational: separate aggregators per edge type (imports, contains, references, relates_to, derives_from)
- Inductive learning: generates embeddings for NEW nodes without retraining (critical for dynamic experiential memory)
- Query projection layer (2048→512) bridges Jina embeddings to GNN output space
- Training: contrastive loss with positive/negative query-node pairs
- Performance: 43-97ms p50 retrieval latency

**PathRAG Constraints:**
- Query plan enforces caps: beam ≤ 8-16, hops ≤ 2, fanout ≤ 2-3, docs_max ≤ 160
- Global latency SLO: 2s for 95% of queries
- Staged timeouts with reason codes (graphsage_candidates, cap_exceeded, timeout, budget_docs, etc.)
- Graceful fallback to vector search if GraphSAGE fails

**ArangoDB Integration**
- Unix socket connections for performance (avoid TCP overhead)
  - Read-only: `/run/hades/readonly/arangod.sock` (0660 group permissions)
  - Read-write: `/run/hades/readwrite/arangod.sock` (0600 owner-only in prod)
- Primary database: `hades_memories`
- Collections: repo_docs, doc_chunks, core_memory, messages, recall_log, archival_docs, memories, concepts
- View: repo_text (links doc_chunks.text with analyzer)
- Go proxies provide additional security layer (compiled binaries in `core/database/arango/proxies/`)

**Performance Targets (SLOs)**
- Version query: p50 ≤ 0.4ms, p95 ≤ 0.9ms
- Small cursor: p50 ≤ 0.8ms, p95 ≤ 1.6ms
- Recall log: p99 ≤ 250ms
- Archival vector search: p99 ≤ 750ms
- User-agent latency: avg < 100ms
- Proxy overhead: ≤ 0.2ms p50

### Module Structure

- `core/extractors/` - Document and code extraction with TreeSitter
- `core/embedders/` - Embedding services (Jina v4, SentenceTransformers)
- `core/config/` - Configuration management with YAML schemas
- `core/database/arango/` - Optimized HTTP/2 ArangoDB client and Go proxies
- `core/gnn/` - GraphSAGE implementation (model, training, inference, graph building)
- `core/runtime/memory/` - Experiential memory system (observations, reflections, entities)
- `core/workflows/` - High-level workflow orchestration
- `scripts/` - CLI tools for ingestion, chat, GraphSAGE training, and administration
- `infra/` - Systemd units and runtime configuration
- `tests/` - Smoke tests and benchmarks mirroring module structure
- `models/` - Trained model artifacts (GraphSAGE checkpoints, graph exports, training data)
- `acheron/` - Archived legacy code (excluded from ingestion)

### Conveyance Framework

HADES measures agent-user communication efficiency using the Conveyance framework:
- Core metric: C = (W·R·H / T) · C_ext^α
- W: Semantic quality, R: Structure, H: Agent capability, T: Latency/cost
- Measurements occur **only at user↔agent boundaries**, not internal operations
- Context amplification α is learned online (target 95% CI: [1.5, 2.0])
- Protocol compatibility P_ij = product of schema, arg types, media, and safety scores

### Experiential Memory Integration

Local library for agent memory operations:
- store_observation: Persist experiences with Jina v4 embeddings
- retrieve_memories: PathRAG-powered semantic search with graph traversal
- search_by_tag: Category-based retrieval
- consolidate_memories: Generate reflections from observations (LLM-powered)

Memory types: observations → reflections → entities → relationships (knowledge graph)
Prompt engineering patterns inspired by: https://github.com/doobidoo/mcp-memory-service
Conveyance metrics tracked at agent-user boundary only

## Testing Guidelines

- Tests mirror the module tree under `tests/`
- Target ≥80% coverage for new code
- Mock external services in unit tests
- Use `pytest.mark.integration` for real-service tests
- Pre-commit hooks run both linting and tests automatically

## Environment Variables

Key configuration (see `.env.example` for full list):
- `ARANGO_DB_NAME=hades_memories` - Primary database
- `ARANGO_RO_SOCKET=/run/hades/readonly/arangod.sock` - Read-only socket
- `ARANGO_RW_SOCKET=/run/hades/readwrite/arangod.sock` - Read-write socket
- `ARANGO_USERNAME`, `ARANGO_PASSWORD` - Authentication (omit for local dev)
- `FF_CONVEYANCE_ON=1` - Enable conveyance metrics
- `FF_ALPHA_LEARNING=1` - Enable context amplification learning
- `QWEN_MODEL_ID=Qwen/Qwen3-30B-A3B-Thinking-2507` - Model identifier
- `QWEN_DEVICE_MAP=balanced` - GPU distribution
- `QWEN_MAX_MEMORY=0=44GiB,1=44GiB` - Per-GPU memory allocation
- `QWEN_USE_FLASH_ATTN=1` - Enable FlashAttention-2 optimization