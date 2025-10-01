# HADES Project Status & Roadmap

**Last Updated:** 2025-09-30
**Program Plan:** v1.1 (Docs/Program_Plan.md)

## Architecture Overview

HADES implements an agent runtime with experiential memory, powered by:

- **ArangoDB** graph database (Unix socket connections)
- **GraphSAGE + PathRAG** retrieval with learned node embeddings and graph traversal
- **Jina v4** embeddings (32k context, 2048 dimensions)
- **Qwen3-30B-A3B-Thinking** model (262k context)

**Experiential memory system** (NOT MemGPT paper architecture):

- Observations: Raw agent experiences
- Reflections: Synthesized insights
- Entities: Extracted concepts/people/components
- Relationships: Knowledge graph edges

Prompt engineering patterns inspired by: <https://github.com/doobidoo/mcp-memory-service>

## Phase Status Summary

| Phase | Name | Status | Completion |
|-------|------|--------|------------|
| A | Transport (Sockets & Proxies) | ✅ **COMPLETE** | 100% |
| B | Ingestion Pipeline | ⚠️ **READY** | 95% |
| C | PathRAG v1 Retrieval | ⚠️ **PARTIAL** | 75% |
| D | Telemetry & Metrics | ❌ **TODO** | 0% |
| E | Agent Runtime Integration | ⚠️ **PARTIAL** | 60% |
| F | Learned Scoring (GraphSAGE) | ✅ **COMPLETE** | 100% |

---

## Phase A: Transport ✅ COMPLETE

**Goal:** High-performance ArangoDB access via Unix sockets with strict SLOs

### Achievements

✅ Optimized HTTP/2 client (`core/database/arango/optimized_client.py`)
✅ Go proxy binaries (RO/RW in `core/database/arango/proxies/`)
✅ Socket permissions configured (RO: 0660 group, RW: 0600 owner)
✅ Performance benchmarks meet SLOs:

- Upstream version: p50=0.192ms, p95=0.241ms ✅ (target: p50≤0.4ms, p95≤0.9ms)
- RO proxy cursor: p50=0.356ms, p95=0.455ms ✅ (target: p50≤0.8ms, p95≤1.6ms)
- Proxy overhead: ~0.06ms ✅ (target: ≤0.2ms)

### Acceptance Criteria Met

✅ SLOs met with margin
✅ Permissions validated
✅ Access logs/counters visible

---

## Phase B: Ingestion Pipeline ⚠️ READY (Needs Execution)

**Goal:** Idempotent ingestion of code/docs with embeddings

### Achievements

✅ Ingestion script with pipeline: normalize → embed → chunk → write
✅ `.hades.ignore` support (gitignore-like exclusions)
✅ PDF support via Docling
✅ TreeSitter code extraction
✅ Jina v4 embedding generation
✅ Late chunking support
✅ Collection schema definitions

### What's Left

⚠️ **Run actual ingestion** to populate database:

```bash
poetry run python scripts/ingest_repo_workflow.py --root . --write --device cuda --batch 16
```

### Acceptance Criteria Status

✅ Idempotent re-runs supported
✅ No analyzer/collection collisions
⚠️ Needs execution with real data

**Files:**

- `scripts/ingest_repo_workflow.py` - Main ingestion orchestrator
- `core/extractors/` - Document/code extraction
- `core/embedders/` - Jina v4 embedding generation
- `.hades.ignore` - Exclusion patterns

---

## Phase C: PathRAG v1 Retrieval ⚠️ PARTIAL (75%)

**Goal:** Graph-augmented retrieval with strict caps and SLOs

### Achievements

✅ PathRAG scaffold with beam search (`core/runtime/memory/pathrag.py`)
✅ Memory-specific PathRAG (`pathrag_memory.py`):

- 3-stage pipeline: vector seeds → graph traversal → result packing
- MMR diversity scoring
- Recency boosting
- Multiple traversal patterns (obs→entity, obs→reflection, entity→entity)
✅ Caps enforced: H≤2, F≤3, B≤8
✅ Timeout enforcement with reason codes
✅ Integration into `MemoryStore.retrieve_memories()`

### What's Left

❌ **Reranker** (Stage 2.5: between PathRAG and final packing)

- Cross-encoder or bi-encoder scoring
- Top-M selection (m=20)
- Timeout: 500ms
- File: `core/runtime/memory/reranker.py` (needs creation)

❌ **Coverage score calculation** (BM25-based token-weighted term coverage)

⚠️ **Integration testing** with actual ingested data

### Acceptance Criteria Status

✅ Caps never exceeded (enforced in code)
⚠️ ≥95% runs meet 2s global SLO (needs testing with real data)
✅ Reasons emitted on trims/early stop

**Files:**

- `core/runtime/memory/pathrag_memory.py` - Memory graph traversal
- `core/runtime/memory/operations.py` - retrieve_memories() implementation
- `core/runtime/memory/plan_executor.py` - Query plan normalization

---

## Phase D: Telemetry & Metrics ❌ TODO (0%)

**Goal:** Per-stage observability with boundary-only conveyance tracking

### What's Needed

❌ **Telemetry tracker module** (`core/runtime/memory/telemetry_tracker.py`):

- Per-stage metrics: reads_stage, latency_stage, dedupe_rate
- Coverage score calculation
- Capped/early_stop flags
- Reason code aggregation

❌ **Boundary metric collection**:

- W·R·H measurements at user↔agent boundaries ONLY
- Internal operation metrics (NOT conveyance)
- Integration with existing telemetry.py

❌ **Phoenix/Arize exporters**:

- Telemetry event export
- Dashboard integration
- Conveyance uplift visualization

### Acceptance Criteria

❌ Per-stage reads/latency/coverage visible
❌ Boundary W·R·H vs T tracked
❌ Boundary-only conveyance events complete (no internal operation pollution)

**Reference:** Docs/theory/updated_conveyance_framework.md

---

## Phase E: Agent Runtime Integration ⚠️ PARTIAL (60%)

**Goal:** Integrate experiential memory into agent chat loop with autonomous usage

### Achievements

✅ **Experiential memory system** (`core/runtime/memory/`):

- Schema defined (observations, reflections, entities, relationships)
- Type-safe models with Pydantic
- Operations: store_observation, retrieve_memories, search_by_tag, list, archive
- PathRAG-powered retrieval
- Consolidation engine for reflection generation

✅ **Prompt engineering**:

- System prompts for autonomous memory management
- Tool descriptions with usage guidelines
- When to store/retrieve patterns
- Quality over quantity guidelines

✅ **Model engine** (`core/runtime/memory/model_engine.py`):

- Qwen3-30B-A3B-Thinking support
- AWQ quantization
- FlashAttention-2 optimization
- 262k context window

### What's Left

❌ **Chat integration** (`scripts/hades_chat.py`):

- Update hades_chat.py with experiential memory operations (MCP pattern)
- System prompt injection (memory guidelines)
- Autonomous store/retrieve decisions with PathRAG
- Session management with memory context

❌ **Plan executor integration**:

- Query plan contract enforcement
- Budget tracking (docs_max, latency_ms)
- Stage coordination

❌ **α̂ estimator** (context amplification learning):

- Online learning from boundary data
- Target: 95% CI in [1.5, 2.0]
- Decay/redundancy enforcement

❌ **Boundary metric tracking**:

- Collect W, R, H, T at user↔agent boundaries
- Compute C_pair = Hmean(C_out, C_in)·C_ext^α·P_ij
- Log to telemetry system

### Acceptance Criteria Status

✅ Memory operations implemented
⚠️ Recall p99 ≤ 250ms (needs testing)
⚠️ Vector p99 ≤ 750ms (needs testing)
❌ User-agent avg < 100ms (not integrated)
❌ α̂ CI in band [1.5, 2.0] (not implemented)

**Files:**

- `core/runtime/memory/operations.py` - Core memory API (MCP pattern)
- `core/runtime/memory/consolidation.py` - Reflection generation
- `core/runtime/memory/templates/` - System prompts
- `scripts/hades_chat.py` - Agent chat with experiential memory

---

## Phase F: GraphSAGE Integration ✅ COMPLETE (100%)

**Goal:** Train GraphSAGE for inductive node embeddings to enable retrieval over dynamic experiential memory graph

**Why Required:**
- Experiential memory graph grows continuously (observations → reflections → entities)
- Need to score relevance of NEW memories not seen during training
- GraphSAGE provides inductive embeddings without retraining
- Enables agent to retrieve its own past experiences semantically
- **Grounds fuzzy heuristics** - learns edge relevance, node importance, path quality from data
- Heuristic v1 works but is hand-tuned; GNN makes it adaptive and data-driven

**Pipeline Architecture (User-Corrected Order):**
```
query → GraphSAGE (find k=50 candidates) → PathRAG (prune to H≤2, F≤3, B≤8) → context
```

GraphSAGE finds many semantically relevant nodes, then PathRAG enforces budget constraints through graph traversal.

### Achievements

✅ **Multi-relational GraphSAGE model** (`core/gnn/graphsage_model.py`):

- Architecture: 2048-dim (Jina v4) → 1024-dim → 512-dim
- Separate aggregators per edge type (imports, contains, references, relates_to, derives_from)
- Inductive learning: generates embeddings for NEW nodes without retraining
- Query projection layer (2048→512) bridges Jina embeddings to GNN output space

✅ **Graph export and training data** (`core/gnn/graph_builder.py`, `scripts/generate_training_data.py`):

- 99 nodes (code files from repo_docs)
- 252 edges (47 imports + 205 references)
- 696 training pairs (232 positive, 464 negative, 33.3% positive ratio)
- Positive examples from file/function names + docstrings
- Negative sampling with random non-matching documents

✅ **Training pipeline** (`core/gnn/trainer.py`, `scripts/train_graphsage.py`):

- Contrastive loss with margin=0.5 (ranking loss for retrieval)
- Best validation: loss=0.153, accuracy=66.2%
- Positive similarity: 0.696, Negative similarity: 0.463
- Model saved to `models/checkpoints/best.pt`

✅ **Fast inference** (`core/gnn/inference.py`):

- Precomputed node embeddings for <50ms retrieval
- Query projection for dimension matching
- Cosine similarity ranking with top-k filtering

✅ **PathRAG integration** (`core/runtime/memory/pathrag_code.py`):

- GraphSAGE as optional enhancement (graceful fallback to vector search)
- Hybrid scoring: GNN candidates → PathRAG beam search
- Tested latency: 43-97ms (within <100ms target)

✅ **Chat integration** (`scripts/hades_chat.py`):

- `--graphsage` flag to enable GNN-enhanced retrieval
- Loads checkpoint and precomputes embeddings at startup
- Integrated with PathRAG retrieval in conversational loop

### Acceptance Criteria Met

✅ GraphSAGE embeddings for new nodes without retraining (inductive learning)
✅ Retrieval quality: pos_sim=0.696 > neg_sim=0.463 (clear separation)
✅ Latency within budget: 43-97ms p50 (target: <150ms p95)
✅ Works with dynamic code graph (can extend to experiential memory)

**Files:**

- `core/gnn/graphsage_model.py` - Multi-relational GraphSAGE
- `core/gnn/graph_builder.py` - ArangoDB → PyG graph export
- `core/gnn/trainer.py` - Training loop with contrastive loss
- `core/gnn/inference.py` - Fast inference with precomputed embeddings
- `scripts/export_graph_for_training.py` - Graph data export
- `scripts/generate_training_data.py` - Positive/negative pair generation
- `scripts/train_graphsage.py` - Training orchestrator
- `scripts/hades_chat.py` - Chat with GraphSAGE-enhanced retrieval

---

## Critical Path Forward

### Immediate Next Steps (1-2 weeks)

#### 1. **Run Ingestion** (1 day)

```bash
# Populate database with actual codebase
poetry run python scripts/ingest_repo_workflow.py --root . --write --device cuda --batch 16

# Verify
poetry run python -c "from core.database.arango.memory_client import *; \
  c = ArangoMemoryClient(resolve_memory_config()); \
  print('Docs:', c.execute_query('RETURN COUNT(repo_docs)')); \
  print('Chunks:', c.execute_query('RETURN COUNT(doc_chunks)'))"
```

#### 2. **Implement Reranker** (2-3 days)

Create `core/runtime/memory/reranker.py`:

- Cross-encoder scoring (or bi-encoder if faster)
- Top-M selection with timeout enforcement
- Integration into retrieve_memories() pipeline

#### 3. **Build Chat Integration** (3-5 days)

Create `scripts/memory_chat.py`:

- Load MemoryStore with system prompts
- Qwen3 model with memory operation tools
- Autonomous memory management loop
- Session tracking with observations

#### 4. **Add Telemetry Tracking** (2-3 days)

Create `core/runtime/memory/telemetry_tracker.py`:

- Per-stage metrics collection
- Coverage score calculation
- Boundary metric separation

#### 5. **Integration Testing** (2-3 days)

- Test PathRAG with real data
- Verify SLOs under load
- Tune caps/timeouts
- Validate memory operations

### Medium-term (3-4 weeks)

#### 6. **α̂ Estimator** (1 week)

Implement context amplification learning:

- Collect boundary conveyance data
- Online α estimation
- Validation against target CI

#### 7. **Full Telemetry Integration** (1 week)

- Phoenix/Arize exporters
- Dashboards for conveyance metrics
- Performance monitoring

#### 8. **Production Hardening** (1-2 weeks)

- Error handling and fallbacks
- Rate limiting and backpressure
- Monitoring and alerting
- Documentation and runbooks

### Long-term (1-2 months)

#### 9. **GraphSAGE Training** (Phase F)

- Feature extraction pipeline
- Model training infrastructure
- A/B testing framework
- Scoring swap with validation

---

## Key Risks & Mitigations

### Risk: Ingestion fails or is too slow

**Mitigation:** Test with small subset first; optimize batch sizes; consider distributed processing

### Risk: PathRAG doesn't improve retrieval quality

**Mitigation:** A/B test against simple vector search; tune caps; add more graph edges

### Risk: Agent doesn't use memory autonomously

**Mitigation:** Refine system prompts; add examples; tune when-to-store heuristics

### Risk: SLOs not met under load

**Mitigation:** Profile hot paths; optimize AQL queries; tune timeouts; add caching

### Risk: Context amplification α̂ doesn't converge

**Mitigation:** Simplify α learning; use fixed α initially; more training data

---

## Testing Strategy

### Unit Tests (Target: ≥80% coverage)

- ✅ Models (Observation, Reflection, Entity) serialization
- ⚠️ Memory operations (store, retrieve, search)
- ⚠️ PathRAG traversal with mocked data
- ❌ Telemetry tracking
- ❌ Consolidation engine

### Integration Tests

- ❌ End-to-end ingestion → retrieve workflow
- ❌ PathRAG with real ArangoDB data
- ❌ Chat loop with memory operations
- ❌ SLO validation under load

### Performance Tests

- ⚠️ Socket benchmarks (done, meets targets)
- ❌ PathRAG latency with various graph sizes
- ❌ Memory operation throughput
- ❌ Concurrent chat sessions

---

## Documentation Status

### Complete

✅ `Docs/SCHEMA_Experiential_Memory.md` - Memory collections specification
✅ `core/runtime/memory/README.md` - Module overview
✅ `core/runtime/memory/templates/memory_system_prompt.md` - Agent guidelines
✅ `Docs/Program_Plan.md` - v1.1 with updated terminology
✅ `CLAUDE.md` - Development commands and architecture

### Needs Update

⚠️ `README.md` - Update with experiential memory overview
⚠️ `Docs/SoP_Program_Quickstart.md` - Update phase E terminology
⚠️ Script docstrings - Remove old MemGPT references

---

## Dependencies & Environment

### Required

- Python 3.12+
- Poetry for dependency management
- ArangoDB 3.12+ with Unix socket support
- CUDA-capable GPU for embeddings/model (2x44GiB for Qwen3)

### Key Python Packages

- `httpx[http2]` - ArangoDB HTTP/2 client
- `transformers` - Qwen3 model loading
- `accelerate` - Multi-GPU distribution
- `tree-sitter-languages` - Code parsing
- `pydantic` - Type-safe models
- `numpy` - Vector operations

### Optional

- `docling` - PDF extraction
- `autoawq` - Qwen3 quantization
- Phoenix/Arize clients - Telemetry export

---

## Quick Reference

### Run Ingestion

```bash
poetry run python scripts/ingest_repo_workflow.py --root . --write --device cuda --batch 16
```

### Test Memory Operations

```python
from core.runtime.memory import MemoryStore

store = MemoryStore.from_env()
store.ensure_schema()

# Store observation
key = store.store_observation(
    content="Fixed bug in PathRAG traversal",
    tags=["debugging", "pathrag"],
    memory_type="observation"
)

# Retrieve with PathRAG
results = store.retrieve_memories(
    query="pathrag bugs",
    n_results=5
)
```

### Check Database

```bash
# Via Python
poetry run python -c "from core.database.arango.memory_client import *; \
  c = ArangoMemoryClient(resolve_memory_config()); \
  print(c.execute_query('FOR c IN COLLECTIONS() RETURN {name: c.name, count: COUNT(c)}'))"
```

### Run Tests

```bash
poetry run pytest tests/runtime/  # Runtime tests
poetry run pytest tests/smoke/    # Smoke tests
poetry run pytest --cov=core      # With coverage
```

---

## Contact & Resources

- **Program Plan:** Docs/Program_Plan.md
- **PathRAG PRD:** Docs/PRD_PathRAG_Integration.md
- **Schema Spec:** Docs/SCHEMA_Experiential_Memory.md
- **Inspiration:** <https://github.com/doobidoo/mcp-memory-service>
