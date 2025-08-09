# HADES Processor Pipeline - Phase Status

## Current Status: Phase 1 Complete, Preparing for Phase 1.5

### Document Version

- **Date**: 2025-01-20
- **Pipeline Version**: 1.0
- **Architecture**: Per-relation edge collections with delta updates

---

## Phase Overview

### ✅ Phase 1: Base Data Import (COMPLETE)

**Status**: Core processors operational for ArXiv and GitHub

#### Completed Components

##### ArXiv Processor (v7.0)

- **Location**: `processors/arxiv/scripts/process_pdf_with_citations_v7.0.py`
- **Collections**:
  - `base_arxiv` (papers/vertices)
  - `base_arxiv_chunks` (chunks with embeddings)
- **Features**:
  - Docling v2 PDF processing
  - Jina v4 embeddings with late chunking
  - Citation extraction (numbered, arxiv, DOI)
  - Delta edge updates for citations
  - Content-hash deduplication
- **Production Ready**: YES

##### GitHub Processor (MVP)

- **Location**: `processors/github/`
- **Collections**: `base_github`
- **Status**: MVP complete, needs integration testing
- **Pending**: Performance optimization, batch processing

##### Web Processor (Experimental)

- **Location**: `processors/web/`
- **Status**: Experimental - not needed for initial deployment
- **Decision**: Keep simple with just ArXiv and GitHub for now
- **Future**: Add web collections as needed (documentation sites, blogs)

---

### 🚧 Phase 1.5: Edge Materialization (NEXT)

**Status**: Ready to implement

#### Edge Collections to Create

```yaml
edges_cites         # Paper → Paper citations (partially done)
edges_mentions      # Doc → Paper/Repo lightweight references
edges_implements    # Repo → Paper implementations
edges_depends_on    # Repo → Repo/Package dependencies
edges_contains      # Paper → Chunk containment (done)
edges_equivalent_to # Entity ↔ Entity deduplication
```

#### Implementation Approach

1. **Delta Updates**: Only modify changed edges (implemented in v7.0)
2. **Soft Deletes**: Mark edges inactive instead of deleting
3. **Deterministic Keys**: 32-char hashes for idempotency
4. **PRUNE Support**: Structure for efficient traversals

#### Scripts Needed

- [ ] `phase_1_5_materialize_edges.py` - Main edge creation script
- [ ] `edge_deduplication.py` - Find and mark equivalent entities
- [ ] `edge_validation.py` - Verify edge consistency

---

### 📋 Phase 2: Analytical Graphs (PLANNED)

**Status**: Design complete, awaiting Phase 1.5

#### Named Graphs to Create

```yaml
g_core_knowledge    # Main graph over all collections
g_arxiv_citations   # Citation network analysis
g_implementation    # Theory to practice bridges
```

#### Analysis Types

- Theory-practice bridge discovery
- Citation network analysis
- Implementation coverage metrics
- Knowledge evolution tracking

---

### 🔮 Phase 3: HADES MCP Server (PLANNED)

**Status**: Specification needed

#### Core Functions

- Semantic search across all collections
- Graph traversal queries
- Theory-practice bridge discovery
- Progressive document processing
- Real-time edge updates

#### MCP Tools to Expose

```python
# Search tools
semantic_search(query: str, collections: List[str])
find_implementations(arxiv_id: str)
find_citations(paper_id: str)

# Processing tools
process_arxiv_pdf(arxiv_id: str)
process_github_repo(repo_url: str)
update_edges(entity_id: str)

# Analysis tools
find_bridges(source_type: str, target_type: str)
analyze_citation_network(paper_id: str, depth: int)
calculate_impact_metrics(entity_id: str)
```

---

## Directory Structure (Current)

```
processors/
├── PHASE_STATUS.md                    # This file
├── arxiv/
│   ├── core/
│   │   ├── enhanced_docling_processor_v2.py
│   │   └── jina_v4_embedder.py
│   └── scripts/
│       ├── process_pdf_with_citations_v7.0.py  # PRODUCTION
│       └── process_pdf_with_citations_v6.7.py  # DEPRECATED
├── github/
│   ├── README.md
│   └── [implementation files]
└── web/                               # EXPERIMENTAL
    ├── PHASE_ARCHITECTURE_IMPLEMENTATION.md
    ├── PHASE_ARCHITECTURE_IMPLEMENTATION_V2.md
    └── [experimental scripts]
```

---

## Database Schema (Final)

### Vertex Collections

```javascript
base_arxiv          # ArXiv papers
base_arxiv_chunks   # Paper chunks with embeddings
base_github         # GitHub repositories
base_github_files   # Repository files (future)
```

### Edge Collections (Per-Relation)

```javascript
edges_cites         # Academic citations
edges_mentions      # Lightweight references
edges_implements    # Code implementations
edges_depends_on    # Dependencies
edges_contains      # Parent-child containment
edges_equivalent_to # Deduplication links
```

### Indexes Strategy

- NO redundant `_from`/`_to` indexes (automatic)
- Indexes on: `relation`, `source_system`, `confidence`, `active`
- Vector indexes on embedding fields (when available)
- Composite indexes for versioning

---

## Migration Checklist

### Before Phase 1.5

- [x] Finalize ArXiv processor (v7.0)
- [x] Document edge architecture
- [x] Implement delta updates
- [ ] Clean up experimental code
- [ ] Archive deprecated versions
- [ ] Update README files

### Phase 1.5 Tasks

- [ ] Create edge materialization script
- [ ] Process existing ArXiv data for edges
- [ ] Create GitHub → ArXiv implementation edges
- [ ] Build deduplication edges
- [ ] Validate edge consistency

### Before Phase 2

- [ ] Performance benchmarks
- [ ] Edge traversal optimization
- [ ] Create named graphs
- [ ] Document query patterns

### Before Phase 3

- [ ] Define MCP server API
- [ ] Implement search functions
- [ ] Create bridge discovery algorithms
- [ ] Setup progressive processing

---

## Key Design Decisions

1. **Simplify Sources**: Focus on ArXiv + GitHub, add web as needed
2. **Per-Relation Edges**: Better performance than single edge collection
3. **Delta Updates**: Only modify changed edges (efficiency)
4. **Soft Deletes**: Maintain history with active/inactive flags
5. **32-char Keys**: Prevent collisions at scale
6. **Late Chunking**: Better context preservation in embeddings

---

## Performance Targets

- Edge creation: < 100ms per paper
- Traversal: < 100ms for 3-hop queries
- Search: < 500ms for semantic queries
- Processing: ~30 seconds per PDF

---

## Next Immediate Steps

1. **Clean up `processors/` directory**
   - Archive old versions to `Acheron/`
   - Remove experimental web scripts
   - Update all README files

2. **Create Phase 1.5 scripts**
   - Main edge materialization
   - Validation tools
   - Performance tests

3. **Document for handoff**
   - Update CLAUDE.md files
   - Create operation guides
   - Document known issues

---

## Contact & Resources

- **ArXiv Processor**: See `processors/arxiv/README.md`
- **GitHub Processor**: See `processors/github/README.md`
- **Architecture**: See `PHASE_ARCHITECTURE_IMPLEMENTATION_V2.md`
- **Theory**: See main HADES README.md
