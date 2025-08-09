# GitHub Processor Integration Plan

## Overview

Complete GitHub repository processing system for theory-practice bridge discovery, integrating with ArXiv papers to identify implementations of theoretical concepts.

## Core Objectives

1. **Extract executable knowledge** from code repositories
2. **Build symbol tables** for cross-language semantic search  
3. **Identify theory-practice bridges** between papers and implementations
4. **Track implementation evolution** across versions and forks
5. **Map dependency networks** to understand knowledge propagation

## System Architecture

### Data Flow

```
GitHub URL → Clone → Parse → Extract → Embed → Store → Query
                       ↓
                  Tree-sitter
                   Symbols
```

### Database Design (Unified Collection)

Following ArXiv v6.7 pattern - single polymorphic collection:

```python
# base_github collection (unified like base_arxiv)
{
    "_key": "repo_key or file_key or symbol_key",
    "uid": "github:owner/repo#path",
    "kind": "repo" | "file" | "symbol" | "chunk",
    "repository": "github",
    
    # Repo documents (kind='repo')
    "github_url": "...",
    "clone_status": "...",
    "default_branch": "...",
    "stars": 1234,
    "language_stats": {"python": 0.7, "javascript": 0.3},
    
    # File documents (kind='file')
    "repo_id": "owner_repo",
    "file_path": "src/main.py",
    "language": "python",
    "content_hash": "sha256:...",
    
    # Symbol documents (kind='symbol')
    "symbol_type": "function" | "class" | "import",
    "symbol_name": "calculate_pagerank",
    "signature": "(graph: Graph, alpha: float = 0.85)",
    "docstring": "...",
    "references": ["arxiv:1234.5678"],  # Papers mentioned
    
    # Chunk documents (kind='chunk')
    "chunk_type": "function_body" | "class_def" | "module_doc",
    "embedding": [...],  # 2048-dim Jina v4
    "ast_context": {...}  # Tree-sitter parse tree context
}
```

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1)

- [x] Basic repository cloning
- [ ] Fix Jina v4 integration (use correct API)
- [ ] Migrate to unified collection schema
- [ ] Add content-hash based caching
- [ ] Implement retry logic with exponential backoff

### Phase 2: Tree-sitter Integration (Week 1-2)

- [ ] Install tree-sitter with language parsers
- [ ] Create `TreeSitterSymbolExtractor` class
- [ ] Extract functions, classes, imports, exports
- [ ] Generate symbol-aware chunks (function boundaries)
- [ ] Calculate complexity metrics

### Phase 3: Symbol Table System (Week 2)

- [ ] Design symbol document schema
- [ ] Build cross-language symbol index
- [ ] Create symbol embeddings (name + signature + docstring)
- [ ] Implement fuzzy symbol matching
- [ ] Add call graph extraction

### Phase 4: Citation Extraction (Week 2-3)

- [ ] Extract paper references from comments/docstrings
- [ ] Find ArXiv IDs, DOIs, paper titles
- [ ] Extract algorithm names from comments
- [ ] Link to papers in database
- [ ] Build citation confidence scores

### Phase 5: Theory-Practice Bridge Discovery (Week 3)

- [ ] Create bridge detection queries
- [ ] Calculate conveyance gradients (paper → code)
- [ ] Identify implementation patterns
- [ ] Rank bridge strength
- [ ] Generate bridge reports

### Phase 6: Advanced Features (Week 4+)

- [ ] Version history tracking (git commits)
- [ ] Dependency graph analysis (imports)
- [ ] Fork relationship mapping
- [ ] Implementation evolution tracking
- [ ] Performance benchmarking

## Tree-sitter Symbol Extraction Design

### Supported Languages (Priority Order)

1. Python - Most ML/AI implementations
2. JavaScript/TypeScript - Web implementations  
3. C/C++ - System implementations (Word2Vec)
4. Java - Enterprise implementations
5. Go - Modern system implementations
6. Rust - Performance-critical implementations

### Symbol Extraction Queries

```python
# Universal queries across languages
FUNCTION_QUERY = """
(function_definition
  name: (identifier) @name
  parameters: (parameters) @params
  body: (block) @body)
"""

CLASS_QUERY = """
(class_definition
  name: (identifier) @name
  body: (block) @body)
"""

IMPORT_QUERY = """
(import_statement) @import
"""
```

### Symbol Embedding Strategy

```python
def create_symbol_embedding(symbol):
    """
    Create rich embedding for a symbol.
    """
    # Combine multiple signals
    text = f"{symbol.name} {symbol.signature} {symbol.docstring}"
    
    # Add context
    if symbol.references_papers:
        text += f" References: {' '.join(symbol.references_papers)}"
    
    # Generate embedding
    return embedder.embed_texts([text], task="code")
```

## Query Patterns for Bridge Discovery

### 1. Direct Implementation Search

```aql
FOR paper IN base_arxiv
  FILTER paper.kind == 'paper'
  FOR code IN base_github
    FILTER code.kind == 'symbol'
    FILTER code.symbol_type == 'function'
    LET similarity = COSINE_SIMILARITY(paper.embedding, code.embedding)
    FILTER similarity > 0.8
    RETURN {paper: paper.title, implementation: code.symbol_name, score: similarity}
```

### 2. Citation-Based Bridge

```aql
FOR code IN base_github
  FILTER code.kind == 'symbol'
  FILTER LENGTH(code.references) > 0
  FOR ref IN code.references
    FOR paper IN base_arxiv
      FILTER paper.arxiv_id == ref
      RETURN {paper: paper.title, code: code.uid, explicit_reference: true}
```

### 3. Conveyance Gradient

```python
def calculate_conveyance_gradient(paper_id, repo_id):
    """
    Measure how conveyance increases from paper to code.
    """
    paper = get_paper(paper_id)
    implementations = get_implementations(repo_id)
    
    # Paper has theory (low conveyance)
    paper_conveyance = estimate_conveyance(paper.abstract)
    
    # Code has execution (high conveyance)  
    code_conveyance = 0.95  # Code is directly executable
    
    # Gradient shows transformation
    return code_conveyance - paper_conveyance
```

## Testing Strategy

### Unit Tests

- [ ] Symbol extraction for each language
- [ ] Embedding generation
- [ ] Database operations
- [ ] Citation extraction

### Integration Tests  

- [ ] Clone → Parse → Store pipeline
- [ ] Cross-language symbol search
- [ ] Bridge discovery queries

### Test Repositories

1. **Word2Vec Ecosystem** (Our MVP)
   - `tmikolov/word2vec` - Original paper implementation
   - `RaRe-Technologies/gensim` - Production implementation
   - Multiple language ports

2. **PageRank Implementations**
   - NetworkX (Python)
   - SNAP (C++)
   - GraphX (Scala)

3. **Transformer Implementations**
   - Hugging Face Transformers
   - Original "Attention is All You Need" code
   - Multiple framework implementations

## Success Metrics

1. **Coverage**: Process 1000+ repositories
2. **Symbol Extraction**: 95% of functions/classes extracted
3. **Bridge Discovery**: Find 100+ paper-code bridges
4. **Performance**: Process 10 repos/minute
5. **Accuracy**: 90% precision in bridge detection

## Command-Line Interface

```bash
# Process single repository
python process_github.py https://github.com/tmikolov/word2vec

# Process with specific options
python process_github.py tmikolov/word2vec \
  --extract-symbols \
  --find-citations \
  --link-papers \
  --max-depth 3

# Batch process from list
python process_github.py --batch repos.txt

# Find bridges for specific paper
python find_bridges.py --arxiv-id 1301.3781  # Word2Vec paper

# Generate bridge report
python bridge_report.py --min-score 0.8 --output bridges.json
```

## Environment Variables

```bash
ARANGO_PASSWORD=xxx
GITHUB_TOKEN=xxx  # For API rate limits
CUDA_VISIBLE_DEVICES=0
MAX_REPO_SIZE_MB=500
TREE_SITTER_LANGUAGES=python,javascript,c,cpp,java,go,rust
```

## Next Steps

1. **Immediate**: Fix Jina v4 integration
2. **This Week**: Implement tree-sitter symbol extraction
3. **Next Week**: Build bridge discovery queries
4. **Testing**: Process Word2Vec ecosystem as proof of concept

---

*This plan integrates GitHub code processing with ArXiv papers to discover and quantify theory-practice bridges through symbol analysis and semantic embeddings.*
