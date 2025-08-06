# Theory-Practice Bridge Discovery System

## Formal Build Document

**Version**: 2.0.0  
**Date**: 2025-08-06  
**Status**: Specification  
**Purpose**: Data Infrastructure for Theory-Practice Analysis

---

## 1. Executive Summary

### 1.1 Purpose

Build a two-phase system: First, establish base data infrastructure for ArXiv papers and GitHub repositories. Second, create analytical layers for theory-practice bridge discovery and gap analysis in Information Reconstructionism theory.

### 1.2 Development Philosophy

**Phase 1 (Months 1-2)**: Base container infrastructure - Store raw data faithfully from sources without interpretation or analysis. Focus on reliable data acquisition and storage.

**Phase 2 (Months 3-4)**: Analytical containers - Apply theoretical frameworks, measurements, and experimental analysis on top of stable base data.

### 1.3 Success Metrics

**Phase 1 Success**:

- ✅ ArXiv base container operational (COMPLETE)
- ⬜ GitHub base containers operational
- ⬜ 100+ repositories processed and stored
- ⬜ MCP tools for data management functional

**Phase 2 Success**:

- ⬜ Analytical bridge discovery operational
- ⬜ 1000+ paper-code pairs analyzed
- ⬜ Theoretical gap identification system functional
- ⬜ Statistical validation framework implemented

---

## 2. Phase 1: Base Container Infrastructure

### 2.1 Current State

**ArXiv Base Container** ✅ COMPLETE

- Collection: `base_arxiv`
- Daily updates via cron
- On-demand PDF processing
- Full-text embeddings with Jina v3
- MCP tools operational

**GitHub Base Containers** ⬜ TO BUILD

- Collection: `base_github_repos` - Repository metadata
- Collection: `base_github_files` - File contents  
- Collection: `base_github_embeddings` - Code embeddings
- MCP tools needed

### 2.2 GitHub Base Container Schemas

**Collection: `base_github_repos`**

```javascript
{
  "_key": "owner_repo_name",              // e.g., "networkx_networkx"
  "github_url": "string",                 // Full GitHub URL
  "clone_url": "string",                  // Git clone URL
  "owner": "string",                      // Repository owner
  "name": "string",                       // Repository name
  "description": "string",                // Repository description
  "default_branch": "string",             // Main branch name
  "created_at": "datetime",               // Repository creation date
  "updated_at": "datetime",               // Last update date
  "pushed_at": "datetime",                // Last push date
  "size": "integer",                      // Repository size in KB
  "stargazers_count": "integer",          // GitHub stars
  "language": "string",                   // Primary language
  "languages": "object",                  // Language breakdown
  "topics": "array<string>",              // GitHub topics
  "license": "string",                    // License type
  "clone_status": "enum",                 // not_cloned|cloned|processed
  "local_path": "string",                 // Local storage path
  "last_commit_processed": "string",      // SHA of last processed commit
  "processing_date": "datetime"           // When processed
}
```

**Collection: `base_github_files`**

```javascript
{
  "_key": "owner_repo_filepath",          // Composite key
  "repo_key": "string",                   // FK to base_github_repos
  "file_path": "string",                  // Full file path in repo
  "file_name": "string",                  // File name only
  "file_extension": "string",             // File extension
  "language": "string",                   // Detected language
  "size": "integer",                      // File size in bytes
  "lines": "integer",                     // Line count
  "commit_sha": "string",                 // Commit SHA
  "content": "text",                      // Raw file content
  "encoding": "string",                   // File encoding
  "last_modified": "datetime",            // Last modification date
  "author": "string",                     // Last author
  "file_type": "enum",                    // source|test|doc|config|data|vendor
  "is_generated": "boolean",              // Auto-generated flag
  "is_binary": "boolean"                  // Binary file flag
}
```

**Collection: `base_github_ast`** (OPTIONAL - Phase 1.5)

*Note: AST processing is optional enhancement after basic infrastructure works*

```javascript
{
  "_key": "owner_repo_filepath",          // Matches file key
  "file_key": "string",                   // FK to base_github_files
  "repo_key": "string",                   // FK to base_github_repos
  "language": "string",                   // Programming language
  "parser_version": "string",             // Tree-sitter version
  "ast_summary": "object",                // Simplified AST data
  "full_ast": "text"                      // Complete AST if needed
}
```

**Collection: `base_github_embeddings`**

```javascript
{
  "_key": "owner_repo_filepath_chunk_n",  // Composite key with chunk
  "file_key": "string",                   // FK to base_github_files
  "repo_key": "string",                   // FK to base_github_repos
  "chunk_index": "integer",               // Chunk sequence number
  "chunk_type": "enum",                   // function|class|module|comment
  "start_line": "integer",                // Starting line number
  "end_line": "integer",                  // Ending line number
  "chunk_text": "text",                   // Actual code chunk
  "embedding": "array<float>",            // Jina v3 embedding (1024-dim)
  "embedding_model": "string",            // Model identifier
  "embedded_date": "datetime",            // When embedded
  "ast_context": "object",                // AST context for chunk
  "function_name": "string",              // If chunk is function
  "class_name": "string",                 // If chunk is in class
  "imports": "array<string>",             // Imports used in chunk
  "language_constructs": "array<string>"  // Language features used
}
```

---

## 3. Phase 1 Implementation (Base Containers)

### 3.1 GitHub Repository Processing Pipeline

```python
# processors/github/scripts/process_repo_on_demand.py

class OnDemandRepoProcessor:
    """
    Simple repo processor - clone, store, embed.
    No analysis, no theory, just data storage.
    """
    
    def __init__(self, db_host="192.168.1.69", db_name="academy_store"):
        self.db_host = db_host
        self.db_name = db_name
        self._init_database()
        self.embedder = JinaEmbedderV3()  # Reuse from ArXiv
    
    def process_repo(self, repo_url: str) -> Dict:
        """
        Basic processing: Clone → Store Files → Generate Embeddings
        """
        # 1. Clone repository
        owner, name = parse_github_url(repo_url)
        local_path = self.clone_repo(repo_url)
        
        # 2. Store repo metadata
        self.store_repo_metadata(owner, name, local_path)
        
        # 3. Process each file
        for file_path in self.walk_repo(local_path):
            if self.should_process(file_path):
                # Store file content
                self.store_file(owner, name, file_path)
                
                # Generate embeddings
                content = self.read_file(file_path)
                chunks = self.chunk_code(content)
                embeddings = self.embedder.embed(chunks)
                
                # Store embeddings
                self.store_embeddings(owner, name, file_path, embeddings)
        
        return {"status": "success", "repo": f"{owner}/{name}"}
```

### 3.2 MCP Tools for Base Container Management

```python
# mcp_server/tools/github_base_tools.py

@server.tool()
async def search_github_repos(query: str, limit: int = 10) -> List[Dict]:
    """Search GitHub for repositories."""
    # Use GitHub API to find repos
    return github_api.search_repositories(query, limit)

@server.tool()
async def process_github_repo(repo_url: str) -> Dict:
    """Clone and process a GitHub repository into base containers."""
    processor = OnDemandRepoProcessor()
    return processor.process_repo(repo_url)

@server.tool()
async def update_github_repo(repo_key: str) -> Dict:
    """Pull latest changes and update processed repo."""
    processor = OnDemandRepoProcessor()
    return processor.update_repo(repo_key)

@server.tool()
async def get_repo_files(repo_key: str, file_pattern: str = "*") -> List[Dict]:
    """Retrieve files from a processed repository."""
    return db.base_github_files.find({"repo_key": repo_key, "pattern": file_pattern})

@server.tool()
async def get_code_embeddings(repo_key: str, file_path: str = None) -> List[Dict]:
    """Retrieve embeddings for repository code."""
    filter = {"repo_key": repo_key}
    if file_path:
        filter["file_path"] = file_path
    return db.base_github_embeddings.find(filter)
```

---

## 4. Phase 2: Analytical Containers (FUTURE)

*Note: Phase 2 begins only after Phase 1 base infrastructure is operational*

### 4.1 Analytical Container Schemas

**Collection: `analytical_bridges`** (PHASE 2)

```javascript
{
  "_key": "paper_id_repo_key",            // Composite key
  "bridge_type": "enum",                  // explicit|discovered|hypothesized
  "paper": {
    "arxiv_id": "string",
    "title": "string",
    "has_pseudocode": "boolean",
    "has_equations": "boolean",
    "algorithm_descriptions": "array",
    "key_concepts": "array<string>",
    "embedding_summary": "object"
  },
  "implementation": {
    "repo_key": "string",
    "primary_files": "array<string>",
    "total_files": "integer",
    "primary_language": "string",
    "ast_patterns": "object"
  },
  "measurements": {
    "semantic_similarity": "float",       // Embedding similarity
    "structural_similarity": "float",     // AST pattern match
    "nomenclature_preservation": "float", // Name similarity
    "complexity_delta": "float",          // Added complexity
    "temporal_drift": "float"             // Evolution over time
  },
  "conveyance_score": "float",            // Overall conveyance
  "confidence": "float",                  // Measurement confidence
  "validated": "boolean",                 // Human validated
  "validation_notes": "text"              // Human annotations
}
```

**Collection: `experimental_gaps`** (PHASE 2)

```javascript
{
  "_key": "gap_id",                       // Unique gap identifier
  "gap_type": "enum",                     // Type of theoretical gap
  "evidence": "object",                   // Supporting evidence
  "hypothesis": "text",                   // Gap hypothesis
  "frequency": "integer"                  // Observation count
}
```

### 4.2 Phase 2 Processing (FUTURE)

*Theory-practice bridge discovery and gap analysis will be implemented after base infrastructure is stable.*

---

## 5. Implementation Details

### 5.1 Phase 1: Repository Processing with Error Handling

```python
def process_repository_with_retry(repo_url: str) -> Dict:
    """
    Phase 1: Repository processing with error handling and rate limiting.
    """
    import time
    from tenacity import retry, stop_after_attempt, wait_exponential
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    def clone_with_retry(url: str) -> Path:
        """Clone repository with retry logic."""
        try:
            return clone_repo(url)
        except RateLimitError:
            time.sleep(60)  # GitHub rate limit cooldown
            raise
        except NetworkError as e:
            logger.warning(f"Network error: {e}, retrying...")
            raise
    
    try:
        # Step 1: Clone repository with retry
        local_path = clone_with_retry(repo_url)
        
        # Step 2: Store repository metadata
        repo_metadata = extract_repo_metadata(local_path)
        store_repo_metadata(repo_metadata)
        
        # Step 3: Process code files with error recovery
        files_processed = 0
        files_failed = []
        
        for file_path in walk_directory(local_path):
            if not should_process_file(file_path):
                continue
            
            try:
                # Read file content with encoding detection
                content = read_file_safe(file_path)
                
                # Store in database
                store_file_content(file_path, content)
                
                # Semantic-aware chunking for Jina v4's 32k context
                chunks = chunk_code_semantic(content, max_tokens=28000)
                
                # Late chunking with Jina v4 (32k context)
                if chunks:
                    # Process full file context first, then chunk
                    embeddings = jina_v4_late_chunk_embed(content, chunks)
                    store_embeddings_batch(file_path, chunks, embeddings)
                
                files_processed += 1
                
            except UnicodeDecodeError:
                logger.warning(f"Skipping binary/unreadable file: {file_path}")
                files_failed.append(str(file_path))
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                files_failed.append(str(file_path))
        
        return {
            "status": "success",
            "files_processed": files_processed,
            "files_failed": len(files_failed),
            "failed_files": files_failed[:10]  # First 10 for debugging
        }
        
    except Exception as e:
        logger.error(f"Repository processing failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }

def read_file_safe(file_path: Path) -> str:
    """Read file with multiple encoding attempts."""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'ascii']
    
    for encoding in encodings:
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    
    # If all encodings fail, try with error handling
    return file_path.read_text(encoding='utf-8', errors='ignore')
```

### 5.2 File Filtering

```python
def should_process_file(file_path: str) -> bool:
    """
    Determine if file should be processed.
    Phase 1: Focus on source code files only.
    """
    # Comprehensive skip directories list
    skip_dirs = [
        'node_modules', '.git', 'vendor', 'dist', 'build', 'target',
        '__pycache__', '.pytest_cache', '.tox', 'venv', 'env', '.env',
        'site-packages', 'bower_components', '.next', '.nuxt', 'out',
        'bin', 'obj', '.gradle', '.idea', '.vscode', 'coverage',
        'htmlcov', '.eggs', '*.egg-info', 'wheels', 'lib', 'lib64'
    ]
    
    # Check if path contains any skip directory
    path_parts = file_path.split('/')
    if any(skip_dir in path_parts for skip_dir in skip_dirs):
        return False
    
    # Process common code extensions
    code_extensions = [
        '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h',
        '.go', '.rs', '.rb', '.php', '.cs', '.swift', '.kt', '.scala',
        '.r', '.m', '.sql', '.sh', '.bash', '.ps1', '.yaml', '.yml'
    ]
    return any(file_path.endswith(ext) for ext in code_extensions)
```

### 5.3 Late Chunking with Jina v4

**CRITICAL: We use Jina v4 with late chunking, NOT v3**

```python
def jina_v4_late_chunk_embed(full_content: str, chunks: List[str]) -> List[np.ndarray]:
    """
    Jina v4 late chunking implementation.
    Process full document context (up to 32k tokens) BEFORE chunking.
    """
    from transformers import AutoModel, AutoTokenizer
    import torch
    
    model = AutoModel.from_pretrained('jinaai/jina-embeddings-v4', trust_remote_code=True)
    
    # Process full document through transformer first
    with torch.no_grad():
        # Get token-level embeddings for full context (up to 32k)
        token_embeddings = model.encode(
            full_content,
            output_value='token_embeddings',  # Get token-level outputs
            max_length=32768
        )
        
        # Now apply chunking with full context awareness
        chunk_embeddings = []
        for chunk_start, chunk_end in chunk_boundaries:
            # Each chunk embedding contains surrounding context
            chunk_emb = token_embeddings[chunk_start:chunk_end].mean(dim=0)
            chunk_embeddings.append(chunk_emb)
    
    return chunk_embeddings
```

Key advantages over traditional chunking:
- **Preserves long-range dependencies** across function calls
- **Maintains import context** throughout file
- **4x larger context window** (32k vs 8k tokens)
- **Code-specific LoRA adapter** for better code understanding

### 5.4 Semantic-Aware Code Chunking

```python
def chunk_code_semantic(content: str, max_tokens: int = 28000) -> List[str]:
    """
    Phase 1: Semantic-aware chunking without full AST.
    Uses simple heuristics to preserve code structure.
    """
    import tiktoken
    
    # Initialize tokenizer for accurate token counting
    encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(text: str) -> int:
        """Accurate token counting using tiktoken."""
        return len(encoding.encode(text))
    
    chunks = []
    lines = content.split('\n')
    current_chunk = []
    current_tokens = 0
    
    # Track nesting level for better chunking
    indent_stack = []
    
    for i, line in enumerate(lines):
        line_tokens = count_tokens(line)
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        
        # Detect function/class boundaries (Python example, extend for other languages)
        is_definition = stripped.startswith(('def ', 'class ', 'async def '))
        is_block_end = indent < indent_stack[-1] if indent_stack else False
        
        # Check if we should start a new chunk
        should_split = False
        
        if current_tokens + line_tokens > max_tokens:
            # Must split due to size
            should_split = True
        elif is_definition and current_tokens > max_tokens // 2:
            # Start new chunk at function/class boundary if chunk is reasonably large
            should_split = True
        elif is_block_end and current_tokens > max_tokens * 0.7:
            # Split at block end if chunk is getting large
            should_split = True
        
        if should_split and current_chunk:
            chunks.append('\n'.join(current_chunk))
            current_chunk = []
            current_tokens = 0
            indent_stack = []
        
        # Add line to current chunk
        current_chunk.append(line)
        current_tokens += line_tokens
        
        # Update indent tracking
        if is_definition:
            indent_stack.append(indent)
        elif is_block_end and indent_stack:
            # Pop indent levels that ended
            while indent_stack and indent < indent_stack[-1]:
                indent_stack.pop()
    
    # Add remaining
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks
```

---

## 6. Phase 2: Analytical Processing (FUTURE)

*Phase 2 will implement theory-practice bridge discovery, conveyance measurement, and gap analysis. This work begins only after Phase 1 base infrastructure is stable and tested.*

### 6.1 Identified Deficiencies (To Address in Phase 2)

**Database Architecture**:
- [ ] Graph edge definitions for paper-code relationships
- [ ] Foreign key constraints between collections
- [ ] Migration path from Phase 1 to Phase 2 schemas
- [ ] Backward compatibility guarantees

**Measurement Framework**:
- [ ] Statistical validation methodology undefined
- [ ] Conveyance calculation needs mathematical rigor
- [ ] Null hypothesis testing framework missing
- [ ] Temporal evolution tracking unspecified

**Gap Discovery**:
- [ ] Failure logging system design incomplete
- [ ] Gap detection algorithms need specification
- [ ] Hypothesis generation methodology unclear
- [ ] Cross-domain analysis framework missing

**Integration Issues**:
- [ ] Phase transition procedures unspecified
- [ ] Data migration between phases undefined
- [ ] Analytical container dependencies unclear
- [ ] Performance impact of analysis unknown

### 6.2 Phase 2 Core Components (To Be Detailed)

- Bridge discovery algorithms
- Conveyance measurement (multiplicative model)
- Gap detection and logging
- Statistical validation framework
- Temporal evolution tracking

---

## 7. Testing Strategy

### 7.1 Phase 1 Tests (Base Infrastructure)

```python
# tests/phase1/test_repo_processing.py
def test_repo_cloning():
    """Test repository cloning and storage."""
    
def test_file_extraction():
    """Test code file identification and extraction."""

def test_embedding_generation():
    """Test Jina embedding generation for code."""
    
def test_database_storage():
    """Test storage in base containers."""

# tests/phase1/test_mcp_tools.py
def test_search_repos():
    """Test GitHub repository search."""
    
def test_process_repo():
    """Test end-to-end repo processing."""
```

### 7.2 Phase 2 Tests (Future - Analytical)

```python
# tests/phase2/test_bridge_discovery.py
def test_paper_code_matching():
    """Test finding code for papers."""

# tests/phase2/test_conveyance.py
def test_conveyance_measurement():
    """Test theory-practice conveyance calculation."""
```

---

## 7. Phase 1 Completion Criteria

### 7.1 Technical Requirements
- [ ] All three GitHub collections created with proper indexes
- [ ] Successfully process 100+ diverse repositories
- [ ] Embedding generation success rate > 95%
- [ ] Average processing time < 3 minutes per small repo (<10MB)
- [ ] Error recovery for network failures, rate limits
- [ ] Comprehensive logging and monitoring

### 7.2 Quality Gates
- [ ] Unit test coverage > 80% for core functions
- [ ] Integration tests pass for end-to-end flow
- [ ] Documentation complete for all MCP tools
- [ ] Error handling for all external dependencies
- [ ] Performance benchmarks established

### 7.3 Data Validation
- [ ] Verify embeddings preserve semantic meaning
- [ ] Confirm chunk boundaries respect code structure
- [ ] Validate foreign key relationships in database
- [ ] Ensure no data loss during processing

---

## 8. Deployment Plan

### 8.1 Phase 1: Base Infrastructure (Weeks 1-4)

#### Week 1: GitHub Container Setup

- [ ] Create `base_github_repos` collection in ArangoDB
- [ ] Create `base_github_files` collection
- [ ] Create `base_github_embeddings` collection
- [ ] Set up database indexes

#### Week 2: Repository Processing

- [ ] Implement repository cloning with git
- [ ] Build file extraction and filtering
- [ ] Create simple code chunking (no AST)
- [ ] Test on 5 small repositories

#### Week 3: Embedding Pipeline

- [ ] Integrate Jina v3 for code embeddings
- [ ] Implement batch processing for efficiency
- [ ] Add progress tracking and logging
- [ ] Test on 10 medium repositories

#### Week 4: MCP Integration

- [ ] Implement `search_github_repos` tool
- [ ] Implement `process_github_repo` tool
- [ ] Implement `get_repo_files` tool
- [ ] Implement `get_code_embeddings` tool
- [ ] End-to-end testing with 25+ repositories

### 8.2 Phase 2: Analytical Infrastructure (Weeks 5-8)

#### Week 5: Bridge Discovery

- [ ] Design analytical container schemas
- [ ] Implement basic similarity search
- [ ] Create paper-to-code matching logic

#### Week 6: Conveyance Measurement

- [ ] Implement multiplicative conveyance model
- [ ] Add dimensional measurements
- [ ] Create measurement validation

#### Week 7: Gap Discovery

- [ ] Build failure logging system
- [ ] Implement gap detection algorithms
- [ ] Create hypothesis generation

#### Week 8: Experimental Framework

- [ ] Statistical validation setup
- [ ] Batch experiment runner
- [ ] Results analysis tools

---

## 8. Resource Requirements

### 8.1 Computational Resources

- **GPU**: 2x NVIDIA A6000 (48GB each) for embeddings
- **CPU**: 32+ cores for parallel AST processing
- **RAM**: 128GB for large repository processing
- **Storage**: 10TB for repository clones and embeddings

### 8.2 External Services

- **GitHub API**: Rate-limited access for repository metadata
- **ArXiv API**: Paper metadata and PDF access
- **Tree-sitter**: Language parsers (npm packages)

### 8.3 Time Estimates (Updated for Phase 1)

**Phase 1 Processing**:
- **Small Repository (<10MB)**: ~2-3 minutes total
  - Cloning: 10-30 seconds
  - File processing: 30-60 seconds
  - Embedding generation: 60-90 seconds
  
- **Medium Repository (10-50MB)**: ~5-8 minutes total
  - Cloning: 30-60 seconds
  - File processing: 2-3 minutes
  - Embedding generation: 3-4 minutes

- **Large Repository (>50MB)**: ~10-20 minutes total
  - Cloning: 1-2 minutes
  - File processing: 4-6 minutes
  - Embedding generation: 5-12 minutes

**Phase 2 Analysis** (Future estimates):
- **AST Extraction**: ~1 second per file
- **Bridge Analysis**: ~30 seconds per paper-repo pair
- **Conveyance Calculation**: ~5 seconds per bridge

---

## 9. Risk Analysis

### 9.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| GitHub rate limiting | High | Medium | Implement caching, use authenticated requests |
| Large repository processing | Medium | High | Set size limits, sample large repos |
| AST parser failures | Medium | Medium | Fallback to text-only processing |
| Embedding model limitations | Low | High | Test multiple embedding models |

### 9.2 Experimental Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Insufficient gap discovery | Low | High | Expand search space, vary domains |
| False positive gaps | Medium | Medium | Human validation of discoveries |
| Measurement instability | Medium | Medium | Multiple measurements, statistical validation |

---

## 10. Success Criteria

### 10.1 Quantitative Metrics

- Process ≥1000 repositories
- Analyze ≥1000 paper-code pairs
- Discover ≥5 theoretical gaps
- Achieve p < 0.05 for gap significance
- Measurement stability > 0.9 (test-retest)

### 10.2 Qualitative Outcomes

- Identify new dimensions for Information Reconstructionism
- Validate conveyance as measurable phenomenon
- Generate hypotheses for theory revision
- Produce publishable experimental results

---

## 11. Documentation Requirements

### 11.1 Code Documentation

- Docstrings following Google style guide
- Theory connections in every module
- AST pattern documentation
- Gap discovery explanations

### 11.2 Experimental Documentation

- Lab notebook for all experiments
- Failed experiment analysis
- Gap evidence compilation
- Statistical analysis reports

### 11.3 User Documentation

- MCP tool usage guide
- Bridge discovery tutorials
- Gap interpretation guide
- Troubleshooting guide

---

## 12. Maintenance Plan

### 12.1 Regular Updates

- Weekly: Process new ArXiv papers
- Monthly: Update popular repositories
- Quarterly: Retrain embedding models
- Annually: Theory revision based on gaps

### 12.2 Monitoring

- Track processing failures
- Monitor measurement stability
- Log gap discoveries
- Alert on anomalies

---

## Appendix A: File Structure

```
/home/todd/olympus/HADES/processors/
├── github/
│   ├── scripts/
│   │   ├── process_repo_on_demand.py
│   │   ├── clone_repository.py
│   │   ├── update_repository.py
│   │   └── search_github.py
│   ├── core/
│   │   ├── ast_processor.py
│   │   ├── code_chunker.py
│   │   ├── language_detector.py
│   │   └── repo_walker.py
│   └── tests/
│       ├── test_ast_processor.py
│       └── test_code_chunker.py
│
├── bridges/
│   ├── scripts/
│   │   ├── find_theory_practice_bridges.py
│   │   ├── measure_conveyance.py
│   │   ├── discover_gaps.py
│   │   └── run_experiments.py
│   ├── core/
│   │   ├── gap_discovery.py
│   │   ├── conveyance_calculator.py
│   │   ├── semantic_matcher.py
│   │   ├── ast_comparer.py
│   │   └── temporal_analyzer.py
│   ├── experiments/
│   │   ├── null_hypothesis.py
│   │   ├── divergence_detection.py
│   │   ├── domain_analysis.py
│   │   └── boundary_conditions.py
│   └── tests/
│       ├── test_conveyance.py
│       ├── test_gap_discovery.py
│       └── test_experiments.py
│
└── mcp_server/
    └── tools/
        ├── github_tools.py
        ├── bridge_tools.py
        └── experiment_tools.py
```

## Appendix B: Database Indexes

```javascript
// Optimize for common queries
db.base_github_repos.ensureIndex({ "language": 1 })
db.base_github_repos.ensureIndex({ "topics": 1 })
db.base_github_repos.ensureIndex({ "stargazers_count": -1 })

db.base_github_files.ensureIndex({ "repo_key": 1 })
db.base_github_files.ensureIndex({ "language": 1 })
db.base_github_files.ensureIndex({ "file_type": 1 })

db.base_github_embeddings.ensureIndex({ "repo_key": 1 })
db.base_github_embeddings.ensureIndex({ "file_key": 1 })

db.analytical_bridges.ensureIndex({ "paper.arxiv_id": 1 })
db.analytical_bridges.ensureIndex({ "implementation.repo_key": 1 })
db.analytical_bridges.ensureIndex({ "conveyance_score": -1 })

db.experimental_gaps.ensureIndex({ "gap_type": 1 })
db.experimental_gaps.ensureIndex({ "frequency": -1 })
```

---

**END OF BUILD DOCUMENT**

*This document serves as the authoritative specification for implementing the Theory-Practice Bridge Discovery System. All implementation decisions should reference this document.*
