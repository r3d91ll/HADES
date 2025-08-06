# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Information Reconstructionism is a theoretical framework and infrastructure for validating that information exists as a multiplicative function of dimensional prerequisites, manifesting only through observer interaction.

### Core Equation

```
Information(i→j|S-O) = WHERE × WHAT × CONVEYANCE × TIME × FRAME
```

If ANY dimension = 0, then Information = 0

## Project Structure

### 📚 Paper (`paper/`)

Theoretical framework, mathematical proofs, and academic materials.

- Core theory and mathematical validation
- Wolfram scripts for zero propagation and context amplification
- See `paper/PAPER_CLAUDE.md` for theory-specific guidance

### 🏗️ Infrastructure (`infrastructure/`)

Scalable document processing system for validating the theory.

- 3-collection architecture (metadata, documents, chunks) with atomic transactions
- GPU-accelerated Jina v4 embeddings (2048-dimensional) with late chunking
- ArangoDB for graph storage and similarity queries
- See `infrastructure/INFRA_CLAUDE.md` for implementation details

### 🌐 MCP Server (`mcp_server/`)

Model Context Protocol server for progressive document processing.

- Provides async tools for arXiv paper processing and search
- Supports lazy-loading PDFs: download and process on-demand
- Hybrid system: abstracts for quick search, full-text for deep analysis
- See `mcp_server/CLAUDE.md` for setup with Claude Code

### 🧪 Experiments (`experiments/`)

Hypothesis validation using the infrastructure.

- Experiment 1: Multiplicative model validation
- Experiment 2: GPU-accelerated large-scale processing

### 🌉 Theory-Practice Bridge Processors (`processors/`)

Production infrastructure for discovering and analyzing theory-practice connections.

#### ArXiv Processor (`processors/arxiv/`)
- **Status**: Database rebuild in progress (45.93% complete as of Jan 20, 2025)
- Processing 2.79M papers with dual-GPU Jina v4 embeddings
- Expected completion: Jan 21, 2025 (~15 hours remaining)
- Full documentation in `processors/arxiv/README.md`

#### GitHub Processor (`processors/github/`)
- **Status**: MVP complete, awaiting integration testing
- Clones repositories and generates code embeddings with Jina v4
- Test coverage: ~60-70% unit tests, 0% integration tests
- **Pending Work**:
  - Integration testing with real ArangoDB after ArXiv rebuild
  - Performance benchmarking with GPU acceleration
  - Error recovery and retry mechanisms
  - Batch processing optimizations
- See `processors/github/README.md` for architecture details

#### Web Scraper (`processors/bridges/`)
- **Status**: Design phase - architecture planning underway
- Will extract content from documentation sites, blogs, tutorials
- Focus on technical documentation and implementation guides
- Planned features:
  - Intelligent content extraction with readability algorithms
  - Structured data preservation (code blocks, diagrams)
  - Metadata extraction (authors, dates, tags)
  - Rate limiting and respectful crawling

## Progressive Document Processing Strategy

### Phase 1: Abstract-Only (Current State)

- Metadata and abstracts already processed
- Quick semantic search across ~2M papers
- Lightweight, fast responses

### Phase 2: On-Demand PDF Enhancement (Next Step)

- User requests deeper analysis of specific papers
- System downloads PDFs from arXiv as needed
- Extracts full text, generates embeddings
- Updates database with full-text capabilities

### Phase 3: Intelligent Caching

- Frequently accessed papers kept locally
- Popular papers pre-processed during idle time
- Balance between storage costs and query performance

## Common Development Commands

### Setting Up Environment

```bash
cd infrastructure/
./setup_venv.sh
source venv/bin/activate
```

### MCP Server Setup

```bash
# Install MCP dependencies
pip install -r mcp_server/requirements.txt

# Add to Claude Code (project scope)
claude mcp add hades-arxiv python /home/todd/olympus/HADES/mcp_server/launch.py \
  -e ARANGO_PASSWORD="your-password"

# Start server manually for testing
python mcp_server/launch.py
```

### Installing Dependencies

```bash
# Core infrastructure
pip install -r setup/requirements.txt

# Install package in development mode
pip install -e .

# With dev dependencies (testing/linting)
pip install -e ".[dev]"
```

### Processing Documents (3-Collection Pipeline)

```bash
# Test run with 10 documents
python setup/process_documents_three_collections.py --count 10 --clean-start

# Production run
python setup/process_documents_three_collections.py \
    --count 1000 \
    --source-dir /mnt/data/arxiv_data/pdf \
    --db-name irec_production
```

### Running Tests

```bash
# All tests
pytest tests/

# With coverage
pytest --cov=irec_infrastructure tests/

# Infrastructure components
python tests/test_infrastructure.py
```

### Code Quality

```bash
# Format code
black irec_infrastructure/

# Type checking
mypy irec_infrastructure/
```

### Running Experiments

```bash
# Experiment 1: Multiplicative model
cd experiments/experiment_1
python run_experiment.py

# Experiment 2: GPU pipeline
cd experiments/experiment_2
./pipeline/launch_gpu_pipeline.sh
```

## High-Level Architecture

### Three Essential Dimensions

**WHERE**: Spatial location and accessibility

- File paths, permissions, graph proximity
- If WHERE = 0: Cannot locate information (e.g., missing file)

**WHAT**: Semantic content and understanding  

- Topics, concepts, embeddings
- If WHAT = 0: No semantic understanding (e.g., encrypted)

**CONVEYANCE**: Actionability and implementability

- Theory to practice translation potential
- If CONVEYANCE = 0: Cannot act upon information (e.g., pure philosophy)

### 3-Collection Database Architecture

The infrastructure uses atomic transactions to maintain consistency:

1. **metadata**: Pure arXiv metadata with PDF tracking
   - Basic info: title, authors, categories, abstract
   - PDF status: `pdf_status` ('not_downloaded'|'downloaded'|'processed')
   - Storage: `pdf_local_path` (if downloaded), `pdf_url` (arXiv link)
   - Processing: `full_text_processed`, `processing_date`

2. **documents**: Full Docling-extracted markdown with structure
   - Complete PDF text extraction
   - Document structure preservation
   - Linked to metadata via `arxiv_id`

3. **chunks**: Semantic chunks with Jina v4 embeddings
   - Both abstract and full-text chunks
   - `chunk_type`: 'abstract'|'full_text'
   - Page numbers, sections, positions
   - Enables granular semantic search

All three collections are updated atomically or rolled back together, preventing partial states.

### Key Design Principles

1. **Atomic Transactions**: All-or-nothing storage prevents partial states
2. **Pre-computed Infrastructure**: Heavy computation done once and stored
3. **GPU Optimization**: Dual A6000s with NVLink for batch processing
4. **Graph-Based Storage**: ArangoDB enables efficient similarity queries
5. **Predictable IDs**: Chunks use pattern `{arxiv_id}_chunk_{index}`

## Environment Variables

Create `.env` file:

```bash
# Database
ARANGO_HOST=localhost
ARANGO_PORT=8529
ARANGO_USERNAME=root
ARANGO_PASSWORD=your_password
ARANGO_DATABASE=information_reconstructionism

# GPU Settings
USE_GPU=true
GPU_DEVICES=0,1

# Processing
BATCH_SIZE=32
CHUNK_SIZE=1024
```

## Key Hypotheses Being Tested

### H1: Context Amplification

Context functions as exponential amplifier (Context^α) where α ≈ 1.5-2.0

### H2: Multiplicative Dependency

Information requires ALL dimensions non-zero

### H3: Theory-Practice Bridge Discovery

Strong bridges exhibit specific dimensional patterns (WHERE × WHAT × CONVEYANCE > threshold)

### H4: Observer Effect

Different observers (S-Observer vs A-Observer) create different information realities

## PDF Processing Workflow (Under Development)

### Current Tools (Abstract-Based)

- `process_arxiv_batch`: Process metadata and abstracts
- `semantic_search`: Search across abstract embeddings
- `check_job_status`: Monitor batch processing

### Planned PDF Tools (To Be Implemented)

- `check_pdf_status`: Check if PDF is local/processed
- `download_pdf`: Fetch from arXiv, store locally
- `process_pdf_full_text`: Extract, chunk, and embed full text
- `search_within_document`: Semantic search in specific paper
- `enhance_paper_with_full_text`: Complete PDF processing workflow
- `batch_enhance_papers`: Process multiple PDFs

### Database Fields for PDF Tracking

```python
# In metadata collection
{
    "arxiv_id": "2310.08560",
    "pdf_status": "not_downloaded",  # or "downloaded", "processed"
    "pdf_local_path": null,          # Path when downloaded
    "pdf_url": "https://arxiv.org/pdf/2310.08560.pdf",
    "full_text_processed": false,
    "processing_date": null
}

# In chunks collection
{
    "chunk_type": "abstract",  # or "full_text"
    "page_number": 1,
    "section": "Introduction"
}
```

## Important Notes

- **Progressive Enhancement**: Start with abstracts, add full-text on demand
- **Lazy Loading**: PDFs downloaded only when needed
- **Atomic Transactions**: Always use to prevent partial states
- **GPU Processing**: Uses both A6000s with NVLink
- **Chunk IDs**: Follow predictable pattern for reconstruction
- **Zero Propagation**: Any dimension = 0 means no information exists
- **Context Amplification**: Measured empirically through experiments
- **ArXiv Data Format**: Currently limited to Kaggle dataset format (see processors/arxiv/README.md)

## Available MCP Tools for Development

### ArangoDB Tools (mcp__arango-academy_store__)

- **`arango_query`**: Execute AQL queries directly
  - Use for complex graph traversals and custom queries
  - Example: Finding papers by similarity, citation networks
- **`arango_insert`**: Insert documents into collections
  - Use for adding new papers, metadata, chunks
- **`arango_update`**: Update existing documents
  - Use for marking PDFs as processed, updating embeddings
- **`arango_remove`**: Remove documents from collections
  - Use for cleanup, removing duplicates
- **`arango_list_collections`**: List all database collections
  - Use for exploring database structure
- **`arango_create_collection`**: Create new collections
  - Use for setting up new experiment collections
- **`arango_backup`**: Backup collections to JSON
  - Use for data preservation before major changes

### GitHub Tools (mcp__github__)

- **`get_file_contents`**: Read files from GitHub repos
  - Use for fetching implementation code for papers
- **`search_repositories`**: Search GitHub for repositories
  - Use for finding paper implementations
- **`create_issue`**: Create issues for tracking work
- **`create_pull_request`**: Create PRs for code changes

### LSP Tools (mcp__lsp-python__)

- **`start_lsp`**: Initialize language server
  - Use before analyzing Python code
- **`open_document`**: Open files for analysis
- **`get_diagnostics`**: Get code errors/warnings
- **`get_info_on_location`**: Get type info and docs
  - Use for understanding unfamiliar code
- **`get_completions`**: Get code completion suggestions

### Tree-sitter Tools (mcp__tree-sitter__)

- **`register_project_tool`**: Register HADES for analysis
- **`get_ast`**: Get abstract syntax tree
  - Use for code structure analysis
- **`find_text`**: Search for patterns in code
- **`get_symbols`**: Extract functions, classes, imports
  - Use for mapping code dependencies
- **`analyze_complexity`**: Measure code complexity

### Wolfram Alpha Tool (mcp__MCP-wolfram-alpha__)

- **`query-wolfram-alpha`**: Complex math and computations
  - Use for validating mathematical formulas
  - Use for dimensional analysis calculations

### CodeRabbit Tools (mcp__coderabbit__)

- **`get_coderabbit_reviews`**: Get PR code reviews
- **`get_review_comments`**: Get detailed review feedback
  - Use for understanding code quality issues

### IDE Tools (mcp__ide__)

- **`getDiagnostics`**: Get VS Code diagnostics
- **`executeCode`**: Execute Python in Jupyter kernel
  - Use for testing embeddings, running experiments

### Key Tool Usage Patterns

#### For Database Operations

```python
# Query for papers with embeddings
mcp__arango-academy_store__arango_query(
    query="FOR doc IN metadata FILTER doc.full_text_processed == true RETURN doc",
    bindVars={}
)

# Update PDF status
mcp__arango-academy_store__arango_update(
    collection="metadata",
    key="2310.08560",
    update={"pdf_status": "downloaded", "pdf_local_path": "/path/to/pdf"}
)
```

#### For Code Analysis

```python
# Find paper implementations
mcp__github__search_repositories(
    query="MemGPT implementation",
    limit=10
)

# Analyze code structure
mcp__tree-sitter__get_symbols(
    project="HADES",
    file_path="mcp_server/server.py",
    symbol_types=["functions", "classes"]
)
```

## CRITICAL: Jina Embeddings Version

**WE USE JINA V4 EXCLUSIVELY** - Not v3!

### Jina v4 vs v3: Fundamental Differences
- **Context Length**: 32,768 tokens (vs 8,192 in v3) - 4x increase!
- **Base Model**: Qwen2.5-VL-3B-Instruct (completely different from v3's architecture)
- **Embedding Dimensions**: 2048 with Matryoshka support (128-2048 flexible)
- **Model Name**: `jinaai/jina-embeddings-v4` (NOT v3!)

### Late Chunking with Jina v4
Late chunking is CRITICAL for preserving context:
1. **Traditional chunking breaks context** - splits text into isolated fragments
2. **Late chunking preserves context** - processes entire document FIRST (up to 32k tokens)
3. **Then chunks with awareness** - each chunk embedding contains surrounding context
4. **Result**: Dramatically better semantic understanding

Implementation pattern:
```python
# Process full document through transformer
token_embeddings = model.encode(full_text, output_value='token_embeddings')
# Apply chunking AFTER getting contextual embeddings
chunks = apply_late_chunking(token_embeddings, chunk_size)
```

### Code-Specific LoRA Adapter
Jina v4 includes specialized adapters:
- **jina-embeddings-v4-vllm-code**: Optimized for code understanding
- **Enhanced AST awareness**: Better function/class boundary detection
- **Cross-file context**: Maintains import/dependency relationships

### References:
- [Late Chunking Explanation](https://jina.ai/news/late-chunking-in-long-context-embedding-models/)
- [Jina v4 Model](https://huggingface.co/jinaai/jina-embeddings-v4)
- [Jina v4 Code Adapter](https://huggingface.co/jinaai/jina-embeddings-v4-vllm-code)
- [Jina v4 Paper](https://arxiv.org/pdf/2506.18902)

## Testing Standards

- Unit tests: 85% minimum coverage
- Integration tests for module interactions
- Theory validation tests (zero propagation, context amplification)
- Performance benchmarks for GPU operations
