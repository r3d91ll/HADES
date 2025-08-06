# Web Content Processor for Theory-Practice Bridge Discovery

## Overview

The Web Content Processor captures and analyzes web-based technical content (documentation, tutorials, blog posts) to discover bridges between theoretical papers (ArXiv) and practical implementations (GitHub). It uses FireCrawl for intelligent web scraping and processes content into our 2048-dimensional vector space.

## Theory Connection

In Actor-Network Theory terms, web content represents **OBLIGATORY PASSAGE POINTS** - nodes through which knowledge must flow to transform from theory to practice. Documentation sites, tutorials, and technical blogs serve as **translation mechanisms** that lower CONVEYANCE barriers by providing context, examples, and explanations.

From our Information Reconstructionism framework:

- **WHERE**: Web URLs and site hierarchies (0.6-0.8 typical)
- **WHAT**: Semantic content of documentation (0.7-0.9 typical)
- **CONVEYANCE**: Translation clarity from abstract to concrete (0.8-1.0 for good docs)
- **Bridge Strength** = WHERE × WHAT × CONVEYANCE

## Architecture

### Storage Structure

```
/bulk-store/web-content/
├── raw/                          # FireCrawl JSON outputs
│   ├── {uuid}/                   # Session-based organization
│   │   ├── manifest.json         # Crawl metadata
│   │   └── *.json                # Individual page JSONs
│   └── index.json                # Master index of all crawls
├── processed/                    # Post-processed content
│   ├── sites/                   # Organized by domain
│   │   └── {domain}/
│   │       ├── site_graph.json  # Internal link structure
│   │       └── embeddings/      # Chunk embeddings
│   └── bridges/                 # Discovered connections
│       └── {date}/              # Daily bridge discoveries
└── checkpoints/                  # Processing state
    └── crawl_status.json         # Track processing progress
```

### Database Schema

#### Collection: web_sites

```javascript
{
  "_key": "arize_com_docs_phoenix_2025_01_20",
  "domain": "arize.com",
  "base_url": "https://arize.com/docs/phoenix",
  "crawl_config": {
    "max_depth": 5,
    "max_pages": 100,
    "crawl_date": "2025-01-20T...",
    "firecrawl_session_id": "579377a3-72cc-451f-bfbe-170e838fb867"
  },
  "site_type": "documentation",  // documentation|tutorial|blog|forum
  "content_stats": {
    "total_pages": 100,
    "code_examples": 245,
    "api_references": 67,
    "tutorials": 12
  },
  "discovered_bridges": {
    "arxiv_papers": ["2310.08560", "2401.12345"],
    "github_repos": ["Arize-ai/phoenix", "openai/openai-python"],
    "frameworks": ["phoenix", "opentelemetry", "langchain"]
  },
  "embedding_status": "completed",
  "processing_date": "2025-01-20T..."
}
```

#### Collection: web_pages

```javascript
{
  "_key": "arize_docs_page_{md5_hash}",
  "parent_site": "arize_com_docs_phoenix_2025_01_20",
  "url": "https://arize.com/docs/phoenix/tracing/how-to-tracing/setup-tracing/instrument-python",
  "url_path": "/docs/phoenix/tracing/how-to-tracing/setup-tracing/instrument-python",
  "page_type": "api_reference",  // api_reference|tutorial|guide|example
  "title": "Using Phoenix Decorators",
  "description": "As part of the OpenInference library...",
  "markdown_content": "...",  // Full markdown from FireCrawl
  "content_stats": {
    "word_count": 3456,
    "code_blocks": 28,
    "images": 2,
    "tables": 1
  },
  "code_examples": [
    {
      "language": "python",
      "purpose": "decorator_usage",
      "code": "@tracer.chain\ndef my_func(input: str) -> str:...",
      "imports": ["phoenix.otel", "opentelemetry"],
      "line_count": 15
    }
  ],
  "internal_links": [
    "/docs/phoenix/tracing/how-to-tracing/setup-tracing/custom-spans",
    "/docs/phoenix/tracing/how-to-tracing/setup-tracing/javascript"
  ],
  "external_links": [
    "https://github.com/Arize-ai/phoenix",
    "https://pypi.org/project/arize-phoenix-client/"
  ],
  "references": {
    "github_repos": ["Arize-ai/phoenix"],
    "packages": ["openinference-semantic-conventions", "opentelemetry-api"],
    "frameworks": ["phoenix", "opentelemetry"]
  },
  "embedding_status": "completed",
  "chunk_keys": ["web_chunk_001", "web_chunk_002", "..."]
}
```

#### Collection: web_chunks

```javascript
{
  "_key": "web_chunk_{parent_key}_{index}",
  "parent_page": "arize_docs_page_{md5_hash}",
  "parent_site": "arize_com_docs_phoenix_2025_01_20",
  "chunk_index": 0,
  "chunk_type": "text",  // text|code|mixed
  "content": "...",  // Chunk content
  "context": {
    "section": "Installation",
    "subsection": "Setting Up Tracing",
    "page_position": 0.25  // Relative position in page
  },
  "embedding": [...],  // Jina v4 2048-dim embedding
  "embedding_model": "jinaai/jina-embeddings-v4",
  "embedding_date": "2025-01-20T...",
  "tokens": 512,
  "semantic_type": "instructional"  // instructional|conceptual|example|reference
}
```

#### Collection: web_bridges

```javascript
{
  "_key": "bridge_{source}_{target}_{type}",
  "source": {
    "type": "arxiv",
    "id": "2310.08560",
    "title": "Attention Is All You Need"
  },
  "target": {
    "type": "github",
    "id": "huggingface/transformers",
    "path": "src/transformers/models/bert/modeling_bert.py"
  },
  "intermediary": {
    "type": "web_page",
    "url": "https://huggingface.co/docs/transformers/model_doc/bert",
    "page_key": "huggingface_docs_bert_page"
  },
  "bridge_metrics": {
    "where_score": 0.75,     // Proximity in reference graph
    "what_score": 0.85,      // Semantic similarity
    "conveyance_score": 0.92, // Implementation clarity
    "overall_strength": 0.584 // WHERE × WHAT × CONVEYANCE
  },
  "evidence": {
    "direct_citation": true,
    "code_implementation": true,
    "explanation_quality": "detailed"
  },
  "discovered_date": "2025-01-20T..."
}
```

## Implementation

### Core Components

#### 1. FireCrawlClient (`firecrawl_client.py`)

```python
class FireCrawlClient:
    """
    Client for FireCrawl MCP server integration.
    Handles site crawling, page fetching, and result processing.
    """
    
    def crawl_site(self, url: str, config: CrawlConfig) -> CrawlResult
    def fetch_page(self, url: str) -> PageContent
    def map_site(self, url: str) -> SiteMap
```

#### 2. WebContentProcessor (`web_processor.py`)

```python
class WebContentProcessor:
    """
    Main processor for web content analysis and embedding.
    Orchestrates crawling, processing, and bridge discovery.
    """
    
    def process_documentation_site(self, url: str) -> ProcessingResult
    def process_blog_post(self, url: str) -> ProcessingResult
    def extract_code_examples(self, content: str) -> List[CodeExample]
    def discover_bridges(self, page: WebPage) -> List[Bridge]
```

#### 3. ContentAnalyzer (`content_analyzer.py`)

```python
class ContentAnalyzer:
    """
    Analyzes web content for technical elements and references.
    Extracts papers, repos, code examples, and technical concepts.
    """
    
    def extract_arxiv_references(self, content: str) -> List[str]
    def extract_github_references(self, links: List[str]) -> List[str]
    def classify_content_type(self, content: str) -> ContentType
    def extract_technical_concepts(self, content: str) -> List[Concept]
```

#### 4. BridgeDiscovery (`bridge_discovery.py`)

```python
class BridgeDiscovery:
    """
    Discovers theory-practice bridges through web content.
    Analyzes connections between papers, documentation, and code.
    """
    
    def find_paper_implementations(self, arxiv_id: str) -> List[Bridge]
    def find_code_documentation(self, repo: str) -> List[Bridge]
    def calculate_bridge_strength(self, source, target, intermediary) -> float
    def rank_bridges(self, bridges: List[Bridge]) -> List[Bridge]
```

### Processing Pipeline

#### Phase 1: Crawling

```python
def crawl_documentation_site(url: str) -> CrawlResult:
    """
    Crawl entire documentation site with FireCrawl.
    
    Steps:
    1. Configure crawl parameters (depth, max_pages, filters)
    2. Execute FireCrawl with session tracking
    3. Store raw JSON outputs
    4. Create site manifest
    """
```

#### Phase 2: Content Processing

```python
def process_crawl_results(session_id: str) -> ProcessingResult:
    """
    Process FireCrawl JSON outputs into structured data.
    
    Steps:
    1. Load all JSON files from session
    2. Build site hierarchy from URLs
    3. Extract and classify content
    4. Generate embeddings with Jina v4
    5. Store in ArangoDB with atomic transactions
    """
```

#### Phase 3: Bridge Discovery

```python
def discover_bridges_from_content(site_key: str) -> List[Bridge]:
    """
    Analyze content for theory-practice connections.
    
    Steps:
    1. Extract all references (papers, repos, packages)
    2. Query existing ArXiv and GitHub collections
    3. Calculate semantic similarities
    4. Compute bridge strength metrics
    5. Store discovered bridges
    """
```

## Usage Examples

### Process Documentation Site

```python
from web_processor import WebContentProcessor

processor = WebContentProcessor()

# Process entire Phoenix documentation
result = processor.process_documentation_site(
    url="https://arize.com/docs/phoenix",
    config={
        "max_depth": 5,
        "max_pages": 100,
        "focus_paths": ["/tracing", "/evaluation"],
        "extract_code": True
    }
)

print(f"Processed {result['pages_processed']} pages")
print(f"Found {result['code_examples']} code examples")
print(f"Discovered {result['bridges_found']} theory-practice bridges")
```

### Process Technical Blog Post

```python
# Process single blog post about Word2Vec
result = processor.process_blog_post(
    url="https://medium.com/towards-data-science/word2vec-explained-49c52b8f9f22",
    extract_references=True
)

# Check for ArXiv references
for paper in result['arxiv_references']:
    print(f"References paper: {paper}")

# Check for GitHub implementations
for repo in result['github_references']:
    print(f"Links to implementation: {repo}")
```

### Discover Bridges for Specific Paper

```python
from bridge_discovery import BridgeDiscovery

discoverer = BridgeDiscovery()

# Find all web content bridging to Word2Vec paper
bridges = discoverer.find_paper_implementations("1301.3781")

for bridge in bridges:
    print(f"Bridge via: {bridge['intermediary']['url']}")
    print(f"To implementation: {bridge['target']['id']}")
    print(f"Strength: {bridge['bridge_metrics']['overall_strength']:.3f}")
```

## Configuration

### Environment Variables

```bash
# FireCrawl Configuration
FIRECRAWL_API_KEY=your_api_key
FIRECRAWL_MCP_ENDPOINT=http://localhost:8080

# Storage Configuration
WEB_CONTENT_PATH=/bulk-store/web-content
MAX_CRAWL_DEPTH=5
MAX_PAGES_PER_SITE=100

# Processing Configuration
BATCH_SIZE=32
CHUNK_SIZE=1024
USE_GPU=true
GPU_DEVICE=1

# Database Configuration
ARANGO_HOST=192.168.1.69
ARANGO_PORT=8529
ARANGO_DATABASE=academy_store
ARANGO_PASSWORD=your_password
```

### Crawl Configurations

#### Documentation Sites

```python
DOCUMENTATION_CONFIG = {
    "max_depth": 5,
    "max_pages": 500,
    "include_patterns": ["/docs/", "/api/", "/reference/"],
    "exclude_patterns": ["/blog/", "/forum/", "/search/"],
    "wait_for_js": True,
    "extract_code": True
}
```

#### Tutorial Sites

```python
TUTORIAL_CONFIG = {
    "max_depth": 3,
    "max_pages": 100,
    "include_patterns": ["/tutorials/", "/guides/", "/examples/"],
    "extract_code": True,
    "preserve_structure": True
}
```

#### Blog Posts

```python
BLOG_CONFIG = {
    "max_depth": 0,  # Single page
    "extract_main_content": True,
    "extract_comments": False,
    "extract_related": True
}
```

## Testing

### Unit Tests

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_firecrawl_client.py
pytest tests/test_content_analyzer.py
pytest tests/test_bridge_discovery.py

# Run with coverage
pytest --cov=web_processor tests/
```

### Integration Tests

```python
# Test FireCrawl integration
def test_firecrawl_documentation_crawl():
    """Test crawling a small documentation site."""
    
# Test bridge discovery
def test_bridge_discovery_accuracy():
    """Test accuracy of theory-practice bridge detection."""
    
# Test embedding generation
def test_jina_embedding_generation():
    """Test Jina v4 embeddings for web content."""
```

## Performance Considerations

### Crawling Strategy

- **Incremental crawling**: Only fetch new/changed pages
- **Rate limiting**: Respect robots.txt and server limits
- **Caching**: Store frequently accessed content locally
- **Parallel processing**: Use Ray for multi-page processing

### Storage Optimization

- **Deduplication**: MD5 hash for duplicate detection
- **Compression**: Compress large markdown content
- **Selective storage**: Only store relevant content
- **Cleanup**: Remove outdated crawls after processing

### Embedding Optimization

- **Batch processing**: Process multiple chunks together
- **GPU utilization**: Use CUDA_VISIBLE_DEVICES=1
- **Memory management**: Clear GPU memory between batches
- **Checkpoint recovery**: Save progress for large sites

## Monitoring and Metrics

### Key Metrics

- Pages crawled per minute
- Embeddings generated per second
- Bridges discovered per site
- Storage usage trends
- GPU memory utilization
- Database transaction times

### Health Checks

```python
def check_crawler_health():
    """Verify FireCrawl server is responsive."""
    
def check_storage_capacity():
    """Ensure adequate storage for crawl results."""
    
def check_gpu_availability():
    """Verify GPU is available for embeddings."""
```

## Troubleshooting

### Common Issues

#### FireCrawl Timeout

```python
# Increase timeout for large sites
config = {
    "timeout": 300000,  # 5 minutes
    "retry_on_timeout": True,
    "max_retries": 3
}
```

#### Memory Issues

```python
# Process in smaller batches
BATCH_SIZE = 16  # Reduce from 32
CHUNK_SIZE = 512  # Reduce from 1024

# Clear GPU cache
torch.cuda.empty_cache()
```

#### Duplicate Content

```python
# Use content hashing for deduplication
content_hash = hashlib.md5(content.encode()).hexdigest()
if content_hash not in processed_hashes:
    process_content(content)
```

## Future Enhancements

### Planned Features

1. **Real-time monitoring**: WebSocket updates during crawl
2. **Smart crawling**: ML-based relevance scoring
3. **Differential updates**: Only process changed content
4. **Multi-language support**: Beyond English documentation
5. **API documentation parsing**: OpenAPI/Swagger integration

### Research Directions

1. **Conveyance scoring**: ML model for documentation quality
2. **Bridge prediction**: Predict likely implementation sites
3. **Concept extraction**: Build knowledge graphs from docs
4. **Version tracking**: Track documentation evolution
5. **Community detection**: Find clusters of related content

## References

### Academic Papers

- Information Reconstructionism Theory (our framework)
- Actor-Network Theory (Latour, 2005)
- Knowledge Translation in Software Engineering (various)

### Technical Documentation

- [FireCrawl Documentation](https://firecrawl.com/docs)
- [Jina v4 Embeddings](https://jina.ai/embeddings)
- [ArangoDB Graph Queries](https://www.arangodb.com/docs/stable/aql/)

### Related Projects

- ArXiv Processor (`processors/arxiv/`)
- GitHub Processor (`processors/github/`)
- HERMES Data Pipeline (`olympus/HERMES/`)

## License

Part of the HADES Information Reconstructionism implementation.
See LICENSE file in project root.
