# HADES MCP Server

Model Context Protocol (MCP) server for the HADES arXiv processing pipeline. Provides async tools for processing, searching, and managing academic papers with GPU-accelerated embeddings.

## Features

### Core Tools

- **`process_arxiv_batch`**: Process batches of arXiv papers from metadata files
- **`process_single_paper`**: Process individual papers on-demand
- **`semantic_search`**: Search papers by semantic similarity
- **`check_job_status`**: Monitor processing job progress
- **`get_gpu_status`**: Check GPU availability and memory usage
- **`list_jobs`**: View all processing jobs

### Key Capabilities

- **GPU Acceleration**: Dual GPU support with automatic batching
- **Async Processing**: Non-blocking operations for long-running tasks
- **Job Management**: Track multiple processing jobs with detailed statistics
- **Semantic Search**: Vector similarity search with category and date filtering
- **Checkpointing**: Resume interrupted processing from last checkpoint
- **Error Recovery**: Robust error handling with automatic retries

## Installation

```bash
# Install dependencies
pip install -r mcp_server/requirements.txt

# Install MCP SDK
pip install mcp
```

## Configuration

Edit `mcp_server/config/server_config.yaml`:

```yaml
server:
  name: "hades-arxiv"
  port: 8080
  
database:
  host: "localhost"
  port: 8529
  username: "root"
  password: "${ARANGO_PASSWORD}"
  
processing:
  gpu_batch_size: 1024
  model_name: "jinaai/jina-embeddings-v3"
```

## Usage

### Starting the Server

```bash
# Basic start
python mcp_server/launch.py

# With custom config
python mcp_server/launch.py --config /path/to/config.yaml

# Override settings
python mcp_server/launch.py --port 8081 --log-level DEBUG
```

### Connecting from Claude

```bash
# Connect to the MCP server
mcp connect hades://localhost:8080

# List available tools
mcp tools

# Process papers
mcp call process_arxiv_batch --input-file arxiv_metadata.json --limit 1000

# Search papers
mcp call semantic_search --query "transformer architectures" --limit 10
```

### Example Tool Usage

#### Process a Batch of Papers

```python
result = await process_arxiv_batch(
    input_file="arxiv_cs_2024.json",
    db_name="academy_store",
    collection_name="base_arxiv",
    limit=1000,
    categories=["cs.AI", "cs.LG"],
    resume=True,
    gpu_batch_size=1024
)
# Returns: Job ID for tracking
```

#### Process Single Paper

```python
result = await process_single_paper(
    arxiv_id="2310.08560",
    title="MemGPT: Towards LLMs as Operating Systems",
    abstract="Large language models...",
    categories=["cs.AI"],
    authors=["Charles Packer", "..."]
)
# Returns: Processing confirmation with embedding info
```

#### Semantic Search

```python
result = await semantic_search(
    query="attention mechanisms in transformers",
    limit=10,
    categories=["cs.LG"],
    min_similarity=0.7
)
# Returns: Ranked list of similar papers
```

#### Check Processing Status

```python
result = await check_job_status(job_id="job_20241220_143025")
# Returns: Job status, statistics, and progress
```

#### GPU Status

```python
result = await get_gpu_status()
# Returns: GPU memory usage and availability
```

## Architecture

### Async Design

- All CPU/GPU intensive operations run in thread pools
- Non-blocking I/O for database operations
- Background job processing with status tracking

### State Management

- Processor instance kept alive between calls for efficiency
- Job tracking persists across multiple operations
- Model loaded once and reused

### Error Handling

- Graceful degradation when GPU unavailable
- Automatic retry logic for transient failures
- Detailed error reporting in job status

## Performance

### Batch Processing
- 1000 papers: ~2-3 minutes with dual GPUs
- 10,000 papers: ~20-30 minutes
- Automatic checkpointing every 100 documents

### Single Paper Processing
- Abstract embedding: <100ms
- Database storage: <50ms
- Total latency: <200ms

### Search Performance
- Vector similarity search: <500ms for 1M documents
- With filtering: <1s depending on complexity

## Development

### Adding New Tools

```python
@self.server.tool()
async def custom_tool(param1: str, param2: int) -> ToolResult:
    """Tool description for MCP discovery."""
    # Implementation
    return ToolResult(content=[TextContent(text="Result")])
```

### Extending Processing

The server wraps the production processor from `processors/arxiv/`. To add features:

1. Extend the `ProductionProcessor` class
2. Add corresponding MCP tool wrapper
3. Handle async execution properly

## Monitoring

### Logs
- Server logs: `hades_mcp_server.log`
- Processing logs: Check job status for details

### Metrics
- Processing rate: Documents/second
- GPU utilization: Memory and compute
- Error rate: Failed documents ratio

## Troubleshooting

### GPU Not Available
- Server falls back to CPU mode automatically
- Check CUDA installation: `nvidia-smi`
- Verify PyTorch GPU support: `torch.cuda.is_available()`

### Database Connection Failed
- Check ArangoDB is running: `systemctl status arangodb3`
- Verify credentials in config
- Test connection: `arango-client test`

### Out of Memory
- Reduce `gpu_batch_size` in config
- Enable FP16 mode for larger batches
- Monitor with `nvidia-smi -l 1`

## Future Enhancements

- [ ] PDF download and processing
- [ ] Multi-model support (OpenAI, Cohere)
- [ ] Incremental index updates
- [ ] Cross-collection search
- [ ] Citation network analysis
- [ ] Real-time processing streams