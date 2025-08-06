# CLAUDE.md - MCP Server Guide

This file provides guidance for using the HADES MCP Server with Claude Code (claude.ai/code).

## Overview

The HADES MCP Server provides tools for processing arXiv papers, generating embeddings, and performing semantic search through the Model Context Protocol (MCP) interface.

## Setting Up with Claude Code

### 1. Install Dependencies

First, ensure the server dependencies are installed:

```bash
cd /home/todd/olympus/HADES
pip install -r mcp_server/requirements.txt
```

### 2. Configure Database

Ensure ArangoDB is running and set credentials:

```bash
export ARANGO_HOST="localhost"
export ARANGO_PASSWORD="your-password"
```

### 3. Add Server to Claude Code

There are three ways to add the server to Claude Code:

#### Option A: Project-Specific (Recommended for Development)

```bash
# Add to current project only
claude mcp add hades-arxiv python /home/todd/olympus/HADES/mcp_server/launch.py \
  -e ARANGO_PASSWORD="your-password" \
  -e ARANGO_HOST="localhost"
```

#### Option B: Shared Project Configuration

```bash
# Add to project and share via .mcp.json
claude mcp add hades-arxiv python /home/todd/olympus/HADES/mcp_server/launch.py \
  -s project \
  -e ARANGO_PASSWORD="${ARANGO_PASSWORD}" \
  -e ARANGO_HOST="${ARANGO_HOST}"
```

This creates `.mcp.json` in the project root that can be committed to version control.

#### Option C: User-Wide Configuration

```bash
# Available in all Claude Code sessions
claude mcp add hades-arxiv python /home/todd/olympus/HADES/mcp_server/launch.py \
  -s user \
  -e ARANGO_PASSWORD="${ARANGO_PASSWORD}" \
  -e ARANGO_HOST="${ARANGO_HOST}"
```

### 4. Verify Server Connection

In Claude Code, use the `/mcp` command to check server status:

```
/mcp
```

You should see `hades-arxiv` listed as an available server.

## Available Tools

Once connected, the following tools are available:

### Processing Tools

- **`process_arxiv_batch`** - Process multiple papers from JSON metadata
- **`process_single_paper`** - Process one paper immediately
- **`check_job_status`** - Monitor processing progress
- **`list_jobs`** - View all processing jobs

### Search Tools

- **`semantic_search`** - Find papers by semantic similarity

### Monitoring Tools

- **`get_gpu_status`** - Check GPU memory and utilization

## Usage Examples in Claude Code

### Example 1: Process Papers

```
Can you process the latest arXiv CS papers? The metadata is in /data/arxiv_cs_2024.json
```

Claude will use the `process_arxiv_batch` tool automatically.

### Example 2: Search Papers

```
Find papers similar to "attention is all you need" in the cs.LG category
```

Claude will use the `semantic_search` tool with appropriate parameters.

### Example 3: Monitor Processing

```
Check the status of my processing jobs
```

Claude will use `list_jobs` and `check_job_status` tools.

## Configuration File

The server uses `mcp_server/config/server_config.yaml` for configuration:

```yaml
server:
  name: "hades-arxiv"
  port: 8080
  log_level: "INFO"

database:
  host: "${ARANGO_HOST:-localhost}"
  port: 8529
  username: "${ARANGO_USERNAME:-root}"
  password: "${ARANGO_PASSWORD}"
  default_db: "academy_store"
  default_collection: "base_arxiv"

processing:
  gpu_batch_size: 1024
  model_name: "jinaai/jina-embeddings-v3"
```

Environment variables in the config are automatically expanded.

## Advanced Configuration

### Custom Server Arguments

```bash
# Add with custom port and log level
claude mcp add hades-arxiv python /home/todd/olympus/HADES/mcp_server/launch.py \
  --port 8081 \
  --log-level DEBUG \
  -e ARANGO_PASSWORD="your-password"
```

### Using Different Config Files

```bash
# Use alternative configuration
claude mcp add hades-arxiv python /home/todd/olympus/HADES/mcp_server/launch.py \
  --config /path/to/custom_config.yaml \
  -e ARANGO_PASSWORD="your-password"
```

### Setting Timeout

For long-running operations, increase the timeout:

```bash
# Set 5 minute timeout for server startup
export MCP_TIMEOUT=300
claude mcp add hades-arxiv python /home/todd/olympus/HADES/mcp_server/launch.py
```

## Troubleshooting

### Server Not Starting

1. Check logs: `tail -f hades_mcp_server.log`
2. Verify database connection: Test ArangoDB is accessible
3. Check GPU availability: `nvidia-smi`

### Tools Not Available

1. Restart the server: `/mcp restart hades-arxiv`
2. Check server status: `/mcp`
3. Review server configuration

### Permission Issues

Ensure Claude Code has access to:
- Python environment with dependencies
- ArangoDB credentials
- GPU devices (if using GPU acceleration)

## Development Workflow

### Making Changes

1. Edit server code in `mcp_server/`
2. Restart the server in Claude Code:
   ```
   /mcp restart hades-arxiv
   ```
3. Test the updated tools

### Adding New Tools

1. Add tool method to `server.py`:
   ```python
   @self.server.tool()
   async def new_tool(param: str) -> ToolResult:
       """Tool description."""
       # Implementation
       return ToolResult(content=[TextContent(text="Result")])
   ```

2. Restart server to load new tool
3. Tool will be automatically available in Claude Code

### Debugging

Enable debug logging:
```bash
claude mcp add hades-arxiv python /home/todd/olympus/HADES/mcp_server/launch.py \
  --log-level DEBUG
```

View debug output:
```bash
tail -f hades_mcp_server.log
```

## Best Practices

1. **Use Project Scope** for development - keeps configuration isolated
2. **Environment Variables** for sensitive data - never hardcode passwords
3. **Monitor GPU Usage** - use `get_gpu_status` tool regularly
4. **Check Job Status** - for batch processing operations
5. **Graceful Shutdown** - use `/mcp stop hades-arxiv` before making changes

## Security Notes

- Never commit `.mcp.json` with real passwords
- Use environment variables for sensitive configuration
- Restrict database access to necessary operations only
- Monitor server logs for unauthorized access attempts

## Performance Tips

- Batch operations when possible using `process_arxiv_batch`
- Monitor GPU memory with `get_gpu_status` before large operations
- Use job IDs to track long-running processes
- Adjust `gpu_batch_size` based on available GPU memory

## Integration with Claude Code

The server integrates seamlessly with Claude Code's features:

- **Auto-discovery**: Tools are automatically available after server starts
- **Context awareness**: Claude understands tool capabilities from descriptions
- **Error handling**: Detailed error messages help Claude retry operations
- **Async execution**: Non-blocking operations allow continued interaction

When working with the server in Claude Code, you can simply describe what you want to do, and Claude will automatically select and use the appropriate tools.