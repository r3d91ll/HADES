# HADES Integration Guide

This document outlines the recursive integration pattern where HADES-PathRAG is imported into the core HADES system, enabling self-referential knowledge management and retrieval.

## Recursive Integration Architecture

```
┌───────────────────────────────────────────────────────────┐
│                  HADES Core System                         │
│                                                           │
│  ┌─────────────────────┐       ┌───────────────────────┐  │
│  │                     │       │                       │  │
│  │  Knowledge Engine   │◄─────►│  HADES-PathRAG        │  │
│  │                     │       │  (Imported Module)    │  │
│  └─────────┬───────────┘       └───────────┬───────────┘  │
│            │                               │              │
│            │                               │              │
│            ▼                               ▼              │
│  ┌─────────────────────┐       ┌───────────────────────┐  │
│  │                     │       │                       │  │
│  │  Memory Management  │◄─────►│  Knowledge Graph      │  │
│  │                     │       │  (ArangoDB)           │  │
│  └─────────────────────┘       └───────────────────────┘  │
│                                                           │
└───────────────────────────────────────────────────────────┘
                        │
                        │ MCP Integration
                        ▼
┌───────────────────────────────────────────────────────────┐
│                  Windsurf IDE                             │
│                                                           │
│  ┌─────────────────────┐       ┌───────────────────────┐  │
│  │                     │       │                       │  │
│  │  Cascade AI Agent   │◄─────►│  MCP Client           │  │
│  │                     │       │                       │  │
│  └─────────────────────┘       └───────────────────────┘  │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

## Integration Steps

### 1. Install HADES-PathRAG as a Module

```bash
# From your HADES root directory
pip install -e /path/to/HADES-PathRAG
```

Or add to your requirements.txt:

```
-e git+https://github.com/yourusername/HADES-PathRAG.git#egg=hades-pathrag
```

### 2. Import PathRAG in HADES Core

```python
# In your HADES core code
from hades_pathrag import XnXPathRAG
from hades_pathrag.xnx import XnXParams

# Initialize the PathRAG system
pathrag = XnXPathRAG(config={
    "db_url": "http://localhost:8529",
    "db_name": "hades_knowledge",
    "username": "hades_user",
    "password": "your_secure_password"
})
```

### 3. Configure Self-Referential Knowledge

To enable HADES to understand its own codebase:

```python
# Ingest HADES codebase into the knowledge graph
async def ingest_hades_codebase():
    # Get repository path
    repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Find Python files
    python_files = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    # Process each file
    for file_path in python_files:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Extract code blocks, functions, classes
        code_blocks = extract_code_blocks(content)
        
        # Ingest into knowledge graph with XnX notation
        for block in code_blocks:
            await pathrag.ingest_data({
                "type": "code_block",
                "content": block.content,
                "file_path": file_path,
                "relationships": [
                    {
                        "to": "file:" + file_path,
                        "weight": 1.0,
                        "direction": -1,
                        "type": "contained_in"
                    },
                    # Add more relationships based on imports, function calls, etc.
                ]
            })
```

### 4. Connect to MCP Server

Enable HADES to interact with the IDE through the Model Context Protocol:

```python
# In your HADES initialization
from hades_pathrag.mcp import MCPServer

# Start the MCP server
mcp_server = MCPServer(
    host="localhost",
    port=8123,
    pathrag=pathrag
)

# Register tools
mcp_server.register_tool(
    "mcp0_pathrag_retrieve", 
    pathrag.mcp_retrieve_handler
)
mcp_server.register_tool(
    "mcp0_ingest_data", 
    pathrag.mcp_ingest_handler
)
mcp_server.register_tool(
    "mcp0_xnx_pathrag_retrieve", 
    pathrag.mcp_xnx_retrieve_handler
)

# Start server
await mcp_server.start()
```

## Self-Improvement Workflow

The recursive integration enables HADES to improve itself through the following workflow:

1. **Code Analysis**: HADES analyzes its own codebase using PathRAG
2. **Knowledge Identification**: Identifies areas for improvement or extension
3. **IDE Integration**: Suggests improvements through the Windsurf IDE
4. **Implementation**: Changes are implemented through the IDE
5. **Knowledge Update**: New code is automatically ingested back into the system

### Example: Self-Improving Type Inference

```python
async def improve_type_inference():
    """HADES improves its own type inference system."""
    
    # 1. Query PathRAG for current type inference implementation
    type_inference_paths = await pathrag.query(
        "How does HADES perform type inference?",
        min_weight=0.8
    )
    
    # 2. Analyze current implementation
    analysis = await analyze_code_quality(type_inference_paths)
    
    # 3. Identify improvement areas
    improvement_areas = get_improvement_areas(analysis)
    
    # 4. Generate improvement suggestions
    suggestions = []
    for area in improvement_areas:
        suggestion = await generate_improvement(area)
        suggestions.append(suggestion)
    
    # 5. Send suggestions to IDE through MCP
    await mcp_server.send_suggestions(suggestions)
```

## Debugging and Monitoring

### Monitoring the Integration

```python
# Add monitoring to your HADES-PathRAG integration
from hades_pathrag.monitoring import PathRAGMonitor

monitor = PathRAGMonitor(
    pathrag=pathrag,
    log_level="INFO",
    metrics_port=9090
)

# Enable activity logging
monitor.start()
```

### Common Integration Issues

1. **Database Connection Failures**
   
   ```python
   # Test connection before use
   try:
       await pathrag.test_connection()
   except ConnectionError as e:
       logger.error(f"Failed to connect to ArangoDB: {e}")
       # Implement fallback or retry logic
   ```

2. **MCP Communication Issues**
   
   ```python
   # Heartbeat check for MCP connection
   async def check_mcp_health():
       while True:
           try:
               await mcp_server.ping()
           except ConnectionError:
               logger.error("MCP server connection lost")
               # Attempt to reconnect
               await mcp_server.reconnect()
           await asyncio.sleep(30)
   ```

## Advanced Integration Patterns

### Event-Driven Integration

```python
# Event-driven integration with HADES core
from hades_pathrag.events import EventBus

event_bus = EventBus()

# Register event handlers
@event_bus.on("knowledge_updated")
async def on_knowledge_update(event_data):
    """Handle knowledge graph updates."""
    # Re-index affected paths
    await pathrag.reindex_paths(event_data["affected_paths"])
    
    # Notify HADES core components
    hades_core.notify_knowledge_update(event_data)

# In HADES core, publish events
async def update_knowledge():
    # ... update logic ...
    await event_bus.publish("knowledge_updated", {
        "affected_paths": ["path1", "path2"],
        "update_type": "addition",
        "timestamp": datetime.now().isoformat()
    })
```

### Versioned Integration

```python
# Support multiple versions of HADES in the knowledge graph
async def query_across_versions(query, versions=None):
    """Query across different HADES versions."""
    if not versions:
        # Get available versions
        versions = await pathrag.get_available_versions()
    
    # Query each version
    results = {}
    for version in versions:
        version_results = await pathrag.query(
            query,
            as_of_version=version
        )
        results[version] = version_results
    
    # Compare results across versions
    return {
        "results": results,
        "comparison": compare_version_results(results)
    }
```

## Conclusion

By following this integration guide, you can establish a recursive "HADES builds HADES" pattern where the system can analyze and improve itself through self-referential knowledge management and the Model Context Protocol (MCP).
