# XnX Notation: Implementation Guide

This document provides technical implementation details for integrating XnX notation into the HADES-PathRAG system.

## Architecture Overview

The XnX-enhanced PathRAG implementation consists of the following components:

```
┌───────────────────────────────────────────────────────────┐
│                      HADES System                          │
│                                                           │
│  ┌─────────────────────┐       ┌───────────────────────┐  │
│  │   MCP Server        │       │  PathRAG with XnX     │  │
│  │                     │       │                       │  │
│  │  • API Endpoints    │◄─────►│  • Path Construction  │  │
│  │  • Tool Integrations│       │  • Path Pruning       │  │
│  │  • LLM Interface    │       │  • Path Verification  │  │
│  └─────────┬───────────┘       └───────────┬───────────┘  │
│            │                               │              │
│            │                               │              │
│            ▼                               ▼              │
│  ┌─────────────────────┐       ┌───────────────────────┐  │
│  │ ArangoDB Adapter    │       │  Query Processing     │  │
│  │                     │       │                       │  │
│  │  • Graph Storage    │◄─────►│  • XnX Parameter      │  │
│  │  • Entity Management│       │    Extraction         │  │
│  │  • Relationship     │       │  • Path Selection     │  │
│  │    Management       │       │  • Result Formatting  │  │
│  └─────────────────────┘       └───────────────────────┘  │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

## Source Code Structure

The XnX implementation is organized across several key files:

```
/home/todd/ML-Lab/HADES-PathRAG/
├── src/
│   ├── xnx/
│   │   ├── __init__.py
│   │   ├── params.py        # XnX parameter specification and validation
│   │   ├── parser.py        # XnX notation parsing and formatting
│   │   └── paths.py         # XnX-aware path traversal functions
│   │
│   ├── mcp/
│   │   ├── server.py        # MCP server with XnX-enhanced tools
│   │   └── tools.py         # Tool implementations for MCP
│   │
│   └── pathrag/
│       ├── __init__.py
│       ├── xnx_pathrag.py   # XnX-enhanced PathRAG implementation
│       ├── db_adapter.py    # Database adapter for ArangoDB
│       └── query.py         # Query processing logic
```

## Key Classes and Functions

### XnX Parameter Class

```python
# src/xnx/params.py
class XnXParams:
    """Parameters for XnX-enhanced path queries."""
    
    def __init__(
        self,
        min_weight: float = 0.5,
        max_distance: int = 3,
        direction: Optional[int] = None,
        valid_from: Optional[str] = None,
        valid_until: Optional[str] = None,
        permanence: Optional[str] = None
    ):
        """
        Initialize XnX parameters.
        
        Args:
            min_weight: Minimum relationship weight (0.0-1.0)
            max_distance: Maximum path distance to traverse
            direction: Path direction (positive=inbound, negative=outbound, None=both)
            valid_from: Start of validity period (ISO date)
            valid_until: End of validity period (ISO date)
            permanence: Permanence category (∞, L, C, T[date])
        """
        self.min_weight = self._validate_weight(min_weight)
        self.max_distance = max_distance
        self.direction = direction
        self.valid_from = valid_from
        self.valid_until = valid_until
        self.permanence = permanence
    
    def _validate_weight(self, weight: float) -> float:
        """Validate weight is between 0.0 and 1.0."""
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"Weight must be between 0.0 and 1.0, got {weight}")
        return weight
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary for database queries."""
        return {
            "min_weight": self.min_weight,
            "max_distance": self.max_distance,
            "direction": self.direction,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "permanence": self.permanence
        }
    
    @classmethod
    def from_xnx_string(cls, xnx_string: str) -> "XnXParams":
        """Parse XnX notation string into parameters."""
        # Implementation details for parsing strings like "0.8 EntityName -2"
        # ...
```

### XnX-Enhanced PathRAG Implementation

```python
# src/pathrag/xnx_pathrag.py
class XnXPathRAG:
    """
    PathRAG implementation with XnX enhancements.
    """
    
    def __init__(self, db_adapter=None):
        """Initialize the XnX-enhanced PathRAG system."""
        self.db_adapter = db_adapter or ArangoAdapter()
    
    async def query(self, query_text: str, **xnx_params):
        """
        Query the knowledge graph using XnX-enhanced PathRAG.
        
        Args:
            query_text: Natural language query
            **xnx_params: XnX parameters (min_weight, direction, etc.)
        
        Returns:
            List of paths relevant to the query
        """
        # 1. Convert kwargs to XnXParams object
        params = XnXParams(**xnx_params)
        
        # 2. Find entry points in the graph
        entry_points = await self._find_entry_points(query_text)
        
        # 3. Construct paths with XnX parameters
        paths = await self._construct_paths(entry_points, params)
        
        # 4. Prune irrelevant paths using XnX weights
        relevant_paths = self._prune_paths(paths, query_text, params)
        
        # 5. Format results
        return self._format_results(relevant_paths)
    
    async def _find_entry_points(self, query_text: str):
        """Find entry points in the knowledge graph."""
        # Implementation details
        # ...
    
    async def _construct_paths(self, entry_points, params: XnXParams):
        """
        Construct paths from entry points using XnX parameters.
        
        Direction control based on XnX notation:
        - If direction > 0: Follow inbound relationships
        - If direction < 0: Follow outbound relationships
        - If direction is None: Follow both directions
        """
        # Implementation details
        # ...
    
    def _prune_paths(self, paths, query_text: str, params: XnXParams):
        """
        Prune paths based on XnX parameters.
        
        Filtering based on:
        - Relationship weight (min_weight)
        - Path relevance to query
        - Temporal validity if specified
        """
        # Implementation details
        # ...
```

### MCP Tool Implementation

```python
# src/mcp/tools.py
async def mcp0_xnx_pathrag_retrieve(query, min_weight=0.5, max_distance=3, direction=None, 
                                valid_from=None, valid_until=None, domain_filter=None):
    """
    MCP tool for retrieving paths with XnX parameters.
    
    Args:
        query: The query to retrieve paths for
        min_weight: Minimum relationship weight (0.0-1.0)
        max_distance: Maximum path distance to traverse
        direction: Path direction (positive=inbound, negative=outbound)
        valid_from: Start of validity period (ISO date)
        valid_until: End of validity period (ISO date)
        domain_filter: Optional domain to filter results by
    
    Returns:
        List of paths relevant to the query
    """
    # Initialize PathRAG with XnX
    pathrag = XnXPathRAG()
    
    # Execute query with XnX parameters
    paths = await pathrag.query(
        query,
        min_weight=min_weight,
        max_distance=max_distance,
        direction=direction,
        valid_from=valid_from,
        valid_until=valid_until
    )
    
    # Filter by domain if specified
    if domain_filter:
        paths = [p for p in paths if domain_filter in p.get("domains", [])]
    
    return {
        "paths": paths,
        "query_parameters": {
            "min_weight": min_weight,
            "max_distance": max_distance,
            "direction": direction,
            "valid_from": valid_from,
            "valid_until": valid_until,
            "domain_filter": domain_filter
        }
    }
```

## Database Schema

The XnX-enhanced PathRAG system uses ArangoDB with the following collections:

### Nodes Collection

Represents entities in the knowledge graph:

```json
{
  "_id": "nodes/12345",
  "_key": "12345",
  "name": "User Authentication Function",
  "entity_type": "code_function",
  "vector": [...],  // Embedding vector
  "metadata": {
    "file_path": "/src/auth.py",
    "line_number": 42,
    "last_modified": "2025-03-15T12:34:56Z"
  }
}
```

### Edges Collection

Represents relationships with XnX properties:

```json
{
  "_id": "edges/67890",
  "_key": "67890",
  "_from": "nodes/12345",
  "_to": "nodes/67890",
  "weight": 0.85,
  "direction": -1,
  "valid_from": "2025-01-01T00:00:00Z",
  "valid_until": "2026-01-01T00:00:00Z",
  "permanence": "L",
  "relationship_type": "calls",
  "metadata": {
    "creator": "static_analysis",
    "confidence_source": "call_frequency"
  }
}
```

## AQL Query Examples

### Basic Path Query with XnX Parameters

```aql
FOR v, e, p IN 1..@max_distance OUTBOUND @start_vertex edges
  FILTER e.weight >= @min_weight
  FILTER (@valid_from == null OR e.valid_from <= @valid_from)
  FILTER (@valid_until == null OR e.valid_until >= @valid_until)
  FILTER (@direction == null) OR 
         (@direction < 0 AND e.direction < 0) OR
         (@direction > 0 AND e.direction > 0)
  RETURN p
```

### Temporal XnX Query

```aql
LET as_of_date = "2025-03-22"
FOR v, e, p IN 1..3 OUTBOUND @start_vertex edges
  FILTER e.weight >= 0.7
  FILTER (e.valid_from <= as_of_date OR e.valid_from == null)
  FILTER (e.valid_until >= as_of_date OR e.valid_until == null)
  RETURN {
    "path": p,
    "average_weight": AVERAGE(p.edges[*].weight),
    "valid_at": as_of_date
  }
```

## Phase 1 Implementation (Rule-Based)

The initial implementation of XnX-enhanced PathRAG uses a rule-based approach:

1. **XnX Parameter Parsing**: Extracting parameters from XnX notation or explicit parameters
2. **Rule-Based Path Construction**: Using AQL to traverse the graph with XnX parameters
3. **Rule-Based Path Pruning**: Filtering paths based on weights and other parameters
4. **Path Ranking**: Sorting paths by combined confidence/relevance score

## Phase 2 Implementation (GNN-Enhanced)

Future enhancements will incorporate a Graph Neural Network (GNN) approach:

1. **Learned Path Importance**: Using a GNN to learn optimal path weights
2. **Dynamic Weight Adjustment**: Adjusting weights based on usage patterns and feedback
3. **Context-Aware Path Selection**: Selecting paths based on query context
4. **Multi-Hop Reasoning**: Enhancing path discovery with learned patterns

## Performance Considerations

### Query Optimization

For large knowledge graphs:

```python
def optimize_query(params: XnXParams):
    """Optimize query parameters for performance."""
    optimized = params.copy()
    
    # Reduce max distance for low-weight queries
    if params.min_weight < 0.3 and params.max_distance > 2:
        optimized.max_distance = 2
    
    # Add index hints for temporal queries
    if params.valid_from or params.valid_until:
        optimized.index_hints = ["temporal_index"]
    
    return optimized
```

### Caching Strategy

```python
class PathCache:
    """Cache for path queries."""
    
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
    
    def get(self, query_key):
        """Get cached paths for query."""
        return self.cache.get(query_key)
    
    def set(self, query_key, paths):
        """Cache paths for query."""
        if len(self.cache) >= self.max_size:
            # Evict least recently used item
            self.cache.pop(next(iter(self.cache)))
        
        self.cache[query_key] = {
            "paths": paths,
            "timestamp": time.time()
        }
```

## Testing and Validation

### Unit Tests

Example test for XnX parameter validation:

```python
def test_xnx_params_validation():
    """Test XnX parameter validation."""
    # Valid parameters
    params = XnXParams(min_weight=0.7, max_distance=2, direction=-1)
    assert params.min_weight == 0.7
    assert params.max_distance == 2
    assert params.direction == -1
    
    # Invalid weight
    with pytest.raises(ValueError):
        XnXParams(min_weight=1.5)
```

### Integration Tests

Example integration test with ArangoDB:

```python
@pytest.mark.asyncio
async def test_path_retrieval():
    """Test path retrieval with XnX parameters."""
    # Setup test data
    await setup_test_graph()
    
    # Initialize PathRAG
    pathrag = XnXPathRAG()
    
    # Execute query
    paths = await pathrag.query(
        "What functions handle authentication?",
        min_weight=0.7,
        direction=-1
    )
    
    # Validate results
    assert len(paths) > 0
    assert all(p["average_weight"] >= 0.7 for p in paths)
    assert any("authenticate" in p["path_text"] for p in paths)
```
