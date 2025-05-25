# ArangoDB Integration with HADES-PathRAG

This document provides an overview of the ArangoDB integration with HADES-PathRAG, including setup instructions, configuration details, and usage examples.

## Overview

ArangoDB is used as a storage backend for graph data in the HADES-PathRAG system. The integration uses the XnX notation for weighted path traversal and supports the following features:

- Storing nodes with embedded vector representations
- Creating edges with weights according to XnX notation
- Querying paths with XnX weighted traversal
- Semantic similarity search using embeddings

## Prerequisites

- ArangoDB server (>= 3.9.0) installed and running
- Python 3.10 or higher
- Required Python packages: `python-arango`, `numpy`, `scipy`

## Configuration

The ArangoDB connection is configured using environment variables:

```bash
# ArangoDB connection settings
HADES_ARANGO_URL=http://localhost:8529
HADES_ARANGO_HOST=localhost
HADES_ARANGO_PORT=8529
HADES_ARANGO_USER=root
HADES_ARANGO_PASSWORD=your_password  # Use a secure password in production
HADES_ARANGO_DATABASE=pathrag_demo
```

You can set these variables in a `.env` file at the root of your project. A template is provided in `.env.template`.

## Setup

1. Install ArangoDB on your system:
   ```bash
   # Example for Debian/Ubuntu
   curl -OL https://download.arangodb.com/arangodb39/DEBIAN/Release.key
   sudo apt-key add - < Release.key
   echo 'deb https://download.arangodb.com/arangodb39/DEBIAN/ /' | sudo tee /etc/apt/sources.list.d/arangodb.list
   sudo apt-get update
   sudo apt-get install arangodb3
   ```

2. Start the ArangoDB service:
   ```bash
   sudo systemctl start arangodb3
   ```

3. Initialize the required database and collections:
   ```bash
   python scripts/reset_arango_db.py
   ```

## Usage

### Connecting to ArangoDB

```python
from src.db.arango_connection import ArangoConnection
from src.xnx.arango_adapter import ArangoPathRAGAdapter

# Connect to ArangoDB
connection = ArangoConnection(db_name="pathrag_demo")

# Initialize the adapter
adapter = ArangoPathRAGAdapter(
    arango_connection=connection,
    db_name="pathrag_demo",
    nodes_collection="pathrag_nodes",
    edges_collection="pathrag_edges",
    graph_name="pathrag_graph"
)
```

### Storing Nodes and Creating Edges

```python
# Store a node with embedded representation
node_id = adapter.store_node(
    node_id="node1",
    content="This is a sample node",
    embedding=[0.1, 0.2, 0.3, ...],  # Vector embedding
    metadata={"domain": "general", "source": "example"}
)

# Create an edge between nodes
edge_id = adapter.create_edge(
    from_node="node1",
    to_node="node2",
    weight=0.8,
    metadata={"relation": "refers_to"}
)
```

### Querying Paths

```python
# Get all paths starting from a node
paths = adapter.get_paths_from_node("node1", max_depth=2)

# Get weighted paths using XnX query syntax
xnx_query = "X(domain='code')2"  # Boost code-related domains by factor 2
weighted_paths = adapter.get_weighted_paths("node1", xnx_query, max_depth=3)
```

### Semantic Search

```python
# Find nodes similar to a query embedding
similar_nodes = adapter.find_similar_nodes(query_embedding, top_k=3)
```

## Example Script

A complete example script showing these operations is available at `examples/arango_pathrag_example.py`. Run it to see the ArangoDB integration in action:

```bash
python examples/arango_pathrag_example.py
```

## Advanced Features

### XnX Notation in ArangoDB

ArangoDB stores XnX-style edges with the following properties:

- `weight`: Numeric weight (0.0 to 1.0)
- `direction`: Integer representing direction (-1, 0, 1)
- `xnx_notation`: String representation in XnX format (e.g. "0.80 node2 -1")
- `temporal_bounds`: Optional temporal information for time-aware traversal

### Vector Similarity

For vector similarity search, we currently use Python-based cosine similarity. For production deployments with large datasets, consider using ArangoDB with the ArangoSearch Vector Search capabilities.

## Troubleshooting

- **Connection issues**: Verify ArangoDB is running with `sudo systemctl status arangodb3`
- **Authentication errors**: Check your credentials in environment variables
- **Missing collections**: Run the reset script to initialize the database

## Testing

The ArangoDB integration includes a comprehensive test suite in `tests/db/test_arango_adapter.py`. These tests verify all core functionality of the ArangoDB adapter.

### Running Tests

To run the tests, use the following command:

```bash
python -m unittest tests/db/test_arango_adapter.py
```

The tests will create and use a separate test database (default name: `pathrag_test`), and clean up after themselves.

### Writing New Tests

When extending the ArangoDB functionality, add appropriate tests in the test suite. The test suite follows these principles:

1. Tests run in a specific order using numeric prefixes (`test_01_...`, `test_02_...`, etc.).
2. Tests create a separate database and collections to avoid interfering with production data.
3. Test cleanup happens in the `tearDownClass` method to ensure resources are properly released.

### Test Database Configuration

You can specify a custom test database name by setting the `HADES_ARANGO_DATABASE` environment variable before running the tests:

```bash
HADES_ARANGO_DATABASE=my_test_db python -m unittest tests/db/test_arango_adapter.py
```

## Resources

- [ArangoDB Documentation](https://www.arangodb.com/docs/)
- [Python-Arango API Reference](https://python-driver-for-arangodb.readthedocs.io/)
- [HADES-PathRAG XnX Documentation](../xnx/XnX_README.md)
