# GitHub Repository Ingestion

This document explains how to use the GitHub repository ingestion functionality in HADES-PathRAG, which allows you to ingest code repositories into the PathRAG database with structured relationship mapping.

## Overview

The repository ingestion system:

1. Clones a GitHub repository to your local machine
2. Parses all code files and documentation files
3. Extracts code elements (modules, classes, functions) and their relationships
4. Creates nodes and edges in the ArangoDB database representing the repository structure
5. Maps relationships between code elements (imports, inheritance, function calls)
6. Cross-references documentation with code elements

## Prerequisites

- ArangoDB server running (see [ArangoDB Setup](./arango_setup.md))
- Python 3.8+ with dependencies installed
- Git installed and configured

## Usage

### Command Line Interface

The easiest way to ingest a repository is using the provided CLI script:

```bash
python scripts/ingest_repo.py https://github.com/username/repo-name
```

#### Options

- `--name`: Custom name for the repository directory (optional)
- `--base-dir`: Base directory to clone repositories into (default: `/home/todd/ML-Lab`)
- `--host`: ArangoDB host (default: 'localhost')
- `--port`: ArangoDB port (default: 8529)
- `--database`: ArangoDB database name (default: 'pathrag')
- `--username`: ArangoDB username (default: 'root')
- `--password`: ArangoDB password
- `--output`: Output file for ingestion stats (optional)

Example with custom options:

```bash
python scripts/ingest_repo.py https://github.com/r3d91ll/HADES-PathRAG \
  --name custom-repo-name \
  --database pathrag_test \
  --output stats.json
```

### Programmatic Usage

You can also use the ingestion functionality programmatically:

```python
from src.ingest.ingestor import RepositoryIngestor

# Initialize ingestor
ingestor = RepositoryIngestor(
    database="pathrag",
    host="localhost",
    port=8529,
    username="root",
    password=""
)

# Ingest repository
success, message, stats = ingestor.ingest_repository(
    repo_url="https://github.com/r3d91ll/HADES-PathRAG",
    repo_name="custom-name",  # Optional
    base_dir="/home/todd/ML-Lab"  # Base directory for cloning
)

# Check results
if success:
    print(f"Repository ingested successfully! Created {stats['nodes_created']} nodes and {stats['edges_created']} edges.")
else:
    print(f"Ingestion failed: {message}")
```

## Data Model

The ingestion process creates the following types of nodes and edges:

### Node Types

- `repository`: The repository itself
- `file`: Source code or documentation files
- `module`: Python modules
- `class`: Python classes
- `function`: Top-level functions
- `method`: Class methods
- `documentation`: Documentation files
- `doc_section`: Sections within documentation files

### Edge Types

- `contains`: Parent-child relationship between code elements
- `imports`: Module import relationships
- `inherits`: Class inheritance relationships
- `calls`: Function call relationships
- `documents`: Documentation references to code elements

## Repository Structure

The repository ingestion functionality is located in the following directories:

```
src/ingest/
├── __init__.py
├── git_operations.py    # Git repository operations
├── code_parser.py       # Code parsing and analysis
├── doc_parser.py        # Documentation parsing
└── ingestor.py          # Main ingestion orchestration

scripts/
└── ingest_repo.py       # CLI script for repository ingestion
```

## Integration with PathRAG

Once a repository is ingested, you can use the regular PathRAG query functions to explore the repository structure and find relevant code elements. For example:

```python
from src.db.arango_connection import ArangoConnection
from src.xnx.arango_adapter import ArangoPathRAGAdapter

# Connect to database
db_connection = ArangoConnection(
    host="localhost",
    port=8529,
    username="root",
    password="",
    database="pathrag"
)

# Initialize PathRAG adapter
adapter = ArangoPathRAGAdapter(db_connection)

# Find repositories
repositories = adapter.get_nodes_by_type("repository")

# Find functions that match a description
matching_functions = adapter.semantic_search("function that handles authentication", node_type="function", limit=5)

# Traverse the repository structure
paths = adapter.traverse_with_xnx(
    start_node="code_nodes/repository_name",
    min_weight=0.7
)
```

## Troubleshooting

- **Error cloning repository**: Ensure you have proper Git credentials and network access
- **No nodes created**: Check that the repository contains supported file types
- **Database connection issues**: Verify ArangoDB is running and credentials are correct
- **Missing relationships**: Some complex code relationships may require manual refinement
