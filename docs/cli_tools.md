# HADES-PathRAG CLI Tools

This document describes the command-line interface (CLI) tools available in HADES-PathRAG for managing ArangoDB databases, ingesting documentation, and querying the knowledge graph.

## Overview

HADES-PathRAG provides the following CLI tools:

| Command | Description |
|---------|-------------|
| `pathrag-setup-db` | Initialize or recreate ArangoDB database structure |
| `pathrag-verify-db` | Verify ArangoDB database contents and structure |
| `pathrag-reset-db` | Delete an ArangoDB database |
| `pathrag-ingest` | Ingest documentation into the knowledge graph |
| `pathrag-query` | Query the knowledge graph |

## Database Management Tools

### pathrag-setup-db

Sets up the ArangoDB graph structure for HADES-PathRAG.

```bash
./scripts/pathrag-setup-db [options]
```

**Options:**

- `--db-name DB_NAME`: Database name (default: "hades")
- `--node-collection NODE_COLLECTION`: Node collection name (default: "nodes")
- `--edge-collection EDGE_COLLECTION`: Edge collection name (default: "edges")
- `--graph-name GRAPH_NAME`: Graph name (default: "pathrag")
- `--force`: Force recreation even if collections/graph exist
- `--username USERNAME`: ArangoDB username (default: "root")
- `--password PASSWORD`: ArangoDB password (default: "")

**Example:**

```bash
# Create a new database with default structure
./scripts/pathrag-setup-db --db-name my_knowledge_base

# Recreate an existing database (drops existing collections)
./scripts/pathrag-setup-db --db-name my_knowledge_base --force
```

### pathrag-verify-db

Verifies ArangoDB database contents and structure.

```bash
./scripts/pathrag-verify-db [options]
```

**Options:**

- `--db-name DB_NAME`: Database name to verify (default: "hades")
- `--username USERNAME`: ArangoDB username (default: "root")
- `--password PASSWORD`: ArangoDB password (default: "")
- `--output-file OUTPUT_FILE`: Optional file to save database info

**Example:**

```bash
# Verify database structure
./scripts/pathrag-verify-db --db-name my_knowledge_base

# Save verification results to a file
./scripts/pathrag-verify-db --db-name my_knowledge_base --output-file db_info.json
```

### pathrag-reset-db

Resets (deletes) an ArangoDB database.

```bash
./scripts/pathrag-reset-db [options]
```

**Options:**

- `--db-name DB_NAME`: Database name to reset (required)
- `--username USERNAME`: ArangoDB username (default: "root")
- `--password PASSWORD`: ArangoDB password (default: "")
- `--confirm`: Confirm deletion (required to prevent accidental deletion)

**Example:**

```bash
# Delete a database (requires confirmation flag)
./scripts/pathrag-reset-db --db-name my_knowledge_base --confirm
```

## Ingestion and Query Tools

### pathrag-ingest

Ingests documentation into HADES-PathRAG.

```bash
./scripts/pathrag-ingest [options]
```

**Options:**

- `--docs-dir DOCS_DIR`: Directory containing documentation files (required)
- `--dataset-name DATASET_NAME`: Optional name for the dataset
- `--db-name DB_NAME`: Name of the ArangoDB database (default: "pathrag_docs")
- `--db-mode {create,append}`: Database mode (default: "append")
  - `create`: Initializes collections and graph structure
  - `append`: Adds to existing collections
- `--force`: Force recreation of collections even if they exist
- `--output-file OUTPUT_FILE`: Optional file to save ingestion statistics

**Example:**

```bash
# Ingest documentation in append mode (default)
./scripts/pathrag-ingest --docs-dir ./docs --dataset-name "project_docs"

# Ingest with clean database (recreate collections)
./scripts/pathrag-ingest --docs-dir ./docs --db-mode create --force
```

### pathrag-query

Queries the HADES-PathRAG knowledge graph.

```bash
./scripts/pathrag-query [options]
```

**Options:**

- `--query QUERY`: Natural language query (required)
- `--db-name DB_NAME`: Name of the ArangoDB database (default: "pathrag_docs")
- `--top-k TOP_K`: Number of results to return (default: 5)
- `--collection COLLECTION`: Optional specific collection to query
- `--model MODEL`: Model for embedding generation (default: "mirth/chonky_modernbert_large_1")
- `--output-file OUTPUT_FILE`: Optional file to save query results

**Example:**

```bash
# Simple query
./scripts/pathrag-query --query "How does the ingestion pipeline work?"

# Query with specific parameters
./scripts/pathrag-query --query "What is PathRAG?" --db-name my_knowledge_base --top-k 10
```

## Collection Management Modes

The ingestion tool supports two modes for managing ArangoDB collections:

1. **Create Mode** (`--db-mode=create`):
   - Initializes collections and graph structure
   - When used with `--force`, drops existing collections and recreates them
   - Useful for starting with a clean slate or initial setup

2. **Append Mode** (`--db-mode=append`):
   - Adds documents to existing collections (default behavior)
   - Does not modify existing collection structure
   - Useful for incremental updates to the knowledge base

## Implementation Details

The CLI tools are implemented as thin wrappers around core functionality in the `src/cli/` modules:

- `src/cli/admin.py`: Database administration functions
- `src/cli/ingest.py`: Documentation ingestion functions
- `src/cli/query.py`: Knowledge graph query functions

These modules can also be imported and used programmatically in Python code.
