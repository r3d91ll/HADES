# JSON Structure Mockup - Directory Tree Format with Dual-Level Keywords

## Directory Structure Representation

The JSON structure mirrors the actual file system hierarchy, with each level representing a directory or file in the tree.

### 1. Project Root Structure

```json
{
  "project": "HADES",
  "root_path": "/workspace/HADES",
  "structure": {
    "src": {
      "path": "/workspace/HADES/src",
      "components": {
        "path": "/workspace/HADES/src/components",
        "rag_strategy": {
          "path": "/workspace/HADES/src/components/rag_strategy",
          "pathrag": {
            "path": "/workspace/HADES/src/components/rag_strategy/pathrag",
            "files": {
              "pathrag_rag_strategy.py": {
                "full_path": "/workspace/HADES/src/components/rag_strategy/pathrag/pathrag_rag_strategy.py",
                "document_id": "HADES_src_components_rag_strategy_pathrag_pathrag_rag_strategy.py",
                "document_data": {
                  // Full document structure here
                }
              }
            }
          }
        },
        "graph_engine": {
          "path": "/workspace/HADES/src/components/graph_engine",
          "pathrag": {
            "path": "/workspace/HADES/src/components/graph_engine/pathrag",
            "files": {
              "pathrag_graph_engine.py": {
                "full_path": "/workspace/HADES/src/components/graph_engine/pathrag/pathrag_graph_engine.py",
                "document_id": "HADES_src_components_graph_engine_pathrag_pathrag_graph_engine.py"
              }
            }
          }
        }
      },
      "storage": {
        "path": "/workspace/HADES/src/storage",
        "files": {
          "vector_store.py": {
            "full_path": "/workspace/HADES/src/storage/vector_store.py",
            "document_id": "HADES_src_storage_vector_store.py"
          }
        }
      },
      "api": {
        "path": "/workspace/HADES/src/api",
        "engines": {
          "path": "/workspace/HADES/src/api/engines",
          "files": {
            "pathrag_engine.py": {
              "full_path": "/workspace/HADES/src/api/engines/pathrag_engine.py",
              "document_id": "HADES_src_api_engines_pathrag_engine.py"
            }
          }
        }
      }
    }
  }
}
```

### 2. Individual Document Structure (After DocProc + Document Keywords)

```json
{
  "full_path": "/workspace/HADES/src/components/rag_strategy/pathrag/pathrag_rag_strategy.py",
  "document_id": "HADES_src_components_rag_strategy_pathrag_pathrag_rag_strategy.py",
  "path_components": {
    "project": "HADES",
    "directories": ["src", "components", "rag_strategy", "pathrag"],
    "filename": "pathrag_rag_strategy.py",
    "relative_path": "src/components/rag_strategy/pathrag/pathrag_rag_strategy.py"
  },
  "document_metadata": {
    "type": "python",
    "full_path": "/workspace/HADES/src/components/rag_strategy/pathrag/pathrag_rag_strategy.py",
    "size": 12453,
    "last_modified": "2024-01-15T10:30:00Z",
    "language": "python",
    "encoding": "utf-8"
  },
  "document_keywords": [
    "pathrag",
    "retrieval",
    "graph traversal",
    "knowledge graph",
    "semantic search",
    "rag strategy",
    "path construction",
    "query processing",
    "arangodb",
    "vector similarity"
  ],
  "document_summary": "PathRAG implementation combining vector search with graph traversal for enhanced retrieval",
  "content": "\"\"\"PathRAG Strategy Implementation...\"\"\"\n\nfrom typing import List, Dict...\n\nclass PathRAGProcessor:..."
}
```

### 3. After Chunking + Chunk Keywords (With Path Context)

```json
{
  "full_path": "/workspace/HADES/src/components/rag_strategy/pathrag/pathrag_rag_strategy.py",
  "document_id": "HADES_src_components_rag_strategy_pathrag_pathrag_rag_strategy.py",
  "path_components": {
    "project": "HADES",
    "directories": ["src", "components", "rag_strategy", "pathrag"],
    "filename": "pathrag_rag_strategy.py",
    "relative_path": "src/components/rag_strategy/pathrag/pathrag_rag_strategy.py"
  },
  "document_metadata": {
    "type": "python",
    "full_path": "/workspace/HADES/src/components/rag_strategy/pathrag/pathrag_rag_strategy.py",
    "size": 12453,
    "last_modified": "2024-01-15T10:30:00Z",
    "language": "python",
    "encoding": "utf-8"
  },
  "document_keywords": [
    "pathrag",
    "retrieval",
    "graph traversal",
    "knowledge graph",
    "semantic search",
    "rag strategy",
    "path construction",
    "query processing",
    "arangodb",
    "vector similarity"
  ],
  "document_summary": "PathRAG implementation combining vector search with graph traversal for enhanced retrieval",
  "chunks": [
    {
      "chunk_id": "HADES_src_components_rag_strategy_pathrag_pathrag_rag_strategy.py_chunk_0",
      "chunk_path": "/workspace/HADES/src/components/rag_strategy/pathrag/pathrag_rag_strategy.py#chunk_0",
      "chunk_index": 0,
      "content": "class PathRAGProcessor(RAGStrategy):\n    \"\"\"PathRAG implementation for graph-enhanced retrieval.\"\"\"\n    \n    def __init__(self, graph_client: ArangoClient, vector_store: VectorStore):\n        self.graph = graph_client\n        self.vectors = vector_store\n        self.path_builder = PathConstructor()",
      "chunk_keywords": [
        "class definition",
        "PathRAGProcessor",
        "initialization",
        "graph client",
        "vector store",
        "path constructor"
      ],
      "chunk_metadata": {
        "file_path": "/workspace/HADES/src/components/rag_strategy/pathrag/pathrag_rag_strategy.py",
        "start_line": 45,
        "end_line": 52,
        "ast_type": "class_definition",
        "ast_name": "PathRAGProcessor",
        "indentation_level": 0,
        "contains_docstring": true
      }
    },
    {
      "chunk_id": "HADES_src_components_rag_strategy_pathrag_pathrag_rag_strategy.py_chunk_1",
      "chunk_path": "/workspace/HADES/src/components/rag_strategy/pathrag/pathrag_rag_strategy.py#chunk_1",
      "chunk_index": 1,
      "content": "def query(self, query_text: str, top_k: int = 10) -> List[QueryResult]:\n    \"\"\"Execute PathRAG query combining vector and graph search.\"\"\"\n    # Get initial vector matches\n    vector_results = self.vectors.similarity_search(query_text, k=top_k*2)\n    \n    # Build paths from vector matches\n    paths = self.path_builder.construct_paths(vector_results)\n    \n    # Score and rank paths\n    scored_paths = self._score_paths(paths, query_text)",
      "chunk_keywords": [
        "query method",
        "vector search",
        "similarity search",
        "path construction",
        "path scoring",
        "ranking"
      ],
      "chunk_metadata": {
        "file_path": "/workspace/HADES/src/components/rag_strategy/pathrag/pathrag_rag_strategy.py",
        "start_line": 75,
        "end_line": 85,
        "ast_type": "method_definition",
        "ast_name": "query",
        "parent_class": "PathRAGProcessor",
        "indentation_level": 1,
        "parameters": ["query_text", "top_k"]
      }
    }
  ],
  "chunk_relationships": {
    "sequential": [
      ["chunk_0", "chunk_1"]
    ],
    "ast_parent_child": [
      ["chunk_0", "chunk_1"]
    ],
    "file_hierarchy": {
      "parent_directory": "/workspace/HADES/src/components/rag_strategy/pathrag",
      "sibling_files": [],
      "imports_from": [
        "/workspace/HADES/src/pathrag_base.py",
        "/workspace/HADES/src/storage/vector_store.py"
      ]
    }
  }
}
```

### 4. Multi-File Aggregation Structure

```json
{
  "project_root": "/workspace/HADES",
  "index_timestamp": "2024-01-15T10:30:00Z",
  "directory_tree": {
    "/workspace/HADES": {
      "type": "directory",
      "src": {
        "type": "directory",
        "full_path": "/workspace/HADES/src",
        "components": {
          "type": "directory",
          "full_path": "/workspace/HADES/src/components",
          "rag_strategy": {
            "type": "directory",
            "full_path": "/workspace/HADES/src/components/rag_strategy",
            "pathrag": {
              "type": "directory",
              "full_path": "/workspace/HADES/src/components/rag_strategy/pathrag",
              "files": [
                {
                  "filename": "pathrag_rag_strategy.py",
                  "full_path": "/workspace/HADES/src/components/rag_strategy/pathrag/pathrag_rag_strategy.py",
                  "document_id": "HADES_src_components_rag_strategy_pathrag_pathrag_rag_strategy.py",
                  "summary": "PathRAG implementation combining vector search with graph traversal",
                  "keywords": ["pathrag", "retrieval", "graph traversal"],
                  "chunk_count": 15,
                  "relationships": {
                    "imports": [
                      "/workspace/HADES/src/pathrag_base.py",
                      "/workspace/HADES/src/storage/vector_store.py"
                    ],
                    "imported_by": [
                      "/workspace/HADES/src/api/engines/pathrag_engine.py"
                    ]
                  }
                }
              ]
            }
          }
        },
        "storage": {
          "type": "directory",
          "full_path": "/workspace/HADES/src/storage",
          "files": [
            {
              "filename": "vector_store.py",
              "full_path": "/workspace/HADES/src/storage/vector_store.py",
              "document_id": "HADES_src_storage_vector_store.py",
              "summary": "Vector storage implementation for embeddings",
              "keywords": ["vector", "storage", "embeddings"],
              "chunk_count": 8
            }
          ]
        }
      }
    }
  },
  "cross_directory_relationships": {
    "import_graph": {
      "nodes": [
        {
          "id": "HADES_src_components_rag_strategy_pathrag_pathrag_rag_strategy.py",
          "full_path": "/workspace/HADES/src/components/rag_strategy/pathrag/pathrag_rag_strategy.py",
          "directory": "/workspace/HADES/src/components/rag_strategy/pathrag"
        },
        {
          "id": "HADES_src_storage_vector_store.py",
          "full_path": "/workspace/HADES/src/storage/vector_store.py",
          "directory": "/workspace/HADES/src/storage"
        }
      ],
      "edges": [
        {
          "source": "HADES_src_components_rag_strategy_pathrag_pathrag_rag_strategy.py",
          "target": "HADES_src_storage_vector_store.py",
          "relationship": "imports",
          "import_statement": "from src.storage.vector_store import VectorStore"
        }
      ]
    }
  }
}
```

### 5. Query Response Structure with Path Context

```json
{
  "query": "How does PathRAG implement graph traversal?",
  "results": [
    {
      "chunk_id": "HADES_src_components_rag_strategy_pathrag_pathrag_rag_strategy.py_chunk_2",
      "full_path": "/workspace/HADES/src/components/rag_strategy/pathrag/pathrag_rag_strategy.py",
      "path_context": {
        "project": "HADES",
        "module_path": ["src", "components", "rag_strategy", "pathrag"],
        "file": "pathrag_rag_strategy.py",
        "chunk_location": "lines 120-129"
      },
      "content": "def _build_graph_context(self, node_id: str, max_hops: int = 2)...",
      "relevance_score": 0.92,
      "path_relevance": {
        "directory_keywords": ["rag_strategy", "pathrag"],
        "matches_query_intent": true
      },
      "related_files_in_directory": [
        {
          "filename": "pathrag_config.py",
          "full_path": "/workspace/HADES/src/components/rag_strategy/pathrag/pathrag_config.py",
          "relevance": "configuration for same component"
        }
      ],
      "cross_directory_references": [
        {
          "full_path": "/workspace/HADES/src/components/graph_engine/pathrag/pathrag_graph_engine.py",
          "relevance": "implements graph operations used here"
        }
      ]
    }
  ]
}
```

## Key Improvements

1. **Full Path Usage from Root**:
   - Every path starts from the filesystem root `/`
   - Complete directory hierarchy is represented in the JSON structure
   - Path components include all levels from root to file

2. **Tree Structure Representation**:
   - JSON structure exactly mirrors the filesystem hierarchy
   - Each nested object represents a directory level
   - The structure starts from `/` and builds down through each directory

3. **Path Context and Components**:
   - Files include both absolute paths and relative paths
   - Path hierarchy shows the complete tree structure
   - Depth tracking from both root and project levels

4. **Cross-Directory Relationships**:
   - Import relationships maintain full absolute paths
   - Directory depth information for understanding file relationships
   - Complete path context for every reference

5. **Query Results with Full Path Context**:
   - Results include complete filesystem paths
   - Directory hierarchy is preserved throughout
   - Easy navigation from any point to any other point in the tree

### Alternative Flat Representation with Path Arrays

For certain use cases, you might also want a flat representation that's easier to process:

```json
{
  "filesystem_index": {
    "root": "/",
    "files": [
      {
        "path_array": ["/", "home", "todd", "ML-Lab", "Olympus", "HADES", "src", "components", "rag_strategy", "pathrag", "pathrag_rag_strategy.py"],
        "full_path": "/home/todd/ML-Lab/Olympus/HADES/src/components/rag_strategy/pathrag/pathrag_rag_strategy.py",
        "depth": 11,
        "project_depth": 5,
        "document_id": "HADES_src_components_rag_strategy_pathrag_pathrag_rag_strategy.py",
        "type": "file",
        "extension": "py"
      },
      {
        "path_array": ["/", "home", "todd", "ML-Lab", "Olympus", "HADES", "src", "storage", "vector_store.py"],
        "full_path": "/home/todd/ML-Lab/Olympus/HADES/src/storage/vector_store.py",
        "depth": 9,
        "project_depth": 3,
        "document_id": "HADES_src_storage_vector_store.py",
        "type": "file",
        "extension": "py"
      }
    ],
    "directories": [
      {
        "path_array": ["/", "home", "todd", "ML-Lab", "Olympus", "HADES"],
        "full_path": "/home/todd/ML-Lab/Olympus/HADES",
        "depth": 6,
        "type": "project_root"
      }
    ]
  }
}
```

This structure makes it easy to:

- Navigate the complete filesystem hierarchy
- Understand the exact location of every file
- Process paths programmatically with the path_array
- Query based on any directory level
- Build accurate filesystem visualizations
