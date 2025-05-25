# HADES-PathRAG Integration

This document outlines the integration of XnX notation with the BUPT-GAMMA PathRAG implementation for the HADES project.

## Overview

The HADES-PathRAG integration extends the original PathRAG system with XnX notation to enable:

1. **Weighted Path Tuning** - Control path traversal based on relationship strengths
2. **Directional Control** - Specify inbound/outbound relationship traversal 
3. **Access Control** - Enhanced security through distance-based access models
4. **Identity Assumption** - Allow entities to act on behalf of other entities
5. **ArangoDB Integration** - Graph database backend for scalable path storage

## Architecture

The integration consists of the following components:

### 1. XnX Core Components

- `XnXQueryParams` - Parameters for configuring XnX-enhanced path queries
- `XnXIdentityToken` - Token for identity assumption operations
- `XnXPathRAG` - Core class extending PathRAG with XnX capabilities

### 2. ArangoDB Adapter

- `ArangoPathRAGAdapter` - Database adapter connecting PathRAG to ArangoDB
- Supports weighted paths, directional filters, and temporal constraints

### 3. MCP Server Tools

- `mcp0_xnx_pathrag_retrieve` - Query paths with XnX parameters
- `mcp0_xnx_create_relationship` - Create weighted relationships
- `mcp0_xnx_assume_identity` - Generate identity assumption tokens
- `mcp0_xnx_verify_access` - Verify access permissions via XnX paths

## Directory Structure

```
HADES-PathRAG/
├── PathRAG/                  # Original BUPT-GAMMA PathRAG
├── old_hades_imports/        # Components imported from original HADES
│   ├── docs/                 # Documentation
│   │   └── xnx_notation.md   # XnX notation specification
│   ├── src/
│       ├── db/               # Database components
│       │   ├── arango_connection.py
│       │   └── database_setup.py
│       └── mcp/              # Original MCP server
├── src/
│   ├── xnx/                  # XnX-enhanced PathRAG
│   │   ├── __init__.py
│   │   ├── xnx_params.py     # XnX parameter definitions
│   │   ├── xnx_pathrag.py    # XnX-enhanced PathRAG implementation
│   │   └── arango_adapter.py # ArangoDB adapter
│   └── mcp/                  # New MCP tools
│       ├── __init__.py
│       └── xnx_tools.py      # XnX-enhanced MCP tools
└── README.md                 # Original PathRAG README
```

## Implementation Details

### XnX Notation

The XnX notation (`w x d`) has been implemented to enhance path traversal:

- `w` (Weight): Relationship strength (0.0 to 1.0)
- `x` (Node Identifier): Target node label
- `d` (Distance/Flow): Signed integer for direction (+1=inbound, -1=outbound)

### ArangoDB Adapter

The ArangoDB adapter replaces the default PathRAG storage with a graph database:

- Stores nodes with embeddings and metadata
- Creates edges with XnX properties (weight, direction, temporal bounds)
- Supports complex path queries with XnX constraints

### Identity Assumption

The identity assumption feature allows entities to act on behalf of others:

- Creates time-bound tokens with diminishing trust over time
- Access control uses effective weights to determine permissions
- Provides security through distance-based access model

### MCP Tools Integration

New MCP tools provide a bridge between the MCP server and XnX-enhanced PathRAG:

- Path retrieval with XnX filters
- Relationship creation with XnX notation
- Identity assumption for enhanced access control
- Access verification via weighted paths

## Getting Started

1. Set up ArangoDB instance
2. Install dependencies
3. Configure API keys for LLM models
4. Use the MCP tools to interact with the XnX-enhanced PathRAG

## Acknowledgments

This integration builds upon the PathRAG implementation by BUPT-GAMMA:

```
@article{chen2025pathrag,
  title={PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths},
  author={Chen, Boyu and Guo, Zirui and Yang, Zidan and Chen, Yuluo and Chen, Junze and Liu, Zhenghao and Shi, Chuan and Yang, Cheng},
  journal={arXiv preprint arXiv:2502.14902},
  year={2025}
}
```

## Next Steps

1. **Implementation Refinement**:
   - Optimize ArangoDB queries for performance
   - Add comprehensive testing for XnX path traversal

2. **Advanced Features**:
   - Implement GNN-based path weight optimization
   - Develop visualization tools for XnX-weighted paths

3. **Integration**:
   - Connect to existing HADES authentication system
   - Develop UI components for path visualization and management
