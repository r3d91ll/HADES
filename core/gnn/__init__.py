"""
GraphSAGE GNN Module for HADES
===============================

WHAT: Graph Neural Network for inductive node embeddings on code + memory graph
WHERE: core/gnn/ - graph learning subsystem
WHO: Retrieval system requiring learned node representations
TIME: Inference <50ms per batch, training 1-2 hours

Architecture:
- Input: Jina v4 embeddings (2048-dim) for rich semantic space
- Multi-relational GraphSAGE for heterogeneous graphs
- Hidden: 1024-dim, Output: 512-dim for fast retrieval
- Inductive learning: works on unseen nodes without retraining
- Edge types: imports, contains, references, relates_to, derives_from

Pipeline Integration:
```
query → GraphSAGE (find relevant nodes) → PathRAG (prune to budget) → context
```

GraphSAGE finds semantically relevant nodes across entire graph (may be many).
PathRAG enforces structural constraints (H≤2, F≤3, B≤8) to control fanout.

Boundary Notes:
- Training: RO socket for graph export
- Inference: <50ms latency target for retrieval integration
- Checkpoints stored in models/ directory
"""

from .graphsage_model import MultiRelationalGraphSAGE  # noqa: F401
from .inference import GraphSAGEInference  # noqa: F401

__all__ = [
    "MultiRelationalGraphSAGE",
    "GraphSAGEInference",
]
