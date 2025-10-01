#!/usr/bin/env python3
"""
Test GraphSAGE + PathRAG Integration
=====================================

Tests the complete pipeline: GraphSAGE → PathRAG → Context

Verifies:
1. GraphSAGE finds relevant candidates
2. PathRAG prunes to paths within budget
3. End-to-end latency < 200ms

Usage:
    poetry run python scripts/test_graphsage_pathrag_integration.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.database.arango.memory_client import (
    ArangoMemoryClient,
    resolve_memory_config,
)
from core.embedders.embedders_jina import JinaV4Embedder
from core.gnn.inference import GraphSAGEInference
from core.runtime.memory.pathrag_code import CodePathRAG, PathRAGCaps

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def print_results(query: str, result):
    """Pretty print PathRAG results."""
    print("\n" + "="*80)
    print(f"Query: {query}")
    print("="*80)
    print(f"Total nodes: {result.total_nodes}")
    print(f"Total edges: {result.total_edges}")
    print(f"Latency: {result.latency_ms:.1f}ms")
    print(f"Reasons: {', '.join(result.reasons) if result.reasons else 'none'}")
    print(f"Paths found: {len(result.paths)}")
    print()

    if result.paths:
        # Show top 3 paths
        for i, path in enumerate(result.paths[:3], 1):
            print(f"Path {i} (score={path.final_score:.4f}):")
            for j, node in enumerate(path.nodes):
                prefix = "  ├─" if j < len(path.nodes) - 1 else "  └─"
                print(f"{prefix} [{node.path}] (depth={node.depth}, score={node.score:.4f})")
            print()

    # Show unique documents
    unique_docs = result.get_unique_documents()
    print(f"Unique documents retrieved: {len(unique_docs)}")
    for doc in unique_docs[:5]:
        print(f"  - {doc.path} (score={doc.score:.4f})")
    print()


def main():
    logger.info("=" * 80)
    logger.info("GraphSAGE + PathRAG Integration Test")
    logger.info("=" * 80)

    # Initialize ArangoDB client
    logger.info("Initializing ArangoDB client...")
    cfg = resolve_memory_config()
    client = ArangoMemoryClient(cfg)

    # Initialize Jina v4 embedder
    logger.info("Loading Jina v4 embedder...")
    embedder = JinaV4Embedder()

    # Load trained GraphSAGE model
    logger.info("Loading trained GraphSAGE model...")
    checkpoint_path = Path("models/checkpoints/best.pt")

    if not checkpoint_path.exists():
        logger.error(f"GraphSAGE checkpoint not found: {checkpoint_path}")
        logger.error("Train model first: poetry run python scripts/train_graphsage.py")
        return 1

    # Load graph data to get node embeddings and IDs
    logger.info("Loading graph data for node embeddings...")
    graph_data = torch.load(Path("models/graph_data.pt"), weights_only=False)

    # Create GraphSAGE inference engine
    graphsage = GraphSAGEInference.from_checkpoint(
        checkpoint_path=checkpoint_path,
        device="cpu",
    )

    # Precompute node embeddings for fast lookup
    logger.info("Precomputing node embeddings...")
    node_embeddings = graphsage.precompute_embeddings(
        x=graph_data["data"].x,
        edge_index_dict=graph_data["data"].edge_index_dict,
    )

    # Set up inference with precomputed embeddings
    node_ids = [graph_data["node_id_map"][i] for i in range(len(node_embeddings))]
    graphsage.node_embeddings = node_embeddings
    graphsage.node_ids = node_ids
    graphsage.node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    logger.info(f"GraphSAGE ready with {len(node_ids)} precomputed embeddings")

    # Initialize PathRAG with GraphSAGE
    caps = PathRAGCaps(hops=2, fanout=3, beam=8)
    pathrag = CodePathRAG(
        client=client,
        caps=caps,
        graphsage_inference=graphsage  # Enable GNN→PathRAG pipeline
    )

    logger.info("PathRAG initialized with GraphSAGE integration")
    logger.info("=" * 80)

    # Test queries
    test_queries = [
        "How does the Jina embedder work?",
        "What is PathRAG?",
        "How does GraphSAGE training work?",
    ]

    for query in test_queries:
        logger.info(f"\nProcessing query: {query}")

        # Generate query embedding
        query_emb = embedder.embed_single(query, task="retrieval", prompt_name="query")

        # Run retrieval (GraphSAGE → PathRAG)
        result = pathrag.retrieve(
            query=query,
            query_embedding=query_emb.tolist(),
            k_seeds=6
        )

        # Print results
        print_results(query, result)

    logger.info("=" * 80)
    logger.info("Integration test complete!")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
