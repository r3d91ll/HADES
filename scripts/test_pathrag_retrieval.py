#!/usr/bin/env python3
"""
Test PathRAG Code Graph Retrieval
==================================

Verifies end-to-end PathRAG retrieval with heuristic v1 scoring.

Usage:
    poetry run python scripts/test_pathrag_retrieval.py
    poetry run python scripts/test_pathrag_retrieval.py --query "How does embedding work?"
    poetry run python scripts/test_pathrag_retrieval.py --beam 16 --hops 3
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.database.arango.memory_client import (
    ArangoMemoryClient,
    resolve_memory_config
)
from core.embedders.embedders_jina import JinaV4Embedder
from core.runtime.memory.pathrag_code import CodePathRAG, PathRAGCaps

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def print_path_results(result):
    """Pretty print PathRAG results."""
    print("\n" + "="*80)
    print(f"PathRAG Retrieval Results")
    print("="*80)
    print(f"Total nodes visited: {result.total_nodes}")
    print(f"Total edges traversed: {result.total_edges}")
    print(f"Latency: {result.latency_ms:.1f}ms")
    print(f"Reason codes: {', '.join(result.reasons) if result.reasons else 'none'}")
    print(f"Paths found: {len(result.paths)}")
    print()

    if not result.paths:
        print("No paths found!")
        return

    # Show top-k paths with details
    for i, path in enumerate(result.paths[:5], 1):
        print(f"\n--- Path {i} (score={path.final_score:.4f}) ---")

        # Show path nodes
        for j, node in enumerate(path.nodes):
            prefix = "├─" if j < len(path.nodes) - 1 else "└─"
            print(f"{prefix} [{node.path}] (depth={node.depth}, score={node.score:.4f})")

            # Show snippet of content
            content_preview = node.content[:100].replace('\n', ' ')
            print(f"   {content_preview}...")

        # Show edges
        if path.edges:
            print(f"\n   Edges: {len(path.edges)}")
            for edge in path.edges:
                edge_type = edge.get('type', 'unknown')
                weight = edge.get('weight', 0.0)
                print(f"     - {edge_type} (weight={weight:.2f})")

        # Show reasoning
        if path.reasoning:
            print(f"\n   Reasoning: {', '.join(path.reasoning)}")

    # Show unique documents
    print("\n" + "="*80)
    print("Unique Documents Retrieved")
    print("="*80)
    unique_docs = result.get_unique_documents()
    for doc in unique_docs[:10]:
        print(f"  - {doc.path} (score={doc.score:.4f})")

    if len(unique_docs) > 10:
        print(f"  ... and {len(unique_docs) - 10} more")


def run_test_queries(
    pathrag: CodePathRAG,
    embedder: JinaV4Embedder,
    queries: List[str],
    k_seeds: int = 6
):
    """Run test queries through PathRAG."""

    for query in queries:
        print("\n" + "="*80)
        print(f"QUERY: {query}")
        print("="*80)

        # Generate query embedding
        logger.info("Generating query embedding...")
        query_emb = embedder.embed_single(query, task="retrieval", prompt_name="query")

        # Run PathRAG retrieval
        logger.info("Running PathRAG retrieval...")
        result = pathrag.retrieve(
            query=query,
            query_embedding=query_emb.tolist(),
            k_seeds=k_seeds
        )

        # Display results
        print_path_results(result)


def main():
    parser = argparse.ArgumentParser(description="Test PathRAG code graph retrieval")
    parser.add_argument(
        "--query",
        type=str,
        help="Custom query (default: run predefined test queries)"
    )
    parser.add_argument(
        "--hops",
        type=int,
        default=2,
        help="Max traversal depth (default: 2)"
    )
    parser.add_argument(
        "--fanout",
        type=int,
        default=3,
        help="Max neighbors per node (default: 3)"
    )
    parser.add_argument(
        "--beam",
        type=int,
        default=8,
        help="Beam width (default: 8)"
    )
    parser.add_argument(
        "--k-seeds",
        type=int,
        default=6,
        help="Number of seed documents (default: 6)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=500,
        help="Timeout per stage in ms (default: 500)"
    )

    args = parser.parse_args()

    # Initialize ArangoDB client
    logger.info("Initializing ArangoDB client...")
    cfg = resolve_memory_config()
    client = ArangoMemoryClient(cfg)

    # Initialize Jina v4 embedder
    logger.info("Loading Jina v4 embedder...")
    embedder = JinaV4Embedder()

    # Initialize PathRAG with caps
    caps = PathRAGCaps(
        hops=args.hops,
        fanout=args.fanout,
        beam=args.beam,
        timeout_ms=args.timeout
    )

    logger.info(f"Initializing PathRAG with caps: H={caps.hops}, F={caps.fanout}, B={caps.beam}")
    pathrag = CodePathRAG(client=client, caps=caps)

    # Define test queries
    if args.query:
        queries = [args.query]
    else:
        queries = [
            "How does the Jina embedder work?",
            "How is the ArangoDB client initialized?",
            "What does the document extractor do?",
            "How does PathRAG beam search work?",
        ]

    # Run queries
    run_test_queries(pathrag, embedder, queries, k_seeds=args.k_seeds)

    logger.info("PathRAG testing complete!")


if __name__ == "__main__":
    main()
