#!/usr/bin/env python3
"""
Quick test of PathRAG integration in hades_chat.py

This script tests just the PathRAG retrieval part without loading the full model.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.database.arango.memory_client import (
    ArangoMemoryClient,
    resolve_memory_config,
)
from core.embedders.embedders_jina import JinaV4Embedder
from core.runtime.memory.pathrag_code import CodePathRAG, PathRAGCaps


def test_pathrag_integration():
    """Test PathRAG initialization and retrieval as used in hades_chat.py"""

    print("=" * 80)
    print("Testing PathRAG Integration for hades_chat.py")
    print("=" * 80)

    # Initialize PathRAG (mimics hades_chat.py setup)
    print("\n[1/3] Initializing PathRAG...")
    cfg = resolve_memory_config()
    pathrag_client = ArangoMemoryClient(cfg)
    caps = PathRAGCaps(hops=2, fanout=3, beam=8)
    pathrag = CodePathRAG(client=pathrag_client, caps=caps)
    print(f"✓ PathRAG ready: H={caps.hops}, F={caps.fanout}, B={caps.beam}")

    # Initialize embedder
    print("\n[2/3] Loading Jina v4 embedder...")
    embedder = JinaV4Embedder()
    print("✓ Embedder loaded")

    # Test retrieval with a sample query
    print("\n[3/3] Testing retrieval with sample query...")
    test_queries = [
        "How does the Jina embedder work?",
        "What is PathRAG?",
        "How does ArangoDB connect?"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Query {i}: {query}")

        # Generate query embedding
        query_emb = embedder.embed_single(query, task="retrieval", prompt_name="query")

        # Run PathRAG retrieval
        result = pathrag.retrieve(
            query=query,
            query_embedding=query_emb.tolist(),
            k_seeds=4
        )

        # Format results (mimics hades_chat.py formatting)
        if result.paths:
            unique_docs = result.get_unique_documents()[:5]
            print(f"✓ Retrieved {len(unique_docs)} documents in {result.latency_ms:.0f}ms")

            for j, doc in enumerate(unique_docs[:3], 1):  # Show top 3
                preview = doc.content[:100].replace('\n', ' ')
                print(f"  [{j}] {doc.path}")
                print(f"      {preview}...")
        else:
            print("✗ No documents found")

    print("\n" + "=" * 80)
    print("PathRAG Integration Test Complete!")
    print("=" * 80)
    print("\nNext: Run actual chat session with:")
    print("  poetry run python scripts/hades_chat.py --session demo --retrieve")


if __name__ == "__main__":
    try:
        test_pathrag_integration()
    except Exception as exc:
        print(f"\n✗ Test failed: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
