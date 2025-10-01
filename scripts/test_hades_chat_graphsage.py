#!/usr/bin/env python3
"""
Quick test: Load GraphSAGE in hades_chat context without full model load.
"""

from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.database.arango.memory_client import ArangoMemoryClient, resolve_memory_config
from core.embedders.embedders_jina import JinaV4Embedder
from core.gnn.inference import GraphSAGEInference
from core.runtime.memory.pathrag_code import CodePathRAG, PathRAGCaps

print("=" * 80)
print("Testing GraphSAGE + PathRAG integration in chat context")
print("=" * 80)

# Initialize ArangoDB client
print("\n[1/5] Initializing ArangoDB client...")
cfg = resolve_memory_config()
client = ArangoMemoryClient(cfg)
print("✓ Client ready")

# Initialize Jina embedder
print("\n[2/5] Loading Jina v4 embedder...")
embedder = JinaV4Embedder()
print("✓ Embedder ready")

# Load GraphSAGE
print("\n[3/5] Loading GraphSAGE model...")
checkpoint_path = ROOT / "models" / "checkpoints" / "best.pt"
graph_path = ROOT / "models" / "graph_data.pt"

if not checkpoint_path.exists():
    print(f"✗ Checkpoint not found: {checkpoint_path}")
    sys.exit(1)

if not graph_path.exists():
    print(f"✗ Graph data not found: {graph_path}")
    sys.exit(1)

# Load graph data
graph_data = torch.load(graph_path, weights_only=False)

# Create inference
graphsage_inference = GraphSAGEInference.from_checkpoint(
    checkpoint_path=checkpoint_path,
    device="cpu",
)

# Precompute embeddings
node_embeddings = graphsage_inference.precompute_embeddings(
    x=graph_data["data"].x,
    edge_index_dict=graph_data["data"].edge_index_dict,
)

# Update inference with precomputed embeddings
node_ids = [graph_data["node_id_map"][i] for i in range(len(node_embeddings))]
graphsage_inference.node_embeddings = node_embeddings
graphsage_inference.node_ids = node_ids
graphsage_inference.node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

print(f"✓ GraphSAGE ready with {len(node_ids)} precomputed embeddings")

# Initialize PathRAG with GraphSAGE
print("\n[4/5] Initializing PathRAG with GraphSAGE...")
caps = PathRAGCaps(hops=2, fanout=3, beam=8)
pathrag = CodePathRAG(client=client, caps=caps, graphsage_inference=graphsage_inference)
print(f"✓ PathRAG ready (H={caps.hops}, F={caps.fanout}, B={caps.beam})")

# Test retrieval
print("\n[5/5] Testing retrieval...")
test_query = "How does the Jina embedder work?"
query_emb = embedder.embed_single(test_query, task="retrieval", prompt_name="query")

result = pathrag.retrieve(
    query=test_query,
    query_embedding=query_emb.tolist(),
    k_seeds=6,
)

print(f"\nQuery: {test_query}")
print(f"Results: {len(result.paths)} paths, {result.total_nodes} nodes, {result.latency_ms:.1f}ms")
print(f"Reasons: {', '.join(result.reasons)}")

if "graphsage_candidates" in result.reasons:
    print("✓ GraphSAGE is actively contributing to retrieval")
else:
    print("✗ GraphSAGE not used (fell back to vector search)")

print("\n" + "=" * 80)
print("Integration test complete!")
print("=" * 80)
