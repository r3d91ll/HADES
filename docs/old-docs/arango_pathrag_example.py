#!/usr/bin/env python3
"""
ArangoDB Integration Example for HADES-PathRAG

This script demonstrates how to use ArangoDB with the XnX PathRAG implementation,
creating and querying graph paths with weighted XnX notation.
"""

import os
import logging
import asyncio
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import PathRAG components
from src.storage.arango.connection import ArangoConnection
from src.xnx.arango_adapter import ArangoPathRAGAdapter

# Optional: Import Ollama for embeddings if needed
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama not available. Will use random embeddings for demonstration.")


def generate_sample_embedding(dim=1536):
    """Generate a random embedding vector for demonstration."""
    import numpy as np
    return list(np.random.rand(dim))


async def get_embedding(text, model="tinyllama"):
    """Get embedding from Ollama if available, otherwise use random."""
    if OLLAMA_AVAILABLE:
        client = ollama.AsyncClient(host="http://localhost:11434")
        response = await client.embeddings(model=model, prompt=text)
        return response["embedding"]
    else:
        return generate_sample_embedding()


async def main():
    """Run the ArangoDB integration example."""
    logger.info("Starting ArangoDB PathRAG integration example")
    
    # Connect to ArangoDB
    db_name = "pathrag_demo"  # Use a dedicated demo database
    connection = ArangoConnection(db_name=db_name)
    logger.info(f"Connected to ArangoDB database: {db_name}")
    
    # Initialize the PathRAG adapter
    adapter = ArangoPathRAGAdapter(
        arango_connection=connection,
        db_name=db_name,
        nodes_collection="demo_nodes",
        edges_collection="demo_edges",
        graph_name="demo_graph"
    )
    logger.info("Initialized PathRAG adapter")
    
    # Sample data
    nodes = [
        {
            "id": "node1", 
            "content": "This is the first node in our knowledge graph.",
            "metadata": {"domain": "general", "source": "example"}
        },
        {
            "id": "node2", 
            "content": "Python is a popular programming language for AI development.",
            "metadata": {"domain": "code", "source": "example"}
        },
        {
            "id": "node3", 
            "content": "ArangoDB is a multi-model database supporting graphs.",
            "metadata": {"domain": "database", "source": "example"}
        },
        {
            "id": "node4", 
            "content": "PathRAG enables weighted path-based retrieval augmented generation.",
            "metadata": {"domain": "ai", "source": "example"}
        },
    ]
    
    # Store nodes with embeddings
    logger.info("Storing sample nodes with embeddings")
    for node in nodes:
        embedding = await get_embedding(node["content"])
        node_id = adapter.store_node(
            node_id=node["id"],
            content=node["content"],
            embedding=embedding,
            metadata=node["metadata"]
        )
        logger.info(f"Stored node: {node_id}")
    
    # Create edges with XnX weights
    logger.info("Creating edges with XnX weights")
    edges = [
        {"from": "node1", "to": "node2", "weight": 0.8, "relation": "refers_to"},
        {"from": "node2", "to": "node3", "weight": 0.6, "relation": "uses"},
        {"from": "node3", "to": "node4", "weight": 0.9, "relation": "powers"},
        {"from": "node4", "to": "node1", "weight": 0.5, "relation": "mentions"},
    ]
    
    for edge in edges:
        edge_id = adapter.create_edge(
            from_node=edge["from"],
            to_node=edge["to"],
            weight=edge["weight"],
            metadata={"relation": edge["relation"]}
        )
        logger.info(f"Created edge: {edge_id}")
    
    # Demonstrate path retrieval
    logger.info("Demonstrating path retrieval")
    
    # Simple path retrieval - get all paths from node1
    paths = adapter.get_paths_from_node("node1", max_depth=2)
    logger.info(f"Found {len(paths)} paths from node1 (max depth 2)")
    for path in paths:
        path_str = " -> ".join([node.get("_key", node.get("_id", "unknown")).split("/")[-1] for node in path["nodes"]])
        logger.info(f"Path: {path_str} (Weight: {path['total_weight']:.2f})")
    
    # XnX weighted path - prioritize code domain
    xnx_query = "X(domain='code')2"  # XnX notation to boost code-related nodes
    weighted_paths = adapter.get_weighted_paths("node1", xnx_query, max_depth=3)
    logger.info(f"Found {len(weighted_paths)} weighted paths with XnX query: {xnx_query}")
    for path in weighted_paths:
        path_str = " -> ".join([node.get("_key", node.get("_id", "unknown")).split("/")[-1] for node in path["nodes"]])
        logger.info(f"Weighted Path: {path_str} (XnX Score: {path['xnx_score']:.2f})")
    
    # Semantic search - find nodes similar to a query
    query = "How does PathRAG work with databases?"
    query_embedding = await get_embedding(query)
    similar_nodes = adapter.find_similar_nodes(query_embedding, top_k=2)
    logger.info(f"Found {len(similar_nodes)} nodes similar to query: '{query}'")
    for node in similar_nodes:
        node_key = node.get("_key", "unknown")
        content = node.get("content", "No content")
        similarity = node.get("similarity", 0.0)
        logger.info(f"Similar node: {node_key} - '{content}' (Score: {similarity:.4f})")
    
    logger.info("ArangoDB PathRAG integration example completed")


if __name__ == "__main__":
    asyncio.run(main())
