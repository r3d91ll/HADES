"""
Utility functions for working with graph data structures in PyTorch Geometric.

This module provides utility functions for converting document collections to 
PyTorch Geometric data structures, managing neighborhood sampling, and 
performing other graph operations needed for ISNE.
"""

from typing import Dict, List, Optional, Union, Any, Tuple, Set
import torch
import numpy as np
from torch import Tensor
import logging

# Set up logging
logger = logging.getLogger(__name__)

try:
    import torch_geometric
    from torch_geometric.data import Data as GraphData
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch Geometric not available. Install with: pip install torch-geometric")
    TORCH_GEOMETRIC_AVAILABLE = False
    # Define a placeholder class for type checking
    class _PlaceholderGraphData:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch Geometric is not installed")
    
    # Use the placeholder for type hints
    GraphData = _PlaceholderGraphData


def check_torch_geometric() -> None:
    """
    Check if PyTorch Geometric is available and raise an error if not.
    
    Raises:
        ImportError: If PyTorch Geometric is not installed
    """
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise ImportError(
            "PyTorch Geometric is required but not installed. "
            "Install with: pip install torch-geometric"
        )


def documents_to_graph_data(
    documents: List[Dict[str, Any]],
    embedding_field: str = "embedding",
    id_field: str = "id",
    edge_index: Optional[Tensor] = None,
    edge_attr: Optional[Tensor] = None
) -> GraphData:
    """
    Convert a collection of documents with embeddings to PyTorch Geometric graph data.
    
    Args:
        documents: List of document dictionaries with embeddings
        embedding_field: Name of the field containing embeddings
        id_field: Name of the field containing document IDs
        edge_index: Optional pre-constructed edge indices [2, num_edges]
        edge_attr: Optional edge attributes [num_edges, num_features]
        
    Returns:
        PyTorch Geometric graph data object
    """
    check_torch_geometric()
    
    # Extract document IDs and create ID to index mapping
    doc_ids = [doc.get(id_field, f"doc_{i}") for i, doc in enumerate(documents)]
    id_to_index = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    
    # Extract node features (embeddings)
    node_features = []
    for doc in documents:
        embedding = doc.get(embedding_field)
        if embedding is None:
            raise ValueError(f"Document is missing embedding field: {embedding_field}")
        
        # Convert embedding to tensor if it's not already
        if isinstance(embedding, list):
            embedding = torch.tensor(embedding, dtype=torch.float)
        elif isinstance(embedding, np.ndarray):
            embedding = torch.from_numpy(embedding).float()
        
        node_features.append(embedding)
    
    # Stack embeddings into a single tensor
    x = torch.stack(node_features)
    
    # Create graph data object
    graph_data = GraphData(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=len(documents)
    )
    
    # Add metadata to graph data
    graph_data.doc_ids = doc_ids
    graph_data.id_to_index = id_to_index
    
    return graph_data


def create_edge_index_from_relations(
    relations: List[Dict[str, Any]],
    id_to_index: Dict[str, int],
    source_field: str = "source_id",
    target_field: str = "target_id"
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Create edge index tensor from a list of document relations.
    
    Args:
        relations: List of relation dictionaries
        id_to_index: Mapping from document IDs to indices
        source_field: Field name for source document ID
        target_field: Field name for target document ID
        
    Returns:
        Tuple containing:
            - Edge index tensor [2, num_edges]
            - Edge attributes tensor [num_edges, num_features] (if available)
    """
    check_torch_geometric()
    
    # Create lists to store source and target indices
    sources = []
    targets = []
    weights = []
    
    for relation in relations:
        source_id = relation.get(source_field)
        target_id = relation.get(target_field)
        
        # Skip relations with invalid IDs
        if source_id not in id_to_index or target_id not in id_to_index:
            continue
            
        # Convert document IDs to indices
        source_idx = id_to_index[source_id]
        target_idx = id_to_index[target_id]
        
        # Add edge in both directions (undirected graph)
        sources.append(source_idx)
        targets.append(target_idx)
        
        # If relation has weight, add it to weights list
        if "weight" in relation:
            weights.append(relation["weight"])
    
    # Create edge index tensor
    if sources and targets:
        edge_index = torch.tensor([sources + targets, targets + sources], dtype=torch.long)
        
        # Create edge attribute tensor if weights are available
        edge_attr = None
        if weights:
            weights = weights + weights  # Duplicate weights for bidirectional edges
            edge_attr = torch.tensor(weights, dtype=torch.float).view(-1, 1)
            
        return edge_index, edge_attr
    else:
        # Return empty tensors if no relations found
        return torch.zeros((2, 0), dtype=torch.long), None


def sample_neighbors(
    node_idx: Union[int, Tensor],
    edge_index: Tensor,
    num_samples: int,
    replace: bool = False
) -> Tensor:
    """
    Sample neighbors for a given node or batch of nodes.
    
    Args:
        node_idx: Node index or tensor of indices
        edge_index: Edge index tensor [2, num_edges]
        num_samples: Number of neighbors to sample
        replace: Whether to sample with replacement
        
    Returns:
        Tensor of sampled neighbor indices
    """
    check_torch_geometric()
    
    # Convert single index to tensor
    if isinstance(node_idx, int):
        node_idx = torch.tensor([node_idx])
    
    # Find all neighbors
    neighbors = edge_index[1][edge_index[0] == node_idx]
    
    # If no neighbors, return empty tensor
    if len(neighbors) == 0:
        return torch.tensor([], dtype=torch.long)
    
    # Sample neighbors
    if len(neighbors) <= num_samples:
        return neighbors
    else:
        perm = torch.randperm(len(neighbors))
        return neighbors[perm[:num_samples]]


def create_graph_from_documents(docs: List[Dict[str, Any]]) -> Tuple[Optional[GraphData], List[Dict[str, Any]], Dict[str, int]]:
    """
    Create a graph from documents for ISNE processing.
    
    This function creates a graph representation from a collection of documents,
    extracting embeddings and building edges based on document relationships.
    
    Args:
        docs: List of processed documents with embeddings
        
    Returns:
        Tuple containing:
            - PyTorch Geometric graph data object (or None if creation fails)
            - List of node metadata dictionaries
            - Mapping from chunk IDs to node indices
    """
    check_torch_geometric()
    
    # Extract embeddings and metadata
    node_embeddings = []
    node_metadata = []
    node_model_types = []  # Track embedding model types
    edge_index_src = []
    edge_index_dst = []
    edge_attr = []
    
    # Node index mapping
    node_idx_map = {}
    current_idx = 0
    
    # First pass: collect all nodes
    for doc in docs:
        if "chunks" not in doc or not doc["chunks"]:
            continue
            
        for chunk_idx, chunk in enumerate(doc["chunks"]):
            # Skip chunks without embeddings
            if "embedding" not in chunk or chunk["embedding"] is None:
                continue
                
            # Create a unique ID for this chunk
            chunk_id = f"{doc.get('file_id', f'doc_{id(doc)}')}_{ chunk_idx}"
            
            # Store the node index mapping
            node_idx_map[chunk_id] = current_idx
            current_idx += 1
            
            # Add the embedding and metadata
            node_embeddings.append(chunk["embedding"])
            
            # Store embedding model information for ISNE awareness
            embedding_model = chunk.get("embedding_model", "unknown")
            embedding_type = chunk.get("embedding_type", "unknown")
            
            node_model_types.append(embedding_type)
            
            node_metadata.append({
                "file_id": doc.get("file_id", f"doc_{id(doc)}"),
                "chunk_id": chunk_id,
                "text": chunk.get("text", ""),
                "embedding_model": embedding_model,
                "embedding_type": embedding_type,
                "metadata": chunk.get("metadata", {})
            })
    
    # Second pass: create edges
    for doc in docs:
        if "chunks" not in doc or len(doc["chunks"]) < 2:
            continue
            
        # Create sequential edges between chunks in the same document
        for i in range(len(doc["chunks"]) - 1):
            src_id = f"{doc.get('file_id', f'doc_{id(doc)}')}_{ i}"
            dst_id = f"{doc.get('file_id', f'doc_{id(doc)}')}_{ i+1}"
            
            if src_id in node_idx_map and dst_id in node_idx_map:
                src_idx = node_idx_map[src_id]
                dst_idx = node_idx_map[dst_id]
                
                # Check if embeddings are from the same model type
                src_type = node_model_types[src_idx] if src_idx < len(node_model_types) else "unknown"
                dst_type = node_model_types[dst_idx] if dst_idx < len(node_model_types) else "unknown"
                
                # Adjust edge weight based on embedding model compatibility
                # Same model type gets full weight, different types get reduced weight
                edge_weight = 1.0 if src_type == dst_type else 0.7
                
                # Add sequential edge
                edge_index_src.append(src_idx)
                edge_index_dst.append(dst_idx)
                edge_attr.append([edge_weight])  # Sequential relationship with model-aware weight
                
                # Add reverse edge for undirected graph
                edge_index_src.append(dst_idx)
                edge_index_dst.append(src_idx)
                edge_attr.append([edge_weight])
        
        # Process code-specific relationships if this is a Python file
        # These relationships come from metadata produced by code chunkers
        if doc.get("file_name", "").endswith(".py"):
            for chunk_idx, chunk in enumerate(doc["chunks"]):
                # Skip chunks without metadata or references
                if "metadata" not in chunk or "references" not in chunk["metadata"]:
                    continue
                
                # Get source ID
                src_id = f"{doc.get('file_id', f'doc_{id(doc)}')}_{ chunk_idx}"
                if src_id not in node_idx_map:
                    continue
                
                src_idx = node_idx_map[src_id]
                
                # Process each reference (relationship) in the chunk metadata
                for reference in chunk["metadata"]["references"]:
                    target_chunk_id = reference.get("target")
                    if not target_chunk_id or target_chunk_id not in node_idx_map:
                        continue
                    
                    dst_idx = node_idx_map[target_chunk_id]
                    weight = reference.get("weight", 0.9)  # Code relationships get high weight
                    
                    # Add directed edge for code relationship
                    edge_index_src.append(src_idx)
                    edge_index_dst.append(dst_idx)
                    edge_attr.append([weight])
    
    # Create the PyTorch Geometric data object
    if not node_embeddings:
        logger.warning("No valid embeddings found in documents")
        return None, [], {}
        
    try:
        node_embeddings_tensor = torch.tensor(node_embeddings, dtype=torch.float)
        edge_index = torch.tensor([edge_index_src, edge_index_dst], dtype=torch.long)
        edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float) if edge_attr else None
        
        graph = GraphData(
            x=node_embeddings_tensor,
            edge_index=edge_index,
            edge_attr=edge_attr_tensor
        )
        
        logger.info(f"Created graph with {graph.num_nodes} nodes and {graph.num_edges} edges")
        
        return graph, node_metadata, node_idx_map
    except Exception as e:
        logger.error(f"Error creating graph: {e}")
        return None, [], {}


def compute_similarity_matrix(embeddings: Tensor, threshold: Optional[float] = None) -> Tensor:
    """
    Compute cosine similarity matrix between embeddings.
    
    Args:
        embeddings: Tensor of node embeddings [num_nodes, embedding_dim]
        threshold: Optional threshold to zero out low similarities
        
    Returns:
        Similarity matrix [num_nodes, num_nodes]
    """
    # Normalize embeddings
    norm = torch.norm(embeddings, p=2, dim=1, keepdim=True)
    normalized_embeddings = embeddings / norm.clamp(min=1e-10)
    
    # Compute similarity matrix
    similarity = torch.mm(normalized_embeddings, normalized_embeddings.t())
    
    # Apply threshold if provided
    if threshold is not None:
        similarity = torch.where(similarity >= threshold, similarity, torch.zeros_like(similarity))
    
    return similarity
