#!/usr/bin/env python3
"""
Graph Builder - ArangoDB to PyTorch Geometric
==============================================

Exports code + memory graph from ArangoDB to PyG Data format for GraphSAGE training.

Collections:
- repo_docs: Code and text files with Jina v4 embeddings (2048-dim)
- code_edges: imports, contains, references edges
- directories: Directory nodes (future)
- memories: Observations, reflections, entities (future - Phase E)

Output: PyG Data object ready for GraphSAGE training
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from torch_geometric.data import Data
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    Data = None

from core.database.arango.memory_client import ArangoMemoryClient

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Build PyG Data object from ArangoDB graph.
    """

    def __init__(self, client: ArangoMemoryClient):
        """
        Initialize graph builder.

        Args:
            client: ArangoDB client for read-only access
        """
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError(
                "PyTorch Geometric not installed. "
                "Install with: poetry install --extras gnn"
            )

        self.client = client

    def build_graph(
        self,
        include_collections: Optional[List[str]] = None,
        edge_types: Optional[List[str]] = None,
    ) -> Tuple[Data, Dict[int, str]]:
        """
        Build PyG Data object from ArangoDB collections.

        Args:
            include_collections: Node collections to include (default: ["repo_docs"])
            edge_types: Edge types to include (default: all available)

        Returns:
            (data, node_id_map) where:
            - data: PyG Data object
            - node_id_map: Mapping from node index → ArangoDB _id
        """
        include_collections = include_collections or ["repo_docs"]

        logger.info(f"Building graph from collections: {include_collections}")

        # Step 1: Extract nodes
        nodes, node_id_map = self._extract_nodes(include_collections)

        # Step 2: Extract edges
        edge_index_dict, edge_attr_dict = self._extract_edges(node_id_map, edge_types)

        # Step 3: Create PyG Data object
        data = Data(
            x=nodes["features"],  # [num_nodes, 2048]
            node_metadata=nodes["metadata"],  # List of dicts
            edge_index_dict=edge_index_dict,  # Dict[edge_type, [2, num_edges]]
            edge_attr_dict=edge_attr_dict,  # Dict[edge_type, [num_edges, feat_dim]]
        )

        logger.info(
            f"Graph built: {data.x.shape[0]} nodes, "
            f"{sum(e.shape[1] for e in edge_index_dict.values())} edges"
        )

        return data, node_id_map

    def _extract_nodes(
        self,
        collections: List[str],
    ) -> Tuple[Dict, Dict[int, str]]:
        """
        Extract nodes from ArangoDB collections.

        Returns:
            (nodes_dict, node_id_map) where:
            - nodes_dict: {"features": Tensor, "metadata": List[dict]}
            - node_id_map: {node_idx: arango_id}
        """
        features_list = []
        metadata_list = []
        node_id_map = {}
        node_idx = 0

        for collection in collections:
            aql = f"""
            FOR doc IN {collection}
              FILTER doc.embedding != null
              RETURN {{
                _id: doc._id,
                _key: doc._key,
                path: doc.path,
                embedding: doc.embedding,
                metadata: doc.metadata
              }}
            """

            docs = self.client.execute_query(aql, {})

            for doc in docs:
                # Extract embedding
                embedding = doc.get("embedding")
                if not embedding or len(embedding) != 2048:
                    logger.warning(f"Skipping {doc['_id']}: invalid embedding (expected 2048-dim, got {len(embedding) if embedding else 0})")
                    continue

                features_list.append(embedding)

                # Store metadata
                metadata_list.append({
                    "arango_id": doc["_id"],
                    "key": doc["_key"],
                    "path": doc.get("path", ""),
                    "collection": collection,
                })

                # Map node index → ArangoDB ID
                node_id_map[node_idx] = doc["_id"]
                node_idx += 1

        if not features_list:
            raise ValueError("No nodes with embeddings found!")

        # Convert to tensor
        features_tensor = torch.tensor(features_list, dtype=torch.float32)

        logger.info(f"Extracted {len(features_list)} nodes from {collections}")

        return {
            "features": features_tensor,
            "metadata": metadata_list,
        }, node_id_map

    def _extract_edges(
        self,
        node_id_map: Dict[int, str],
        edge_types: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Extract edges from code_edges collection.

        Args:
            node_id_map: Mapping from node_idx → arango_id
            edge_types: Edge types to include (None = all)

        Returns:
            (edge_index_dict, edge_attr_dict) where:
            - edge_index_dict: {edge_type: [2, num_edges]}
            - edge_attr_dict: {edge_type: [num_edges, feat_dim]}
        """
        # Reverse map: arango_id → node_idx
        id_to_idx = {aid: idx for idx, aid in node_id_map.items()}

        edge_index_dict = {}
        edge_attr_dict = {}

        # Query all edges
        aql = """
        FOR edge IN code_edges
          RETURN {
            _from: edge._from,
            _to: edge._to,
            type: edge.type,
            weight: edge.weight
          }
        """

        edges = self.client.execute_query(aql, {})

        # Group by edge type
        edges_by_type = {}

        for edge in edges:
            edge_type = edge.get("type", "unknown")

            # Filter by edge types if specified
            if edge_types and edge_type not in edge_types:
                continue

            from_id = edge["_from"]
            to_id = edge["_to"]

            # Skip if nodes not in graph
            if from_id not in id_to_idx or to_id not in id_to_idx:
                continue

            from_idx = id_to_idx[from_id]
            to_idx = id_to_idx[to_id]
            weight = edge.get("weight", 1.0)

            if edge_type not in edges_by_type:
                edges_by_type[edge_type] = []

            edges_by_type[edge_type].append((from_idx, to_idx, weight))

        # Convert to PyG format
        for edge_type, edge_list in edges_by_type.items():
            if not edge_list:
                continue

            # Separate source, target, weights
            sources = [e[0] for e in edge_list]
            targets = [e[1] for e in edge_list]
            weights = [e[2] for e in edge_list]

            # Create edge_index [2, num_edges]
            edge_index = torch.tensor([sources, targets], dtype=torch.long)

            # Create edge_attr [num_edges, 1]
            edge_attr = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)

            edge_index_dict[edge_type] = edge_index
            edge_attr_dict[edge_type] = edge_attr

            logger.info(f"Edge type '{edge_type}': {len(edge_list)} edges")

        if not edge_index_dict:
            logger.warning("No edges found in graph!")

        return edge_index_dict, edge_attr_dict

    def create_train_val_split(
        self,
        num_nodes: int,
        val_ratio: float = 0.2,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create inductive train/validation split.

        For inductive learning, validation nodes should be "unseen" during training.

        Args:
            num_nodes: Total number of nodes
            val_ratio: Fraction of nodes for validation
            seed: Random seed

        Returns:
            (train_mask, val_mask) boolean arrays
        """
        np.random.seed(seed)

        indices = np.arange(num_nodes)
        np.random.shuffle(indices)

        val_size = int(num_nodes * val_ratio)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        train_mask = np.zeros(num_nodes, dtype=bool)
        val_mask = np.zeros(num_nodes, dtype=bool)

        train_mask[train_indices] = True
        val_mask[val_indices] = True

        logger.info(f"Split: {len(train_indices)} train, {len(val_indices)} val")

        return train_mask, val_mask
