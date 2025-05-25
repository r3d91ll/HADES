"""
Graph Dataset Loader for ISNE.

This module provides functionality to convert document collections into
PyTorch Geometric data structures for use with the ISNE model. It supports
both in-memory and file-based data loading, with special handling for
different document modalities.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Union, Any, Tuple, Set, Callable
import os
import json
import numpy as np
import torch
from torch import Tensor
# Import the base PyTorch Geometric data classes
# Using only the core functionality which doesn't require specialized extensions
from torch_geometric.data import Data, HeteroData
import logging
from pathlib import Path

from ..types.models import (
    DocumentType,
    RelationType, 
    IngestDocument,
    DocumentRelation,
    LoaderResult,
    EmbeddingVector
)

# Configure logging
logger = logging.getLogger(__name__)


class GraphDatasetLoader:
    """
    Converts document collections into PyTorch Geometric data structures.
    
    This class handles the conversion of documents and their relationships into
    graph structures suitable for the ISNE model. It supports both homogeneous
    and heterogeneous graph construction, with special handling for different
    document modalities.
    """
    
    def __init__(
        self,
        use_heterogeneous_graph: bool = False,
        embedding_dim: int = 768,
        device: Optional[str] = None
    ):
        """
        Initialize the GraphDatasetLoader.
        
        Args:
            use_heterogeneous_graph: Whether to create a heterogeneous graph with
                                     node and edge types (True) or a homogeneous
                                     graph (False).
            embedding_dim: Dimension of document embeddings.
            device: Device to place tensors on ('cpu', 'cuda', etc.). If None,
                   will use CUDA if available, otherwise CPU.
        """
        self.use_heterogeneous_graph = use_heterogeneous_graph
        self.embedding_dim = embedding_dim
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Mapping dictionaries for node and edge indices
        self.node_id_to_idx: Dict[str, int] = {}
        self.edge_type_to_idx: Dict[str, int] = {}
        
        # Counter for nodes without embeddings
        self.missing_embedding_count = 0
        
    def load_from_documents(
        self,
        documents: List[IngestDocument],
        relations: List[DocumentRelation],
        include_node_types: bool = True,
        include_edge_attributes: bool = True
    ) -> Union[Data, HeteroData]:
        """
        Load data from document and relation collections.
        
        Args:
            documents: List of IngestDocument objects.
            relations: List of DocumentRelation objects.
            include_node_types: Whether to include node types as features.
            include_edge_attributes: Whether to include edge weights and metadata.
            
        Returns:
            PyTorch Geometric Data or HeteroData object representing the graph.
        """
        logger.info(f"Loading graph from {len(documents)} documents and {len(relations)} relations")
        
        if self.use_heterogeneous_graph:
            return self._create_heterogeneous_graph(
                documents, relations, include_node_types, include_edge_attributes
            )
        else:
            return self._create_homogeneous_graph(
                documents, relations, include_node_types, include_edge_attributes
            )
    
    def load_from_loader_result(
        self,
        loader_result: LoaderResult,
        include_node_types: bool = True,
        include_edge_attributes: bool = True
    ) -> Union[Data, HeteroData]:
        """
        Load data from a LoaderResult object.
        
        Args:
            loader_result: LoaderResult object containing documents and relations.
            include_node_types: Whether to include node types as features.
            include_edge_attributes: Whether to include edge weights and metadata.
            
        Returns:
            PyTorch Geometric Data or HeteroData object representing the graph.
        """
        return self.load_from_documents(
            loader_result.documents,
            loader_result.relations,
            include_node_types,
            include_edge_attributes
        )
    
    def load_from_file(
        self,
        file_path: Union[str, Path],
        include_node_types: bool = True,
        include_edge_attributes: bool = True
    ) -> Union[Data, HeteroData]:
        """
        Load data from a JSON file containing documents and relations.
        
        Args:
            file_path: Path to JSON file containing documents and relations.
            include_node_types: Whether to include node types as features.
            include_edge_attributes: Whether to include edge weights and metadata.
            
        Returns:
            PyTorch Geometric Data or HeteroData object representing the graph.
        """
        logger.info(f"Loading graph from file: {file_path}")
        
        # Ensure path is a Path object
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load data from file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract documents and relations
        documents = [IngestDocument(**doc) for doc in data.get('documents', [])]
        relations = [DocumentRelation(**rel) for rel in data.get('relations', [])]
        
        return self.load_from_documents(
            documents, relations, include_node_types, include_edge_attributes
        )
    
    def _create_homogeneous_graph(
        self,
        documents: List[IngestDocument],
        relations: List[DocumentRelation],
        include_node_types: bool = True,
        include_edge_attributes: bool = True
    ) -> Data:
        """
        Create a homogeneous graph from documents and relations.
        
        Args:
            documents: List of IngestDocument objects.
            relations: List of DocumentRelation objects.
            include_node_types: Whether to include node types as features.
            include_edge_attributes: Whether to include edge weights and metadata.
            
        Returns:
            PyTorch Geometric Data object representing the homogeneous graph.
        """
        # Reset node mapping
        self.node_id_to_idx = {}
        self.missing_embedding_count = 0
        
        # Create node mapping
        for i, doc in enumerate(documents):
            self.node_id_to_idx[doc.id] = i
        
        # Prepare node features (embeddings)
        node_features = self._prepare_node_features(documents, include_node_types)
        
        # Prepare edge indices and attributes
        edge_indices, edge_attrs = self._prepare_edges(relations, include_edge_attributes)
        
        # Create graph data object
        data = Data(
            x=node_features,
            edge_index=edge_indices
        )
        
        # Add edge attributes if requested
        if include_edge_attributes and edge_attrs is not None:
            data.edge_attr = edge_attrs
        
        # Add document IDs as reference
        data.doc_ids = [doc.id for doc in documents]
        
        logger.info(f"Created homogeneous graph with {len(documents)} nodes and {len(relations)} edges")
        if self.missing_embedding_count > 0:
            logger.warning(f"{self.missing_embedding_count} documents had missing embeddings")
        
        return data.to(self.device)
    
    def _create_heterogeneous_graph(
        self,
        documents: List[IngestDocument],
        relations: List[DocumentRelation],
        include_node_types: bool = True,
        include_edge_attributes: bool = True
    ) -> HeteroData:
        """
        Create a heterogeneous graph from documents and relations.
        
        Args:
            documents: List of IngestDocument objects.
            relations: List of DocumentRelation objects.
            include_node_types: Whether to include node types as features.
            include_edge_attributes: Whether to include edge weights and metadata.
            
        Returns:
            PyTorch Geometric HeteroData object representing the heterogeneous graph.
        """
        # Reset mappings
        self.node_id_to_idx = {}
        self.edge_type_to_idx = {}
        self.missing_embedding_count = 0
        
        # Group documents by type
        documents_by_type: Dict[str, List[IngestDocument]] = {}
        for doc in documents:
            doc_type = doc.document_type
            if doc_type not in documents_by_type:
                documents_by_type[doc_type] = []
            documents_by_type[doc_type].append(doc)
        
        # Create heterogeneous data object
        data = HeteroData()
        
        # Add nodes for each document type
        node_type_to_global_idx: Dict[str, Dict[str, int]] = {}
        global_idx = 0
        
        for doc_type, docs in documents_by_type.items():
            # Create node mapping for this type
            node_type_to_global_idx[doc_type] = {}
            local_idx_to_id = {}
            
            for local_idx, doc in enumerate(docs):
                node_type_to_global_idx[doc_type][doc.id] = local_idx
                local_idx_to_id[local_idx] = doc.id
                self.node_id_to_idx[doc.id] = global_idx
                global_idx += 1
            
            # Prepare node features for this type
            type_node_features = self._prepare_node_features(docs, include_node_types)
            
            # Add to data object
            data[doc_type].x = type_node_features
            data[doc_type].doc_ids = [doc.id for doc in docs]
        
        # Group relations by source and target types
        relation_triplets: Dict[Tuple[str, str, str], List[DocumentRelation]] = {}
        
        for rel in relations:
            source_id = rel.source_id
            target_id = rel.target_id
            
            # Find document types
            source_doc = next((d for d in documents if d.id == source_id), None)
            target_doc = next((d for d in documents if d.id == target_id), None)
            
            if source_doc is None or target_doc is None:
                logger.warning(f"Skipping relation with missing document: {rel.source_id} -> {rel.target_id}")
                continue
            
            source_type = source_doc.document_type
            target_type = target_doc.document_type
            rel_type = rel.relation_type.value
            
            triplet = (source_type, rel_type, target_type)
            if triplet not in relation_triplets:
                relation_triplets[triplet] = []
            
            relation_triplets[triplet].append(rel)
        
        # Add edges for each relation type
        for (src_type, rel_type, dst_type), rels in relation_triplets.items():
            # Extract source and target indices
            edge_indices = []
            edge_attrs_list = []
            
            for rel in rels:
                src_local_idx = node_type_to_global_idx[src_type].get(rel.source_id)
                dst_local_idx = node_type_to_global_idx[dst_type].get(rel.target_id)
                
                if src_local_idx is None or dst_local_idx is None:
                    continue
                
                edge_indices.append([src_local_idx, dst_local_idx])
                
                if include_edge_attributes:
                    edge_attrs_list.append([rel.weight])
            
            if not edge_indices:
                continue
            
            # Convert to tensor
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            
            # Add to data object
            data[src_type, rel_type, dst_type].edge_index = edge_index
            
            if include_edge_attributes and edge_attrs_list:
                edge_attr = torch.tensor(edge_attrs_list, dtype=torch.float)
                data[src_type, rel_type, dst_type].edge_attr = edge_attr
        
        logger.info(f"Created heterogeneous graph with {global_idx} nodes across {len(documents_by_type)} types")
        logger.info(f"Added edges for {len(relation_triplets)} relation types")
        if self.missing_embedding_count > 0:
            logger.warning(f"{self.missing_embedding_count} documents had missing embeddings")
        
        return data.to(self.device)
    
    def _prepare_node_features(
        self,
        documents: List[IngestDocument],
        include_node_types: bool = True
    ) -> Tensor:
        """
        Prepare node features from document embeddings.
        
        Args:
            documents: List of documents to extract features from.
            include_node_types: Whether to include node type encoding.
            
        Returns:
            Tensor containing node features.
        """
        # If no documents, return empty tensor
        if not documents:
            return torch.zeros((0, self.embedding_dim), dtype=torch.float)
        
        features_list = []
        
        for doc in documents:
            # Use enhanced embedding if available, otherwise use original
            # Need to handle numpy arrays carefully
            if doc.enhanced_embedding is not None:
                embedding = doc.enhanced_embedding
            else:
                embedding = doc.embedding
            
            if embedding is None:
                # Default to zero vector if no embedding is available
                embedding = np.zeros(self.embedding_dim, dtype=np.float32)
                self.missing_embedding_count += 1  # Increment missing count
            
            # Convert to numpy array if not already
            if isinstance(embedding, list):
                embedding = np.array(embedding, dtype=np.float32)
            
            # Ensure correct shape
            if len(embedding.shape) == 1:
                features_list.append(embedding)
            else:
                # Take the first embedding if multiple exist
                features_list.append(embedding[0])
        
        # Convert to tensor
        node_features = torch.tensor(np.stack(features_list), dtype=torch.float)
        
        # Add node type encodings if requested
        if include_node_types:
            type_encodings = self._encode_document_types(documents)
            node_features = torch.cat([node_features, type_encodings], dim=1)
        
        return node_features
    
    def _encode_document_types(self, documents: List[IngestDocument]) -> Tensor:
        """
        Create one-hot encoding for document types.
        
        Args:
            documents: List of documents to encode types for.
            
        Returns:
            Tensor containing one-hot encoded document types.
        """
        # Get all possible document types
        all_types = list(DocumentType.__members__.values())
        type_to_idx = {t: i for i, t in enumerate(all_types)}
        
        # Create one-hot encoding
        encodings = np.zeros((len(documents), len(all_types)), dtype=np.float32)
        
        for i, doc in enumerate(documents):
            type_idx = type_to_idx.get(doc.document_type, type_to_idx[DocumentType.UNKNOWN])
            encodings[i, type_idx] = 1.0
        
        return torch.tensor(encodings, dtype=torch.float)
    
    def _prepare_edges(
        self,
        relations: List[DocumentRelation],
        include_attributes: bool = True
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Prepare edge indices and attributes from relations.
        
        Args:
            relations: List of document relations.
            include_attributes: Whether to include edge weights as attributes.
            
        Returns:
            Tuple of (edge_indices, edge_attributes) tensors.
        """
        # If no relations, return empty tensor with 2 rows and 0 columns
        # This ensures we maintain the expected shape (2, N) even when N=0
        if not relations:
            return torch.zeros((2, 0), dtype=torch.long), None
        
        edge_list = []
        edge_attrs_list = []
        
        for rel in relations:
            source_idx = self.node_id_to_idx.get(rel.source_id)
            target_idx = self.node_id_to_idx.get(rel.target_id)
            
            if source_idx is None or target_idx is None:
                logger.warning(f"Skipping relation with unknown node ID: {rel.source_id} -> {rel.target_id}")
                continue
            
            edge_list.append([source_idx, target_idx])
            
            if include_attributes:
                # For now, just use the relation weight as the edge attribute
                edge_attrs_list.append([rel.weight])
        
        # Convert to tensors
        edge_indices = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        edge_attrs = None
        if include_attributes and edge_attrs_list:
            edge_attrs = torch.tensor(edge_attrs_list, dtype=torch.float)
        
        return edge_indices, edge_attrs
    
    def split_dataset(
        self,
        data: Union[Data, HeteroData],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        shuffle: bool = True,
        random_seed: int = 42
    ) -> Tuple[Data, Data, Data]:
        """
        Split a graph dataset into training, validation, and test sets.
        
        Args:
            data: PyTorch Geometric Data object to split.
            train_ratio: Ratio of nodes to use for training.
            val_ratio: Ratio of nodes to use for validation.
            test_ratio: Ratio of nodes to use for testing.
            shuffle: Whether to shuffle nodes before splitting.
            random_seed: Random seed for reproducibility.
            
        Returns:
            Tuple of (train_data, val_data, test_data) graph objects.
        """
        if isinstance(data, HeteroData):
            raise NotImplementedError("Dataset splitting not yet implemented for heterogeneous graphs")
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        # Get the number of nodes
        num_nodes = data.x.size(0)
        
        # Create indices and shuffle if requested
        indices = np.arange(num_nodes)
        if shuffle:
            np.random.shuffle(indices)
        
        # Calculate split sizes
        train_size = int(train_ratio * num_nodes)
        val_size = int(val_ratio * num_nodes)
        
        # Get node indices for each split
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Create node masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True
        
        # Add masks to data
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        
        return data
