"""
Unit tests for the Graph Dataset Loader.

These tests ensure that the GraphDatasetLoader correctly converts document collections
into PyTorch Geometric data structures while maintaining type safety and data integrity.
"""

import os
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import torch
from torch import Tensor
# Import the base PyTorch Geometric data classes
# Using only the core functionality which doesn't require specialized extensions
from torch_geometric.data import Data, HeteroData

# Import the module to test
from src.isne.loaders.graph_dataset_loader import GraphDatasetLoader
from src.isne.types.models import (
    DocumentType,
    RelationType,
    IngestDocument,
    DocumentRelation,
    LoaderResult,
)


class TestGraphDatasetLoader(unittest.TestCase):
    """Test suite for GraphDatasetLoader."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create a loader instance for tests
        self.loader = GraphDatasetLoader(
            use_heterogeneous_graph=False,
            embedding_dim=4,  # Small dimension for testing
            device="cpu"
        )
        
        # Create sample documents
        self.sample_documents = [
            IngestDocument(
                id="doc1",
                content="Sample document 1",
                source="test_source",
                document_type=DocumentType.TEXT,
                embedding=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
            ),
            IngestDocument(
                id="doc2",
                content="Sample document 2",
                source="test_source",
                document_type=DocumentType.TEXT,
                embedding=np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32)
            ),
            IngestDocument(
                id="doc3",
                content="Sample code document",
                source="test_source",
                document_type=DocumentType.CODE_PYTHON,
                embedding=np.array([0.9, 1.0, 1.1, 1.2], dtype=np.float32)
            )
        ]
        
        # Create sample relations
        self.sample_relations = [
            DocumentRelation(
                source_id="doc1",
                target_id="doc2",
                relation_type=RelationType.SIMILAR_TO,
                weight=0.8
            ),
            DocumentRelation(
                source_id="doc2",
                target_id="doc3",
                relation_type=RelationType.REFERS_TO,
                weight=0.5
            ),
            DocumentRelation(
                source_id="doc3",
                target_id="doc1",
                relation_type=RelationType.RELATED_TO,
                weight=0.3
            )
        ]
        
        # Create a sample loader result
        self.sample_loader_result = LoaderResult(
            documents=self.sample_documents,
            relations=self.sample_relations
        )
    
    def test_init(self) -> None:
        """Test initialization with different parameters."""
        # Test with default parameters
        loader = GraphDatasetLoader()
        self.assertFalse(loader.use_heterogeneous_graph)
        self.assertEqual(loader.embedding_dim, 768)
        self.assertIn(loader.device, ['cpu', 'cuda'])
        
        # Test with custom parameters
        loader = GraphDatasetLoader(
            use_heterogeneous_graph=True,
            embedding_dim=512,
            device="cpu"
        )
        self.assertTrue(loader.use_heterogeneous_graph)
        self.assertEqual(loader.embedding_dim, 512)
        self.assertEqual(loader.device, "cpu")
    
    def test_load_from_documents_homogeneous(self) -> None:
        """Test loading homogeneous graph from documents."""
        # Load graph from sample documents
        graph = self.loader.load_from_documents(
            self.sample_documents,
            self.sample_relations
        )
        
        # Check graph type
        self.assertIsInstance(graph, Data)
        
        # Check node features
        self.assertIsInstance(graph.x, Tensor)
        self.assertEqual(graph.x.shape[0], 3)  # 3 documents
        
        # The feature dimension includes the embedding plus one-hot encodings for node types
        # 4 (embedding) + len(DocumentType) for node types
        self.assertEqual(graph.x.shape[1], 4 + len(DocumentType))
        
        # Check edge indices
        self.assertIsInstance(graph.edge_index, Tensor)
        self.assertEqual(graph.edge_index.shape[0], 2)  # Source, target pairs
        self.assertEqual(graph.edge_index.shape[1], 3)  # 3 relations
        
        # Check edge attributes
        self.assertIsInstance(graph.edge_attr, Tensor)
        self.assertEqual(graph.edge_attr.shape[0], 3)  # 3 relations
        self.assertEqual(graph.edge_attr.shape[1], 1)  # Weight attribute
        
        # Check document IDs are preserved
        self.assertEqual(len(graph.doc_ids), 3)
        self.assertEqual(graph.doc_ids[0], "doc1")
        self.assertEqual(graph.doc_ids[1], "doc2")
        self.assertEqual(graph.doc_ids[2], "doc3")
    
    def test_load_from_documents_heterogeneous(self) -> None:
        """Test loading heterogeneous graph from documents."""
        # Create a heterogeneous loader
        hetero_loader = GraphDatasetLoader(
            use_heterogeneous_graph=True,
            embedding_dim=4,
            device="cpu"
        )
        
        # Load graph from sample documents
        graph = hetero_loader.load_from_documents(
            self.sample_documents,
            self.sample_relations
        )
        
        # Check graph type
        self.assertIsInstance(graph, HeteroData)
        
        # Check node types exist
        self.assertIn('text', graph.node_types)
        self.assertIn('code_python', graph.node_types)
        
        # Check text node features
        self.assertEqual(graph['text'].x.shape[0], 2)  # 2 text documents
        
        # Check code_python node features
        self.assertEqual(graph['code_python'].x.shape[0], 1)  # 1 code document
        
        # Check some edge types
        # Note: Edge types may be different depending on the documents
        edge_types_found = False
        for edge_type in graph.edge_types:
            edge_types_found = True
            # Should be a tuple of (src_type, edge_type, dst_type)
            self.assertEqual(len(edge_type), 3)
        
        # Ensure we found some edge types
        self.assertTrue(edge_types_found)
    
    def test_load_from_loader_result(self) -> None:
        """Test loading from a LoaderResult object."""
        # Load graph from sample loader result
        graph = self.loader.load_from_loader_result(self.sample_loader_result)
        
        # Check graph type
        self.assertIsInstance(graph, Data)
        
        # Check node features
        self.assertEqual(graph.x.shape[0], 3)  # 3 documents
        
        # Check edge indices
        self.assertEqual(graph.edge_index.shape[1], 3)  # 3 relations
    
    def test_load_from_file(self) -> None:
        """Test loading from a JSON file."""
        # Create a temporary file path
        temp_dir = tempfile.gettempdir()
        tmp_path = os.path.join(temp_dir, f"test_graph_dataset_{os.getpid()}.json")
        
        try:
            # Prepare data for serialization
            data = {
                "documents": [
                    {
                        "id": doc.id,
                        "content": doc.content,
                        "source": doc.source,
                        "document_type": doc.document_type,
                        "embedding": doc.embedding.tolist() if doc.embedding is not None else None
                    } for doc in self.sample_documents
                ],
                "relations": [
                    {
                        "source_id": rel.source_id,
                        "target_id": rel.target_id,
                        "relation_type": rel.relation_type,
                        "weight": rel.weight
                    } for rel in self.sample_relations
                ]
            }
            
            # Write to file in text mode
            with open(tmp_path, 'w', encoding='utf-8') as tmp_file:
                json.dump(data, tmp_file)
            
            # Load graph from file
            graph = self.loader.load_from_file(tmp_path)
            
            # Check graph type
            self.assertIsInstance(graph, Data)
            
            # Check node features
            self.assertEqual(graph.x.shape[0], 3)  # 3 documents
            
            # Check edge indices
            self.assertEqual(graph.edge_index.shape[1], 3)  # 3 relations
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_load_from_file_not_found(self) -> None:
        """Test loading from a non-existent file."""
        with self.assertRaises(FileNotFoundError):
            self.loader.load_from_file("/non/existent/path.json")
    
    def test_prepare_node_features(self) -> None:
        """Test preparing node features from documents."""
        # Test with normal documents
        features = self.loader._prepare_node_features(self.sample_documents)
        
        self.assertIsInstance(features, Tensor)
        self.assertEqual(features.shape[0], 3)  # 3 documents
        self.assertEqual(features.shape[1], 4 + len(DocumentType))  # embedding + node types
        
        # Test with documents having enhanced embeddings
        enhanced_docs = self.sample_documents.copy()
        enhanced_docs[0].enhanced_embedding = np.array([1.1, 1.2, 1.3, 1.4], dtype=np.float32)
        
        features = self.loader._prepare_node_features(enhanced_docs)
        
        self.assertEqual(features.shape[0], 3)
        self.assertEqual(features.shape[1], 4 + len(DocumentType))
        
        # Check that enhanced embedding is used when available
        self.assertAlmostEqual(features[0, 0].item(), 1.1, places=5)
        
        # Test with documents missing embeddings in a separate test instance
        # to avoid counter state from previous tests
        test_loader = GraphDatasetLoader(embedding_dim=4, device="cpu")
        self.assertEqual(test_loader.missing_embedding_count, 0)  # Should start at 0
        
        missing_docs = [
            IngestDocument(
                id="doc1",
                content="Missing embedding document",
                source="test_source",
                document_type=DocumentType.TEXT,
                embedding=None  # Explicitly None embedding
            )
        ]
        
        features = test_loader._prepare_node_features(missing_docs)
        
        self.assertEqual(features.shape[0], 1)  # 1 document
        self.assertEqual(features.shape[1], 4 + len(DocumentType))
        
        # Check the missing embedding counter was incremented
        self.assertEqual(test_loader.missing_embedding_count, 1)
    
    def test_encode_document_types(self) -> None:
        """Test encoding document types as one-hot vectors."""
        encodings = self.loader._encode_document_types(self.sample_documents)
        
        self.assertIsInstance(encodings, Tensor)
        self.assertEqual(encodings.shape[0], 3)  # 3 documents
        self.assertEqual(encodings.shape[1], len(DocumentType))  # One-hot encoding size
        
        # Check text document encoding
        text_idx = list(DocumentType.__members__.values()).index(DocumentType.TEXT)
        self.assertEqual(encodings[0, text_idx].item(), 1.0)
        
        # Check code document encoding
        code_idx = list(DocumentType.__members__.values()).index(DocumentType.CODE_PYTHON)
        self.assertEqual(encodings[2, code_idx].item(), 1.0)
    
    def test_prepare_edges(self) -> None:
        """Test preparing edges from relations."""
        # First update the node_id_to_idx mapping
        self.loader.node_id_to_idx = {
            "doc1": 0,
            "doc2": 1,
            "doc3": 2
        }
        
        # Test with normal relations
        edge_indices, edge_attrs = self.loader._prepare_edges(self.sample_relations)
        
        self.assertIsInstance(edge_indices, Tensor)
        self.assertEqual(edge_indices.shape[0], 2)  # Source, target pairs
        self.assertEqual(edge_indices.shape[1], 3)  # 3 relations
        
        self.assertIsInstance(edge_attrs, Tensor)
        self.assertEqual(edge_attrs.shape[0], 3)  # 3 relations
        self.assertEqual(edge_attrs.shape[1], 1)  # Weight attribute
        
        # Check specific edge indices
        # doc1 -> doc2
        self.assertEqual(edge_indices[0, 0].item(), 0)
        self.assertEqual(edge_indices[1, 0].item(), 1)
        
        # doc2 -> doc3
        self.assertEqual(edge_indices[0, 1].item(), 1)
        self.assertEqual(edge_indices[1, 1].item(), 2)
        
        # Check specific edge attributes
        # doc1 -> doc2 (weight 0.8)
        self.assertAlmostEqual(edge_attrs[0, 0].item(), 0.8, places=5)
        
        # Test with unknown node IDs
        bad_relations = [
            DocumentRelation(
                source_id="unknown",
                target_id="doc1",
                relation_type=RelationType.SIMILAR_TO
            )
        ]
        
        edge_indices, edge_attrs = self.loader._prepare_edges(bad_relations)
        
        # For empty relations, we should get an empty tensor
        self.assertTrue(isinstance(edge_indices, torch.Tensor))
        # Check that it's in the expected format - either a 2D tensor with shape [2, 0] 
        # or a 1D tensor with shape [0] depending on PyTorch Geometric version
        if len(edge_indices.shape) == 2:
            self.assertListEqual(list(edge_indices.shape), [2, 0])  # Should be shape [2, 0]
        else:
            self.assertEqual(edge_indices.shape[0], 0)  # Should have no elements
        self.assertIsNone(edge_attrs)
    
    def test_split_dataset(self) -> None:
        """Test splitting a dataset into train/val/test."""
        # Load graph from sample documents
        graph = self.loader.load_from_documents(
            self.sample_documents,
            self.sample_relations
        )
        
        # Split the dataset
        split_graph = self.loader.split_dataset(
            graph,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            shuffle=True,
            random_seed=42
        )
        
        # The split operation modifies the graph in-place
        self.assertIs(split_graph, graph)
        
        # Check masks
        self.assertTrue(hasattr(graph, 'train_mask'))
        self.assertTrue(hasattr(graph, 'val_mask'))
        self.assertTrue(hasattr(graph, 'test_mask'))
        
        self.assertEqual(graph.train_mask.sum().item(), 1)  # 60% of 3 nodes ≈ 1.8 ≈ 1 node
        self.assertEqual(graph.val_mask.sum().item() + graph.test_mask.sum().item(), 2)  # Remaining nodes
        
        # Check that each node is assigned to exactly one split
        for i in range(len(self.sample_documents)):
            # Count how many splits this node is assigned to
            splits = int(graph.train_mask[i]) + int(graph.val_mask[i]) + int(graph.test_mask[i])
            self.assertEqual(splits, 1)


if __name__ == '__main__':
    unittest.main()
