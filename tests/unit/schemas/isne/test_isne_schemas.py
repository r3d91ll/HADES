"""
Unit tests for the ISNE schemas in the HADES-PathRAG system.

Tests ISNE schema functionality including validation, model configurations,
document representations, and relationship structures.
"""

import unittest
import uuid
from datetime import datetime
from pydantic import ValidationError

from src.schemas.isne.models import (
    ISNEDocumentType, 
    ISNERelationType,
    ISNEModelConfigSchema,
    ISNETrainingConfigSchema,
    ISNEConfigSchema
)
from src.schemas.isne.documents import (
    ISNEChunkSchema,
    ISNEDocumentSchema
)
from src.schemas.isne.relations import (
    ISNEDocumentRelationSchema,
    ISNEGraphSchema
)


class TestISNEModelConfigSchema(unittest.TestCase):
    """Test the ISNEModelConfigSchema functionality."""
    
    def test_model_config_instantiation(self):
        """Test that ISNEModelConfigSchema can be instantiated with default values."""
        # Test with default values
        config = ISNEModelConfigSchema()
        
        # Check default values
        self.assertEqual(config.hidden_dim, 256)
        self.assertEqual(config.output_dim, 768)
        self.assertEqual(config.num_layers, 2)
        self.assertEqual(config.num_heads, 8)
        self.assertEqual(config.dropout, 0.1)
        self.assertEqual(config.activation, "relu")
        self.assertTrue(config.normalization)
        self.assertEqual(config.attention_type, "dot_product")
        
        # Test with custom values
        config = ISNEModelConfigSchema(
            hidden_dim=512,
            output_dim=1024,
            num_layers=3,
            num_heads=16,
            dropout=0.2,
            activation="gelu",
            normalization=False,
            attention_type="additive"
        )
        
        self.assertEqual(config.hidden_dim, 512)
        self.assertEqual(config.output_dim, 1024)
        self.assertEqual(config.num_layers, 3)
        self.assertEqual(config.num_heads, 16)
        self.assertEqual(config.dropout, 0.2)
        self.assertEqual(config.activation, "gelu")
        self.assertFalse(config.normalization)
        self.assertEqual(config.attention_type, "additive")
    
    def test_integer_validation(self):
        """Test validation of integer values."""
        # Valid values
        config = ISNEModelConfigSchema(
            hidden_dim=64,
            output_dim=128,
            num_layers=1,
            num_heads=1
        )
        
        # Invalid values - should be positive
        with self.assertRaises(ValidationError):
            ISNEModelConfigSchema(hidden_dim=0)
        
        with self.assertRaises(ValidationError):
            ISNEModelConfigSchema(output_dim=-10)
        
        with self.assertRaises(ValidationError):
            ISNEModelConfigSchema(num_layers=0)
        
        with self.assertRaises(ValidationError):
            ISNEModelConfigSchema(num_heads=-1)
    
    def test_dropout_validation(self):
        """Test validation of dropout value."""
        # Valid values
        config = ISNEModelConfigSchema(dropout=0.0)
        config = ISNEModelConfigSchema(dropout=0.5)
        config = ISNEModelConfigSchema(dropout=1.0)
        
        # Invalid values - should be between 0 and 1
        with self.assertRaises(ValidationError):
            ISNEModelConfigSchema(dropout=-0.1)
        
        with self.assertRaises(ValidationError):
            ISNEModelConfigSchema(dropout=1.1)


class TestISNETrainingConfigSchema(unittest.TestCase):
    """Test the ISNETrainingConfigSchema functionality."""
    
    def test_training_config_instantiation(self):
        """Test that ISNETrainingConfigSchema can be instantiated with default values."""
        # Test with default values
        config = ISNETrainingConfigSchema()
        
        # Check default values
        self.assertEqual(config.learning_rate, 0.001)
        self.assertEqual(config.weight_decay, 0.0001)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.epochs, 100)
        self.assertEqual(config.early_stopping_patience, 10)
        self.assertEqual(config.validation_fraction, 0.1)
        self.assertEqual(config.optimizer, "adam")
        self.assertIsNone(config.scheduler)
        self.assertEqual(config.max_grad_norm, 1.0)
        
        # Test with custom values
        config = ISNETrainingConfigSchema(
            learning_rate=0.01,
            weight_decay=0.001,
            batch_size=64,
            epochs=50,
            early_stopping_patience=5,
            validation_fraction=0.2,
            optimizer="sgd",
            scheduler="linear",
            max_grad_norm=5.0,
            device="cuda:0",
            checkpoint_dir="/tmp/checkpoints",
            checkpoint_interval=5
        )
        
        self.assertEqual(config.learning_rate, 0.01)
        self.assertEqual(config.weight_decay, 0.001)
        self.assertEqual(config.batch_size, 64)
        self.assertEqual(config.epochs, 50)
        self.assertEqual(config.early_stopping_patience, 5)
        self.assertEqual(config.validation_fraction, 0.2)
        self.assertEqual(config.optimizer, "sgd")
        self.assertEqual(config.scheduler, "linear")
        self.assertEqual(config.max_grad_norm, 5.0)
        self.assertEqual(config.device, "cuda:0")
        self.assertEqual(config.checkpoint_dir, "/tmp/checkpoints")
        self.assertEqual(config.checkpoint_interval, 5)
    
    def test_validation_fraction(self):
        """Test validation of validation_fraction value."""
        # Valid values
        config = ISNETrainingConfigSchema(validation_fraction=0.0)
        config = ISNETrainingConfigSchema(validation_fraction=0.5)
        config = ISNETrainingConfigSchema(validation_fraction=1.0)
        
        # Invalid values - should be between 0 and 1
        with self.assertRaises(ValidationError):
            ISNETrainingConfigSchema(validation_fraction=-0.1)
        
        with self.assertRaises(ValidationError):
            ISNETrainingConfigSchema(validation_fraction=1.1)
    
    def test_positive_value_validation(self):
        """Test validation of values that must be positive."""
        # Invalid values
        with self.assertRaises(ValidationError):
            ISNETrainingConfigSchema(batch_size=0)
        
        with self.assertRaises(ValidationError):
            ISNETrainingConfigSchema(epochs=0)
        
        with self.assertRaises(ValidationError):
            ISNETrainingConfigSchema(early_stopping_patience=0)


class TestISNEConfigSchema(unittest.TestCase):
    """Test the ISNEConfigSchema functionality."""
    
    def test_isne_config_instantiation(self):
        """Test that ISNEConfigSchema can be instantiated with default values."""
        # Test with default values
        config = ISNEConfigSchema()
        
        # Check default values
        self.assertTrue(config.use_isne)
        self.assertIsNone(config.isne_model_path)
        self.assertIsInstance(config.model, ISNEModelConfigSchema)
        self.assertIsInstance(config.training, ISNETrainingConfigSchema)
        self.assertEqual(config.modality, "text")
        self.assertEqual(config.edge_threshold, 0.7)
        self.assertEqual(config.max_edges_per_node, 10)
        
        # Test with custom values
        model_config = ISNEModelConfigSchema(hidden_dim=512)
        training_config = ISNETrainingConfigSchema(batch_size=64)
        
        config = ISNEConfigSchema(
            use_isne=False,
            isne_model_path="/path/to/model.pt",
            model=model_config,
            training=training_config,
            modality="code",
            edge_threshold=0.5,
            max_edges_per_node=5
        )
        
        self.assertFalse(config.use_isne)
        self.assertEqual(config.isne_model_path, "/path/to/model.pt")
        self.assertEqual(config.model, model_config)
        self.assertEqual(config.training, training_config)
        self.assertEqual(config.modality, "code")
        self.assertEqual(config.edge_threshold, 0.5)
        self.assertEqual(config.max_edges_per_node, 5)
    
    def test_edge_threshold_validation(self):
        """Test validation of edge_threshold value."""
        # Valid values
        config = ISNEConfigSchema(edge_threshold=0.0)
        config = ISNEConfigSchema(edge_threshold=0.5)
        config = ISNEConfigSchema(edge_threshold=1.0)
        
        # Invalid values - should be between 0 and 1
        with self.assertRaises(ValidationError):
            ISNEConfigSchema(edge_threshold=-0.1)
        
        with self.assertRaises(ValidationError):
            ISNEConfigSchema(edge_threshold=1.1)
    
    def test_max_edges_validation(self):
        """Test validation of max_edges_per_node value."""
        # Valid values
        config = ISNEConfigSchema(max_edges_per_node=1)
        config = ISNEConfigSchema(max_edges_per_node=100)
        
        # Invalid values - should be positive
        with self.assertRaises(ValidationError):
            ISNEConfigSchema(max_edges_per_node=0)
        
        with self.assertRaises(ValidationError):
            ISNEConfigSchema(max_edges_per_node=-5)


class TestISNEDocumentSchema(unittest.TestCase):
    """Test the ISNEDocumentSchema and ISNEChunkSchema functionality."""
    
    def test_chunk_schema_instantiation(self):
        """Test that ISNEChunkSchema can be instantiated with required attributes."""
        # Test minimal chunk
        chunk = ISNEChunkSchema(
            id="chunk1",
            parent_id="doc1",
            start_index=0,
            end_index=100,
            content="This is a test chunk."
        )
        
        self.assertEqual(chunk.id, "chunk1")
        self.assertEqual(chunk.parent_id, "doc1")
        self.assertEqual(chunk.start_index, 0)
        self.assertEqual(chunk.end_index, 100)
        self.assertEqual(chunk.content, "This is a test chunk.")
        self.assertIsNone(chunk.embedding)
        self.assertEqual(chunk.metadata, {})
        
        # Test with embedding
        chunk = ISNEChunkSchema(
            id="chunk1",
            parent_id="doc1",
            start_index=0,
            end_index=100,
            content="This is a test chunk.",
            embedding=[0.1, 0.2, 0.3],
            metadata={"importance": "high"}
        )
        
        self.assertEqual(chunk.id, "chunk1")
        self.assertEqual(chunk.embedding, [0.1, 0.2, 0.3])
        self.assertEqual(chunk.metadata, {"importance": "high"})
    
    def test_document_schema_instantiation(self):
        """Test that ISNEDocumentSchema can be instantiated with required attributes."""
        # Create test chunks
        chunk1 = ISNEChunkSchema(
            id="chunk1",
            parent_id="doc1",
            start_index=0,
            end_index=100,
            content="First chunk."
        )
        
        chunk2 = ISNEChunkSchema(
            id="chunk2",
            parent_id="doc1",
            start_index=101,
            end_index=200,
            content="Second chunk."
        )
        
        # Test minimal document
        doc = ISNEDocumentSchema(
            id="doc1",
            source="test.txt",
            embedding_model="test-model"
        )
        
        self.assertEqual(doc.id, "doc1")
        self.assertEqual(doc.source, "test.txt")
        self.assertEqual(doc.type, ISNEDocumentType.TEXT)  # default value
        self.assertIsNone(doc.content)
        self.assertEqual(doc.chunks, [])
        self.assertEqual(doc.embeddings, {})
        self.assertEqual(doc.embedding_model, "test-model")
        
        # Test with all attributes
        doc = ISNEDocumentSchema(
            id="doc1",
            type=ISNEDocumentType.CODE,
            content="Full document content.",
            chunks=[chunk1, chunk2],
            metadata={"author": "test"},
            embeddings={"document": [0.1, 0.2, 0.3]},
            embedding_model="test-model",
            source="test.py",
            created_at="2023-01-01",
            processed_at="2023-01-02"
        )
        
        self.assertEqual(doc.id, "doc1")
        self.assertEqual(doc.type, ISNEDocumentType.CODE)
        self.assertEqual(doc.content, "Full document content.")
        self.assertEqual(doc.chunks, [chunk1, chunk2])
        self.assertEqual(doc.metadata, {"author": "test"})
        self.assertEqual(doc.embeddings, {"document": [0.1, 0.2, 0.3]})
        self.assertEqual(doc.embedding_model, "test-model")
        self.assertEqual(doc.source, "test.py")
        self.assertEqual(doc.created_at, "2023-01-01")
        self.assertEqual(doc.processed_at, "2023-01-02")
    
    def test_parent_id_validation(self):
        """Test that parent_id is set correctly in chunks."""
        # Create chunks with different parent_id
        chunk1 = ISNEChunkSchema(
            id="chunk1",
            parent_id="old_doc",
            start_index=0,
            end_index=100,
            content="First chunk."
        )
        
        chunk2 = ISNEChunkSchema(
            id="chunk2",
            parent_id="doc1",
            start_index=101,
            end_index=200,
            content="Second chunk."
        )
        
        # Create document with these chunks
        doc = ISNEDocumentSchema(
            id="doc1",
            source="test.txt",
            embedding_model="test-model",
            chunks=[chunk1, chunk2]
        )
        
        # The parent_id of chunk1 should be changed to match the document id
        self.assertEqual(doc.chunks[0].parent_id, "doc1")
        self.assertEqual(doc.chunks[1].parent_id, "doc1")
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        doc = ISNEDocumentSchema(
            id="doc1",
            source="test.txt",
            embedding_model="test-model",
            metadata={"key": "value"}
        )
        
        doc_dict = doc.to_dict()
        
        # Check fields were preserved
        self.assertEqual(doc_dict["id"], "doc1")
        self.assertEqual(doc_dict["source"], "test.txt")
        self.assertEqual(doc_dict["embedding_model"], "test-model")
        self.assertEqual(doc_dict["metadata"], {"key": "value"})


class TestISNERelationSchema(unittest.TestCase):
    """Test the ISNEDocumentRelationSchema and ISNEGraphSchema functionality."""
    
    def test_relation_schema_instantiation(self):
        """Test that ISNEDocumentRelationSchema can be instantiated with required attributes."""
        # Test minimal relation
        relation = ISNEDocumentRelationSchema(
            source_id="doc1",
            target_id="doc2",
            relation_type=ISNERelationType.SIMILARITY
        )
        
        self.assertEqual(relation.source_id, "doc1")
        self.assertEqual(relation.target_id, "doc2")
        self.assertEqual(relation.relation_type, ISNERelationType.SIMILARITY)
        self.assertEqual(relation.weight, 1.0)  # default value
        self.assertFalse(relation.bidirectional)  # default value
        self.assertEqual(relation.metadata, {})  # default value
        self.assertIsNotNone(relation.id)  # auto-generated
        self.assertIsNotNone(relation.created_at)  # auto-generated
        
        # Test with all attributes
        relation = ISNEDocumentRelationSchema(
            source_id="doc1",
            target_id="doc2",
            relation_type=ISNERelationType.REFERENCE,
            weight=0.7,
            bidirectional=True,
            metadata={"author": "test"},
            id="relation1",
            created_at=datetime.now()
        )
        
        self.assertEqual(relation.source_id, "doc1")
        self.assertEqual(relation.target_id, "doc2")
        self.assertEqual(relation.relation_type, ISNERelationType.REFERENCE)
        self.assertEqual(relation.weight, 0.7)
        self.assertTrue(relation.bidirectional)
        self.assertEqual(relation.metadata, {"author": "test"})
        self.assertEqual(relation.id, "relation1")
    
    def test_weight_validation(self):
        """Test validation of weight value."""
        # Valid values
        relation = ISNEDocumentRelationSchema(
            source_id="doc1",
            target_id="doc2",
            relation_type=ISNERelationType.SIMILARITY,
            weight=0.0
        )
        
        relation = ISNEDocumentRelationSchema(
            source_id="doc1",
            target_id="doc2",
            relation_type=ISNERelationType.SIMILARITY,
            weight=0.5
        )
        
        relation = ISNEDocumentRelationSchema(
            source_id="doc1",
            target_id="doc2",
            relation_type=ISNERelationType.SIMILARITY,
            weight=1.0
        )
        
        # Invalid values - should be between 0 and 1
        with self.assertRaises(ValidationError):
            ISNEDocumentRelationSchema(
                source_id="doc1",
                target_id="doc2",
                relation_type=ISNERelationType.SIMILARITY,
                weight=-0.1
            )
        
        with self.assertRaises(ValidationError):
            ISNEDocumentRelationSchema(
                source_id="doc1",
                target_id="doc2",
                relation_type=ISNERelationType.SIMILARITY,
                weight=1.1
            )
    
    def test_custom_relation_type(self):
        """Test handling of custom relation types."""
        # Using the CUSTOM relation type directly
        relation = ISNEDocumentRelationSchema(
            source_id="doc1",
            target_id="doc2",
            relation_type=ISNERelationType.CUSTOM
        )
        
        self.assertEqual(relation.relation_type, ISNERelationType.CUSTOM)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        relation = ISNEDocumentRelationSchema(
            source_id="doc1",
            target_id="doc2",
            relation_type=ISNERelationType.SIMILARITY,
            weight=0.8,
            metadata={"key": "value"}
        )
        
        relation_dict = relation.to_dict()
        
        # Check fields were preserved
        self.assertEqual(relation_dict["source_id"], "doc1")
        self.assertEqual(relation_dict["target_id"], "doc2")
        self.assertEqual(relation_dict["relation_type"], "similarity")
        self.assertEqual(relation_dict["weight"], 0.8)
        self.assertEqual(relation_dict["metadata"], {"key": "value"})


class TestISNEGraphSchema(unittest.TestCase):
    """Test the ISNEGraphSchema functionality."""
    
    def test_graph_schema_instantiation(self):
        """Test that ISNEGraphSchema can be instantiated with default values."""
        # Test with default values
        graph = ISNEGraphSchema()
        
        self.assertEqual(graph.nodes, {})
        self.assertEqual(graph.edges, [])
        self.assertEqual(graph.metadata, {})
        self.assertIsNotNone(graph.id)
        self.assertIsNotNone(graph.created_at)
        self.assertIsNone(graph.name)
        self.assertIsNone(graph.description)
        
        # Test with custom values
        graph = ISNEGraphSchema(
            id="graph1",
            name="Test Graph",
            description="A test graph",
            metadata={"project": "test"}
        )
        
        self.assertEqual(graph.id, "graph1")
        self.assertEqual(graph.name, "Test Graph")
        self.assertEqual(graph.description, "A test graph")
        self.assertEqual(graph.metadata, {"project": "test"})
    
    def test_adding_nodes_and_edges(self):
        """Test adding nodes and edges to the graph."""
        graph = ISNEGraphSchema()
        
        # Add nodes
        graph.add_node("node1", {"label": "Node 1", "type": "document"})
        graph.add_node("node2", {"label": "Node 2", "type": "document"})
        
        self.assertEqual(len(graph.nodes), 2)
        self.assertEqual(graph.nodes["node1"]["label"], "Node 1")
        self.assertEqual(graph.nodes["node2"]["type"], "document")
        
        # Add an edge
        relation = ISNEDocumentRelationSchema(
            source_id="node1",
            target_id="node2",
            relation_type=ISNERelationType.SIMILARITY,
            weight=0.9,
            bidirectional=False
        )
        
        graph.add_edge(relation)
        
        self.assertEqual(len(graph.edges), 1)
        self.assertEqual(graph.edges[0].source_id, "node1")
        self.assertEqual(graph.edges[0].target_id, "node2")
        
        # Add a bidirectional edge
        relation = ISNEDocumentRelationSchema(
            source_id="node1",
            target_id="node2",
            relation_type=ISNERelationType.SAME_DOCUMENT,
            weight=1.0,
            bidirectional=True
        )
        
        graph.add_edge(relation)
        
        # Should add two edges for bidirectional
        self.assertEqual(len(graph.edges), 3)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        graph = ISNEGraphSchema(
            name="Test Graph",
            description="A test graph",
            metadata={"project": "test"}
        )
        
        # Add nodes
        graph.add_node("node1", {"label": "Node 1"})
        graph.add_node("node2", {"label": "Node 2"})
        
        # Add an edge
        relation = ISNEDocumentRelationSchema(
            source_id="node1",
            target_id="node2",
            relation_type=ISNERelationType.SIMILARITY,
            weight=0.9
        )
        
        graph.add_edge(relation)
        
        graph_dict = graph.to_dict()
        
        # Check fields were preserved
        self.assertEqual(graph_dict["name"], "Test Graph")
        self.assertEqual(graph_dict["description"], "A test graph")
        self.assertEqual(graph_dict["metadata"], {"project": "test"})
        self.assertEqual(len(graph_dict["nodes"]), 2)
        self.assertEqual(len(graph_dict["edges"]), 1)


if __name__ == "__main__":
    unittest.main()
