"""
Test script for the ISNE pipeline with ModernBERT outputs.

This script tests the end-to-end functionality of processing ModernBERT outputs
through the ISNE pipeline, validating each step of the process.
"""

import os
import sys
import logging
import unittest
from pathlib import Path
import json
import shutil
import torch
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.isne.pipeline import ISNEPipeline, PipelineConfig
from src.isne.types.models import ISNEConfig, RelationType
from src.isne.loaders.modernbert_loader import ModernBERTLoader

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test constants
TEST_DATA_DIR = Path(__file__).parent.parent / "data" / "test_data"
TEST_OUTPUT_DIR = Path(__file__).parent.parent / "data" / "test_output" / "isne"
MODERNBERT_TEST_FILE = TEST_DATA_DIR / "modernbert_sample_output.json"


class TestModernBERTISNEPipeline(unittest.TestCase):
    """Test cases for the ISNE pipeline with ModernBERT outputs."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Create test directories
        os.makedirs(TEST_DATA_DIR, exist_ok=True)
        os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
        
        # Create sample ModernBERT output if it doesn't exist
        if not MODERNBERT_TEST_FILE.exists():
            cls._create_sample_modernbert_output()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        # Clean up output directory
        if TEST_OUTPUT_DIR.exists():
            shutil.rmtree(TEST_OUTPUT_DIR)
    
    @classmethod
    def _create_sample_modernbert_output(cls):
        """Create a sample ModernBERT output file for testing."""
        # Create a simple document structure with embeddings
        sample_data = {
            "id": "test_document",
            "title": "Test Document",
            "format": "document",
            "content": "This is a test document for ISNE processing.",
            "source": "test_source.txt",
            "metadata": {
                "source": "test",
                "category": "test"
            },
            "chunks": [
                {
                    "id": "chunk1",
                    "content": "This is chunk 1 of the test document.",
                    "embedding": [0.1] * 768,  # 768-dimensional embedding
                    "type": "text",
                    "metadata": {"position": 1}
                },
                {
                    "id": "chunk2",
                    "content": "This is chunk 2 of the test document.",
                    "embedding": [0.2] * 768,
                    "type": "text",
                    "metadata": {"position": 2}
                },
                {
                    "id": "chunk3",
                    "content": "This is chunk 3 of the test document.",
                    "embedding": [0.3] * 768,
                    "type": "text",
                    "metadata": {"position": 3}
                }
            ],
            "relations": [
                {
                    "source_id": "chunk1",
                    "target_id": "chunk2",
                    "relation_type": "follows",
                    "weight": 0.8
                },
                {
                    "source_id": "chunk2",
                    "target_id": "chunk3",
                    "relation_type": "follows",
                    "weight": 0.9
                }
            ]
        }
        
        # Save to file
        with open(MODERNBERT_TEST_FILE, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2)
    
    def test_modernbert_loader(self):
        """Test the ModernBERT loader functionality."""
        # Create loader
        loader = ModernBERTLoader()
        
        # Load data
        result = loader.load(MODERNBERT_TEST_FILE)
        
        # Basic validation
        self.assertIsNotNone(result)
        self.assertTrue(len(result.documents) > 0)
        self.assertTrue(len(result.relations) > 0)
        
        # Validate document structure
        doc = result.documents[0]  # Parent document
        self.assertIsNotNone(doc.embedding)
        self.assertIsNotNone(doc.chunks)
        self.assertEqual(len(doc.chunks), 3)
        
        # Validate relationships - expect both sequential and similarity relationships
        # We expect: 
        # - 2 sequential relationships (chunk1->chunk2, chunk2->chunk3)
        # - 3 parent-child relationships (parent->chunk1, parent->chunk2, parent->chunk3)
        # - 3 similarity relationships (chunk1<->chunk2, chunk2<->chunk3, chunk1<->chunk3)
        self.assertGreaterEqual(len(result.relations), 5)  # At minimum, expect sequential + parent-child
        
        # Check for specific relationship types
        follows_relations = [r for r in result.relations if r.relation_type == RelationType.FOLLOWS]
        contains_relations = [r for r in result.relations if r.relation_type == RelationType.CONTAINS]
        similarity_relations = [r for r in result.relations if r.relation_type == RelationType.SIMILAR_TO]
        
        self.assertGreaterEqual(len(follows_relations), 2)  # At least 2 sequential relationships
        self.assertGreaterEqual(len(contains_relations), 3)  # At least 3 parent-child relationships
    
    def test_modernbert_isne_pipeline_end_to_end(self):
        """Test end-to-end processing of ModernBERT output through ISNE pipeline."""
        # Create pipeline config
        config = PipelineConfig(
            pipeline_name="modernbert_isne_test",
            output_dir=str(TEST_OUTPUT_DIR),
            enable_isne_model=True,
            use_gpu=torch.cuda.is_available()
        )
        
        # Create pipeline
        pipeline = ISNEPipeline(config)
        
        # Process ModernBERT output
        output_file = TEST_OUTPUT_DIR / "isne_enhanced_output.json"
        stats = pipeline.process_modernbert_output(
            input_file=MODERNBERT_TEST_FILE,
            output_file=output_file,
            use_gpu=False  # Force CPU for testing
        )
        
        # Validate results
        self.assertIsNotNone(stats)
        self.assertFalse("error" in stats)
        self.assertTrue(output_file.exists())
        
        # Check output file content
        with open(output_file, 'r', encoding='utf-8') as f:
            output_data = json.load(f)
        
        # Basic validation of output structure
        self.assertIn("documents", output_data)
        self.assertIn("relations", output_data)
        self.assertIn("pipeline_info", output_data)
        
        # Validate that documents have enhanced embeddings
        doc = output_data["documents"][0]
        self.assertIn("embedding", doc)
        self.assertIn("embedding_model", doc)
        self.assertIn("isne", doc["embedding_model"].lower())
        
        # Validate metadata includes ISNE processing information
        self.assertIn("metadata", doc)
        self.assertIn("isne_processed", doc["metadata"])
        self.assertTrue(doc["metadata"]["isne_processed"])
    
    def test_graph_construction(self):
        """Test the graph construction component of the ISNE pipeline."""
        # Create pipeline
        config = PipelineConfig(
            pipeline_name="graph_test",
            output_dir=str(TEST_OUTPUT_DIR),
            enable_graph_processing=True,
            enable_isne_model=False  # Disable ISNE model to focus on graph construction
        )
        pipeline = ISNEPipeline(config)
        
        # Load data
        loader = ModernBERTLoader()
        result = loader.load(MODERNBERT_TEST_FILE)
        
        # Process through graph processor
        from src.isne.processors.isne_graph_processor import ISNEGraphProcessor
        graph_processor = ISNEGraphProcessor()
        graph_result = graph_processor.process(
            documents=result.documents,
            relations=result.relations,
            dataset=result.dataset
        )
        
        # Validate graph construction
        self.assertIsNotNone(graph_result)
        self.assertIn("node_features", graph_result.metadata)
        self.assertIn("edge_index", graph_result.metadata)
        self.assertIn("edge_weights", graph_result.metadata)
        
        # Check that node features match document count
        node_features = graph_result.metadata["node_features"]
        self.assertEqual(len(node_features), len(result.documents))
        
        # Check edge construction
        edge_index = graph_result.metadata["edge_index"]
        self.assertIsNotNone(edge_index)
        if edge_index:  # If we have edges
            self.assertEqual(len(edge_index), 2)  # Source and target lists
            
            # Edge count should match relation count (plus bidirectional edges and self-loops)
            expected_edge_count = len(result.relations) * 2 + len(result.documents)  # *2 for bidirectional, +docs for self-loops
            self.assertEqual(len(edge_index[0]), expected_edge_count)


if __name__ == "__main__":
    unittest.main()
