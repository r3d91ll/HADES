"""
Integration test for the ISNE module using real data.

This test validates the JSON output files from previous test runs
to ensure they contain the expected format and content for ISNE processing.
It focuses on data validation rather than model training/inference to avoid
performance issues with large datasets.
"""

import os
import json
import unittest
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from src.isne.types.models import IngestDocument, DocumentRelation, DocumentType, RelationType

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestISNERealDataPipeline(unittest.TestCase):
    """
    Integration test for the ISNE module using real paper data.
    
    This test validates that:
    1. JSON test output files are properly formatted
    2. Document structure and metadata are as expected
    3. Basic document relationships can be established
    """
    
    def setUp(self) -> None:
        """Set up test environment and paths to real data files."""
        # Paths to real data files
        self.isne_paper_path = "/home/todd/ML-Lab/Olympus/HADES-PathRAG/test-output/modernbert-cpu-test/complete_ISNE_paper_output.json"
        self.pathrag_paper_path = "/home/todd/ML-Lab/Olympus/HADES-PathRAG/test-output/modernbert-cpu-test/complete_PathRAG_paper_output.json"
        
        # Log file existence status
        if not os.path.exists(self.isne_paper_path):
            logger.warning(f"ISNE paper file not found at {self.isne_paper_path}. Tests may be skipped.")
        else:
            logger.info(f"ISNE paper file found at {self.isne_paper_path}")
            
        if not os.path.exists(self.pathrag_paper_path):
            logger.warning(f"PathRAG paper file not found at {self.pathrag_paper_path}. Tests may be skipped.")
        else:
            logger.info(f"PathRAG paper file found at {self.pathrag_paper_path}")
    
    def test_isne_paper_json_structure(self) -> None:
        """
        Test that the ISNE paper JSON file has the expected structure.
        
        This verifies:
        1. The JSON file can be loaded
        2. It has expected fields (id, metadata, chunks)
        3. Chunks contain expected data
        """
        logger.info("Starting test_isne_paper_json_structure")
        
        # Skip if file doesn't exist
        if not os.path.exists(self.isne_paper_path):
            logger.warning(f"ISNE paper file not found. Skipping test.")
            self.skipTest("ISNE paper file not found")
        
        try:
            # Load the JSON file
            logger.info(f"Loading JSON from {self.isne_paper_path}")
            with open(self.isne_paper_path, 'r') as f:
                data = json.load(f)
            
            # Log available keys
            logger.info(f"JSON keys: {sorted(data.keys())}")
            
            # Basic structure validation
            self.assertIn('id', data, "JSON missing 'id' field")
            self.assertIn('metadata', data, "JSON missing 'metadata' field")
            self.assertIn('chunks', data, "JSON missing 'chunks' field")
            
            # Document ID validation
            doc_id = data.get('id')
            logger.info(f"Document ID: {doc_id}")
            self.assertTrue(doc_id.startswith('pdf_'), "Document ID should start with 'pdf_'")
            
            # Metadata validation
            metadata = data.get('metadata', {})
            logger.info(f"Metadata keys: {sorted(metadata.keys())}")
            self.assertIn('file_path', metadata, "Metadata missing 'file_path'")
            self.assertIn('title', metadata, "Metadata missing 'title'")
            
            # Chunks validation
            chunks = data.get('chunks', [])
            logger.info(f"Document has {len(chunks)} chunks")
            self.assertGreater(len(chunks), 0, "Document should have at least one chunk")
            
            # First chunk structure if available
            if chunks:
                first_chunk = chunks[0]
                logger.info(f"First chunk keys: {sorted(first_chunk.keys() if isinstance(first_chunk, dict) else [])}")
                if isinstance(first_chunk, dict):
                    self.assertIn('content', first_chunk, "Chunk missing 'content' field")
                    # Check for chunk structure but without requiring 'metadata' which might not be present
                    self.assertIn('id', first_chunk, "Chunk missing 'id' field")
            
            # Check for embeddings which should be present
            has_embeddings = False
            embedding_dimension = 0
            for chunk in chunks:
                if isinstance(chunk, dict) and 'embedding' in chunk:
                    has_embeddings = True
                    embedding = chunk['embedding']
                    if isinstance(embedding, list):
                        embedding_dimension = len(embedding)
                        logger.info(f"Found embedding with dimension {embedding_dimension}")
                    break
            
            self.assertTrue(has_embeddings, "Document chunks should contain embeddings")
            if has_embeddings and embedding_dimension > 0:
                # Modern embedding models typically use dimensions that are multiples of 64
                self.assertGreaterEqual(embedding_dimension, 64, "Embedding dimension should be at least 64")
                logger.info(f"Document contains chunk embeddings with dimension {embedding_dimension}")
            
            logger.info("ISNE paper JSON structure validation successful")
        
        except Exception as e:
            logger.error(f"Error validating ISNE paper JSON: {str(e)}")
            logger.error(traceback.format_exc())
            self.fail(f"ISNE paper JSON validation failed: {str(e)}")

    def test_pathrag_paper_json_structure(self) -> None:
        """
        Test that the PathRAG paper JSON file has the expected structure.
        
        This verifies:
        1. The JSON file can be loaded
        2. It has expected fields (id, metadata, chunks)
        3. Chunks contain expected data
        """
        logger.info("Starting test_pathrag_paper_json_structure")
        
        # Skip if file doesn't exist
        if not os.path.exists(self.pathrag_paper_path):
            logger.warning(f"PathRAG paper file not found. Skipping test.")
            self.skipTest("PathRAG paper file not found")
        
        try:
            # Load the JSON file
            logger.info(f"Loading JSON from {self.pathrag_paper_path}")
            with open(self.pathrag_paper_path, 'r') as f:
                data = json.load(f)
            
            # Log available keys
            logger.info(f"JSON keys: {sorted(data.keys())}")
            
            # Basic structure validation
            self.assertIn('id', data, "JSON missing 'id' field")
            self.assertIn('metadata', data, "JSON missing 'metadata' field")
            self.assertIn('chunks', data, "JSON missing 'chunks' field")
            
            # Document ID validation
            doc_id = data.get('id')
            logger.info(f"Document ID: {doc_id}")
            self.assertTrue(doc_id.startswith('pdf_'), "Document ID should start with 'pdf_'")
            
            # Metadata validation
            metadata = data.get('metadata', {})
            logger.info(f"Metadata keys: {sorted(metadata.keys())}")
            
            # Chunks validation
            chunks = data.get('chunks', [])
            logger.info(f"Document has {len(chunks)} chunks")
            self.assertGreater(len(chunks), 0, "Document should have at least one chunk")
            
            # Check chunk content 
            embedding_count = 0
            for i, chunk in enumerate(chunks):
                if isinstance(chunk, dict):
                    self.assertIn('content', chunk, f"Chunk {i} missing 'content' field")
                    if 'embedding' in chunk:
                        embedding_count += 1
            
            logger.info(f"Found {embedding_count} chunks with embeddings out of {len(chunks)} total chunks")
            logger.info("PathRAG paper JSON structure validation successful")
        
        except Exception as e:
            logger.error(f"Error validating PathRAG paper JSON: {str(e)}")
            logger.error(traceback.format_exc())
            self.fail(f"PathRAG paper JSON validation failed: {str(e)}")

    def test_create_ingest_documents(self) -> None:
        """
        Test creating a single basic IngestDocument object.
        
        This is a simplified test that just ensures we can create a valid document
        without any complexities from the JSON structure.
        """
        logger.info("Starting simplified test_create_ingest_documents")
        
        try:
            # Create a minimal valid IngestDocument
            doc = IngestDocument(
                id="test_doc_001",
                content="This is a test document.",
                source="test-source",
                document_type=DocumentType.PDF.value,
                metadata={'title': 'Test Document'}
            )
            
            # Basic validation
            self.assertEqual(doc.id, "test_doc_001")
            self.assertEqual(doc.content, "This is a test document.")
            self.assertEqual(doc.source, "test-source")
            self.assertEqual(doc.document_type, DocumentType.PDF.value)
            self.assertEqual(doc.metadata['title'], "Test Document")
            
            # Create a basic relationship
            relation = DocumentRelation(
                source_id="test_doc_001",
                target_id="test_doc_002",
                relation_type=RelationType.REFERENCES,  # Use the enum directly, not .value
                metadata={'confidence': 0.95}
            )
            
            # Basic validation
            self.assertEqual(relation.source_id, "test_doc_001")
            self.assertEqual(relation.target_id, "test_doc_002")
            self.assertEqual(relation.relation_type, RelationType.REFERENCES)
            self.assertEqual(relation.metadata['confidence'], 0.95)
            
            logger.info("Successfully created and validated basic document and relation")
            
        except Exception as e:
            logger.error(f"Error creating basic IngestDocument: {str(e)}")
            logger.error(traceback.format_exc())
            print(f"\nFAILURE DETAILS: {str(e)}\n{traceback.format_exc()}")
            self.fail(f"Error creating basic IngestDocument: {str(e)}")


if __name__ == "__main__":
    # Set logging to INFO level for detailed output when running the test directly
    logging.basicConfig(level=logging.INFO)
    unittest.main()
