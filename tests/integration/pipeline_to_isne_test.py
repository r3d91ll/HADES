"""
Integration test for pipeline components leading up to ISNE module.

This script tests the document processing, chunking, and embedding stages
of the pipeline to verify they work correctly with our updated schemas
and can properly prepare data for the ISNE module.
"""

import os
import sys
import logging
import unittest
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.docproc.manager import DocumentProcessorManager
from src.chunking.text_chunkers.chonky_chunker import chunk_text
from src.embedding.base import get_adapter


class PipelineToISNETest(unittest.TestCase):
    """Integration test for pipeline components leading up to ISNE."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test resources."""
        # Initialize components
        cls.doc_processor = DocumentProcessorManager()
        cls.embedding_adapter = get_adapter("modernbert")
        
        # Create test directories if they don't exist
        cls.test_data_dir = Path("test-data/sample_docs")
        cls.test_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a simple test document if it doesn't exist
        cls.test_doc_path = cls.test_data_dir / "test_document.txt"
        if not cls.test_doc_path.exists():
            cls.test_doc_path.write_text("""
# Test Document

This is a sample document for testing the pipeline components.

## Section 1

This section contains information about the first topic.
The pipeline should process this document, chunk it, and generate embeddings.

## Section 2

This section covers a second topic with different semantic content.
We want to ensure that the chunking process properly separates these sections.
            """)
    
    def test_pipeline_integration(self):
        """Test that the pipeline components work together correctly."""
        # Step 1: Process document
        logger.info("Processing document...")
        processed_doc = self.doc_processor.process_file(str(self.test_doc_path))
        self.assertIsNotNone(processed_doc, "Document processing should return a result")
        self.assertIn("content", processed_doc, "Processed document should contain content")
        
        # Step 2: Chunk the document
        logger.info("Chunking document...")
        chunks = chunk_text({"id": "test-doc-1", "content": processed_doc["content"]})
        self.assertIsNotNone(chunks, "Chunking should return results")
        self.assertTrue(len(chunks) > 0, "Should have at least one chunk")
        
        logger.info(f"Created {len(chunks)} chunks")
        
        # Step 3: Generate embeddings
        logger.info("Generating embeddings...")
        texts = [chunk["content"] for chunk in chunks]
        embeddings = self.embedding_adapter.embed_batch(texts)
        
        self.assertEqual(len(embeddings), len(chunks),
                        "Should have one embedding per chunk")
        
        # Print a sample of the embedding
        first_embedding = embeddings[0]
        logger.info(f"Embedding shape: {np.array(first_embedding).shape}")
        logger.info(f"Embedding sample (first 5 values): {np.array(first_embedding)[:5]}")
        
        # Step 4: Prepare the data structure for ISNE
        logger.info("Preparing data for ISNE...")
        documents = []
        document_id = "test-doc-1"
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{document_id}_chunk_{i}"
            documents.append({
                "id": chunk_id,
                "content": chunk["content"],
                "embedding": embedding,
                "metadata": {
                    "document_id": document_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            })
        
        # Verify the structure
        self.assertEqual(len(documents), len(chunks),
                        "Should have one document entry per chunk")
                        
        # Print the final structure that would be passed to ISNE
        logger.info(f"Final document structure ready for ISNE (showing first entry):")
        sample_doc = {k: v if k != 'embedding' else f"<embedding vector with shape {np.array(v).shape}>"
                    for k, v in documents[0].items()}
        logger.info(f"{sample_doc}")
        
        # Return the result for any further testing
        return documents


if __name__ == "__main__":
    unittest.main()
