#!/usr/bin/env python3
"""
End-to-end integration test for HADES-PathRAG documentation processing.

This test verifies the entire pipeline from ingestion through querying:
1. Chonky semantic chunking for documentation
2. ISNE embedding generation
3. ArangoDB storage and retrieval
4. PathRAG path ranking algorithm
"""

import os
import sys
import unittest
import tempfile
import shutil
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Import the scripts we want to test
sys.path.append(str(project_root / "scripts"))

# Create a test logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our modules
from src.storage.arango.connection import ArangoConnection
from src.ingest.repository.arango_repository import ArangoRepository
from scripts.ingest_documentation import create_documentation_config, ingest_documentation
from scripts.query_documentation import query_documentation


class DocumentationE2ETest(unittest.TestCase):
    """End-to-end test for documentation processing pipeline."""
    
    # Test database name (will be created and destroyed during test)
    TEST_DB_NAME = "pathrag_e2e_test"
    
    # Sample documentation files for testing
    SAMPLE_DOCS = {
        "architecture.md": """# HADES-PathRAG Architecture
        
## Overview
HADES-PathRAG is a hybrid system combining semantic chunking with graph-based retrieval.

## Components
* Chonky: Semantic chunking for non-code content
* ISNE: Inductive Shallow Node Embedding
* PathRAG: Path-based retrieval and ranking algorithm

## Data Flow
1. Documents are semantically chunked with Chonky
2. ISNE builds a graph representation
3. PathRAG ranks paths through the graph
""",
        "setup.md": """# Setup Guide
        
## Prerequisites
* Python 3.9+
* ArangoDB 3.8+
* CUDA-compatible GPU (recommended)

## Installation
```bash
pip install -r requirements.txt
```

## Configuration
Edit `config.yaml` to set up your embedding model and database connection.
""",
        "usage.md": """# Usage Guide
        
## Ingesting Documents
Use the `ingest_documentation.py` script to add documents to the system.

## Querying
The `query_documentation.py` script allows semantic search through the document graph.

## Advanced Features
* Path ranking with customizable weights
* Inter-document relationships
* Hybrid code-documentation understanding
"""
    }
    
    @classmethod
    def setUpClass(cls):
        """Create temporary directory and sample docs."""
        # Create temp directory
        cls.temp_dir = tempfile.mkdtemp()
        
        # Create sample docs
        for filename, content in cls.SAMPLE_DOCS.items():
            doc_path = Path(cls.temp_dir) / filename
            with open(doc_path, "w") as f:
                f.write(content)
        
        # Connect to ArangoDB and prepare test database
        config = create_documentation_config()
        config["storage"]["database"] = cls.TEST_DB_NAME
        
        try:
            # Create test connection (to _system db)
            connection = ArangoConnection(
                db_name="_system",
                host=config["storage"]["host"],
                username=config["storage"]["username"],
                password=config["storage"]["password"]
            )
            
            # Drop test db if it exists
            if connection.database_exists(cls.TEST_DB_NAME):
                connection.client.delete_database(cls.TEST_DB_NAME)
                logger.info(f"Deleted existing test database: {cls.TEST_DB_NAME}")
                
            # Create test database
            connection.client.create_database(cls.TEST_DB_NAME)
            logger.info(f"Created test database: {cls.TEST_DB_NAME}")
            
        except Exception as e:
            logger.error(f"Failed to prepare test database: {e}")
            raise
    
    @classmethod
    def tearDownClass(cls):
        """Clean up temp directory and test database."""
        # Remove temporary directory
        shutil.rmtree(cls.temp_dir)
        
        # Connect to ArangoDB and drop test database
        config = create_documentation_config()
        
        try:
            # Create test connection (to _system db)
            connection = ArangoConnection(
                db_name="_system",
                host=config["storage"]["host"],
                username=config["storage"]["username"],
                password=config["storage"]["password"]
            )
            
            # Drop test db
            if connection.database_exists(cls.TEST_DB_NAME):
                connection.client.delete_database(cls.TEST_DB_NAME)
                logger.info(f"Deleted test database: {cls.TEST_DB_NAME}")
                
        except Exception as e:
            logger.warning(f"Failed to clean up test database: {e}")
    
    def test_documentation_pipeline(self):
        """Test the entire pipeline from ingestion through querying."""
        # Phase 1: Ingest documentation
        try:
            # Override database name for testing
            docs_dir = Path(self.temp_dir)
            test_config = create_documentation_config()
            test_config["storage"]["database"] = self.TEST_DB_NAME
            
            # Ingest documentation
            logger.info(f"Starting test ingestion from {docs_dir}")
            ingest_result = ingest_documentation(docs_dir, "e2e_test_dataset")
            
            # Check that ingestion succeeded
            self.assertIsNotNone(ingest_result, "Ingestion failed to return results")
            self.assertIn("results", ingest_result, "Ingestion results missing stats")
            self.assertIn("entities_processed", ingest_result["results"], "No entities processed")
            
            # Ensure we processed docs and created chunks
            entities_processed = ingest_result["results"]["entities_processed"]
            self.assertGreaterEqual(entities_processed, len(self.SAMPLE_DOCS), 
                             f"Expected at least {len(self.SAMPLE_DOCS)} entities, got {entities_processed}")
            
            logger.info(f"Ingestion completed: {entities_processed} entities processed")
            
            # Allow time for database to commit the changes
            time.sleep(2)
            
            # Phase 2: Query documentation
            # Test several queries to verify different aspects of the system
            test_queries = [
                {
                    "query": "How do I install the system?",
                    "expected_docs": ["setup.md"],
                    "min_relevance": 0.6
                },
                {
                    "query": "What is PathRAG?",
                    "expected_docs": ["architecture.md"],
                    "min_relevance": 0.6
                },
                {
                    "query": "How to search documents?",
                    "expected_docs": ["usage.md"],
                    "min_relevance": 0.6
                }
            ]
            
            for test_case in test_queries:
                logger.info(f"Testing query: {test_case['query']}")
                
                # Run query through PathRAG
                results = query_documentation(
                    query=test_case["query"],
                    max_results=5,
                    filters=None
                )
                
                # Verify query succeeded
                self.assertIsNotNone(results, "Query failed to return results")
                self.assertIn("paths", results, "Query results missing paths")
                
                # Check that we found paths
                paths = results["paths"]
                self.assertGreater(len(paths), 0, f"No paths found for query: {test_case['query']}")
                
                # Check that top path has expected relevance
                top_path = paths[0]
                self.assertGreaterEqual(
                    top_path["score"], 
                    test_case["min_relevance"],
                    f"Top path score {top_path['score']} below minimum {test_case['min_relevance']}"
                )
                
                # Check that expected documents are in the results
                found_expected = False
                for path in paths:
                    for node in path.get("nodes", []):
                        source_file = node.get("source_file", "")
                        if any(expected in source_file for expected in test_case["expected_docs"]):
                            found_expected = True
                            break
                    if found_expected:
                        break
                
                self.assertTrue(
                    found_expected, 
                    f"Expected documents {test_case['expected_docs']} not found in results"
                )
                
                logger.info(f"Query test passed: {test_case['query']}")
            
            logger.info("All query tests passed!")
            
        except Exception as e:
            logger.error(f"Test failed: {e}")
            raise
        

if __name__ == "__main__":
    unittest.main()
