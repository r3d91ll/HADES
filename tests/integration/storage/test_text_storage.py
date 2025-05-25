"""
Integration test for text storage service.

Tests storing processed documents from the pipeline into ArangoDB.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.storage.arango.connection import ArangoConnection
from src.storage.arango.text_repository import TextArangoRepository
from src.storage.text_storage import TextStorageService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_store_document(document_output_file: str) -> None:
    """
    Test storing a processed document in ArangoDB.
    
    Args:
        document_output_file: Path to the processed document output JSON file
    """
    logger.info(f"Testing text storage with file: {document_output_file}")
    
    try:
        # Load the processed document data
        with open(document_output_file, 'r') as f:
            document_data = json.load(f)
        
        logger.info(f"Loaded document data with ID: {document_data.get('id')}")
        
        # Initialize ArangoDB connection and repository
        # Use a test database to avoid affecting production data
        connection = ArangoConnection(db_name="hades_test")
        repository = TextArangoRepository(connection)
        
        # Initialize the storage service
        storage_service = TextStorageService(repository=repository)
        
        # Store the document
        document_id = await storage_service.store_processed_document(document_data)
        logger.info(f"Successfully stored document with ID: {document_id}")
        
        # Retrieve the document with its chunks
        document_with_chunks = await storage_service.get_document_with_chunks(document_id)
        if document_with_chunks:
            document = document_with_chunks.get("document", {})
            chunks = document_with_chunks.get("chunks", [])
            
            logger.info(f"Retrieved document: {document.get('title')}")
            logger.info(f"Retrieved {len(chunks)} chunks")
            
            # Verify chunk count matches
            original_chunk_count = len(document_data.get("chunks", []))
            retrieved_chunk_count = len(chunks)
            
            if original_chunk_count == retrieved_chunk_count:
                logger.info(f"Chunk count matches: {original_chunk_count}")
            else:
                logger.warning(
                    f"Chunk count mismatch: original={original_chunk_count}, "
                    f"retrieved={retrieved_chunk_count}"
                )
        else:
            logger.error(f"Failed to retrieve document with ID: {document_id}")
        
        # Test vector search if embeddings are available
        if any(chunk.get("embedding") is not None for chunk in document_data.get("chunks", [])):
            # Get first chunk with embedding for testing
            test_chunk = next(
                (c for c in document_data.get("chunks", []) if c.get("embedding") is not None), 
                None
            )
            
            if test_chunk:
                logger.info("Testing vector search with chunk embedding")
                query_vector = test_chunk.get("embedding")
                
                results = await storage_service.search_by_vector(query_vector, limit=5)
                logger.info(f"Vector search returned {len(results)} results")
                
                # Top result should be the same chunk or very similar
                if results:
                    top_result, score = results[0]
                    logger.info(f"Top result: {top_result.get('id')} with score {score:.4f}")
                    
                    # Test with ISNE embeddings if available
                    if test_chunk.get("isne_enhanced_embedding") is not None:
                        logger.info("Testing vector search with ISNE embedding")
                        isne_query_vector = test_chunk.get("isne_enhanced_embedding")
                        
                        isne_results = await storage_service.search_by_vector(
                            isne_query_vector, 
                            limit=5,
                            use_isne=True
                        )
                        logger.info(f"ISNE vector search returned {len(isne_results)} results")
        
        logger.info("Text storage test completed successfully")
        
    except Exception as e:
        logger.error(f"Error in test_store_document: {e}")
        raise

async def main() -> None:
    """Run the integration test."""
    # Path to the test output directory
    output_dir = Path("/home/todd/ML-Lab/Olympus/HADES-PathRAG/test-output/isne-fixed-test")
    
    # Test files to process
    test_files = [
        output_dir / "complete_ISNE_paper_output.json",
        output_dir / "complete_PathRAG_paper_output.json"
    ]
    
    # Process each test file
    for test_file in test_files:
        if test_file.exists():
            logger.info(f"Processing test file: {test_file}")
            await test_store_document(str(test_file))
        else:
            logger.error(f"Test file not found: {test_file}")

if __name__ == "__main__":
    asyncio.run(main())
