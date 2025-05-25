"""
Integration test for PDF processing in the full pipeline.

This test validates that the full pipeline can process PDF files correctly,
including document processing, chunking, embedding generation, and storage.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipelines.ingest.orchestrator.ingestor import RepositoryIngestor
from src.docproc.manager import DocumentProcessorManager
from src.embedding.base import get_adapter
from src.storage.arango.connection import ArangoConnection

# Explicitly import the ModernBERT adapter to ensure it gets registered
from src.embedding.adapters.modernbert_adapter import ModernBERTEmbeddingAdapter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_pdf_pipeline():
    """Test the full pipeline with PDF files."""
    # Create a temporary directory with the PDF files
    temp_dir = Path("temp_pdf_test")
    temp_dir.mkdir(exist_ok=True)
    
    # Copy the PDF files to the temporary directory
    pdf_files = [
        Path("docs/PathRAG_paper.pdf"),
        Path("test-data/ISNE_paper.pdf")
    ]
    
    for pdf_file in pdf_files:
        if not pdf_file.exists():
            logger.error(f"Test PDF not found: {pdf_file}")
            continue
        
        # Create a copy in the temp directory
        target_path = temp_dir / pdf_file.name
        with open(pdf_file, "rb") as src, open(target_path, "wb") as dst:
            dst.write(src.read())
        
        logger.info(f"Copied {pdf_file} to {target_path}")
    
    # Initialize the pipeline components
    doc_processor = DocumentProcessorManager()
    
    # Use our new ModernBERT adapter for embeddings
    embedding_adapter = get_adapter(
        "modernbert", 
        pooling_strategy="cls",
        max_length=8192,
        normalize_embeddings=True
    )
    
    # Create a connection to ArangoDB with a unique database name for testing
    connection = ArangoConnection(
        db_name="pdf_test_db",
        host=os.environ.get("ARANGO_HOST", "http://localhost:8529"),
        username=os.environ.get("ARANGO_USERNAME", "root"),
        password=os.environ.get("ARANGO_PASSWORD", ""),
    )
    
    # Initialize the ingestor with our components
    ingestor = RepositoryIngestor(
        connection=connection,
        doc_processor=doc_processor,
        embedding_adapter=embedding_adapter,
        initialize_db=True,  # Create the database structure
        batch_size=2,
        max_concurrency=2,
    )
    
    try:
        # Run the ingestion process
        logger.info("Starting PDF pipeline test...")
        stats = await ingestor.ingest(
            repo_path=temp_dir,
            include_patterns=["**/*.pdf"],
            exclude_patterns=[],
        )
        
        # Log the results
        logger.info("PDF pipeline test completed")
        logger.info(f"Statistics: {stats.to_dict()}")
        
        # Validate the results
        assert stats.processed_files > 0, "No files were processed"
        assert stats.entities_created > 0, "No entities were created"
        assert stats.embeddings_created > 0, "No embeddings were created"
        assert stats.nodes_created > 0, "No nodes were stored in the database"
        
        # Check if we have document and chunk entities in the database
        doc_count = await connection.db.collection("nodes").count()
        logger.info(f"Total nodes in database: {doc_count}")
        
        # Query for document entities
        doc_cursor = await connection.db.aql.execute(
            "FOR doc IN nodes FILTER doc.type == 'document' RETURN doc"
        )
        documents = await doc_cursor.all()
        logger.info(f"Document entities: {len(documents)}")
        
        # Query for chunk entities
        chunk_cursor = await connection.db.aql.execute(
            "FOR chunk IN nodes FILTER chunk.type == 'chunk' RETURN chunk"
        )
        chunks = await chunk_cursor.all()
        logger.info(f"Chunk entities: {len(chunks)}")
        
        # Check if we have embeddings
        embed_cursor = await connection.db.aql.execute(
            "FOR embed IN embeddings RETURN embed"
        )
        embeddings = await embed_cursor.all()
        logger.info(f"Embeddings: {len(embeddings)}")
        
        # Validate ModernBERT embedding dimensions
        if embeddings:
            # Check a sample embedding's dimensions
            sample_embedding = embeddings[0].get('vector', [])
            embedding_dim = len(sample_embedding)
            logger.info(f"ModernBERT embedding dimensions: {embedding_dim}")
            
            # ModernBERT should produce 768-dimensional embeddings
            assert embedding_dim == 768, f"Expected ModernBERT embedding dimension 768, got {embedding_dim}"
        
        # Check if we have relationships
        edge_cursor = await connection.db.aql.execute(
            "FOR edge IN edges RETURN edge"
        )
        edges = await edge_cursor.all()
        logger.info(f"Edges: {len(edges)}")
        
        # Get processing metadata to check embedding performance
        meta_cursor = await connection.db.aql.execute(
            "FOR doc IN nodes FILTER doc.type == 'document' AND doc.processing_metadata != null "  
            "RETURN doc.processing_metadata"
        )
        processing_metadata = await meta_cursor.all()
        
        # Extract embedding performance metrics if available
        embedding_metrics = {}
        if processing_metadata:
            for meta in processing_metadata:
                if meta and 'embedding' in meta:
                    embedding_info = meta['embedding']
                    for key, value in embedding_info.items():
                        # Aggregate the metrics
                        if key in embedding_metrics:
                            if isinstance(value, (int, float)):
                                embedding_metrics[key] += value
                            else:
                                embedding_metrics[key] = value
                        else:
                            embedding_metrics[key] = value
        
        logger.info(f"Embedding performance metrics: {embedding_metrics}")
        
        # Verify ModernBERT is being used
        if embedding_metrics:
            adapter_name = embedding_metrics.get('adapter', '')
            assert 'modernbert' in adapter_name.lower(), f"Expected ModernBERT adapter, got {adapter_name}"
        
        # Calculate performance metrics
        benchmark_data = {
            "documents_per_second": stats.processed_files / stats.elapsed_time if stats.elapsed_time > 0 else 0,
            "chunks_per_second": len(chunks) / stats.elapsed_time if stats.elapsed_time > 0 else 0,
            "embedding_dimensions": embedding_dim if embeddings else 0,
            "average_tokens_per_chunk": sum(len(c.get('content', '').split()) for c in chunks) / len(chunks) if chunks else 0
        }
        
        # Organize test results for alignment with standard protocol
        return {
            "stats": stats.to_dict(),
            "counts": {
                "documents": len(documents),
                "chunks": len(chunks),
                "embeddings": len(embeddings),
                "edges": len(edges),
            },
            "embedding": {
                "adapter": embedding_metrics.get('adapter', 'modernbert'),
                "model": embedding_metrics.get('model', 'answerdotai/ModernBERT-base'),
                "dimensions": embedding_dim if embeddings else 0,
                "pooling_strategy": embedding_metrics.get('pooling_strategy', 'cls')
            },
            "benchmark": benchmark_data
        }
    
    finally:
        # Clean up the temporary directory
        for file in temp_dir.glob("*"):
            file.unlink()
        temp_dir.rmdir()
        
        # Drop the test database
        try:
            await connection.sys_db.delete_database("pdf_test_db")
            logger.info("Cleaned up test database")
        except Exception as e:
            logger.warning(f"Error cleaning up test database: {e}")


async def main():
    """Run the PDF pipeline test."""
    try:
        results = await test_pdf_pipeline()
        logger.info("Test completed successfully")
        logger.info(f"Results: {results}")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
