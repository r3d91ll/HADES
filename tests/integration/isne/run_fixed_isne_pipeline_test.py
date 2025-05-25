"""
Fixed ISNE pipeline test with working embeddings.

This script processes PDFs using the patched pipeline with properly loaded embedding adapters,
demonstrating the complete ISNE enhancement workflow.
"""

import asyncio
import sys
import os
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the fixed run_pipeline function
# Import and register adapters to ensure they're available
from src.embedding.adapters.modernbert_adapter import ModernBERTEmbeddingAdapter
from src.embedding.base import register_adapter
register_adapter("modernbert", ModernBERTEmbeddingAdapter)

# Now import the pipeline components
from src.pipelines.ingest.orchestrator.patched_pdf_pipeline import PDFPipeline, run_pipeline

async def main():
    """Run the fixed ISNE-enhanced pipeline on test PDFs."""
    # Set up paths
    test_data_dir = Path("/home/todd/ML-Lab/Olympus/HADES-PathRAG/test-data")
    output_dir = Path("/home/todd/ML-Lab/Olympus/HADES-PathRAG/test-output/isne-fixed-test")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find PDF files
    pdf_files = list(test_data_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files found in {test_data_dir}")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Configure ISNE
    isne_options = {
        "use_isne": True,  # Enable ISNE in collection mode (no trained model yet)
    }
    
    # Set up embedding options
    embedding_options = {
        "adapter_name": "modernbert",
        "model_name": "answerdotai/ModernBERT-base",
        "device": "cpu",
        "batch_size": 4,
        "pooling_strategy": "cls",
        "normalize_embeddings": True
    }
    
    # Verify adapter is registered
    from src.embedding.base import _adapter_registry
    logger.info(f"Available adapters: {list(_adapter_registry.keys())}")
    
    # Initialize pipeline directly to test
    pipeline = PDFPipeline(
        output_dir=output_dir,
        embedding_options=embedding_options,
        isne_options=isne_options,
        save_intermediate_results=True
    )
    
    # Verify adapter is initialized
    logger.info(f"Pipeline embedding adapter: {pipeline.embedding_adapter}")
    
    # Process each PDF file
    for pdf_file in pdf_files:
        logger.info(f"Processing {pdf_file}...")
        result = await pipeline.process_pdf(pdf_file)
        if result:
            logger.info(f"Successfully processed {pdf_file}")
            # Check if embeddings were generated
            chunks = result.get("chunks", [])
            embedding_count = sum(1 for chunk in chunks if isinstance(chunk, dict) and chunk.get("embedding") is not None)
            logger.info(f"Generated {embedding_count} embeddings out of {len(chunks)} chunks")
        else:
            logger.error(f"Failed to process {pdf_file}")
    
    logger.info("ISNE pipeline test with fixed embeddings completed")

if __name__ == "__main__":
    asyncio.run(main())
