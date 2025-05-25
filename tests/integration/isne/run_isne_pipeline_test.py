"""
Test script to run the enhanced PDF pipeline with ISNE integration.

This script processes PDF files using the enhanced pipeline that includes ISNE
to create knowledge graph-aware embeddings.
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

# Import the pipeline
from src.pipelines.ingest.orchestrator.pdf_pipeline_prototype import run_pipeline


async def main():
    """Run the ISNE-enhanced pipeline on test PDFs."""
    # Set up paths
    test_data_dir = Path("/home/todd/ML-Lab/Olympus/HADES-PathRAG/test-data")
    output_dir = Path("/home/todd/ML-Lab/Olympus/HADES-PathRAG/test-output/isne-enhanced")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find PDF files
    pdf_files = list(test_data_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files found in {test_data_dir}")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Configure ISNE (using collection mode since we don't have a trained model yet)
    isne_options = {
        "use_isne": True,  # Enable ISNE in collection mode
        # No model path, so it will use collection mode only
    }
    
    # Import embedding configuration at runtime to ensure it's available
    try:
        from src.embedding.processors import add_embeddings_to_document
        from src.embedding.base import get_adapter
        logger.info("Successfully imported embedding modules")
        
        # Initialize the adapter to verify it works
        adapter_name = "modernbert"
        adapter = get_adapter(
            adapter_name,
            model_name="answerdotai/ModernBERT-base",
            device="cpu",
            batch_size=4,
            normalize_embeddings=True
        )
        logger.info(f"Successfully created {adapter_name} adapter")
    except Exception as e:
        logger.error(f"Error initializing embedding adapter: {e}")
        logger.error("Pipeline will run without embeddings")
    
    # Set up embedding options
    embedding_options = {
        "adapter_name": "modernbert",
        "model_name": "answerdotai/ModernBERT-base",
        "device": "cpu",
        "batch_size": 4,
        "normalize_embeddings": True
    }
    
    # Run the pipeline
    await run_pipeline(
        pdf_paths=pdf_files,
        output_dir=output_dir,
        isne_options=isne_options,
        embedding_options=embedding_options,
        save_intermediate_results=True
    )
    
    logger.info("ISNE pipeline test completed")


if __name__ == "__main__":
    asyncio.run(main())
