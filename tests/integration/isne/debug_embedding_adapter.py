"""
Debug script to test and fix the embedding adapter registration issue.

This script diagnoses why the 'modernbert' adapter is not being registered properly
and demonstrates how to fix it.
"""

import logging
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_adapter_registration():
    """Test and fix the embedding adapter registration."""
    logger.info("Testing embedding adapter registration...")
    
    # First, check if adapters are registered normally
    try:
        from src.embedding.base import _adapter_registry, get_adapter
        logger.info(f"Initial adapter registry state: {_adapter_registry}")
    except Exception as e:
        logger.error(f"Error accessing adapter registry: {e}")
    
    # Force import of adapters to ensure registration happens
    logger.info("Explicitly importing adapter modules...")
    try:
        from src.embedding.adapters import modernbert_adapter
        logger.info("Successfully imported modernbert_adapter module")
    except Exception as e:
        logger.error(f"Error importing modernbert_adapter: {e}")
    
    # Check registry state after explicit import
    try:
        from src.embedding.base import _adapter_registry
        logger.info(f"Adapter registry after explicit imports: {_adapter_registry}")
    except Exception as e:
        logger.error(f"Error accessing adapter registry after imports: {e}")
    
    # Try to get the modernbert adapter
    try:
        from src.embedding.base import get_adapter
        adapter = get_adapter(
            "modernbert",
            model_name="answerdotai/ModernBERT-base",
            device="cpu",
            batch_size=4,
            normalize_embeddings=True
        )
        logger.info(f"Successfully created modernbert adapter: {adapter}")
        return True
    except Exception as e:
        logger.error(f"Error creating modernbert adapter: {e}")
        return False

def fix_adapter_issue():
    """Apply a fix for the adapter registration issue."""
    logger.info("Applying fix for adapter registration...")
    
    # Method 1: Make sure the adapter is registered
    try:
        from src.embedding.adapters.modernbert_adapter import ModernBERTEmbeddingAdapter
        from src.embedding.base import register_adapter
        
        # Re-register the adapter to ensure it's in the registry
        register_adapter("modernbert", ModernBERTEmbeddingAdapter)
        logger.info("Successfully re-registered ModernBERTEmbeddingAdapter")
        
        # Verify registration
        from src.embedding.base import _adapter_registry
        logger.info(f"Adapter registry after fix: {_adapter_registry}")
        
        # Test adapter creation
        from src.embedding.base import get_adapter
        adapter = get_adapter(
            "modernbert",
            model_name="answerdotai/ModernBERT-base",
            device="cpu",
            batch_size=4,
            normalize_embeddings=True
        )
        logger.info(f"Successfully created modernbert adapter after fix: {adapter}")
        return True
    except Exception as e:
        logger.error(f"Error in fix_adapter_issue: {e}")
        return False

def test_embedding_generation():
    """Test generating embeddings with the fixed adapter."""
    logger.info("Testing embedding generation...")
    
    try:
        import asyncio
        from src.embedding.base import get_adapter
        
        # Get the adapter
        adapter = get_adapter(
            "modernbert",
            model_name="answerdotai/ModernBERT-base",
            device="cpu",
            batch_size=4,
            normalize_embeddings=True
        )
        
        # Test text
        test_texts = [
            "This is a test document for embedding generation.",
            "ISNE enhances embeddings with graph structural information."
        ]
        
        # Generate embeddings
        async def generate_embeddings():
            embeddings = await adapter.embed(test_texts)
            return embeddings
        
        # Run the async function
        embeddings = asyncio.run(generate_embeddings())
        
        # Check results
        for i, embedding in enumerate(embeddings):
            logger.info(f"Generated embedding for text {i+1}, dimensions: {len(embedding)}")
        
        return True
    except Exception as e:
        logger.error(f"Error in test_embedding_generation: {e}")
        return False

def create_patched_pipeline():
    """Create a patched version of the PDF pipeline that ensures adapters are registered."""
    logger.info("Creating patched version of PDF pipeline...")
    
    patch_file_path = "/home/todd/ML-Lab/Olympus/HADES-PathRAG/src/pipelines/ingest/orchestrator/patched_pdf_pipeline.py"
    
    try:
        # Read original file
        with open("/home/todd/ML-Lab/Olympus/HADES-PathRAG/src/pipelines/ingest/orchestrator/pdf_pipeline_prototype.py", "r") as f:
            content = f.read()
        
        # Add adapter registration fix at the beginning after imports
        import_section_end = content.find("class PDFPipeline")
        if import_section_end == -1:
            logger.error("Could not find PDFPipeline class in file")
            return False
        
        patched_content = content[:import_section_end] + """
# Explicitly import and register adapters to ensure they're available
try:
    from src.embedding.adapters.modernbert_adapter import ModernBERTEmbeddingAdapter
    from src.embedding.base import register_adapter
    # Re-register the adapter to ensure it's in the registry
    register_adapter("modernbert", ModernBERTEmbeddingAdapter)
    logger.info("Successfully registered ModernBERTEmbeddingAdapter")
except Exception as e:
    logger.error(f"Error registering ModernBERTEmbeddingAdapter: {e}")

""" + content[import_section_end:]
        
        # Write patched file
        with open(patch_file_path, "w") as f:
            f.write(patched_content)
        
        logger.info(f"Successfully created patched pipeline at {patch_file_path}")
        return True
    except Exception as e:
        logger.error(f"Error creating patched pipeline: {e}")
        return False

if __name__ == "__main__":
    # Run the tests
    logger.info("Starting embedding adapter debugging...")
    
    # Test initial state
    initial_test = test_adapter_registration()
    logger.info(f"Initial adapter registration test {'passed' if initial_test else 'failed'}")
    
    # Apply fix
    fix_result = fix_adapter_issue()
    logger.info(f"Adapter fix {'succeeded' if fix_result else 'failed'}")
    
    # Test embedding generation
    if fix_result:
        embedding_test = test_embedding_generation()
        logger.info(f"Embedding generation test {'passed' if embedding_test else 'failed'}")
    
    # Create patched pipeline
    patch_result = create_patched_pipeline()
    logger.info(f"Creating patched pipeline {'succeeded' if patch_result else 'failed'}")
    
    logger.info("Embedding adapter debugging complete.")
