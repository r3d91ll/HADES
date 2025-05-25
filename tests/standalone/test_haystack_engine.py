"""
Test script for the Haystack model engine.

This script tests the initialization and startup of the Haystack model engine.
"""

import sys
import logging
import traceback
from pathlib import Path

# Configure logging to see all the details
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_haystack_engine")

def test_haystack_engine():
    """Test the Haystack model engine initialization and startup."""
    logger.info("Testing Haystack model engine...")
    
    try:
        # Import the HaystackModelEngine
        from src.model_engine.engines.haystack import HaystackModelEngine
        logger.info("Successfully imported HaystackModelEngine")
        
        # Create an instance of the engine
        engine = HaystackModelEngine()
        logger.info("Successfully created HaystackModelEngine instance")
        
        # Start the engine
        logger.info("Attempting to start the engine...")
        result = engine.start()
        logger.info(f"Engine start result: {result}")
        
        # Check if the engine is running
        logger.info(f"Engine running status: {engine.running}")
        logger.info(f"Engine client: {engine.client}")
        
        if engine.client:
            # Try to ping the server
            logger.info("Attempting to ping the server...")
            ping_result = engine.client.ping()
            logger.info(f"Ping result: {ping_result}")
        
        # Try to load a model
        if engine.running:
            logger.info("Attempting to load a model...")
            model_id = "bert-base-uncased"  # A small model for testing
            try:
                load_result = engine.load_model(model_id)
                logger.info(f"Model load result: {load_result}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                logger.error(traceback.format_exc())
        
        # Stop the engine
        logger.info("Attempting to stop the engine...")
        stop_result = engine.stop()
        logger.info(f"Engine stop result: {stop_result}")
        
        return result
    except Exception as e:
        logger.error(f"Error testing Haystack model engine: {e}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    result = test_haystack_engine()
    if result:
        logger.info("Haystack model engine test PASSED")
        sys.exit(0)
    else:
        logger.error("Haystack model engine test FAILED")
        sys.exit(1)
