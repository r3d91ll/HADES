#!/usr/bin/env python3
"""
Test ISNE API Training Pipeline

This script tests the formalized ISNE training pipeline through the API endpoint.
It simulates what would happen during a production training run.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.api.isne_training import ISNEProductionTrainer
from src.alerts.alert_manager import AlertManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_training_pipeline():
    """Test the ISNE training pipeline with a small dataset."""
    
    # Use the 10% test data as our test case
    test_graph_path = "output/test_data3_10percent_hybrid/graph_data.json"
    test_model_path = "output/test_data3_10percent_hybrid/isne_model_final.pth"
    output_dir = "output/api_test"
    
    # Check if test data exists
    if not Path(test_graph_path).exists():
        logger.error(f"Test graph data not found at {test_graph_path}")
        logger.error("Please run the 10% hybrid test first to generate test data")
        return False
    
    # Initialize components
    alert_manager = AlertManager(alert_dir="./alerts")
    trainer = ISNEProductionTrainer(alert_manager=alert_manager)
    
    # Simulate job data structure
    job_id = "test-job-001"
    job_data = {
        "job_id": job_id,
        "status": "pending",
        "progress_percent": 0.0,
        "stage": "initialization",
        "results": {}
    }
    
    # Training configuration for quick test
    training_config = {
        "epochs": 5,  # Just 5 epochs for testing
        "learning_rate": 0.001,
        "batch_size": 32,
        "hidden_dim": 128,
        "num_layers": 3,
        "num_heads": 4,
        "dropout": 0.1,
        "weight_decay": 1e-5,
        "patience": 3,
        "min_delta": 0.001
    }
    
    logger.info("Starting ISNE API training pipeline test...")
    logger.info(f"Graph data: {test_graph_path}")
    logger.info(f"Existing model: {test_model_path}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Test 1: Train from scratch
        logger.info("\n=== Test 1: Training from scratch ===")
        results = await trainer.train(
            job_id=f"{job_id}-scratch",
            job_data=job_data.copy(),
            graph_data_path=test_graph_path,
            output_dir=output_dir,
            training_config=training_config,
            model_path=None,  # No existing model
            model_name="test_scratch_model"
        )
        
        logger.info("Training from scratch completed!")
        logger.info(f"Results: {json.dumps(results, indent=2, default=str)}")
        
        # Test 2: Continue training from existing model
        logger.info("\n=== Test 2: Continue training from existing model ===")
        job_data_continued = job_data.copy()
        job_data_continued["job_id"] = f"{job_id}-continued"
        
        results_continued = await trainer.train(
            job_id=f"{job_id}-continued",
            job_data=job_data_continued,
            graph_data_path=test_graph_path,
            output_dir=output_dir,
            training_config=training_config,
            model_path=test_model_path,  # Use existing model
            model_name="test_continued_model"
        )
        
        logger.info("Continued training completed!")
        logger.info(f"Results: {json.dumps(results_continued, indent=2, default=str)}")
        
        # Verify outputs
        output_path = Path(output_dir)
        expected_files = [
            "test_scratch_model_final.pth",
            "test_scratch_model_best.pth",
            "test_continued_model_final.pth",
            "test_continued_model_best.pth"
        ]
        
        logger.info("\n=== Verifying outputs ===")
        all_files_exist = True
        for file_name in expected_files:
            file_path = output_path / file_name
            if file_path.exists():
                logger.info(f"✓ {file_name} - {file_path.stat().st_size / 1024 / 1024:.2f} MB")
            else:
                logger.error(f"✗ {file_name} - NOT FOUND")
                all_files_exist = False
        
        # Test W&B integration
        logger.info("\n=== W&B Integration Status ===")
        if trainer.wandb_logger and trainer.wandb_logger.enabled:
            logger.info("✓ W&B integration is active")
            logger.info("Check your W&B dashboard for run details")
        else:
            logger.warning("✗ W&B integration is not active")
            logger.info("Set up W&B credentials to enable tracking")
        
        return all_files_exist
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_api_endpoint():
    """Test the API endpoint directly."""
    import aiohttp
    from src.api.pipelines import ISNETrainingRequest
    
    logger.info("\n=== Testing API Endpoint ===")
    
    # Check if API server is running
    api_url = "http://localhost:8000/api/v1/pipelines/isne_training"
    
    try:
        async with aiohttp.ClientSession() as session:
            # Create request
            request_data = ISNETrainingRequest(
                training_data_path="output/test_data3_10percent_hybrid/graph_data.json",
                model_settings={
                    "output_dir": "output/api_endpoint_test",
                    "model_name": "api_test_model"
                },
                training_config={
                    "epochs": 3,
                    "learning_rate": 0.001
                }
            )
            
            # Send request
            async with session.post(api_url, json=request_data.model_dump()) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"API endpoint test successful!")
                    logger.info(f"Job ID: {result['job_id']}")
                    logger.info(f"Progress endpoint: {result['progress_endpoint']}")
                    return True
                else:
                    logger.error(f"API request failed: {response.status}")
                    return False
                    
    except aiohttp.ClientConnectorError:
        logger.warning("API server not running. Start with: python -m src.api.server")
        return False
    except Exception as e:
        logger.error(f"API test failed: {e}")
        return False


async def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("ISNE API Training Pipeline Test")
    logger.info("=" * 60)
    
    # Test the training pipeline directly
    pipeline_success = await test_training_pipeline()
    
    # Test the API endpoint
    api_success = await test_api_endpoint()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    logger.info(f"Pipeline Test: {'✓ PASSED' if pipeline_success else '✗ FAILED'}")
    logger.info(f"API Endpoint Test: {'✓ PASSED' if api_success else '✗ FAILED (server may not be running)'}")
    
    if pipeline_success:
        logger.info("\n✅ ISNE training pipeline is ready for production use!")
        logger.info("You can now start the 30-hour training run with confidence.")
    else:
        logger.error("\n❌ Tests failed. Please fix issues before starting long training run.")


if __name__ == "__main__":
    asyncio.run(main())