#!/usr/bin/env python3
"""
Quick ISNE Training Test

A faster version of the training test with minimal epochs.
"""

import asyncio
import logging
import sys
from pathlib import Path

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


async def quick_test():
    """Quick test with minimal training."""
    
    # Use the 10% test data
    test_graph_path = "output/test_data3_10percent_hybrid/graph_data.json"
    output_dir = "output/quick_test"
    
    # Check if test data exists
    if not Path(test_graph_path).exists():
        logger.error(f"Test graph data not found at {test_graph_path}")
        return False
    
    # Initialize components
    alert_manager = AlertManager(alert_dir="./alerts")
    trainer = ISNEProductionTrainer(alert_manager=alert_manager)
    
    # Job data
    job_id = "quick-test-001"
    job_data = {
        "job_id": job_id,
        "status": "pending",
        "progress_percent": 0.0,
        "stage": "initialization",
        "results": {}
    }
    
    # Minimal training configuration
    training_config = {
        "epochs": 1,  # Just 1 epoch for quick test
        "learning_rate": 0.001,
        "batch_size": 32,
        "hidden_dim": 64,  # Smaller for speed
        "num_layers": 2,   # Fewer layers
        "num_heads": 2,    # Fewer heads
        "dropout": 0.1,
        "weight_decay": 1e-5,
        "patience": 1,
        "min_delta": 0.001
    }
    
    logger.info("Starting quick ISNE training test...")
    logger.info(f"Config: {training_config}")
    
    try:
        # Run training
        results = await trainer.train(
            job_id=job_id,
            job_data=job_data,
            graph_data_path=test_graph_path,
            output_dir=output_dir,
            training_config=training_config,
            model_path=None,
            model_name="quick_test_model"
        )
        
        logger.info("✅ Quick test completed successfully!")
        logger.info(f"Model saved to: {results['model_path']}")
        logger.info(f"Training time: {results['training_time_seconds']:.2f}s")
        logger.info(f"Final loss: {results['final_loss']:.6f}")
        
        # Verify output file exists
        model_path = Path(results['model_path'])
        if model_path.exists():
            logger.info(f"✅ Model file verified: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
            return True
        else:
            logger.error("❌ Model file not found")
            return False
        
    except Exception as e:
        logger.error(f"❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(quick_test())
    if success:
        logger.info("\n🎉 Quick test passed! The pipeline is ready for production.")
    else:
        logger.error("\n❌ Quick test failed. Check the issues above.")
        sys.exit(1)