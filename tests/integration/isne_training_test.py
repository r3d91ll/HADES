#!/usr/bin/env python
"""
Integration test for ISNE model training.

This test demonstrates the complete workflow of training an ISNE model
using the outputs from the document processing pipeline. It shows how to
connect the document pipeline with the ISNE trainer.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import local modules
from src.isne.trainer.training_orchestrator import ISNETrainingOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def run_isne_training_test(
    input_dir: str = "./test-output/pipeline-mp-test",
    output_dir: str = "./test-output/isne-training",
    model_dir: str = "./models/isne",
    epochs: int = 20,
    device: str = "cpu"
):
    """
    Run the ISNE training integration test.
    
    Args:
        input_dir: Directory containing processed documents from the pipeline
        output_dir: Directory for saving training artifacts
        model_dir: Directory for saving trained models
        epochs: Number of training epochs
        device: Device to use for training (cpu or cuda:x)
    """
    logger.info("=== Starting ISNE Training Integration Test ===")
    
    # Ensure input directory exists
    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error(f"Input directory {input_path} does not exist. "
                    f"Please run the pipeline_multiprocess_test.py first.")
        return
    
    # Check for required input files
    isne_input_file = input_path / "isne_input_sample.json"
    if not isne_input_file.exists():
        logger.error(f"ISNE input file {isne_input_file} not found. "
                    f"Please run the pipeline_multiprocess_test.py first.")
        return
    
    # Start timing
    start_time = time.time()
    
    # Create training config override
    config_override = {
        "training": {
            "epochs": epochs,
            "device": device
        }
    }
    
    try:
        # Initialize the trainer orchestrator
        orchestrator = ISNETrainingOrchestrator(
            input_dir=input_dir,
            output_dir=output_dir,
            model_output_dir=model_dir,
            config_override=config_override
        )
        
        # Run the training process
        logger.info(f"Starting ISNE training with {epochs} epochs on {device}...")
        training_metrics = orchestrator.train()
        
        # Report results
        duration = time.time() - start_time
        logger.info(f"ISNE training completed in {duration:.2f} seconds")
        logger.info(f"Trained for {training_metrics['epochs']} epochs")
        logger.info(f"Final loss values:")
        
        if 'losses' in training_metrics and 'total_loss' in training_metrics['losses']:
            final_loss = training_metrics['losses']['total_loss'][-1]
            logger.info(f"  Total loss: {final_loss:.4f}")
        
        # Load the model for verification
        logger.info("Loading trained model to verify it was saved correctly...")
        model = orchestrator.load_model()
        logger.info(f"Successfully loaded model with {model.num_layers} layers")
        
        logger.info("=== ISNE Training Integration Test Completed Successfully ===")
        
    except Exception as e:
        logger.error(f"Error during ISNE training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.error("=== ISNE Training Integration Test Failed ===")


def main():
    """Main entry point for the ISNE training integration test."""
    parser = argparse.ArgumentParser(description='Test ISNE model training with pipeline outputs')
    parser.add_argument('--input-dir', type=str, default='./test-output/pipeline-mp-test',
                       help='Directory containing processed documents from pipeline')
    parser.add_argument('--output-dir', type=str, default='./test-output/isne-training',
                       help='Directory for saving training artifacts')
    parser.add_argument('--model-dir', type=str, default='./models/isne',
                       help='Directory for saving trained models')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use for training (cpu or cuda:x)')
    
    args = parser.parse_args()
    
    # Run the test
    run_isne_training_test(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        epochs=args.epochs,
        device=args.device
    )


if __name__ == "__main__":
    main()
