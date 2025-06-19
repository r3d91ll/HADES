#!/usr/bin/env python
"""
Script to create a test ISNE model for integration testing.

This script creates a state dictionary-based model that is easier to load across contexts.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# Add the project root to the Python path so we can import project modules
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def create_test_isne_model(output_path: Path, embedding_dim: int = 768):
    """
    Create a simple test ISNE model state dict for integration testing.
    
    Args:
        output_path: Path to save the model
        embedding_dim: Dimension of the embeddings
    """
    # Create parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Instead of saving a custom class model, save a state dict with random weights
    # This avoids pickling issues with PyTorch 2.6+
    test_state_dict = {
        'model_type': 'test_isne_model',
        'embedding_dim': embedding_dim,
        'weights': {
            'layer1.weight': torch.randn(embedding_dim * 2, embedding_dim),
            'layer1.bias': torch.randn(embedding_dim * 2),
            'layer2.weight': torch.randn(embedding_dim, embedding_dim * 2),
            'layer2.bias': torch.randn(embedding_dim),
        },
        'isne_test_model': True,
    }
    
    # Save the state dict
    torch.save(test_state_dict, output_path)
    
    logger.info(f"Created test ISNE model state dict at {output_path} with PyTorch {torch.__version__}")
    return test_state_dict

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create a test ISNE model for integration testing")
    parser.add_argument(
        "output_path", nargs="?", default="../models/isne/isne_model_latest.pt", 
        help="Path to save the model (default: ../models/isne/isne_model_latest.pt)"
    )
    args = parser.parse_args()
    
    # Convert to Path object
    model_path = Path(args.output_path)
    
    # Convert to absolute path if needed
    if not model_path.is_absolute():
        model_path = Path(os.getcwd()) / model_path
    
    # Create the model
    create_test_isne_model(model_path)
    
    logger.info(f"Test ISNE model created at: {model_path}")
    logger.info("This model can be used for integration testing of the ISNE pipeline.")

if __name__ == "__main__":
    main()
