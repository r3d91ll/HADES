#!/usr/bin/env python3
"""
Test script to verify Docling device configuration behavior.
This tests our monkey-patching implementation to ensure Docling
respects the configured device settings.
"""
import os
import sys
import logging
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import after setting up path
from src.docproc.adapters.docling_adapter import DoclingAdapter
import docling.utils.accelerator_utils

def test_docling_device_config():
    """Test that Docling respects our device configuration."""
    logger.info("Testing Docling device configuration")
    
    # Check original device detection
    # The decide_device function requires an accelerator_device parameter
    logger.info(f"Original device detection: {docling.utils.accelerator_utils.decide_device('auto')}")
    
    # Test with CPU configuration
    logger.info("Creating DoclingAdapter with CPU configuration")
    cpu_adapter = DoclingAdapter(options={
        'device': 'cpu',
        'use_gpu': False
    })
    logger.info(f"After CPU config, Docling device: {docling.utils.accelerator_utils.decide_device('auto')}")
    
    # Test with GPU configuration
    logger.info("Creating DoclingAdapter with GPU configuration")
    gpu_adapter = DoclingAdapter(options={
        'device': 'cuda:0',
        'use_gpu': True
    })
    logger.info(f"After GPU config, Docling device: {docling.utils.accelerator_utils.decide_device('auto')}")
    
    # Test with no configuration (should use original device detection)
    logger.info("Creating DoclingAdapter with no device configuration")
    default_adapter = DoclingAdapter()
    logger.info(f"After default config, Docling device: {docling.utils.accelerator_utils.decide_device('auto')}")
    
    # Also test with explicit device parameters
    logger.info(f"With 'cpu' parameter: {docling.utils.accelerator_utils.decide_device('cpu')}")
    logger.info(f"With 'cuda:0' parameter: {docling.utils.accelerator_utils.decide_device('cuda:0')}")
    logger.info(f"With 'cuda:1' parameter: {docling.utils.accelerator_utils.decide_device('cuda:1')}")
    logger.info(f"With 'mps' parameter: {docling.utils.accelerator_utils.decide_device('mps')}")
    

if __name__ == "__main__":
    test_docling_device_config()
