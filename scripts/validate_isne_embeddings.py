#!/usr/bin/env python
"""
Script to validate ISNE embeddings in an existing dataset.

This script performs validation checks on a JSON dataset containing ISNE embeddings
to identify potential issues and inconsistencies.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# Add project root to Python path to allow imports from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    """Run ISNE embedding validation on a JSON dataset."""
    parser = argparse.ArgumentParser(description="Validate ISNE embeddings in a dataset")
    
    parser.add_argument(
        '--input', 
        type=str, 
        required=True,
        help='Path to JSON file with ISNE-enhanced documents'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default='./validation-report.json',
        help='Path to save validation report'
    )
    
    args = parser.parse_args()
    
    # Start timing
    start_time = time.time()
    
    # Check if input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    # Load documents
    logger.info(f"Loading documents from {input_path}")
    try:
        with open(input_path, 'r') as f:
            documents = json.load(f)
        logger.info(f"Loaded {len(documents)} documents")
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        sys.exit(1)
    
    # Import validation module
    try:
        from src.validation.embedding_validator import (
            validate_embeddings_before_isne,
            validate_embeddings_after_isne,
            create_validation_summary
        )
    except ImportError as e:
        logger.error(f"Error importing validation module: {e}")
        logger.error("Make sure you're running this script from the project root directory")
        sys.exit(1)
    
    # Run pre-validation
    logger.info("Running pre-validation checks...")
    pre_validation = validate_embeddings_before_isne(documents)
    logger.info(f"Pre-validation complete")
    
    # Run post-validation
    logger.info("Running post-validation checks...")
    post_validation = validate_embeddings_after_isne(documents, pre_validation)
    logger.info(f"Post-validation complete")
    
    # Create validation summary
    validation_summary = create_validation_summary(pre_validation, post_validation)
    
    # Print summary
    logger.info("\n=== Validation Summary ===")
    logger.info(f"Documents: {pre_validation['total_docs']}")
    logger.info(f"Documents with chunks: {pre_validation['docs_with_chunks']}")
    logger.info(f"Total chunks: {pre_validation['total_chunks']}")
    logger.info(f"Chunks with base embeddings: {pre_validation['chunks_with_base_embeddings']}")
    logger.info(f"Chunks with ISNE embeddings: {post_validation['chunks_with_isne']}")
    
    # Save validation report
    output_path = Path(args.output)
    os.makedirs(output_path.parent, exist_ok=True)
    
    try:
        with open(output_path, 'w') as f:
            json.dump(validation_summary, f, indent=2)
        logger.info(f"Saved validation report to {output_path}")
    except Exception as e:
        logger.error(f"Error saving validation report: {e}")
    
    # Report any discrepancies
    discrepancies = validation_summary["discrepancies"]
    if any(discrepancies.values()):
        logger.warning("\n=== Discrepancies Detected ===")
        for key, value in discrepancies.items():
            if value:
                logger.warning(f"{key}: {value}")
        logger.warning("\nReview the validation report for details")
        exit_code = 1
    else:
        logger.info("\nNo discrepancies detected! All embeddings are correctly applied.")
        exit_code = 0
    
    # Complete timing
    total_time = time.time() - start_time
    logger.info(f"\nValidation completed in {total_time:.2f} seconds")
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
