#!/usr/bin/env python
"""
ISNE model validation test.

This script implements validation for ISNE model embeddings to detect inconsistencies
during the data ingestion pipeline and ensure each chunk receives a single embedding.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_json_documents(file_path: str) -> List[Dict[str, Any]]:
    """
    Load JSON documents from a file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        List of document dictionaries
    """
    try:
        with open(file_path, 'r') as f:
            documents = json.load(f)
        logger.info(f"Loaded {len(documents)} documents from {file_path}")
        return documents
    except Exception as e:
        logger.error(f"Error loading documents from {file_path}: {e}")
        return []

def save_validation_report(output_path: str, report: Dict[str, Any]) -> None:
    """
    Save validation report to a JSON file.
    
    Args:
        output_path: Path to save the report
        report: Validation report dictionary
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved validation report to {output_path}")
    except Exception as e:
        logger.error(f"Error saving validation report: {e}")

def run_validation(input_file: str, output_file: str) -> Dict[str, Any]:
    """
    Run validation on documents from a JSON file.
    
    Args:
        input_file: Path to input JSON file with documents
        output_file: Path to save validation report
        
    Returns:
        Validation results
    """
    # Start timing
    start_time = time.time()
    
    # Load documents
    documents = load_json_documents(input_file)
    if not documents:
        logger.error("No documents loaded, aborting validation")
        return {"error": "No documents loaded"}
    
    # Import validation module
    try:
        from src.validation.embedding_validator import (
            validate_embeddings_before_isne,
            validate_embeddings_after_isne,
            create_validation_summary
        )
    except ImportError as e:
        logger.error(f"Error importing validation module: {e}")
        return {"error": f"Validation module not available: {e}"}
    
    # Run pre-validation
    logger.info("Running pre-validation checks...")
    pre_validation = validate_embeddings_before_isne(documents)
    logger.info(f"Pre-validation complete: {len(documents)} documents analyzed")
    
    # Run post-validation (since we're validating already processed documents)
    logger.info("Running post-validation checks...")
    post_validation = validate_embeddings_after_isne(documents, pre_validation)
    logger.info(f"Post-validation complete: found {post_validation['chunks_with_isne']} chunks with ISNE embeddings")
    
    # Create validation summary
    validation_summary = create_validation_summary(pre_validation, post_validation)
    
    # Log results
    logger.info("\n=== Validation Results ===")
    logger.info(f"Documents analyzed: {pre_validation['total_docs']}")
    logger.info(f"Total chunks: {pre_validation['total_chunks']}")
    logger.info(f"Chunks with base embeddings: {pre_validation['chunks_with_base_embeddings']}")
    logger.info(f"Chunks with ISNE embeddings: {post_validation['chunks_with_isne']}")
    
    # Log discrepancies
    discrepancies = validation_summary["discrepancies"]
    if any(discrepancies.values()):
        logger.warning("\n=== Discrepancies Detected ===")
        for key, value in discrepancies.items():
            if value:
                logger.warning(f"{key}: {value}")
    else:
        logger.info("\nNo discrepancies detected! All embeddings are correctly applied.")
    
    # Save validation report
    save_validation_report(output_file, validation_summary)
    
    # Complete timing
    validation_time = time.time() - start_time
    logger.info(f"\nValidation completed in {validation_time:.2f} seconds")
    
    return validation_summary

def main():
    """Run the ISNE validation test as a standalone script."""
    parser = argparse.ArgumentParser(description="ISNE Model Validation Test")
    
    # Required arguments
    parser.add_argument(
        '--input', 
        type=str, 
        required=True,
        help='Path to JSON file with ISNE-enhanced documents'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output', 
        type=str, 
        default='./test-output/validation-report.json',
        help='Path to save validation report'
    )
    
    args = parser.parse_args()
    
    # Run validation
    validation_results = run_validation(args.input, args.output)
    
    # Determine exit code based on validation results
    if "error" in validation_results:
        logger.error(f"Validation failed: {validation_results['error']}")
        sys.exit(1)
    elif validation_results["discrepancies"] and any(validation_results["discrepancies"].values()):
        logger.warning("Validation completed with discrepancies")
        sys.exit(2)
    else:
        logger.info("Validation completed successfully with no discrepancies")
        sys.exit(0)

if __name__ == "__main__":
    main()
