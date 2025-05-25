#!/usr/bin/env python
"""
ISNE Validation Integration Test.

This integration test demonstrates the validation system working with
real ISNE embeddings, providing a comprehensive test case and example.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import validation module
from src.validation.embedding_validator import (
    validate_embeddings_before_isne,
    validate_embeddings_after_isne,
    create_validation_summary,
    attach_validation_summary
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_documents(input_path: str) -> List[Dict[str, Any]]:
    """
    Load documents from JSON file.
    
    Args:
        input_path: Path to JSON file with documents
        
    Returns:
        List of document dictionaries
    """
    try:
        with open(input_path, 'r') as f:
            documents = json.load(f)
        logger.info(f"Loaded {len(documents)} documents from {input_path}")
        return documents
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        return []

def save_validation_report(report: Dict[str, Any], output_path: str) -> None:
    """
    Save validation report to JSON file.
    
    Args:
        report: Validation report dictionary
        output_path: Path to save report
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved validation report to {output_path}")
    except Exception as e:
        logger.error(f"Error saving validation report: {e}")

def run_validation_pipeline(input_path: str, output_path: str) -> Dict[str, Any]:
    """
    Run the complete validation pipeline on a document set.
    
    Args:
        input_path: Path to JSON file with documents
        output_path: Path to save validation report
        
    Returns:
        Validation summary
    """
    # Load documents
    documents = load_documents(input_path)
    if not documents:
        return {"error": "No documents loaded"}
    
    start_time = time.time()
    
    # Run pre-validation
    logger.info("Running pre-ISNE validation...")
    pre_validation = validate_embeddings_before_isne(documents)
    logger.info(f"Pre-validation completed for {pre_validation['total_docs']} documents with {pre_validation['total_chunks']} chunks")
    
    # Run post-validation
    logger.info("Running post-ISNE validation...")
    post_validation = validate_embeddings_after_isne(documents, pre_validation)
    logger.info(f"Post-validation completed with {post_validation['chunks_with_isne']} chunks having ISNE embeddings")
    
    # Create validation summary
    validation_summary = create_validation_summary(pre_validation, post_validation)
    
    # Attach summary to documents (for potential later use)
    documents_with_summary = attach_validation_summary(documents, validation_summary)
    
    # Log validation time
    validation_time = time.time() - start_time
    logger.info(f"Validation completed in {validation_time:.2f} seconds")
    
    # Log validation results
    log_validation_results(validation_summary)
    
    # Save validation report
    if output_path:
        save_validation_report(validation_summary, output_path)
    
    return validation_summary

def log_validation_results(validation_summary: Dict[str, Any]) -> None:
    """
    Log validation results in a human-readable format.
    
    Args:
        validation_summary: Validation summary dictionary
    """
    pre = validation_summary["pre_validation"]
    post = validation_summary["post_validation"]
    discrepancies = validation_summary["discrepancies"]
    
    logger.info("\n=== Validation Results ===")
    logger.info(f"Documents: {pre['total_docs']}")
    logger.info(f"Documents with chunks: {pre['docs_with_chunks']}")
    logger.info(f"Total chunks: {pre['total_chunks']}")
    logger.info(f"Chunks with base embeddings: {pre['chunks_with_base_embeddings']}")
    logger.info(f"Chunks with ISNE embeddings: {post['chunks_with_isne']}")
    
    if any(discrepancies.values()):
        logger.warning("\n=== Discrepancies Detected ===")
        for key, value in discrepancies.items():
            if value:
                logger.warning(f"{key}: {value}")
        logger.warning("\nReview the validation report for details")
    else:
        logger.info("\nNo discrepancies detected! All embeddings are correctly applied.")

def main():
    """Run the ISNE validation integration test."""
    parser = argparse.ArgumentParser(description="ISNE Validation Integration Test")
    
    parser.add_argument(
        '--input', 
        type=str, 
        required=True,
        help='Path to JSON file with ISNE-enhanced documents'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default='./test-output/validation/validation_report.json',
        help='Path to save validation report'
    )
    
    args = parser.parse_args()
    
    # Run validation pipeline
    logger.info("=== Starting ISNE Validation Integration Test ===")
    validation_summary = run_validation_pipeline(args.input, args.output)
    
    # Determine exit code based on validation results
    if "error" in validation_summary:
        logger.error(f"Validation failed: {validation_summary['error']}")
        sys.exit(1)
    elif validation_summary["discrepancies"] and any(validation_summary["discrepancies"].values()):
        logger.warning("Validation completed with discrepancies")
        sys.exit(2)
    else:
        logger.info("Validation completed successfully with no discrepancies")
        sys.exit(0)

if __name__ == "__main__":
    main()
