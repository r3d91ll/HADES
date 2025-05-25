#!/usr/bin/env python
"""
Test script for demonstrating the AlertManager integration with ISNE pipeline.
This script creates sample documents with various validation issues and shows
how alerts are generated during the validation process.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import local modules
from src.alerts import AlertManager, AlertLevel
from src.validation.embedding_validator import (
    validate_embeddings_before_isne,
    validate_embeddings_after_isne
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def create_test_documents(num_docs=5, chunks_per_doc=3, embedding_dim=384):
    """Create test documents with various validation issues."""
    import random
    import numpy as np
    
    documents = []
    
    for doc_idx in range(num_docs):
        doc_id = f"test_doc_{doc_idx}"
        chunks = []
        
        for chunk_idx in range(chunks_per_doc):
            # Determine if this chunk should have issues
            has_base_embedding = random.random() > 0.2  # 20% chance of missing base embedding
            has_isne_embedding = random.random() > 0.3 if has_base_embedding else False  # 30% chance of missing ISNE if base exists
            
            chunk = {
                "text": f"This is chunk {chunk_idx} of document {doc_idx}",
                "metadata": {
                    "chunk_id": f"{doc_id}_chunk_{chunk_idx}",
                    "embedding_model": "test_model"
                }
            }
            
            # Add base embedding if it exists
            if has_base_embedding:
                chunk["embedding"] = np.random.rand(embedding_dim).tolist()
            
            # Add ISNE embedding if it exists
            if has_isne_embedding:
                chunk["isne_embedding"] = np.random.rand(embedding_dim).tolist()
            
            chunks.append(chunk)
        
        document = {
            "file_id": doc_id,
            "file_path": f"/path/to/{doc_id}.txt",
            "metadata": {
                "title": f"Test Document {doc_idx}"
            },
            "chunks": chunks
        }
        
        documents.append(document)
    
    return documents


def test_alert_integration(output_dir="./alert_test_output", alert_threshold="MEDIUM"):
    """Test the AlertManager integration with validation."""
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    alert_dir = output_path / "alerts"
    alert_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize AlertManager
    alert_level = getattr(AlertLevel, alert_threshold, AlertLevel.MEDIUM)
    alert_manager = AlertManager(
        alert_dir=str(alert_dir),
        min_level=alert_level,
        email_config=None  # No email in test mode
    )
    
    # Create test documents
    logger.info("Creating test documents with validation issues")
    documents = create_test_documents(num_docs=5, chunks_per_doc=10)
    
    # Save original documents
    orig_docs_path = output_path / "original_documents.json"
    with open(orig_docs_path, "w") as f:
        json.dump(documents, f, indent=2)
    logger.info(f"Saved original test documents to {orig_docs_path}")
    
    # Run pre-ISNE validation
    logger.info("Running pre-ISNE validation")
    pre_validation = validate_embeddings_before_isne(documents)
    
    # Generate alerts for pre-validation issues
    missing_base = pre_validation.get("missing_base_embeddings", 0)
    if missing_base > 0:
        alert_manager.alert(
            message=f"Missing base embeddings detected in {missing_base} chunks",
            level=AlertLevel.MEDIUM if missing_base < 5 else AlertLevel.HIGH,
            source="isne_validation",
            context={
                "missing_count": missing_base,
                "affected_chunks": pre_validation.get('missing_base_embedding_ids', [])
            }
        )
        logger.warning(f"Found {missing_base} chunks missing base embeddings")
    
    # Save pre-validation report
    pre_validation_path = output_path / "pre_validation.json"
    with open(pre_validation_path, "w") as f:
        json.dump(pre_validation, f, indent=2)
    logger.info(f"Saved pre-validation report to {pre_validation_path}")
    
    # Run post-ISNE validation
    logger.info("Running post-ISNE validation")
    post_validation = validate_embeddings_after_isne(documents, pre_validation)
    
    # Generate alerts for post-validation issues
    discrepancies = post_validation.get("discrepancies", {})
    total_discrepancies = post_validation.get("total_discrepancies", 0)
    
    if total_discrepancies > 0:
        alert_level = AlertLevel.HIGH if total_discrepancies > 5 else AlertLevel.MEDIUM
        alert_manager.alert(
            message=f"Found {total_discrepancies} embedding discrepancies after ISNE application",
            level=alert_level,
            source="isne_validation",
            context={
                "discrepancies": discrepancies,
                "total_discrepancies": total_discrepancies,
                "expected_counts": post_validation.get("expected_counts", {}),
                "actual_counts": post_validation.get("actual_counts", {})
            }
        )
        logger.warning(f"Found {total_discrepancies} total embedding discrepancies")
    
    # Save post-validation report
    post_validation_path = output_path / "post_validation.json"
    with open(post_validation_path, "w") as f:
        json.dump(post_validation, f, indent=2)
    logger.info(f"Saved post-validation report to {post_validation_path}")
    
    # Get all alerts
    alerts = alert_manager.get_alerts()
    
    # Print alert summary
    print("\n===== Alert Summary =====")
    alert_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    
    for alert in alerts:
        # Alert objects have properties rather than dictionary keys
        alert_counts[alert.level.name] += 1
    
    print(f"LOW:      {alert_counts['LOW']}")
    print(f"MEDIUM:   {alert_counts['MEDIUM']}")
    print(f"HIGH:     {alert_counts['HIGH']}")
    print(f"CRITICAL: {alert_counts['CRITICAL']}")
    
    # Save alerts to file - convert Alert objects to dictionaries
    alerts_path = output_path / "alert_summary.json"
    alert_dicts = [alert.to_dict() for alert in alerts]
    with open(alerts_path, "w") as f:
        json.dump(alert_dicts, f, indent=2)
    logger.info(f"Saved alert summary to {alerts_path}")
    
    print(f"\nAll test outputs saved to {output_dir}")
    print(f"Alert logs saved to {alert_dir}")
    
    return {
        "documents": documents,
        "pre_validation": pre_validation,
        "post_validation": post_validation,
        "alerts": alerts
    }


def main():
    """Run the script from command line."""
    parser = argparse.ArgumentParser(description='Test AlertManager integration with ISNE validation')
    parser.add_argument('--output-dir', type=str, default='./alert_test_output',
                      help='Directory for test outputs')
    parser.add_argument('--alert-threshold', type=str, choices=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
                      default='MEDIUM', help='Alert threshold level')
    
    args = parser.parse_args()
    
    test_alert_integration(
        output_dir=args.output_dir,
        alert_threshold=args.alert_threshold
    )


if __name__ == "__main__":
    main()
