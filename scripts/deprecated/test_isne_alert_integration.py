#!/usr/bin/env python
"""
Test script to demonstrate the alert system integration with ISNE pipeline.

This script intentionally creates documents with validation issues to show
how the alert system detects and reports them during ISNE processing.
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import local modules
from src.alerts import AlertManager, AlertLevel
from src.validation.embedding_validator import (
    validate_embeddings_before_isne,
    validate_embeddings_after_isne
)
from tests.integration.pipeline_multiprocess_test import PipelineMultiprocessTester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def create_test_documents(num_docs=5, chunks_per_doc=5, embedding_dim=384):
    """Create test documents with intentional validation issues."""
    import random
    import numpy as np
    
    documents = []
    
    for doc_idx in range(num_docs):
        doc_id = f"test_doc_{doc_idx}"
        chunks = []
        
        for chunk_idx in range(chunks_per_doc):
            # Deliberately create specific validation issues
            has_base = True
            has_isne = False
            
            # Every third document has missing base embeddings
            if doc_idx % 3 == 1:
                has_base = chunk_idx % 2 == 0  # Every other chunk missing base
            
            # Every fourth document has pre-existing ISNE embeddings (unexpected)
            if doc_idx % 4 == 2:
                has_isne = chunk_idx % 3 == 0  # Every third chunk has ISNE
            
            chunk = {
                "text": f"This is chunk {chunk_idx} of document {doc_idx}",
                "metadata": {
                    "chunk_id": f"{doc_id}_chunk_{chunk_idx}",
                    "embedding_model": "test_model"
                }
            }
            
            # Add base embedding if configured
            if has_base:
                chunk["embedding"] = np.random.rand(embedding_dim).tolist()
            
            # Add ISNE embedding if configured (this will trigger alerts)
            if has_isne:
                chunk["isne_embedding"] = np.random.rand(embedding_dim).tolist()
            
            chunks.append(chunk)
        
        document = {
            "file_id": doc_id,
            "file_name": f"{doc_id}.txt",
            "file_path": f"/path/to/{doc_id}.txt",
            "metadata": {
                "title": f"Test Document {doc_idx}"
            },
            "chunks": chunks,
            "worker_id": 0,
            "timing": {
                "document_processing": 0.1,
                "chunking": 0.2,
                "embedding": 0.3,
                "total": 0.6
            }
        }
        
        documents.append(document)
    
    return documents


def run_test_with_alerts(output_dir="./alert-test-output", alert_threshold="MEDIUM"):
    """Run a test of the ISNE pipeline with alert integration."""
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Define embedding dimension for test documents and model
    embedding_dim = 384
    
    # Create a test dataset with validation issues
    logger.info("Creating test documents with validation issues")
    documents = create_test_documents(num_docs=10, chunks_per_doc=8)
    
    # Save the test documents
    docs_path = output_path / "test_documents.json"
    with open(docs_path, "w") as f:
        json.dump(documents, f, indent=2)
    logger.info(f"Saved {len(documents)} test documents to {docs_path}")
    
    # Initialize the pipeline tester
    logger.info("Initializing PipelineMultiprocessTester with alert system")
    tester = PipelineMultiprocessTester(
        test_data_dir=str(output_path),
        output_dir=str(output_path),
        num_files=len(documents),
        max_workers=1,
        batch_size=len(documents),
        alert_threshold=alert_threshold
    )
    
    # Create a minimal model file for testing
    import torch
    model_path = output_path / "test_isne_model.pt"
    test_model = {
        "state_dict": {"weight": torch.ones(1, embedding_dim)},
        "config": {"embedding_dim": embedding_dim, "isne_dim": embedding_dim}
    }
    torch.save(test_model, model_path)
    logger.info(f"Created test ISNE model at {model_path}")
    
    # Instead of trying to patch methods, we'll directly use the implementation
    # in a more controlled way to demonstrate the alert system
    
    # Initialize AlertManager with our settings
    alert_dir = output_path / "alerts"
    alert_dir.mkdir(exist_ok=True, parents=True)
    alert_level = getattr(AlertLevel, alert_threshold, AlertLevel.MEDIUM)
    
    # Create a direct AlertManager instance
    alert_manager = AlertManager(
        alert_dir=str(alert_dir),
        min_level=alert_level,
        email_config=None  # No email alerts in test mode
    )
    
    logger.info("Running validation and ISNE application with alerts")
    
    # Run pre-ISNE validation
    logger.info("Validating documents before ISNE application")
    pre_validation = validate_embeddings_before_isne(documents)
    
    # Log validation results
    total_chunks = sum(len(doc.get("chunks", [])) for doc in documents)
    chunks_with_base = pre_validation.get("chunks_with_base_embeddings", 0)
    logger.info(f"Pre-ISNE Validation: {len(documents)} documents, {total_chunks} total chunks")
    logger.info(f"Found {chunks_with_base}/{total_chunks} chunks with base embeddings")
    
    # Generate alerts for validation issues
    missing_base = pre_validation.get("missing_base_embeddings", 0)
    if missing_base > 0:
        alert_level = AlertLevel.HIGH if missing_base > total_chunks * 0.2 else AlertLevel.MEDIUM
        alert_manager.alert(
            message=f"Missing base embeddings detected in {missing_base} chunks",
            level=alert_level,
            source="isne_pipeline",
            context={
                "missing_count": missing_base,
                "total_chunks": total_chunks,
                "affected_chunks": pre_validation.get('missing_base_embedding_ids', [])
            }
        )
    
    # Simulate ISNE application
    logger.info("Applying ISNE to documents (simulated)")
    import torch
    import numpy as np
    
    # Apply ISNE embeddings to some but not all chunks (to create validation issues)
    enhanced_docs = [doc.copy() for doc in documents]
    for doc in enhanced_docs:
        for chunk_idx, chunk in enumerate(doc.get("chunks", [])):
            # Only add ISNE to chunks with base embeddings, and only some of them
            if "embedding" in chunk and chunk_idx % 2 == 0:
                chunk["isne_embedding"] = np.random.rand(embedding_dim).tolist()
    
    # Run post-ISNE validation
    logger.info("Validating documents after ISNE application")
    post_validation = validate_embeddings_after_isne(enhanced_docs, pre_validation)
    
    # Generate alerts for validation issues
    discrepancies = post_validation.get("discrepancies", {})
    total_discrepancies = post_validation.get("total_discrepancies", 0)
    
    if total_discrepancies > 0:
        alert_level = AlertLevel.HIGH if total_discrepancies > total_chunks * 0.1 else AlertLevel.MEDIUM
        alert_manager.alert(
            message=f"Found {total_discrepancies} embedding discrepancies after ISNE application",
            level=alert_level,
            source="isne_pipeline",
            context={
                "discrepancies": discrepancies,
                "total_discrepancies": total_discrepancies,
                "expected_counts": post_validation.get("expected_counts", {}),
                "actual_counts": post_validation.get("actual_counts", {})
            }
        )
    
    # Save enhanced documents
    enhanced_docs_path = output_path / "enhanced_documents.json"
    with open(enhanced_docs_path, "w") as f:
        json.dump(enhanced_docs, f, indent=2)
    logger.info(f"Saved enhanced documents to {enhanced_docs_path}")
    
    # Save validation reports
    pre_validation_path = output_path / "pre_validation.json"
    with open(pre_validation_path, "w") as f:
        json.dump(pre_validation, f, indent=2)
    
    post_validation_path = output_path / "post_validation.json"
    with open(post_validation_path, "w") as f:
        json.dump(post_validation, f, indent=2)
    
    # Collect all alerts
    alerts = alert_manager.get_alerts()
    alert_dicts = [alert.to_dict() for alert in alerts]
    
    # Save alert summary
    alerts_path = output_path / "alert_summary.json"
    with open(alerts_path, "w") as f:
        json.dump(alert_dicts, f, indent=2)
    logger.info(f"Saved alert summary to {alerts_path}")
    
    # Print alert summary
    alert_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    for alert in alert_dicts:
        if "level" in alert:
            alert_counts[alert["level"]] += 1
    
    print("\n===== Alert Summary =====")
    print(f"LOW:      {alert_counts['LOW']}")
    print(f"MEDIUM:   {alert_counts['MEDIUM']}")
    print(f"HIGH:     {alert_counts['HIGH']}")
    print(f"CRITICAL: {alert_counts['CRITICAL']}")
    
    print(f"\nAlert logs saved to {output_path}/alerts")
    print(f"All test outputs saved to {output_path}")
    
    # Open the alert directory to check for log files
    alert_dir = output_path / "alerts"
    if alert_dir.exists():
        alert_files = list(alert_dir.glob("*"))
        if alert_files:
            print("\nAlert log files:")
            for file in alert_files:
                print(f"  - {file.name}")
    
    return {
        "documents": documents,
        "enhanced_documents": enhanced_docs,
        "alerts": alert_dicts,
        "pre_validation": pre_validation,
        "post_validation": post_validation
    }


def main():
    """Run the script from command line."""
    parser = argparse.ArgumentParser(description='Test Alert System Integration with ISNE')
    parser.add_argument('--output-dir', type=str, default='./alert-test-output',
                      help='Directory for test outputs')
    parser.add_argument('--alert-threshold', type=str, choices=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
                      default='LOW', help='Alert threshold level')
    
    args = parser.parse_args()
    
    run_test_with_alerts(
        output_dir=args.output_dir,
        alert_threshold=args.alert_threshold
    )


if __name__ == "__main__":
    main()
