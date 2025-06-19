#!/usr/bin/env python
"""
Test the alert system with intentionally corrupted embedding data.

This script demonstrates how the alert system detects and reports embedding
discrepancies by introducing controlled data quality issues into existing
documents.
"""

import argparse
import copy
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the project root directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.isne.pipeline.isne_pipeline import ISNEPipeline
from src.alerts import AlertManager, AlertLevel
from src.validation.embedding_validator import (
    validate_embeddings_before_isne,
    validate_embeddings_after_isne,
    create_validation_summary
)


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers
    )


def corrupt_documents(documents: List[Dict[str, Any]], 
                     corruption_type: str,
                     corruption_rate: float = 0.2) -> List[Dict[str, Any]]:
    """
    Introduce controlled corruption into document embeddings.
    
    Args:
        documents: List of documents to corrupt
        corruption_type: Type of corruption to introduce
        corruption_rate: Percentage of chunks to corrupt
        
    Returns:
        Corrupted documents
    """
    # Create a deep copy to avoid modifying the original
    corrupted_docs = copy.deepcopy(documents)
    
    # Count total chunks
    total_chunks = sum(len(doc.get("chunks", [])) for doc in corrupted_docs)
    chunks_to_corrupt = int(total_chunks * corruption_rate)
    
    logging.info(f"Corrupting {chunks_to_corrupt} out of {total_chunks} chunks ({corruption_rate*100:.1f}%)")
    
    # Flatten list of all chunks for easier random selection
    all_chunks = []
    for doc in corrupted_docs:
        for chunk in doc.get("chunks", []):
            all_chunks.append((doc, chunk))
    
    # Randomly select chunks to corrupt
    if len(all_chunks) > 0:
        chunks_to_corrupt = min(chunks_to_corrupt, len(all_chunks))
        selected_chunks = random.sample(all_chunks, chunks_to_corrupt)
        
        for doc, chunk in selected_chunks:
            if corruption_type == "missing_base":
                # Remove base embeddings
                if "embedding" in chunk:
                    del chunk["embedding"]
                    if "embedding_model" in chunk:
                        del chunk["embedding_model"]
                    if "embedding_type" in chunk:
                        del chunk["embedding_type"]
            
            elif corruption_type == "missing_isne":
                # Remove ISNE embeddings
                if "isne_embedding" in chunk:
                    del chunk["isne_embedding"]
                    if "isne_embedding_model" in chunk:
                        del chunk["isne_embedding_model"]
            
            elif corruption_type == "wrong_dimensions":
                # Corrupt embedding dimensions
                if "embedding" in chunk and isinstance(chunk["embedding"], list):
                    # Either truncate or extend the embedding
                    orig_dim = len(chunk["embedding"])
                    if random.choice([True, False]):
                        # Truncate
                        new_dim = max(orig_dim - 50, 10)
                        chunk["embedding"] = chunk["embedding"][:new_dim]
                    else:
                        # Extend
                        new_dim = orig_dim + 50
                        chunk["embedding"].extend([0.0] * 50)
            
            elif corruption_type == "mixed":
                # Apply a random corruption type
                corruption_types = ["missing_base", "missing_isne", "wrong_dimensions"]
                random_type = random.choice(corruption_types)
                
                if random_type == "missing_base" and "embedding" in chunk:
                    del chunk["embedding"]
                    if "embedding_model" in chunk:
                        del chunk["embedding_model"]
                    if "embedding_type" in chunk:
                        del chunk["embedding_type"]
                
                elif random_type == "missing_isne" and "isne_embedding" in chunk:
                    del chunk["isne_embedding"]
                    if "isne_embedding_model" in chunk:
                        del chunk["isne_embedding_model"]
                
                elif random_type == "wrong_dimensions" and "embedding" in chunk and isinstance(chunk["embedding"], list):
                    orig_dim = len(chunk["embedding"])
                    if random.choice([True, False]):
                        new_dim = max(orig_dim - 50, 10)
                        chunk["embedding"] = chunk["embedding"][:new_dim]
                    else:
                        new_dim = orig_dim + 50
                        chunk["embedding"].extend([0.0] * 50)
    
    return corrupted_docs


def test_alert_system(args):
    """Test the alert system with corrupted document data."""
    # Set up output directories
    output_dir = Path(args.output_dir)
    corrupted_dir = output_dir / "corrupted"
    validation_dir = output_dir / "validation"
    alerts_dir = output_dir / "alerts"
    
    for dir_path in [output_dir, corrupted_dir, validation_dir, alerts_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Load input documents
    try:
        with open(args.input_file, "r") as f:
            documents = json.load(f)
        
        if isinstance(documents, dict) and "documents" in documents:
            documents = documents["documents"]
        
        logging.info(f"Loaded {len(documents)} documents from {args.input_file}")
        
        # Count chunks and embeddings
        total_chunks = sum(len(doc.get("chunks", [])) for doc in documents)
        base_embeddings = sum(
            1 for doc in documents 
            for chunk in doc.get("chunks", []) 
            if "embedding" in chunk
        )
        isne_embeddings = sum(
            1 for doc in documents 
            for chunk in doc.get("chunks", []) 
            if "isne_embedding" in chunk
        )
        
        logging.info(f"Total chunks: {total_chunks}")
        logging.info(f"Chunks with base embeddings: {base_embeddings}/{total_chunks}")
        logging.info(f"Chunks with ISNE embeddings: {isne_embeddings}/{total_chunks}")
        
    except Exception as e:
        logging.error(f"Error loading documents: {e}")
        return 1
    
    # Corrupt the documents
    corrupted_docs = corrupt_documents(
        documents,
        corruption_type=args.corruption_type,
        corruption_rate=args.corruption_rate
    )
    
    # Save corrupted documents
    corrupted_file = corrupted_dir / "corrupted_documents.json"
    with open(corrupted_file, "w") as f:
        json.dump(corrupted_docs, f, indent=2)
    
    logging.info(f"Saved corrupted documents to {corrupted_file}")
    
    # Initialize alert manager
    alert_manager = AlertManager(
        alert_dir=str(alerts_dir),
        min_level=getattr(AlertLevel, args.alert_threshold.upper())
    )
    
    # Initialize pipeline
    pipeline = ISNEPipeline(
        validate=True,
        alert_threshold=args.alert_threshold.lower(),
        alert_manager=alert_manager
    )
    
    # Process documents in validation-only mode
    logging.info("Running validation with the corrupted documents...")
    
    # Run pre-validation
    pre_validation = validate_embeddings_before_isne(corrupted_docs)
    
    # Save pre-validation results
    with open(validation_dir / "pre_validation.json", "w") as f:
        json.dump(pre_validation, f, indent=2)
    
    # Check if documents have ISNE embeddings
    has_isne = any("isne_embedding" in chunk for doc in corrupted_docs for chunk in doc.get("chunks", []))
    
    if has_isne:
        # Run post-validation
        post_validation = validate_embeddings_after_isne(corrupted_docs, pre_validation)
        
        # Save post-validation results
        with open(validation_dir / "post_validation.json", "w") as f:
            json.dump(post_validation, f, indent=2)
        
        # Create validation summary
        validation_summary = create_validation_summary(pre_validation, post_validation)
        
        # Save validation summary
        with open(validation_dir / "validation_summary.json", "w") as f:
            json.dump(validation_summary, f, indent=2)
        
        # Process discrepancies and create alerts
        discrepancies = validation_summary.get("discrepancies", {})
        total_discrepancies = sum(abs(value) for value in discrepancies.values())
        
        if total_discrepancies > 0:
            # Create detailed context
            context = {
                "discrepancies": discrepancies,
                "total_discrepancies": total_discrepancies,
                "expected_counts": validation_summary.get("expected_counts", {}),
                "actual_counts": validation_summary.get("actual_counts", {})
            }
            
            # Determine alert level based on discrepancy count
            alert_level = AlertLevel.LOW
            title_prefix = "NOTICE"
            
            if total_discrepancies >= 10:
                alert_level = AlertLevel.HIGH
                title_prefix = "CRITICAL"
            elif total_discrepancies >= 5:
                alert_level = AlertLevel.MEDIUM
                title_prefix = "WARNING"
            
            # Create the alert
            message = f"{title_prefix}: Found {total_discrepancies} total embedding discrepancies"
            
            details = []
            for key, value in discrepancies.items():
                if value != 0:
                    details.append(f"{key}: {value}")
            
            if details:
                message += f" - {', '.join(details)}"
            
            alert_manager.alert(
                message=message,
                level=alert_level,
                source="test_corruption",
                context=context
            )
    else:
        validation_summary = {"error": "No ISNE embeddings found to validate"}
        
        # Create an alert for missing ISNE embeddings
        alert_manager.alert(
            message="No ISNE embeddings found in documents",
            level=AlertLevel.MEDIUM,
            source="test_corruption"
        )
    
    # Add alerts for missing base embeddings
    if "missing_base_embeddings" in pre_validation:
        # Check if it's a list or an integer
        if isinstance(pre_validation["missing_base_embeddings"], list):
            missing_chunks = pre_validation["missing_base_embeddings"]
            count = len(missing_chunks)
        else:  # It's a count
            count = pre_validation["missing_base_embeddings"]
            missing_chunks = pre_validation.get("missing_base_embedding_ids", [])
        
        if count > 0:
            alert_manager.alert(
                message=f"Missing base embeddings detected in {count} chunks",
                level=AlertLevel.MEDIUM,
                source="test_corruption",
                context={
                    "missing_count": count,
                    "affected_chunks": missing_chunks
                }
            )
    
    # Wait a moment for all logs to be written
    time.sleep(0.5)
    
    # Get alert statistics
    alert_stats = alert_manager.get_alert_stats()
    
    # Print summary
    print("\n===== Alert System Test Summary =====")
    print(f"Corruption type: {args.corruption_type}")
    print(f"Corruption rate: {args.corruption_rate*100:.1f}%")
    print(f"Documents processed: {len(corrupted_docs)}")
    print(f"Total chunks: {total_chunks}")
    
    # Print validation summary if available
    if has_isne:
        print("\n===== Validation Summary =====")
        print(f"Total discrepancies: {total_discrepancies}")
        
        for key, value in discrepancies.items():
            if value != 0:
                print(f"  {key}: {value}")
    else:
        print("\nNo ISNE embeddings found to validate")
    
    # Print alert summary
    print("\n===== Alert Summary =====")
    for level, count in alert_stats.items():
        print(f"{level}: {count}")
    
    # Check if there were any critical alerts
    critical_count = alert_stats.get("CRITICAL", 0) + alert_stats.get("HIGH", 0)
    if critical_count > 0:
        print(f"\n⚠️ WARNING: {critical_count} critical/high alerts were generated.")
        print(f"Review alerts in {alerts_dir}")
    
    # Print output paths
    print("\n===== Output Files =====")
    print(f"Corrupted documents: {corrupted_file}")
    print(f"Validation reports: {validation_dir}")
    print(f"Alert logs: {alerts_dir}")
    
    return 0


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Test the ISNE alert system with corrupted document data"
    )
    
    parser.add_argument(
        "-i", "--input-file",
        required=True,
        help="Input JSON file containing documents with embeddings"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        default="./alert-test-output",
        help="Output directory for test results"
    )
    
    parser.add_argument(
        "-c", "--corruption-type",
        choices=["missing_base", "missing_isne", "wrong_dimensions", "mixed"],
        default="mixed",
        help="Type of corruption to introduce"
    )
    
    parser.add_argument(
        "-r", "--corruption-rate",
        type=float,
        default=0.2,
        help="Percentage of chunks to corrupt (0.0-1.0)"
    )
    
    parser.add_argument(
        "-a", "--alert-threshold",
        choices=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        default="LOW",
        help="Threshold for triggering alerts"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--log-file",
        help="Log file path"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level, args.log_file)
    
    try:
        return test_alert_system(args)
    except Exception as e:
        logging.error(f"Error in alert system test: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
