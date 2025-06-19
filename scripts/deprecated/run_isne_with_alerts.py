#!/usr/bin/env python
"""
Run ISNE pipeline with integrated alert system.

This script demonstrates the integration of the AlertManager with the ISNE pipeline,
including validation before and after ISNE application and alerts for discrepancies.
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import local modules
from src.alerts import AlertManager, AlertLevel
from src.validation.embedding_validator import (
    validate_embeddings_before_isne,
    validate_embeddings_after_isne,
    create_validation_summary,
    attach_validation_summary
)
from src.isne.trainer.training_orchestrator import ISNETrainingOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def run_isne_with_alerts(
    input_file: str,
    model_path: str,
    output_dir: str,
    alert_threshold: str = "MEDIUM"
):
    """
    Run ISNE pipeline with alert system integration.
    
    Args:
        input_file: Path to input JSON file with document embeddings
        model_path: Path to trained ISNE model
        output_dir: Directory for output files
        alert_threshold: Alert threshold level (LOW, MEDIUM, HIGH, CRITICAL)
    """
    start_time = time.time()
    logger.info(f"Starting ISNE pipeline with alert system")
    
    # Create output directories
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    validation_dir = output_path / "validation"
    validation_dir.mkdir(exist_ok=True, parents=True)
    
    alert_dir = output_path / "alerts"
    alert_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize alert manager
    alert_level = getattr(AlertLevel, alert_threshold, AlertLevel.MEDIUM)
    alert_manager = AlertManager(
        alert_dir=str(alert_dir),
        min_level=alert_level,
        email_config=None  # No email alerts in test mode
    )
    
    # Load documents from input file
    logger.info(f"Loading documents from {input_file}")
    try:
        with open(input_file, "r") as f:
            documents = json.load(f)
        logger.info(f"Loaded {len(documents)} documents")
    except Exception as e:
        error_msg = f"Error loading input file: {str(e)}"
        logger.error(error_msg)
        alert_manager.alert(
            message=error_msg,
            level=AlertLevel.CRITICAL,
            source="isne_pipeline",
            context={"input_file": input_file}
        )
        return
    
    # Validate documents before ISNE application
    logger.info("Validating documents before ISNE application")
    pre_validation = validate_embeddings_before_isne(documents)
    
    # Log validation results
    total_chunks = sum(len(doc.get("chunks", [])) for doc in documents)
    chunks_with_base = pre_validation.get("chunks_with_base_embeddings", 0)
    logger.info(f"Pre-ISNE Validation: {len(documents)} documents, {total_chunks} total chunks")
    logger.info(f"Found {chunks_with_base}/{total_chunks} chunks with base embeddings")
    
    # Check for validation issues and create alerts if needed
    missing_base = pre_validation.get("missing_base_embeddings", 0)
    if missing_base > 0:
        alert_manager.alert(
            message=f"Missing base embeddings detected in {missing_base} chunks",
            level=AlertLevel.MEDIUM,
            source="isne_pipeline",
            context={
                "missing_count": missing_base,
                "affected_chunks": pre_validation.get('missing_base_embedding_ids', [])
            }
        )
    
    # Save pre-validation report
    pre_validation_path = validation_dir / "pre_validation.json"
    with open(pre_validation_path, "w") as f:
        json.dump(pre_validation, f, indent=2)
    logger.info(f"Saved pre-validation report to {pre_validation_path}")
    
    # Apply ISNE model if exists and is valid
    if not Path(model_path).exists():
        error_msg = f"ISNE model not found at {model_path}"
        logger.error(error_msg)
        alert_manager.alert(
            message=error_msg,
            level=AlertLevel.HIGH,
            source="isne_pipeline",
            context={"model_path": model_path}
        )
        return
    
    # Apply ISNE model to enhance embeddings
    logger.info(f"Applying ISNE model from {model_path}")
    isne_start_time = time.time()
    
    try:
        # Load model
        logger.info("Loading ISNE model")
        model = ISNETrainingOrchestrator.load_model(model_path)
        
        # Build graph from documents
        logger.info("Building document graph for ISNE inference")
        data = build_graph_from_documents(documents)
        
        # Apply ISNE model to get enhanced embeddings
        logger.info("Applying ISNE model to enhance embeddings")
        enhanced_embeddings = model(data.x, data.edge_index, data.edge_attr)
        
        # Add enhanced embeddings back to documents
        logger.info("Adding enhanced embeddings to documents")
        node_idx_map = {}
        current_idx = 0
        
        # First pass to build node index mapping
        for doc in documents:
            if "chunks" not in doc or not doc["chunks"]:
                continue
            
            for chunk_idx, chunk in enumerate(doc["chunks"]):
                if "embedding" not in chunk or not chunk["embedding"]:
                    continue
                
                chunk_id = f"{doc['file_id']}_{chunk_idx}"
                node_idx_map[chunk_id] = current_idx
                current_idx += 1
        
        # Second pass to add ISNE embeddings to chunks
        for doc in documents:
            if "chunks" not in doc or not doc["chunks"]:
                continue
            
            for chunk_idx, chunk in enumerate(doc["chunks"]):
                if "embedding" not in chunk or not chunk["embedding"]:
                    continue
                
                chunk_id = f"{doc['file_id']}_{chunk_idx}"
                if chunk_id in node_idx_map:
                    node_idx = node_idx_map[chunk_id]
                    # Add the enhanced embedding to the chunk
                    chunk["isne_embedding"] = enhanced_embeddings[node_idx].tolist()
        
        # Validate documents after ISNE application
        logger.info("Validating documents after ISNE application")
        post_validation = validate_embeddings_after_isne(documents, pre_validation)
        
        # Check for validation issues after ISNE application
        discrepancies = post_validation.get("discrepancies", {})
        total_discrepancies = post_validation.get("total_discrepancies", 0)
        
        # Save post-validation report
        post_validation_path = validation_dir / "post_validation.json"
        with open(post_validation_path, "w") as f:
            json.dump(post_validation, f, indent=2)
        logger.info(f"Saved post-validation report to {post_validation_path}")
        
        if total_discrepancies > 0:
            alert_level = AlertLevel.HIGH if total_discrepancies > 5 else AlertLevel.MEDIUM
            alert_manager.alert(
                message=f"Found {total_discrepancies} total embedding discrepancies - isne_vs_chunks: {discrepancies.get('isne_vs_chunks', 0)}, missing_isne: {discrepancies.get('missing_isne', 0)}",
                level=alert_level,
                source="isne_pipeline",
                context={
                    "discrepancies": discrepancies,
                    "total_discrepancies": total_discrepancies,
                    "expected_counts": post_validation.get("expected_counts", {}),
                    "actual_counts": post_validation.get("actual_counts", {})
                }
            )
        
        # Attach validation summary to each document
        for doc in documents:
            attach_validation_summary(doc, pre_validation, post_validation)
        
        logger.info("Document enhancement with ISNE embeddings completed")
        
    except Exception as e:
        error_msg = f"Error during ISNE application: {str(e)}"
        logger.error(error_msg, exc_info=True)
        alert_manager.alert(
            message=error_msg,
            level=AlertLevel.CRITICAL,
            source="isne_pipeline",
            context={"exception": str(e)}
        )
        # Continue execution to save partial results
    
    # Calculate ISNE enhancement time
    isne_enhancement_time = time.time() - isne_start_time
    logger.info(f"ISNE enhancement completed in {isne_enhancement_time:.2f}s")
    
    # Save enhanced documents to JSON file
    json_start_time = time.time()
    isne_output_path = output_path / "isne_enhanced_documents.json"
    with open(isne_output_path, "w") as f:
        json.dump(documents, f, indent=2)
    
    # Save a sample of documents for inspection
    sample_docs = documents[:min(5, len(documents))]
    sample_path = output_path / "isne_sample_documents.json"
    with open(sample_path, "w") as f:
        json.dump(sample_docs, f, indent=2)
    
    json_save_time = time.time() - json_start_time
    logger.info(f"Saved {len(documents)} enhanced documents to {isne_output_path}")
    logger.info(f"Saved {len(sample_docs)} sample documents to {sample_path}")
    
    # Calculate total processing time
    total_time = time.time() - start_time
    
    # Generate report
    stats = {
        "total_documents": len(documents),
        "total_chunks": total_chunks,
        "isne_enhancement_time": isne_enhancement_time,
        "json_save_time": json_save_time,
        "total_time": total_time,
        "output_path": str(isne_output_path),
        "sample_path": str(sample_path),
        "alerts": alert_manager.get_alerts()
    }
    
    # Generate report
    generate_report(stats, output_path, alert_dir)
    
    return documents


def build_graph_from_documents(docs):
    """Build a graph from document chunks for ISNE processing.
    
    Args:
        docs: List of documents with chunks and embeddings
        
    Returns:
        PyTorch Geometric Data object representing the graph
    """
    import torch
    from torch_geometric.data import Data
    
    # Extract embeddings and metadata
    node_embeddings = []
    node_metadata = []
    node_model_types = []  # Track embedding model types
    edge_index_src = []
    edge_index_dst = []
    edge_attr = []
    
    # Node index mapping
    node_idx_map = {}
    current_idx = 0
    
    # First pass: collect all nodes
    for doc in docs:
        if "chunks" not in doc or not doc["chunks"]:
            continue
        
        for chunk_idx, chunk in enumerate(doc["chunks"]):
            if "embedding" not in chunk or not chunk["embedding"]:
                continue
            
            chunk_id = f"{doc['file_id']}_{chunk_idx}"
            node_idx_map[chunk_id] = current_idx
            
            # Add the base embedding
            node_embeddings.append(chunk["embedding"])
            
            # Add metadata
            metadata = {
                "doc_id": doc["file_id"],
                "chunk_idx": chunk_idx,
                "text": chunk.get("text", "")[:100]  # Truncate for metadata
            }
            node_metadata.append(metadata)
            
            # Track embedding model type
            model_type = chunk.get("metadata", {}).get("embedding_model", "default")
            node_model_types.append(model_type)
            
            current_idx += 1
    
    # Second pass: build edges
    for doc in docs:
        if "chunks" not in doc or not doc["chunks"]:
            continue
        
        # Create sequential edges between chunks in the same document
        for i in range(len(doc["chunks"]) - 1):
            source_id = f"{doc['file_id']}_{i}"
            target_id = f"{doc['file_id']}_{i+1}"
            
            if source_id in node_idx_map and target_id in node_idx_map:
                edge_index_src.append(node_idx_map[source_id])
                edge_index_dst.append(node_idx_map[target_id])
                edge_attr.append([1.0])  # Sequential relationship weight
        
        # Add code-specific relationships from chunk metadata
        for chunk_idx, chunk in enumerate(doc["chunks"]):
            if "metadata" not in chunk or "references" not in chunk["metadata"]:
                continue
            
            source_id = f"{doc['file_id']}_{chunk_idx}"
            if source_id not in node_idx_map:
                continue
            
            # Process each reference (relationship)
            for reference in chunk["metadata"]["references"]:
                target_id = reference.get("target")
                if not target_id or target_id not in node_idx_map:
                    continue
                
                relation_type = reference.get("type", "REFERENCES")
                weight = reference.get("weight", 0.8)
                
                # Add edge to the graph
                edge_index_src.append(node_idx_map[source_id])
                edge_index_dst.append(node_idx_map[target_id])
                edge_attr.append([weight])  # Use the relationship weight
    
    # Convert lists to tensors
    x = torch.tensor(node_embeddings, dtype=torch.float)
    edge_index = torch.tensor([edge_index_src, edge_index_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    logger.info(f"Built graph with {x.shape[0]} nodes and {edge_index.shape[1]} edges")
    return data


def generate_report(stats, output_path, alert_dir):
    """Generate a comprehensive report for the ISNE process.
    
    Args:
        stats: Dictionary of statistics
        output_path: Output directory path
        alert_dir: Alert directory path
    """
    logger.info("\n=== ISNE Pipeline Report ===")
    
    logger.info("\nProcessing Statistics:")
    logger.info(f"  Total Documents:       {stats['total_documents']}")
    logger.info(f"  ISNE Enhancement:      {stats['isne_enhancement_time']:.2f}s")
    logger.info(f"  JSON Saving:           {stats['json_save_time']:.2f}s")
    logger.info(f"  Total Time:            {stats['total_time']:.2f}s")
    
    # Add alert information to report
    if "alerts" in stats and stats["alerts"]:
        alert_counts = {
            "LOW": 0,
            "MEDIUM": 0,
            "HIGH": 0,
            "CRITICAL": 0
        }
        
        for alert in stats["alerts"]:
            if alert["level"] in alert_counts:
                alert_counts[alert["level"]] += 1
        
        logger.info("\nAlert Summary:")
        logger.info(f"  LOW:      {alert_counts['LOW']}")
        logger.info(f"  MEDIUM:   {alert_counts['MEDIUM']}")
        logger.info(f"  HIGH:     {alert_counts['HIGH']}")
        logger.info(f"  CRITICAL: {alert_counts['CRITICAL']}")
        
        # Check if there are critical or high alerts
        if alert_counts["CRITICAL"] > 0 or alert_counts["HIGH"] > 0:
            logger.warning(f"⚠️ WARNING: {alert_counts['CRITICAL'] + alert_counts['HIGH']} critical/high alerts were generated.")
            logger.warning(f"Review alerts in {alert_dir}")
    
    logger.info("\nOutput Files:")
    logger.info(f"  All Documents:         {stats['output_path']}")
    logger.info(f"  Sample Documents:      {stats['sample_path']}")
    logger.info(f"  Alert Logs:            {alert_dir}")
    
    logger.info("\nISNE Pipeline Completed Successfully!")
    
    # Write report to file
    report_file = Path(output_path) / "isne_report.json"
    with open(report_file, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Detailed report saved to {report_file}")


def main():
    """Run the script as a command-line tool."""
    parser = argparse.ArgumentParser(description='Run ISNE Pipeline with Alert System')
    parser.add_argument('--input-file', type=str, required=True,
                      help='Path to input JSON file with document embeddings')
    parser.add_argument('--model-path', type=str, required=True,
                      help='Path to trained ISNE model')
    parser.add_argument('--output-dir', type=str, default='./isne-output',
                      help='Directory for output files')
    parser.add_argument('--alert-threshold', type=str, choices=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'], 
                      default='MEDIUM', help='Alert threshold level')
    
    args = parser.parse_args()
    
    run_isne_with_alerts(
        input_file=args.input_file,
        model_path=args.model_path,
        output_dir=args.output_dir,
        alert_threshold=args.alert_threshold
    )


if __name__ == "__main__":
    main()
