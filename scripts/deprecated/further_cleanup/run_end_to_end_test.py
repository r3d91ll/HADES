#!/usr/bin/env python
"""
End-to-end test of the ISNE pipeline with validation and alerts.

This script demonstrates the complete workflow from PDF ingestion to ISNE
enhancement with validation and alerts.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the project root directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.isne.pipeline.isne_pipeline import ISNEPipeline
from src.alerts import AlertManager, AlertLevel


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


def process_pdfs(args):
    """Process PDFs through the ingestion, ISNE, and validation pipeline."""
    # Create output directories
    output_dir = Path(args.output_dir)
    ingestion_dir = output_dir / "ingestion"
    isne_dir = output_dir / "isne"
    alerts_dir = output_dir / "alerts"
    
    for dir_path in [output_dir, ingestion_dir, isne_dir, alerts_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Set up alert manager
    alert_manager = AlertManager(
        alert_dir=str(alerts_dir),
        min_level=getattr(AlertLevel, args.alert_threshold.upper())
    )
    
    # 1. PDF Ingestion
    logging.info(f"Starting PDF ingestion from {args.input_dir}")
    documents = []
    
    try:
        # Import ingest module
        from src.ingest.pdf_ingester import process_pdfs as ingest_pdfs
        
        # Get list of PDF files
        pdf_dir = Path(args.input_dir)
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            logging.error(f"No PDF files found in {args.input_dir}")
            alert_manager.alert(
                message=f"No PDF files found in {args.input_dir}",
                level=AlertLevel.HIGH,
                source="end_to_end_test"
            )
            return 1
        
        logging.info(f"Found {len(pdf_files)} PDF files")
        
        # Process only a subset if specified
        if args.max_files:
            pdf_files = pdf_files[:args.max_files]
            logging.info(f"Processing {len(pdf_files)} PDFs (limited by --max-files)")
        
        # Process PDFs
        start_time = time.time()
        documents = ingest_pdfs(
            pdf_files,
            output_dir=str(ingestion_dir),
            base_embeddings=True,
            chunk_size=args.chunk_size
        )
        ingestion_time = time.time() - start_time
        
        # Save ingested documents
        with open(ingestion_dir / "ingested_documents.json", "w") as f:
            json.dump(documents, f, indent=2)
        
        logging.info(f"PDF ingestion completed in {ingestion_time:.2f} seconds")
        
    except ImportError:
        logging.error("PDF ingestion module not found")
        alert_manager.alert(
            message="PDF ingestion module not found",
            level=AlertLevel.HIGH,
            source="end_to_end_test"
        )
        
        # For testing, try to load documents from a file if ingestion failed
        if args.fallback_docs:
            try:
                logging.info(f"Trying to load documents from {args.fallback_docs}")
                with open(args.fallback_docs, "r") as f:
                    documents = json.load(f)
                logging.info(f"Loaded {len(documents)} documents from fallback file")
            except Exception as e:
                logging.error(f"Failed to load fallback documents: {e}")
                return 1
        else:
            return 1
    
    # Skip further processing if no documents were generated
    if not documents:
        logging.error("No documents were generated during ingestion")
        alert_manager.alert(
            message="No documents were generated during ingestion",
            level=AlertLevel.HIGH,
            source="end_to_end_test"
        )
        return 1
    
    # Calculate ingestion statistics
    total_chunks = sum(len(doc.get("chunks", [])) for doc in documents)
    chunks_with_embeddings = sum(
        1 for doc in documents 
        for chunk in doc.get("chunks", []) 
        if "embedding" in chunk
    )
    
    logging.info(f"Ingested {len(documents)} documents with {total_chunks} chunks")
    logging.info(f"Chunks with base embeddings: {chunks_with_embeddings}/{total_chunks}")
    
    # Alert if some chunks are missing embeddings
    if chunks_with_embeddings < total_chunks:
        alert_manager.alert(
            message=f"Missing base embeddings in {total_chunks - chunks_with_embeddings} chunks",
            level=AlertLevel.MEDIUM,
            source="end_to_end_test",
            context={
                "total_chunks": total_chunks,
                "chunks_with_embeddings": chunks_with_embeddings
            }
        )
    
    # 2. ISNE Pipeline
    logging.info("Starting ISNE pipeline with validation")
    
    # Initialize pipeline
    pipeline = ISNEPipeline(
        model_path=args.model_path,
        validate=True,
        alert_threshold=args.alert_threshold,
        alert_manager=alert_manager
    )
    
    # Process documents
    start_time = time.time()
    
    try:
        enhanced_docs, stats = pipeline.process_documents(
            documents=documents,
            save_report=True,
            output_dir=str(isne_dir)
        )
        
        isne_time = time.time() - start_time
        
        logging.info(f"ISNE processing completed in {isne_time:.2f} seconds")
        
        # Get validation summary
        validation_summary = stats.get("validation_summary", {})
        
        # Calculate statistics
        chunks_with_isne = sum(
            1 for doc in enhanced_docs 
            for chunk in doc.get("chunks", []) 
            if "isne_embedding" in chunk
        )
        
        logging.info(f"Chunks with ISNE embeddings: {chunks_with_isne}/{total_chunks}")
        
    except Exception as e:
        logging.error(f"Error in ISNE pipeline: {e}")
        alert_manager.alert(
            message=f"ISNE pipeline error: {e}",
            level=AlertLevel.CRITICAL,
            source="end_to_end_test"
        )
        return 1
    
    # 3. Alert Summary
    alert_stats = alert_manager.get_alert_stats()
    
    # Print summary
    print("\n===== End-to-End Pipeline Summary =====")
    print(f"Documents processed: {len(documents)}")
    print(f"Total chunks: {total_chunks}")
    print(f"Chunks with base embeddings: {chunks_with_embeddings}/{total_chunks}")
    
    try:
        print(f"Chunks with ISNE embeddings: {chunks_with_isne}/{total_chunks}")
    except UnboundLocalError:
        print("ISNE embedding application failed")
    
    print(f"\nTotal processing time: {isne_time + ingestion_time:.2f} seconds")
    
    # Print alert summary
    print("\n===== Alert Summary =====")
    for level, count in alert_stats.items():
        print(f"{level}: {count}")
    
    # Print validation summary if available
    if validation_summary:
        print("\n===== Validation Summary =====")
        
        # Missing embeddings
        if "discrepancies" in validation_summary:
            discrepancies = validation_summary["discrepancies"]
            total_discrepancies = sum(abs(value) for value in discrepancies.values())
            print(f"Total discrepancies: {total_discrepancies}")
            
            for key, value in discrepancies.items():
                if value != 0:
                    print(f"  {key}: {value}")
    
    # Check if there were any critical alerts
    critical_count = alert_stats.get("CRITICAL", 0) + alert_stats.get("HIGH", 0)
    if critical_count > 0:
        print(f"\n⚠️ WARNING: {critical_count} critical/high alerts were generated.")
        print(f"Review alerts in {alerts_dir}")
    
    # Print output paths
    print("\n===== Output Files =====")
    print(f"Ingestion output: {ingestion_dir}")
    print(f"ISNE output: {isne_dir}")
    print(f"Alert logs: {alerts_dir}")
    
    return 0


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Run end-to-end ISNE pipeline with validation and alerts"
    )
    
    parser.add_argument(
        "-i", "--input-dir",
        required=True,
        help="Input directory containing PDF files"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        default="./e2e-test-output",
        help="Output directory for all pipeline stages"
    )
    
    parser.add_argument(
        "-m", "--model-path",
        help="Path to ISNE model file"
    )
    
    parser.add_argument(
        "-c", "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size for PDF ingestion"
    )
    
    parser.add_argument(
        "-a", "--alert-threshold",
        choices=["low", "medium", "high", "critical"],
        default="medium",
        help="Threshold for triggering alerts"
    )
    
    parser.add_argument(
        "--max-files",
        type=int,
        help="Maximum number of PDF files to process"
    )
    
    parser.add_argument(
        "--fallback-docs",
        help="Path to fallback document JSON if ingestion fails"
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
        return process_pdfs(args)
    except Exception as e:
        logging.error(f"Error in end-to-end test: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
