#!/usr/bin/env python
"""
Run the ISNE pipeline with integrated validation and alerts.

This script demonstrates the full integration of the ISNE pipeline
with validation and alert system, providing a command-line interface
for processing documents and monitoring alerts.
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


def load_documents(input_path: str) -> List[Dict[str, Any]]:
    """Load documents from a JSON file."""
    with open(input_path, "r") as f:
        documents = json.load(f)
    
    # Handle both list and dict formats
    if isinstance(documents, dict):
        if "documents" in documents:
            return documents["documents"]
        else:
            return [documents]
    
    return documents


def process_documents(args):
    """Process documents with ISNE pipeline and validation alerts."""
    # Load documents
    documents = load_documents(args.input)
    logging.info(f"Loaded {len(documents)} documents from {args.input}")
    
    # Ensure output directory exists
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create alert directory
    alert_dir = Path(args.alert_dir)
    alert_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up email alerts if configured
    email_config = None
    if args.email_config:
        try:
            with open(args.email_config, "r") as f:
                email_config = json.load(f)
        except Exception as e:
            logging.error(f"Error loading email configuration: {e}")
    
    if args.validation_only:
        # Validation-only mode using the AlertManager directly
        from src.validation.embedding_validator import (
            validate_embeddings_before_isne,
            validate_embeddings_after_isne,
            create_validation_summary,
            attach_validation_summary
        )
        
        # Initialize alert manager
        alert_manager = AlertManager(
            alert_dir=str(alert_dir),
            min_level=getattr(AlertLevel, args.alert_threshold.upper())
        )
        
        # Configure email alerts if available
        if email_config:
            alert_manager.configure_email(email_config)
            logging.info("Email alerts configured")
        
        # Process documents with validation only
        logging.info("Running validation-only mode on documents...")
        start_time = time.time()
        
        # Check if documents already have ISNE embeddings
        has_isne = any("isne_embedding" in chunk for doc in documents for chunk in doc.get("chunks", []))
        
        # Run pre-validation
        pre_validation = validate_embeddings_before_isne(documents)
        
        # Run post-validation if documents have ISNE embeddings
        if has_isne:
            post_validation = validate_embeddings_after_isne(documents, pre_validation)
            validation_summary = create_validation_summary(pre_validation, post_validation)
            
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
                
                if total_discrepancies >= 10:  # Same thresholds as in pipeline
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
                    source="validation_only_mode",
                    context=context
                )
            
            # Attach validation summary to documents
            enhanced_docs = attach_validation_summary(documents, validation_summary)
            
            # Save validation report
            report_path = output_path = Path(output_dir) / "validation_report.json"
            with open(report_path, "w") as f:
                json.dump(validation_summary, f, indent=2)
                
            logging.info(f"Saved validation report to {report_path}")
            
            # Stats for return
            stats = {
                "total_documents": len(documents),
                "total_chunks": pre_validation.get("total_chunks", 0),
                "processing_time": time.time() - start_time,
                "validation_summary": validation_summary
            }
        else:
            # No ISNE embeddings to validate
            enhanced_docs = documents
            stats = {
                "total_documents": len(documents),
                "total_chunks": pre_validation.get("total_chunks", 0),
                "processing_time": time.time() - start_time,
                "validation_summary": {"error": "No ISNE embeddings found to validate"}
            }
            
            # Create an alert for missing ISNE embeddings
            alert_manager.alert(
                message="No ISNE embeddings found in documents",
                level=AlertLevel.MEDIUM,
                source="validation_only_mode"
            )
    else:
        # Regular processing mode with ISNE pipeline
        # Initialize pipeline
        pipeline = ISNEPipeline(
            model_path=args.model,
            validate=not args.disable_validation,
            alert_threshold=args.alert_threshold,
            alert_dir=str(alert_dir)
        )
        
        # Configure email alerts if available
        if email_config:
            pipeline.configure_alert_email(email_config)
            logging.info("Email alerts configured")
        
        # Process documents
        logging.info("Processing documents with ISNE pipeline...")
        start_time = time.time()
        
        enhanced_docs, stats = pipeline.process_documents(
            documents=documents,
            save_report=True,
            output_dir=str(output_dir)
        )
    
    processing_time = time.time() - start_time
    
    # Save results
    output_file = output_dir / "enhanced_documents.json"
    with open(output_file, "w") as f:
        json.dump(enhanced_docs, f, indent=2)
    
    # Get alert statistics
    if args.validation_only:
        alert_stats = alert_manager.get_alert_stats()
    else:
        alert_stats = pipeline.alert_manager.get_alert_stats()
    
    # Print summary
    total_chunks = stats.get("total_chunks", 0)
    chunks_per_second = total_chunks / processing_time if processing_time > 0 else 0
    
    print("\n===== ISNE Processing Summary =====")
    print(f"Documents processed: {len(documents)}")
    print(f"Total chunks: {total_chunks}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Processing speed: {chunks_per_second:.2f} chunks/second")
    
    # Print alert summary
    print("\n===== Alert Summary =====")
    for level, count in alert_stats.items():
        print(f"{level}: {count}")
    
    # Print validation summary
    if "validation_summary" in stats and stats["validation_summary"]:
        print("\n===== Validation Summary =====")
        
        # Missing embeddings
        if "discrepancies" in stats["validation_summary"]:
            discrepancies = stats["validation_summary"]["discrepancies"]
            total_discrepancies = sum(abs(value) for value in discrepancies.values())
            print(f"Total discrepancies: {total_discrepancies}")
            
            for key, value in discrepancies.items():
                if value != 0:
                    print(f"  {key}: {value}")
    
    # Check if there were any critical alerts
    critical_count = alert_stats.get("CRITICAL", 0) + alert_stats.get("HIGH", 0)
    if critical_count > 0:
        print(f"\n⚠️ WARNING: {critical_count} critical/high alerts were generated.")
        print(f"Review alerts in {alert_dir}")
    
    # Print output paths
    print("\n===== Output Files =====")
    print(f"Enhanced documents: {output_file}")
    print(f"Validation report: {output_dir / 'isne_validation_report.json'}")
    print(f"Alert logs: {alert_dir}")
    
    return 0


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Run ISNE pipeline with integrated validation and alerts"
    )
    
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input JSON file containing documents with base embeddings"
    )
    
    parser.add_argument(
        "-o", "--output",
        default="./output",
        help="Output directory for enhanced documents and reports"
    )
    
    parser.add_argument(
        "-m", "--model",
        help="Path to ISNE model file"
    )
    
    parser.add_argument(
        "-a", "--alert-dir",
        default="./alerts",
        help="Directory for alert logs"
    )
    
    parser.add_argument(
        "-t", "--alert-threshold",
        choices=["low", "medium", "high"],
        default="medium",
        help="Threshold for triggering alerts"
    )
    
    parser.add_argument(
        "-e", "--email-config",
        help="Path to email configuration JSON file"
    )
    
    parser.add_argument(
        "--disable-validation",
        action="store_true",
        help="Disable validation checks"
    )
    
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Run in validation-only mode (no ISNE model required)"
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
        return process_documents(args)
    except Exception as e:
        logging.error(f"Error processing documents: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
