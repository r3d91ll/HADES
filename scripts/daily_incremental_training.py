#!/usr/bin/env python3
"""
Daily Incremental ISNE Training Script

Designed to run as a cron job to process yesterday's new documents
with incremental ISNE embeddings and relationship building.

Usage:
  # As cron job (runs on yesterday's data)
  python scripts/daily_incremental_training.py
  
  # Manual run for specific date
  python scripts/daily_incremental_training.py --date 2024-01-15
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.isne.bootstrap import ScheduledISNETrainer
from src.config.config_loader import load_config
from src.isne.pipeline.isne_pipeline import ISNEPipeline

def setup_logging(log_level: str = "INFO"):
    """Set up logging for daily training."""
    log_dir = Path("./training_logs/daily")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = log_dir / f"daily_training_{timestamp}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )

def load_trained_isne_pipeline() -> ISNEPipeline:
    """Load the trained ISNE pipeline from bootstrap or previous training."""
    
    # Look for trained model in standard locations
    model_paths = [
        Path("./bootstrap-output/models/refined_isne_model.pt"),
        Path("./models/current_isne_model.pt"),
        Path("./training_logs/weekly/latest_model.pt")
    ]
    
    for model_path in model_paths:
        if model_path.exists():
            logger.info(f"Loading ISNE model from {model_path}")
            return ISNEPipeline.load_from_file(str(model_path))
    
    raise FileNotFoundError("No trained ISNE model found. Run bootstrap first.")

def main():
    """Run daily incremental training."""
    parser = argparse.ArgumentParser(description="Daily incremental ISNE training")
    parser.add_argument("--date", type=str, 
                       help="Date to process (YYYY-MM-DD, defaults to yesterday)")
    parser.add_argument("--config", type=str, default="scheduled_training_config",
                       help="Training configuration file name")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be processed without training")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    training_config = config.get('scheduled_training', {})
    
    # Set up logging
    log_level = training_config.get('logging', {}).get('log_level', 'INFO')
    setup_logging(log_level)
    
    global logger
    logger = logging.getLogger(__name__)
    
    # Parse date
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            logger.error(f"Invalid date format: {args.date}. Use YYYY-MM-DD")
            return 1
    else:
        target_date = datetime.now() - timedelta(days=1)  # Yesterday
    
    logger.info(f"Starting daily incremental training for {target_date.strftime('%Y-%m-%d')}")
    
    try:
        # Load trained ISNE pipeline
        isne_pipeline = load_trained_isne_pipeline()
        
        # Initialize scheduled trainer
        scheduled_trainer = ScheduledISNETrainer(
            isne_pipeline=isne_pipeline,
            training_schedule_config=training_config
        )
        
        if args.dry_run:
            # Show what would be processed
            new_chunks = scheduled_trainer._get_chunks_by_date(target_date)
            logger.info(f"DRY RUN: Would process {len(new_chunks)} chunks from {target_date.strftime('%Y-%m-%d')}")
            return 0
        
        # Run daily incremental training
        results = scheduled_trainer.run_daily_incremental(target_date)
        
        # Log results
        if results.get('status') == 'no_new_data':
            logger.info(f"No new data found for {target_date.strftime('%Y-%m-%d')}")
        else:
            logger.info(f"Daily training completed successfully:")
            logger.info(f"  - Chunks processed: {results.get('chunks_processed', 0)}")
            logger.info(f"  - Processing time: {results.get('processing_time', 0):.2f} seconds")
            logger.info(f"  - Quality score: {results.get('quality_metrics', {}).get('overall_score', 'N/A')}")
            logger.info(f"  - Daily updates since full retrain: {results.get('daily_updates_since_full_retrain', 0)}")
        
        # Check if full retraining should be triggered
        should_retrain, reason = scheduled_trainer.should_run_full_retrain()
        if should_retrain:
            logger.warning(f"Full retraining recommended: {reason}")
            logger.warning("Consider running weekly_full_retraining.py")
        
        return 0
        
    except Exception as e:
        logger.error(f"Daily incremental training failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Send alert if configured
        if training_config.get('logging', {}).get('alert_on_failure', False):
            # Here you could integrate with your alerting system
            logger.error("ALERT: Daily training failed - manual intervention required")
        
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)