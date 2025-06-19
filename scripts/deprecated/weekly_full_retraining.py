#!/usr/bin/env python3
"""
Weekly Full ISNE Retraining Script

Designed to run as a cron job to perform full ISNE retraining on all data
from the past week/month, ensuring optimal model performance and preventing
embedding drift.

Usage:
  # As cron job (weekly retraining)
  python scripts/weekly_full_retraining.py
  
  # Manual monthly retraining
  python scripts/weekly_full_retraining.py --scope monthly
  
  # Full comprehensive retraining
  python scripts/weekly_full_retraining.py --scope full
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

def setup_logging(log_level: str = "INFO", scope: str = "weekly"):
    """Set up logging for full retraining."""
    log_dir = Path(f"./training_logs/{scope}")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{scope}_retraining_{timestamp}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )

def load_trained_isne_pipeline() -> ISNEPipeline:
    """Load the trained ISNE pipeline from previous training."""
    
    # Look for trained model in standard locations
    model_paths = [
        Path("./models/current_isne_model.pt"),
        Path("./bootstrap-output/models/refined_isne_model.pt"),
        Path("./training_logs/weekly/latest_model.pt")
    ]
    
    for model_path in model_paths:
        if model_path.exists():
            logger.info(f"Loading ISNE model from {model_path}")
            return ISNEPipeline.load_from_file(str(model_path))
    
    raise FileNotFoundError("No trained ISNE model found. Run bootstrap first.")

def save_model_checkpoint(isne_pipeline: ISNEPipeline, scope: str):
    """Save the retrained model as the current checkpoint."""
    
    # Save to multiple locations for redundancy
    model_dir = Path("./models")
    model_dir.mkdir(exist_ok=True)
    
    # Current model (for next incremental training)
    current_model_path = model_dir / "current_isne_model.pt"
    isne_pipeline.save_to_file(str(current_model_path))
    
    # Timestamped backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = model_dir / f"{scope}_retrained_model_{timestamp}.pt"
    isne_pipeline.save_to_file(str(backup_path))
    
    # Latest for the scope
    scope_dir = Path(f"./training_logs/{scope}")
    scope_dir.mkdir(exist_ok=True)
    latest_path = scope_dir / "latest_model.pt"
    isne_pipeline.save_to_file(str(latest_path))
    
    logger.info(f"Model saved to: {current_model_path}")
    logger.info(f"Backup saved to: {backup_path}")

def main():
    """Run full retraining."""
    parser = argparse.ArgumentParser(description="Weekly/Monthly full ISNE retraining")
    parser.add_argument("--scope", type=str, choices=['weekly', 'monthly', 'full'], 
                       default='weekly', help="Retraining scope")
    parser.add_argument("--config", type=str, default="scheduled_training_config",
                       help="Training configuration file name")
    parser.add_argument("--force", action="store_true",
                       help="Force retraining even if not scheduled")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be retrained without actually training")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    training_config = config.get('scheduled_training', {})
    
    # Set up logging
    log_level = training_config.get('logging', {}).get('log_level', 'INFO')
    setup_logging(log_level, args.scope)
    
    global logger
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting {args.scope} full ISNE retraining")
    
    try:
        # Load trained ISNE pipeline
        isne_pipeline = load_trained_isne_pipeline()
        
        # Initialize scheduled trainer
        scheduled_trainer = ScheduledISNETrainer(
            isne_pipeline=isne_pipeline,
            training_schedule_config=training_config
        )
        
        # Check if retraining should run (unless forced)
        if not args.force:
            should_retrain, reason = scheduled_trainer.should_run_full_retrain()
            if not should_retrain:
                logger.info(f"Full retraining not needed: {reason}")
                logger.info("Use --force to override")
                return 0
            else:
                logger.info(f"Full retraining triggered: {reason}")
        
        if args.dry_run:
            # Show what would be processed
            if args.scope == 'weekly':
                start_date = datetime.now() - timedelta(weeks=1)
            elif args.scope == 'monthly':
                start_date = datetime.now() - timedelta(days=30)
            else:
                start_date = None
                
            all_chunks = scheduled_trainer._get_chunks_by_date_range(start_date, datetime.now())
            logger.info(f"DRY RUN: Would retrain with {len(all_chunks)} total chunks")
            return 0
        
        # Run full retraining
        logger.info(f"Starting {args.scope} retraining...")
        start_time = datetime.now()
        
        results = scheduled_trainer.run_full_retraining(scope=args.scope)
        
        training_time = datetime.now() - start_time
        
        # Log results
        logger.info(f"{args.scope.title()} retraining completed successfully:")
        logger.info(f"  - Total chunks: {results.get('total_chunks', 0)}")
        logger.info(f"  - Training time: {training_time.total_seconds():.2f} seconds")
        logger.info(f"  - Final loss: {results.get('training_metrics', {}).get('final_loss', 'N/A')}")
        logger.info(f"  - Validation score: {results.get('training_metrics', {}).get('validation_score', 'N/A')}")
        logger.info(f"  - Quality score: {results.get('quality_metrics', {}).get('overall_score', 'N/A')}")
        
        # Save retrained model
        save_model_checkpoint(isne_pipeline, args.scope)
        
        # Check quality and alert if needed
        quality_score = results.get('quality_metrics', {}).get('overall_score', 0)
        quality_threshold = training_config.get('logging', {}).get('quality_threshold', 0.8)
        
        if quality_score < quality_threshold:
            logger.warning(f"Quality score {quality_score:.3f} below threshold {quality_threshold}")
            if training_config.get('logging', {}).get('alert_on_quality_drop', False):
                logger.warning("ALERT: Model quality drop detected - review training data")
        
        # Reset daily update counter
        scheduled_trainer.daily_updates_since_full_retrain = 0
        
        logger.info(f"{args.scope.title()} retraining completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"{args.scope.title()} retraining failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Send alert if configured
        if training_config.get('logging', {}).get('alert_on_failure', False):
            logger.error(f"ALERT: {args.scope} retraining failed - manual intervention required")
        
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)