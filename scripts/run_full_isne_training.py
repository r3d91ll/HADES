#!/usr/bin/env python3
"""
Run Full ISNE Training on Complete Dataset

This script runs ISNE training on the complete dataset, continuing from
the 10% pre-trained model. Expected to take ~30 hours.
"""

import asyncio
import json
import logging
import sys
import os
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.api.isne_training import ISNEProductionTrainer
from src.alerts.alert_manager import AlertManager

# Configure logging with both file and console output
log_dir = Path("logs/isne_training")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"isne_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


async def run_full_training():
    """Run ISNE training on the full dataset."""
    
    # Configuration  
    full_graph_path = "output/full_dataset/graph_data.json"  # You'll need to generate this
    sample_model_path = "output/10percent_sample_bootstrap"  # Look for latest 10% sample model
    output_dir = "output/production_models"
    model_name = f"isne_production_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Find latest 10% sample model
    pretrained_model_path = None
    if Path(sample_model_path).exists():
        # Look for latest model in the sample bootstrap directory
        model_files = list(Path(sample_model_path).glob("**/isne_model_final.pth"))
        if model_files:
            # Get the most recent one
            pretrained_model_path = str(sorted(model_files, key=lambda x: x.stat().st_mtime)[-1])
    
    # Check prerequisites
    if not pretrained_model_path or not Path(pretrained_model_path).exists():
        logger.error("No 10% sample model found!")
        logger.error("Please run the 10% sample bootstrap first:")
        logger.error("  python scripts/bootstrap_10percent_sample.py")
        logger.error("")
        logger.error("This will create a quick validation model for inspection before scaling up.")
        return False
    
    if not Path(full_graph_path).exists():
        logger.warning(f"Full dataset graph not found at {full_graph_path}")
        logger.info("You need to run the full dataset processing first:")
        logger.info("1. Process all documents through the pipeline")
        logger.info("2. Generate embeddings for all chunks")
        logger.info("3. Build the complete graph")
        logger.info("\nFor now, we'll use the 10% sample data as a placeholder")
        # Use the graph from the same sample model directory
        sample_graph_path = str(Path(pretrained_model_path).parent / "graph_data.json")
        if Path(sample_graph_path).exists():
            full_graph_path = sample_graph_path
        else:
            logger.error("No graph data found for the 10% sample model!")
            return False
    
    # Production training configuration
    training_config = {
        "epochs": 50,  # More epochs for production
        "learning_rate": 0.0005,  # Lower LR since we're fine-tuning
        "batch_size": 64,  # Larger batch if memory allows
        "hidden_dim": 128,
        "num_layers": 3,
        "num_heads": 4,
        "dropout": 0.1,
        "weight_decay": 1e-5,
        "patience": 10,  # More patience for production
        "min_delta": 0.0001
    }
    
    # Initialize components
    alert_manager = AlertManager(alert_dir="./alerts")
    trainer = ISNEProductionTrainer(alert_manager=alert_manager)
    
    # Job tracking
    job_id = f"production-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    job_data = {
        "job_id": job_id,
        "status": "pending",
        "progress_percent": 0.0,
        "stage": "initialization",
        "results": {}
    }
    
    logger.info("=" * 80)
    logger.info("ISNE PRODUCTION TRAINING")
    logger.info("=" * 80)
    logger.info(f"Job ID: {job_id}")
    logger.info(f"Model name: {model_name}")
    logger.info(f"Graph data: {full_graph_path}")
    logger.info(f"Pre-trained model: {pretrained_model_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Training config: {json.dumps(training_config, indent=2)}")
    logger.info("=" * 80)
    
    # Confirm before starting
    if os.getenv("ISNE_TRAINING_AUTO_CONFIRM") != "true":
        logger.info("\n⚠️  This training run may take 10-30 hours to complete.")
        logger.info("Make sure you're running this in a screen/tmux session or background process.")
        response = input("\nProceed with training? (yes/no): ")
        if response.lower() != "yes":
            logger.info("Training cancelled.")
            return False
    
    try:
        logger.info("\n🚀 Starting ISNE production training...")
        start_time = datetime.now()
        
        # Run training
        results = await trainer.train(
            job_id=job_id,
            job_data=job_data,
            graph_data_path=full_graph_path,
            output_dir=output_dir,
            training_config=training_config,
            model_path=pretrained_model_path,
            model_name=model_name
        )
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"Duration: {duration}")
        logger.info(f"Final model: {results['model_path']}")
        logger.info(f"Best model: {results.get('best_model_path', 'N/A')}")
        logger.info(f"Final loss: {results['final_loss']:.6f}")
        logger.info(f"Best loss: {results['best_loss']:.6f}")
        logger.info(f"Training epochs: {results['training_epochs']}")
        
        # Save training summary
        summary_path = Path(output_dir) / f"{model_name}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                "job_id": job_id,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration.total_seconds(),
                "training_config": training_config,
                "results": results
            }, f, indent=2, default=str)
        
        logger.info(f"\n📄 Training summary saved to: {summary_path}")
        
        return True
        
    except KeyboardInterrupt:
        logger.warning("\n\n⚠️  Training interrupted by user!")
        logger.info("The training state has been saved and can be resumed.")
        return False
        
    except Exception as e:
        logger.error(f"\n\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main entry point."""
    success = await run_full_training()
    
    if success:
        logger.info("\n🎉 Production ISNE model is ready!")
        logger.info("Next steps:")
        logger.info("1. Validate the model performance")
        logger.info("2. Deploy to production")
        logger.info("3. Set up incremental update pipeline")
    else:
        logger.error("\n❌ Training did not complete successfully.")
        sys.exit(1)


if __name__ == "__main__":
    # Set up signal handlers for graceful shutdown
    import signal
    
    def signal_handler(signum, frame):
        logger.warning(f"\nReceived signal {signum}. Attempting graceful shutdown...")
        # The training loop should handle this gracefully
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the training
    asyncio.run(main())