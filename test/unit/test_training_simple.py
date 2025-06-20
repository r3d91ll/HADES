#!/usr/bin/env python3
"""
Simple one-time ISNE training test script.
Uses config defaults to train with 10% data and 20 epochs.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.isne.training.pipeline import ISNETrainingPipeline, ISNETrainingConfig

# Configure logging to both console and file
from datetime import datetime, timezone
import os

# Create logs directory if it doesn't exist
logs_dir = Path(__file__).parent / "logs" / "training_scripts"
logs_dir.mkdir(parents=True, exist_ok=True)

# Create log filename with timestamp
log_filename = f"simple_training_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
log_file_path = logs_dir / log_filename

# Configure logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)

print(f"Logging to: {log_file_path}")

def main():
    """Run simple training test with config defaults."""
    
    print("=" * 60)
    print("SIMPLE ISNE TRAINING TEST")
    print("=" * 60)
    print("Using config defaults:")
    print("- Data: 10% of ../test-data3")
    print("- Epochs: 20")
    print("- Source model: output/micro_validation_bootstrap/isne_model_final.pth")
    print("=" * 60)
    
    try:
        # Create training pipeline with default config
        config = ISNETrainingConfig()
        pipeline = ISNETrainingPipeline(config)
        
        # Run training
        result = pipeline.train()
        
        if result.success:
            print("\n" + "=" * 60)
            print("✅ TRAINING COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"Model: {result.model_path}")
            print(f"Name: {result.model_name}")
            print(f"Version: {result.version}")
            print(f"Duration: {result.total_time_seconds / 60:.1f} minutes")
            print(f"Files processed: {result.training_stats.get('files_processed', 0)}")
            
            if result.evaluation_results:
                performance = result.evaluation_results.get('inductive_performance', 0)
                print(f"Performance: {performance:.2%}")
            
            return True
            
        else:
            print("\n" + "=" * 60)
            print("❌ TRAINING FAILED")
            print("=" * 60)
            print(f"Error: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"\n❌ Script failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)