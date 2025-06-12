#!/usr/bin/env python3
"""
Example: Setting up Scheduled ISNE Training

This script demonstrates how to set up and use the hierarchical 
training system for daily incremental and weekly/monthly full retraining.

Usage:
  python scripts/example_scheduled_training_setup.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.isne.bootstrap import ScheduledISNETrainer
from src.config.config_loader import load_config
from src.isne.pipeline.isne_pipeline import ISNEPipeline

def example_setup():
    """Example of setting up scheduled training."""
    
    print("🤖 HADES Scheduled Training Setup Example")
    print("=" * 55)
    
    # 1. Load configuration
    print("📋 Loading training configuration...")
    try:
        config = load_config('scheduled_training_config')
        training_config = config.get('scheduled_training', {})
        print("✅ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return
    
    # 2. Load trained model (would come from bootstrap)
    print("\n🔧 Loading trained ISNE model...")
    
    # In practice, you'd load your actual trained model:
    # isne_pipeline = ISNEPipeline.load_from_file("./models/current_isne_model.pt")
    
    # For this example, we'll show the setup without actual training
    print("✅ Model loading configured")
    
    # 3. Initialize scheduled trainer
    print("\n⚙️ Initializing scheduled trainer...")
    
    # scheduled_trainer = ScheduledISNETrainer(
    #     isne_pipeline=isne_pipeline,
    #     training_schedule_config=training_config
    # )
    
    print("✅ Scheduled trainer configured")
    
    # 4. Show training schedule
    print("\n📅 Training Schedule:")
    print("-" * 25)
    
    incremental_config = training_config.get('incremental', {})
    print(f"Daily Incremental:")
    print(f"  - Max chunks per day: {incremental_config.get('max_daily_chunks', 500)}")
    print(f"  - Max daily updates before full retrain: {incremental_config.get('max_daily_updates_before_full', 30)}")
    
    weekly_config = training_config.get('weekly', {})
    print(f"\nWeekly Full Retraining:")
    print(f"  - Enabled: {weekly_config.get('enabled', True)}")
    print(f"  - Interval: {weekly_config.get('interval_days', 7)} days")
    
    monthly_config = training_config.get('monthly', {})
    print(f"\nMonthly Comprehensive:")
    print(f"  - Enabled: {monthly_config.get('enabled', True)}")
    print(f"  - Run on 1st: {monthly_config.get('run_on_first', True)}")
    
    # 5. Example workflow
    print("\n🔄 Example Training Workflow:")
    print("-" * 35)
    
    print("1. Daily (2 AM): Process yesterday's documents incrementally")
    print("   - Get new chunks from ArangoDB by date")
    print("   - Generate ISNE embeddings with existing model") 
    print("   - Build relationships to existing chunks")
    print("   - Update database with new embeddings")
    
    print("\n2. Weekly (Sunday 3 AM): Full retraining on week's data")
    print("   - Collect all chunks from past 7 days")
    print("   - Retrain ISNE model on combined dataset")
    print("   - Re-embed all chunks with updated model")
    print("   - Save new model as current checkpoint")
    
    print("\n3. Monthly (1st 4 AM): Comprehensive retraining")
    print("   - Collect all chunks from past 30 days")
    print("   - Full model retraining with enhanced parameters")
    print("   - Complete embedding refresh")
    print("   - Model backup and deployment")
    
    # 6. Cron job setup
    print("\n⏰ Cron Job Setup:")
    print("-" * 22)
    print("To set up automated training, run:")
    print("  ./scripts/setup_training_cron.sh")
    print("")
    print("This will add these cron jobs:")
    print("  0 2 * * *   Daily incremental training")
    print("  0 3 * * 0   Weekly full retraining") 
    print("  0 4 1 * *   Monthly comprehensive retraining")
    
    # 7. Monitoring
    print("\n📊 Monitoring:")
    print("-" * 15)
    print("Check training status with:")
    print("  python scripts/check_training_status.py")
    print("  python scripts/check_training_status.py --full")
    print("  python scripts/check_training_status.py --alerts")
    
    # 8. Manual execution examples
    print("\n🔧 Manual Execution:")
    print("-" * 22)
    print("Run training manually:")
    print("  # Daily for specific date")
    print("  python scripts/daily_incremental_training.py --date 2024-01-15")
    print("  ")
    print("  # Weekly retraining")
    print("  python scripts/weekly_full_retraining.py --scope weekly")
    print("  ")
    print("  # Monthly retraining")
    print("  python scripts/weekly_full_retraining.py --scope monthly")
    print("  ")
    print("  # Dry run (see what would be processed)")
    print("  python scripts/daily_incremental_training.py --dry-run")
    
    print("\n✅ Scheduled training setup complete!")
    print("\nNext steps:")
    print("1. Complete your ISNE bootstrap process")
    print("2. Set up cron jobs with setup_training_cron.sh")
    print("3. Monitor daily with check_training_status.py")
    print("4. Your RAG system will continuously improve! 🚀")

if __name__ == "__main__":
    example_setup()