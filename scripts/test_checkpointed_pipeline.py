#!/usr/bin/env python3
"""
Test the checkpointed pipeline with a small sample from the dataset.
This ensures everything works before the overnight run.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from checkpointed_pipeline import CheckpointedPipeline, create_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_small_sample():
    """Test with 10 files from the dataset."""
    logger.info("=" * 60)
    logger.info("TESTING CHECKPOINTED PIPELINE")
    logger.info("=" * 60)
    
    # Dataset location
    data_dir = Path("/home/todd/ML-Lab/Olympus/sequential-ISNE-testdata/isne-testdata")
    
    if not data_dir.exists():
        logger.error(f"Dataset not found at: {data_dir}")
        return False
    
    # Test checkpoint directory
    test_checkpoint_dir = Path("./test_checkpoints")
    test_checkpoint_dir.mkdir(exist_ok=True)
    
    # Create pipeline
    config = create_config()
    pipeline = CheckpointedPipeline(config, test_checkpoint_dir)
    
    # Collect sample files
    logger.info("\nCollecting sample files...")
    all_files = pipeline.collect_files(data_dir)
    
    # Get a diverse sample
    sample_files = []
    file_types = {'.pdf': 2, '.py': 3, '.md': 5}  # Total: 10 files
    
    for ext, count in file_types.items():
        files_of_type = [f for f in all_files if f.suffix == ext]
        sample_files.extend(files_of_type[:count])
    
    logger.info(f"Selected {len(sample_files)} files for testing:")
    for f in sample_files:
        logger.info(f"  - {f.relative_to(data_dir)}")
    
    # Test file categorization
    categories = pipeline.categorize_files(sample_files)
    logger.info(f"\nFile categories: {', '.join(f'{ext}: {len(files)}' for ext, files in categories.items())}")
    
    # Test file to node conversion
    logger.info("\nTesting file to node conversion...")
    test_file = sample_files[0]
    try:
        node = pipeline._file_to_node(test_file)
        logger.info(f"✅ Successfully converted {test_file.name} to node")
        logger.info(f"   Node ID: {node['node_id']}")
        logger.info(f"   File size: {node['size_bytes']} bytes")
    except Exception as e:
        logger.error(f"❌ Failed to convert file to node: {e}")
        return False
    
    # Test batch processing
    logger.info("\nTesting batch processing...")
    try:
        # Process first 3 files
        test_batch = sample_files[:3]
        checkpoints = pipeline.process_batch(test_batch)
        
        successful = sum(1 for cp in checkpoints if cp.status == 'completed')
        logger.info(f"Processed {len(checkpoints)} files: {successful} successful")
        
        for cp in checkpoints:
            status_icon = "✅" if cp.status == 'completed' else "❌"
            logger.info(f"  {status_icon} {Path(cp.file_path).name} - {cp.status}")
            if cp.error_message:
                logger.info(f"     Error: {cp.error_message}")
                
    except Exception as e:
        logger.error(f"❌ Batch processing failed: {e}")
        return False
    
    # Test checkpoint recovery
    logger.info("\nTesting checkpoint recovery...")
    stats = pipeline.checkpoint_manager.get_statistics()
    logger.info(f"Checkpoint stats: {stats}")
    
    remaining = pipeline.checkpoint_manager.get_remaining_files(sample_files)
    logger.info(f"Remaining files: {len(remaining)} of {len(sample_files)}")
    
    # Test scheduling
    logger.info("\nTesting scheduler...")
    logger.info(f"Is overnight window: {pipeline.scheduler.is_overnight_window()}")
    logger.info(f"Should pause: {pipeline.scheduler.should_pause()}")
    
    if pipeline.scheduler.should_pause():
        wait_time = pipeline.scheduler.time_until_next_window()
        logger.info(f"Time until next window: {wait_time/3600:.1f} hours")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ ALL TESTS PASSED")
    logger.info("=" * 60)
    
    return True


def estimate_full_processing_time():
    """Estimate time for full dataset processing."""
    data_dir = Path("/home/todd/ML-Lab/Olympus/sequential-ISNE-testdata/isne-testdata")
    
    config = create_config()
    pipeline = CheckpointedPipeline(config, Path("./temp"))
    
    all_files = pipeline.collect_files(data_dir)
    categories = pipeline.categorize_files(all_files)
    
    # Estimate based on file types (seconds per file)
    time_estimates = {
        '.pdf': 120,   # 2 minutes per PDF
        '.py': 30,     # 30 seconds per Python file
        '.md': 15,     # 15 seconds per Markdown
        '.txt': 10,    # 10 seconds per text
        '.json': 20    # 20 seconds per JSON
    }
    
    total_seconds = 0
    logger.info("\nProcessing time estimates:")
    
    for ext, files in categories.items():
        time_per_file = time_estimates.get(ext, 30)
        category_time = len(files) * time_per_file
        total_seconds += category_time
        
        logger.info(f"  {ext}: {len(files)} files × {time_per_file}s = {category_time/3600:.1f} hours")
    
    logger.info(f"\nTotal estimated time: {total_seconds/3600:.1f} hours")
    logger.info(f"With overhead and retries: {total_seconds/3600 * 1.2:.1f} hours")
    
    # Check if it fits in overnight window
    overnight_hours = 8  # 11 PM to 7 AM
    if total_seconds/3600 > overnight_hours:
        logger.warning(f"⚠️  Processing may exceed overnight window!")
        logger.info(f"Consider processing over {int(total_seconds/3600/overnight_hours) + 1} nights")
    else:
        logger.info(f"✅ Should complete in one overnight session")


if __name__ == "__main__":
    logger.info(f"Test started at: {datetime.now()}")
    
    # Run tests
    if test_small_sample():
        logger.info("\n✅ Pipeline test successful!")
        
        # Show time estimates
        estimate_full_processing_time()
        
        logger.info("\n🚀 Ready for production run:")
        logger.info("   python checkpointed_pipeline.py \\")
        logger.info("     --data-dir /home/todd/ML-Lab/Olympus/sequential-ISNE-testdata/isne-testdata \\")
        logger.info("     --checkpoint-dir ./production_checkpoints")
    else:
        logger.error("\n❌ Pipeline test failed! Fix issues before production run.")