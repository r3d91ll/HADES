#!/usr/bin/env python3
"""
Checkpointed Pipeline for Large Dataset Processing
Supports resume, scheduling, and error recovery.
"""

import sys
import json
import time
import logging
import traceback
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, asdict
import smtplib
from email.mime.text import MIMEText

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.pipelines.bootstrap.supra_weight import SupraWeightBootstrapPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('checkpointed_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingCheckpoint:
    """Checkpoint data for resumable processing."""
    file_path: str
    status: str  # 'completed', 'failed', 'skipped'
    timestamp: str
    error_message: Optional[str] = None
    processing_time_seconds: Optional[float] = None
    relationships_created: Optional[int] = None


class CheckpointManager:
    """Manages checkpointing for resumable processing."""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "processing_checkpoint.json"
        self.processed_files: Dict[str, ProcessingCheckpoint] = self._load_checkpoint()
        
    def _load_checkpoint(self) -> Dict[str, ProcessingCheckpoint]:
        """Load existing checkpoint data."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
                return {
                    path: ProcessingCheckpoint(**checkpoint)
                    for path, checkpoint in data.items()
                }
        return {}
    
    def save_checkpoint(self, checkpoint: ProcessingCheckpoint):
        """Save a processing checkpoint."""
        self.processed_files[checkpoint.file_path] = checkpoint
        
        # Save to file
        data = {
            path: asdict(cp) for path, cp in self.processed_files.items()
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(data, f, indent=2)
            
        # Also save a backup with timestamp
        backup_file = self.checkpoint_dir / f"checkpoint_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        if len(self.processed_files) % 100 == 0:  # Backup every 100 files
            with open(backup_file, 'w') as f:
                json.dump(data, f, indent=2)
    
    def get_remaining_files(self, all_files: List[Path]) -> List[Path]:
        """Get files that haven't been processed successfully."""
        completed = {
            cp.file_path for cp in self.processed_files.values()
            if cp.status == 'completed'
        }
        return [f for f in all_files if str(f) not in completed]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = {
            'total_processed': len(self.processed_files),
            'completed': sum(1 for cp in self.processed_files.values() if cp.status == 'completed'),
            'failed': sum(1 for cp in self.processed_files.values() if cp.status == 'failed'),
            'skipped': sum(1 for cp in self.processed_files.values() if cp.status == 'skipped'),
            'total_time_seconds': sum(
                cp.processing_time_seconds or 0 
                for cp in self.processed_files.values()
            ),
            'total_relationships': sum(
                cp.relationships_created or 0
                for cp in self.processed_files.values()
                if cp.relationships_created
            )
        }
        return stats


class TimeScheduler:
    """Manages time-based scheduling for overnight processing."""
    
    def __init__(self, overnight_hours: tuple = (23, 7), peak_hours: tuple = (9, 17)):
        self.overnight_start, self.overnight_end = overnight_hours
        self.peak_start, self.peak_end = peak_hours
        
    def is_overnight_window(self) -> bool:
        """Check if current time is in overnight processing window."""
        current_hour = datetime.now().hour
        
        if self.overnight_start > self.overnight_end:  # Crosses midnight
            return current_hour >= self.overnight_start or current_hour < self.overnight_end
        else:
            return self.overnight_start <= current_hour < self.overnight_end
    
    def is_peak_hours(self) -> bool:
        """Check if current time is during peak hours."""
        current_hour = datetime.now().hour
        return self.peak_start <= current_hour < self.peak_end
    
    def should_pause(self) -> bool:
        """Determine if processing should pause."""
        # Pause during peak hours on weekdays
        if datetime.now().weekday() < 5:  # Monday-Friday
            return self.is_peak_hours()
        return False
    
    def time_until_next_window(self) -> int:
        """Calculate seconds until next processing window."""
        now = datetime.now()
        current_hour = now.hour
        
        if self.should_pause():
            # Calculate time until overnight window
            if current_hour < self.overnight_start:
                target_hour = self.overnight_start
            else:
                target_hour = self.overnight_start  # Next day
                
            target_time = now.replace(hour=target_hour, minute=0, second=0)
            if target_hour <= current_hour:
                target_time = target_time.replace(day=now.day + 1)
                
            return int((target_time - now).total_seconds())
        
        return 0


class CheckpointedPipeline:
    """Main pipeline with checkpointing, scheduling, and error recovery."""
    
    def __init__(self, config: Dict[str, Any], checkpoint_dir: Path = Path("./checkpoints")):
        self.config = config
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        self.scheduler = TimeScheduler()
        self.pipeline = SupraWeightBootstrapPipeline(config)
        
        # Batch sizes by file type
        self.batch_sizes = {
            '.pdf': 5,      # PDFs are memory intensive
            '.py': 25,      # Code files are lighter
            '.md': 50,      # Markdown is fastest
            '.txt': 50,
            '.json': 25
        }
        
        # Error retry limits
        self.retry_limits = {
            'pdf_parse_error': 3,
            'embedding_timeout': 5,
            'database_connection': 10,
            'memory_error': 1
        }
        
    def collect_files(self, data_dir: Path) -> List[Path]:
        """Collect all processable files from directory."""
        supported_extensions = {'.pdf', '.py', '.md', '.txt', '.json'}
        files = []
        
        for ext in supported_extensions:
            files.extend(data_dir.rglob(f"*{ext}"))
            
        # Sort by size (process smaller files first)
        files.sort(key=lambda f: f.stat().st_size)
        
        return files
    
    def categorize_files(self, files: List[Path]) -> Dict[str, List[Path]]:
        """Categorize files by extension for batch processing."""
        categories = {}
        
        for file in files:
            ext = file.suffix.lower()
            if ext not in categories:
                categories[ext] = []
            categories[ext].append(file)
            
        return categories
    
    def process_batch(self, files: List[Path]) -> List[ProcessingCheckpoint]:
        """Process a batch of files."""
        checkpoints = []
        
        for file in files:
            start_time = time.time()
            
            try:
                logger.info(f"Processing: {file}")
                
                # Convert file to node format
                node = self._file_to_node(file)
                
                # Process through pipeline
                result = self.pipeline.run([node])
                
                processing_time = time.time() - start_time
                
                checkpoint = ProcessingCheckpoint(
                    file_path=str(file),
                    status='completed',
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    processing_time_seconds=processing_time,
                    relationships_created=result.get('edges_created', 0)
                )
                
                logger.info(f"✅ Completed: {file} in {processing_time:.1f}s")
                
            except Exception as e:
                processing_time = time.time() - start_time
                error_msg = f"{type(e).__name__}: {str(e)}"
                
                checkpoint = ProcessingCheckpoint(
                    file_path=str(file),
                    status='failed',
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    error_message=error_msg,
                    processing_time_seconds=processing_time
                )
                
                logger.error(f"❌ Failed: {file} - {error_msg}")
                logger.debug(traceback.format_exc())
            
            checkpoints.append(checkpoint)
            self.checkpoint_manager.save_checkpoint(checkpoint)
            
        return checkpoints
    
    def _file_to_node(self, file_path: Path) -> Dict[str, Any]:
        """Convert a file to a node for processing."""
        # This is simplified - real implementation would parse content
        return {
            "node_id": f"file_{file_path.stem}",
            "node_type": "document",
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_type": file_path.suffix,
            "content": f"Content of {file_path.name}",  # Would read actual content
            "modified_at": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            "size_bytes": file_path.stat().st_size,
            "embedding": [0.1] * 384  # Would generate real embedding
        }
    
    def run(self, data_dir: Path, test_mode: bool = False):
        """Run the checkpointed pipeline."""
        logger.info("=" * 60)
        logger.info("CHECKPOINTED PIPELINE STARTING")
        logger.info("=" * 60)
        
        # Collect all files
        all_files = self.collect_files(data_dir)
        logger.info(f"Found {len(all_files)} total files")
        
        # Get remaining files
        remaining_files = self.checkpoint_manager.get_remaining_files(all_files)
        logger.info(f"Remaining to process: {len(remaining_files)} files")
        
        if test_mode:
            logger.info("TEST MODE: Processing only 10 files")
            remaining_files = remaining_files[:10]
        
        # Show statistics
        stats = self.checkpoint_manager.get_statistics()
        logger.info(f"Previous stats: {stats}")
        
        # Categorize files
        file_categories = self.categorize_files(remaining_files)
        
        # Process by category
        for ext, files in file_categories.items():
            batch_size = self.batch_sizes.get(ext, 10)
            logger.info(f"\nProcessing {len(files)} {ext} files in batches of {batch_size}")
            
            for i in range(0, len(files), batch_size):
                # Check scheduling
                if self.scheduler.should_pause():
                    wait_time = self.scheduler.time_until_next_window()
                    logger.info(f"⏸️  Pausing during peak hours. Resuming in {wait_time/3600:.1f} hours")
                    time.sleep(wait_time)
                
                batch = files[i:i + batch_size]
                logger.info(f"\nProcessing batch {i//batch_size + 1}: {len(batch)} files")
                
                self.process_batch(batch)
                
                # Show progress
                current_stats = self.checkpoint_manager.get_statistics()
                completion_percent = (current_stats['completed'] / len(all_files)) * 100
                logger.info(f"Progress: {completion_percent:.1f}% complete")
                
                # Small delay between batches
                time.sleep(2)
        
        # Final statistics
        final_stats = self.checkpoint_manager.get_statistics()
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Total processed: {final_stats['total_processed']}")
        logger.info(f"Successful: {final_stats['completed']}")
        logger.info(f"Failed: {final_stats['failed']}")
        logger.info(f"Total time: {final_stats['total_time_seconds']/3600:.1f} hours")
        logger.info(f"Total relationships: {final_stats['total_relationships']:,}")


def create_config() -> Dict[str, Any]:
    """Create configuration for the pipeline."""
    return {
        "database": {
            "url": "http://localhost:8529",
            "username": "root",
            "password": "",
            "database": "sequential_isne_graph"
        },
        "bootstrap": {
            "batch_size": 100,
            "max_edges_per_node": 20,
            "min_edge_weight": 0.4,
            "file_extensions": [".py", ".md", ".pdf", ".txt", ".json"]
        },
        "supra_weight": {
            "aggregation_method": "adaptive",
            "semantic_threshold": 0.5,
            "importance_weights": {
                "semantic": 0.8,
                "structural": 0.7,
                "co_location": 0.6,
                "temporal": 0.5
            }
        },
        "density": {
            "target_density": 0.05,
            "local_density_factor": 1.5
        }
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process large dataset with checkpointing and scheduling"
    )
    parser.add_argument("--data-dir", type=Path, required=True,
                       help="Directory containing data to process")
    parser.add_argument("--checkpoint-dir", type=Path, default="./checkpoints",
                       help="Directory for checkpoints")
    parser.add_argument("--test", action="store_true",
                       help="Test mode - process only 10 files")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    try:
        config = create_config()
        pipeline = CheckpointedPipeline(config, args.checkpoint_dir)
        
        pipeline.run(args.data_dir, test_mode=args.test)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.debug(traceback.format_exc())
        raise