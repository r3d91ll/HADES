#!/usr/bin/env python3
"""
Bootstrap 10% Sample - Quick Validation Pipeline

This script runs the bootstrap pipeline on a 10% sample of the full dataset,
providing quick validation before committing to full dataset training.

Workflow:
1. Sample 10% of files from test-data3 (stratified by file type)
2. Run bootstrap pipeline on sample
3. Evaluate model performance 
4. Present results for user inspection
5. If good, user can incrementally scale up training
"""

import asyncio
import logging
import sys
import random
from pathlib import Path
from datetime import datetime, timezone
from typing import List

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.isne.bootstrap.pipeline import ISNEBootstrapPipeline
from src.isne.bootstrap.config import BootstrapConfig

# Configure logging
log_dir = Path("logs/bootstrap")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"10percent_sample_bootstrap_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def stratified_sample_files(input_dir: str, sample_rate: float = 0.1) -> List[Path]:
    """
    Create a stratified sample of files by type.
    
    Args:
        input_dir: Directory to sample from
        sample_rate: Fraction of files to sample (0.1 = 10%)
        
    Returns:
        List of sampled file paths
    """
    input_path = Path(input_dir)
    
    # Collect files by type
    file_types = {
        'pdf': list(input_path.rglob("*.pdf")),
        'py': list(input_path.rglob("*.py")),
        'md': list(input_path.rglob("*.md")),
        'txt': list(input_path.rglob("*.txt")),
        'json': list(input_path.rglob("*.json")),
        'html': list(input_path.rglob("*.html")),
        'yaml': list(input_path.rglob("*.yaml")),
        'yml': list(input_path.rglob("*.yml"))
    }
    
    sampled_files = []
    
    logger.info("📊 STRATIFIED SAMPLING:")
    for file_type, files in file_types.items():
        if not files:
            continue
            
        # Sample from each type
        sample_size = max(1, int(len(files) * sample_rate))
        sampled = random.sample(files, min(sample_size, len(files)))
        sampled_files.extend(sampled)
        
        logger.info(f"  {file_type.upper()}: {len(sampled):,} / {len(files):,} files ({len(sampled)/len(files)*100:.1f}%)")
    
    # Shuffle the final list
    random.shuffle(sampled_files)
    
    logger.info(f"📈 TOTAL SAMPLE: {len(sampled_files):,} files ({len(sampled_files)/sum(len(files) for files in file_types.values())*100:.1f}% of dataset)")
    
    return sampled_files


def bootstrap_10percent_sample():
    """Bootstrap a 10% sample for quick validation."""
    
    # Input and output directories
    input_dir = "/home/todd/ML-Lab/Olympus/test-data3"
    output_dir = "output/10percent_sample_bootstrap"
    model_name = f"sample_10pct_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    
    # Verify input directory exists
    if not Path(input_dir).exists():
        logger.error(f"Input directory not found: {input_dir}")
        return False
    
    logger.info("=" * 80)
    logger.info("10% SAMPLE BOOTSTRAP - QUICK VALIDATION")
    logger.info("=" * 80)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model name: {model_name}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 80)
    
    try:
        # Sample 10% of files
        logger.info("🎯 Sampling 10% of files for quick validation...")
        random.seed(42)  # Reproducible sampling
        sampled_files = stratified_sample_files(input_dir, sample_rate=0.1)
        
        if not sampled_files:
            logger.error("No files were sampled!")
            return False
        
        logger.info("=" * 80)
        
        # Create bootstrap configuration
        config = BootstrapConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            pipeline_name="10percent_sample_bootstrap",
            enable_monitoring=True,
            save_intermediate_results=True
        )
        
        # Enable W&B for tracking
        config.wandb.enabled = True
        config.wandb.project = "hades-10pct-validation"
        config.wandb.tags = ["10percent", "validation", "quick-test"]
        config.wandb.notes = f"Quick validation on {len(sampled_files)} sampled files"
        
        # Configure for hybrid processing
        config.document_processing.use_hybrid_processing = True
        config.document_processing.python_use_ast = True
        config.document_processing.pdf_extraction_method = "docling"
        
        # Optimize for quick processing
        config.chunking.chunk_size = 512  # Smaller chunks for faster processing
        config.chunking.overlap = 50
        
        # Configure embedding for speed
        config.embedding.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        config.embedding.batch_size = 32
        config.embedding.use_gpu = True
        
        # Configure graph construction
        config.graph_construction.similarity_threshold = 0.5  # Higher threshold for smaller dataset
        config.graph_construction.max_edges_per_node = 20
        config.graph_construction.use_gpu = True
        
        # Configure ISNE training for quick validation
        config.isne_training.epochs = 20  # Fewer epochs for quick test
        config.isne_training.learning_rate = 0.001
        config.isne_training.hidden_dim = 128
        config.isne_training.num_layers = 2
        config.isne_training.batch_size = 64
        config.isne_training.patience = 10
        config.isne_training.device = "cuda"
        
        # Configure evaluation
        config.model_evaluation.sample_size_for_visualization = 1000
        config.model_evaluation.target_relative_performance = 0.80  # Lower target for quick test
        
        logger.info("🚀 Starting 10% sample bootstrap pipeline...")
        logger.info("⏱️  Expected time: 30-60 minutes")
        logger.info("💾 Results will be saved to: {output_dir}")
        logger.info("📊 W&B tracking: https://wandb.ai (project: hades-10pct-validation)")
        
        # Initialize and run pipeline
        pipeline = ISNEBootstrapPipeline(config)
        result = pipeline.run(sampled_files, output_dir, model_name)
        
        if result.success:
            logger.info("=" * 80)
            logger.info("✅ 10% SAMPLE BOOTSTRAP COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            logger.info(f"Model saved to: {result.model_path}")
            logger.info(f"Output directory: {result.output_directory}")
            logger.info(f"Total time: {result.total_time_seconds / 60:.1f} minutes")
            
            # Log validation results
            if 'data_flow' in result.final_stats:
                data_flow = result.final_stats['data_flow']
                logger.info("")
                logger.info("📊 SAMPLE VALIDATION RESULTS:")
                logger.info(f"  Documents processed: {data_flow.get('documents_generated', 0):,}")
                logger.info(f"  Chunks generated: {data_flow.get('chunks_generated', 0):,}")
                logger.info(f"  Embeddings created: {data_flow.get('embeddings_generated', 0):,}")
                logger.info(f"  Graph nodes: {data_flow.get('graph_nodes', 0):,}")
                logger.info(f"  Graph edges: {data_flow.get('graph_edges', 0):,}")
                
                if data_flow.get('evaluation_completed'):
                    logger.info("")
                    logger.info("🎯 MODEL PERFORMANCE:")
                    logger.info(f"  Inductive performance: {data_flow.get('inductive_performance', 0):.2%}")
                    target_met = data_flow.get('achieves_90_percent_target', False)
                    logger.info(f"  Target achieved: {'✅' if target_met else '❌'}")
            
            logger.info("")
            logger.info("🔄 NEXT STEPS:")
            logger.info("1. ✅ Review the model performance above")
            logger.info("2. 🔍 Inspect the model outputs in the output directory") 
            logger.info("3. 📈 If satisfied, run incremental training on more data")
            logger.info("4. 🔄 If not satisfied, debug with this small dataset first")
            logger.info("")
            logger.info("💡 TO SCALE UP: Use the training script with more data once validated")
            
            return True
            
        else:
            logger.error("=" * 80)
            logger.error("❌ 10% SAMPLE BOOTSTRAP FAILED")
            logger.error("=" * 80)
            logger.error(f"Error: {result.error_message}")
            logger.error(f"Failed at stage: {result.error_stage}")
            logger.error("")
            logger.error("🔧 DEBUG RECOMMENDATIONS:")
            logger.error("1. Check the log file for detailed errors")
            logger.error("2. Fix issues with small dataset before scaling up") 
            logger.error("3. Re-run this 10% validation once fixed")
            return False
            
    except KeyboardInterrupt:
        logger.warning("⚠️  Bootstrap interrupted by user!")
        logger.info("The pipeline state has been saved and can be resumed.")
        return False
        
    except Exception as e:
        logger.error(f"❌ Bootstrap failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import signal
    
    def signal_handler(signum, frame):
        logger.warning(f"Received signal {signum}. Attempting graceful shutdown...")
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    success = bootstrap_10percent_sample()
    
    if success:
        logger.info("🎉 10% sample validation completed successfully!")
        logger.info("Ready for user inspection and potential scale-up.")
    else:
        logger.error("❌ 10% sample validation did not complete successfully.")
        sys.exit(1)