#!/usr/bin/env python3
"""
Bootstrap Tiny Validation - 5-10 Minute Quality Check

This script runs bootstrap on a TINY sample (~100-200 files) for actual quick validation.
The goal is to verify model QUALITY, not just pipeline stability.

Workflow:
1. Sample ~200 diverse files (5-10 minutes to process)
2. Run full bootstrap pipeline
3. Evaluate model quality metrics
4. Present quality assessment for human inspection
5. If quality looks good → scale up gradually
"""

import logging
import sys
import random
from pathlib import Path
from datetime import datetime
from typing import List

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.isne.bootstrap.pipeline import ISNEBootstrapPipeline
from src.isne.bootstrap.config import BootstrapConfig

# Configure logging for terminal visibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def sample_tiny_diverse_set(input_dir: str, target_files: int = 200) -> List[Path]:
    """
    Sample a tiny but diverse set of files for quality validation.
    
    Args:
        input_dir: Directory to sample from
        target_files: Target number of files (~200 for 5-10 min processing)
        
    Returns:
        List of sampled file paths
    """
    input_path = Path(input_dir)
    
    # Collect files by type with explicit priorities for diversity
    file_types = {
        'pdf': {'files': list(input_path.rglob("*.pdf")), 'priority': 10, 'min_files': 20},
        'py': {'files': list(input_path.rglob("*.py")), 'priority': 8, 'min_files': 50},
        'md': {'files': list(input_path.rglob("*.md")), 'priority': 6, 'min_files': 10},
        'txt': {'files': list(input_path.rglob("*.txt")), 'priority': 4, 'min_files': 10},
        'json': {'files': list(input_path.rglob("*.json")), 'priority': 2, 'min_files': 20},
    }
    
    sampled_files = []
    remaining_quota = target_files
    
    logger.info("🎯 TINY DIVERSE SAMPLING:")
    
    # First pass: ensure minimum representation of each type
    for file_type, info in file_types.items():
        files = info['files']
        min_files = info['min_files']
        
        if not files:
            continue
            
        # Sample minimum required
        sample_size = min(min_files, len(files), remaining_quota)
        if sample_size > 0:
            sampled = random.sample(files, sample_size)
            sampled_files.extend(sampled)
            remaining_quota -= sample_size
            
            logger.info(f"  {file_type.upper()}: {sample_size} files (min diversity)")
    
    # Second pass: fill remaining quota based on priority
    file_types_sorted = sorted(file_types.items(), key=lambda x: x[1]['priority'], reverse=True)
    
    for file_type, info in file_types_sorted:
        if remaining_quota <= 0:
            break
            
        files = info['files']
        already_sampled = [f for f in sampled_files if f.suffix.lower() == f".{file_type}"]
        available = [f for f in files if f not in already_sampled]
        
        if available and remaining_quota > 0:
            additional_sample_size = min(remaining_quota // 2, len(available))
            if additional_sample_size > 0:
                additional = random.sample(available, additional_sample_size)
                sampled_files.extend(additional)
                remaining_quota -= additional_sample_size
                
                logger.info(f"  {file_type.upper()}: +{additional_sample_size} files (priority fill)")
    
    # Shuffle final list
    random.shuffle(sampled_files)
    
    logger.info(f"📈 TOTAL TINY SAMPLE: {len(sampled_files)} files")
    logger.info(f"⏱️  Expected processing time: 5-10 minutes")
    
    return sampled_files


def bootstrap_tiny_validation():
    """Bootstrap tiny sample for quality validation."""
    
    # Input and output directories
    input_dir = "/home/todd/ML-Lab/Olympus/test-data3"
    output_dir = "output/tiny_validation_bootstrap"
    model_name = f"tiny_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Verify input directory exists
    if not Path(input_dir).exists():
        logger.error(f"Input directory not found: {input_dir}")
        return False
    
    logger.info("=" * 60)
    logger.info("TINY VALIDATION BOOTSTRAP - QUALITY CHECK")
    logger.info("=" * 60)
    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Model: {model_name}")
    logger.info("=" * 60)
    
    try:
        # Sample tiny set of files for quality validation
        logger.info("🎯 Sampling tiny diverse set for quality validation...")
        random.seed(42)  # Reproducible sampling
        sampled_files = sample_tiny_diverse_set(input_dir, target_files=200)
        
        if not sampled_files:
            logger.error("No files were sampled!")
            return False
        
        logger.info("=" * 60)
        
        # Create bootstrap configuration optimized for speed
        config = BootstrapConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            pipeline_name="tiny_validation_bootstrap",
            enable_monitoring=True,
            save_intermediate_results=True
        )
        
        # Configure for maximum speed while maintaining quality assessment
        config.document_processing.use_hybrid_processing = True
        config.document_processing.python_use_ast = True
        config.document_processing.pdf_extraction_method = "docling"
        
        # Fast chunking
        config.chunking.chunk_size = 256  # Smaller chunks for speed
        config.chunking.overlap = 25
        
        # Fast embedding
        config.embedding.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        config.embedding.batch_size = 16  # Smaller batch for memory
        config.embedding.use_gpu = True
        
        # Fast graph construction
        config.graph_construction.similarity_threshold = 0.6  # Higher threshold = fewer edges = faster
        config.graph_construction.max_edges_per_node = 10  # Fewer edges for speed
        config.graph_construction.use_gpu = True
        
        # Minimal ISNE training just to validate architecture
        config.isne_training.epochs = 5  # Just enough to see if it learns
        config.isne_training.learning_rate = 0.001
        config.isne_training.hidden_dim = 64  # Smaller model for speed
        config.isne_training.num_layers = 2
        config.isne_training.batch_size = 32
        config.isne_training.patience = 3
        config.isne_training.device = "cuda"
        
        # Minimal evaluation
        config.model_evaluation.sample_size_for_visualization = 100
        config.model_evaluation.target_relative_performance = 0.70  # Lower target for tiny test
        
        logger.info("🚀 Starting tiny validation bootstrap...")
        logger.info("⏱️  Expected time: 5-10 minutes")
        logger.info("🎯 Goal: Validate model QUALITY, not just pipeline stability")
        
        # Initialize and run pipeline
        pipeline = ISNEBootstrapPipeline(config)
        result = pipeline.run(sampled_files, output_dir, model_name)
        
        if result.success:
            logger.info("=" * 60)
            logger.info("✅ TINY VALIDATION COMPLETED!")
            logger.info("=" * 60)
            logger.info(f"Model: {result.model_path}")
            logger.info(f"Time: {result.total_time_seconds / 60:.1f} minutes")
            
            # Detailed quality assessment
            if 'data_flow' in result.final_stats:
                data_flow = result.final_stats['data_flow']
                
                logger.info("")
                logger.info("📊 PROCESSING PIPELINE:")
                logger.info(f"  ✓ Documents: {data_flow.get('documents_generated', 0):,}")
                logger.info(f"  ✓ Chunks: {data_flow.get('chunks_generated', 0):,}")
                logger.info(f"  ✓ Embeddings: {data_flow.get('embeddings_generated', 0):,}")
                logger.info(f"  ✓ Graph nodes: {data_flow.get('graph_nodes', 0):,}")
                logger.info(f"  ✓ Graph edges: {data_flow.get('graph_edges', 0):,}")
                
                if data_flow.get('evaluation_completed'):
                    performance = data_flow.get('inductive_performance', 0)
                    target_met = data_flow.get('achieves_90_percent_target', False)
                    
                    logger.info("")
                    logger.info("🎯 MODEL QUALITY ASSESSMENT:")
                    logger.info(f"  Performance: {performance:.2%}")
                    logger.info(f"  Target met: {'✅ YES' if target_met else '❌ NO'}")
                    
                    # Quality assessment
                    if performance > 0.8:
                        quality = "🟢 EXCELLENT"
                        recommendation = "Scale up to 10% sample"
                    elif performance > 0.6:
                        quality = "🟡 GOOD"
                        recommendation = "Scale up carefully, monitor quality"
                    elif performance > 0.4:
                        quality = "🟠 POOR"
                        recommendation = "Debug issues before scaling"
                    else:
                        quality = "🔴 FAILED"
                        recommendation = "Fix fundamental issues"
                    
                    logger.info(f"  Quality: {quality}")
                    logger.info(f"  Recommendation: {recommendation}")
            
            logger.info("")
            logger.info("🔄 NEXT STEPS:")
            logger.info("1. 🔍 Inspect output files for quality")
            logger.info("2. 📊 Check embedding similarity patterns")
            logger.info("3. 🎯 If quality good → scale up gradually")
            logger.info("4. 🔧 If quality poor → debug on tiny set first")
            logger.info("")
            logger.info(f"📁 Output directory: {result.output_directory}")
            
            return True
            
        else:
            logger.error("=" * 60)
            logger.error("❌ TINY VALIDATION FAILED")
            logger.error("=" * 60)
            logger.error(f"Error: {result.error_message}")
            logger.error(f"Stage: {result.error_stage}")
            return False
            
    except KeyboardInterrupt:
        logger.warning("⚠️  Validation interrupted by user!")
        return False
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import signal
    
    def signal_handler(signum, frame):
        logger.warning(f"Received signal {signum}. Shutting down...")
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    success = bootstrap_tiny_validation()
    
    if success:
        logger.info("🎉 Tiny validation completed - ready for quality inspection!")
    else:
        logger.error("❌ Tiny validation failed - check issues before scaling.")
        sys.exit(1)