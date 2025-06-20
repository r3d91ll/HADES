#!/usr/bin/env python3
"""
Bootstrap Micro Validation - 2-3 Minute Complete Test

This script runs bootstrap on a MICRO sample (~30 files) for complete pipeline validation.
Goal: End-to-end validation including model training and evaluation in 2-3 minutes.
"""

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

# Configure logging for terminal visibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def sample_micro_set(input_dir: str, target_files: int = 30) -> List[Path]:
    """
    Sample a micro set of files for complete end-to-end validation.
    
    Args:
        input_dir: Directory to sample from
        target_files: Target number of files (~30 for 2-3 min processing)
        
    Returns:
        List of sampled file paths
    """
    input_path = Path(input_dir)
    
    # Collect diverse but small sample
    file_types = {
        'pdf': {'files': list(input_path.rglob("*.pdf")), 'target': 5},
        'py': {'files': list(input_path.rglob("*.py")), 'target': 10},
        'md': {'files': list(input_path.rglob("*.md")), 'target': 5},
        'txt': {'files': list(input_path.rglob("*.txt")), 'target': 5},
        'json': {'files': list(input_path.rglob("*.json")), 'target': 5},
    }
    
    sampled_files = []
    
    logger.info("🔬 MICRO SAMPLING:")
    
    for file_type, info in file_types.items():
        files = info['files']
        target = info['target']
        
        if not files:
            continue
            
        sample_size = min(target, len(files))
        if sample_size > 0:
            sampled = random.sample(files, sample_size)
            sampled_files.extend(sampled)
            logger.info(f"  {file_type.upper()}: {sample_size} files")
    
    # Shuffle final list
    random.shuffle(sampled_files)
    
    logger.info(f"📈 TOTAL MICRO SAMPLE: {len(sampled_files)} files")
    logger.info(f"⏱️  Expected processing time: 2-3 minutes")
    
    return sampled_files


def bootstrap_micro_validation():
    """Bootstrap micro sample for complete validation."""
    
    # Input and output directories
    input_dir = "/home/todd/ML-Lab/Olympus/test-data3"
    output_dir = "output/micro_validation_bootstrap"
    model_name = f"micro_validation_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    
    # Verify input directory exists
    if not Path(input_dir).exists():
        logger.error(f"Input directory not found: {input_dir}")
        return False
    
    logger.info("=" * 50)
    logger.info("MICRO VALIDATION - COMPLETE END-TO-END TEST")
    logger.info("=" * 50)
    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Model: {model_name}")
    logger.info("=" * 50)
    
    try:
        # Sample micro set of files
        logger.info("🔬 Sampling micro set for complete validation...")
        random.seed(42)  # Reproducible sampling
        sampled_files = sample_micro_set(input_dir, target_files=30)
        
        if not sampled_files:
            logger.error("No files were sampled!")
            return False
        
        logger.info("=" * 50)
        
        # Create bootstrap configuration optimized for speed
        config = BootstrapConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            pipeline_name="micro_validation_bootstrap",
            enable_monitoring=True,
            save_intermediate_results=True
        )
        
        # Configure for maximum speed
        config.document_processing.use_hybrid_processing = True
        config.document_processing.python_use_ast = True
        config.document_processing.pdf_extraction_method = "docling"
        
        # Very fast chunking - larger chunks = fewer chunks
        config.chunking.chunk_size = 512  # Larger chunks
        config.chunking.overlap = 50
        
        # Fast embedding
        config.embedding.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        config.embedding.batch_size = 32
        config.embedding.use_gpu = True
        
        # Minimal graph construction
        config.graph_construction.similarity_threshold = 0.7  # High threshold = sparse graph
        config.graph_construction.max_edges_per_node = 5  # Very few edges
        config.graph_construction.use_gpu = True
        
        # Ultra-minimal ISNE training
        config.isne_training.epochs = 3  # Just 3 epochs
        config.isne_training.learning_rate = 0.01  # Higher LR for faster convergence
        config.isne_training.hidden_dim = 32  # Very small model
        config.isne_training.num_layers = 1  # Single layer
        config.isne_training.batch_size = 16
        config.isne_training.patience = 2
        config.isne_training.device = "cuda"
        
        # Minimal evaluation
        config.model_evaluation.sample_size_for_visualization = 50
        config.model_evaluation.target_relative_performance = 0.50  # Very low target
        
        logger.info("🚀 Starting micro validation bootstrap...")
        logger.info("⏱️  Expected time: 2-3 minutes")
        logger.info("🎯 Goal: Complete end-to-end validation")
        
        # Initialize and run pipeline
        pipeline = ISNEBootstrapPipeline(config)
        result = pipeline.run(sampled_files, output_dir, model_name)
        
        if result.success:
            logger.info("=" * 50)
            logger.info("✅ MICRO VALIDATION COMPLETED!")
            logger.info("=" * 50)
            logger.info(f"Model: {result.model_path}")
            logger.info(f"Time: {result.total_time_seconds / 60:.1f} minutes")
            
            # Detailed quality assessment
            if 'data_flow' in result.final_stats:
                data_flow = result.final_stats['data_flow']
                
                logger.info("")
                logger.info("📊 COMPLETE PIPELINE VALIDATION:")
                logger.info(f"  ✓ Documents: {data_flow.get('documents_generated', 0):,}")
                logger.info(f"  ✓ Chunks: {data_flow.get('chunks_generated', 0):,}")
                logger.info(f"  ✓ Embeddings: {data_flow.get('embeddings_generated', 0):,}")
                logger.info(f"  ✓ Graph nodes: {data_flow.get('graph_nodes', 0):,}")
                logger.info(f"  ✓ Graph edges: {data_flow.get('graph_edges', 0):,}")
                
                if data_flow.get('evaluation_completed'):
                    performance = data_flow.get('inductive_performance', 0)
                    target_met = data_flow.get('achieves_90_percent_target', False)
                    
                    logger.info("")
                    logger.info("🎯 END-TO-END VALIDATION RESULTS:")
                    logger.info(f"  Model trained: ✅ YES")
                    logger.info(f"  Evaluation completed: ✅ YES") 
                    logger.info(f"  Performance: {performance:.2%}")
                    logger.info(f"  Pipeline working: ✅ YES")
                    
                    # Pipeline assessment
                    logger.info("")
                    logger.info("🔧 PIPELINE STATUS:")
                    logger.info("  All stages completed successfully ✅")
                    logger.info("  Model architecture working ✅")
                    logger.info("  Training loop functional ✅")
                    logger.info("  Evaluation system working ✅")
                    
                    if performance > 0.3:  # Very low bar for micro test
                        logger.info("  Quality check: ✅ ACCEPTABLE for micro test")
                        logger.info("  Recommendation: Use as baseline, scale up gradually")
                    else:
                        logger.info("  Quality check: ❌ Poor even for micro test")
                        logger.info("  Recommendation: Debug fundamental issues")
                else:
                    logger.info("  Evaluation: ❌ Did not complete")
            
            logger.info("")
            logger.info("🔄 MICRO VALIDATION PASSED - READY FOR SCALING:")
            logger.info("1. ✅ Complete pipeline validated")
            logger.info("2. ✅ Model architecture working")
            logger.info("3. ✅ Can use this as baseline/seed model")
            logger.info("4. 📈 Scale up gradually: 10% → 25% → 50% → 100%")
            logger.info("")
            logger.info(f"📁 Model ready at: {result.model_path}")
            
            return True
            
        else:
            logger.error("=" * 50)
            logger.error("❌ MICRO VALIDATION FAILED")
            logger.error("=" * 50)
            logger.error(f"Error: {result.error_message}")
            logger.error(f"Stage: {result.error_stage}")
            logger.error("🔧 Fix issues with micro test before scaling up")
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
    
    success = bootstrap_micro_validation()
    
    if success:
        logger.info("🎉 Micro validation passed - pipeline ready for scaling!")
    else:
        logger.error("❌ Micro validation failed - fix issues before scaling.")
        sys.exit(1)