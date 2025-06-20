#!/usr/bin/env python3
"""
Bootstrap Full Dataset - Complete ISNE Training Pipeline

This script runs the complete bootstrap pipeline on the full test-data3 dataset,
including both PDFs and Python code directories. This creates a production-ready
ISNE model trained on a diverse corpus of research papers and code.
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.isne.bootstrap.pipeline import ISNEBootstrapPipeline
from src.isne.bootstrap.config import BootstrapConfig

# Configure logging
log_dir = Path("logs/bootstrap")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"full_dataset_bootstrap_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def bootstrap_full_dataset():
    """Bootstrap the complete dataset with both PDFs and Python code."""
    
    # Input and output directories
    input_dir = "/home/todd/ML-Lab/Olympus/test-data3"
    output_dir = "output/full_dataset_bootstrap"
    model_name = f"full_dataset_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    
    # Verify input directory exists
    if not Path(input_dir).exists():
        logger.error(f"Input directory not found: {input_dir}")
        return False
    
    # Count files to process
    total_files = 0
    pdf_files = list(Path(input_dir).rglob("*.pdf"))
    py_files = list(Path(input_dir).rglob("*.py"))
    md_files = list(Path(input_dir).rglob("*.md"))
    txt_files = list(Path(input_dir).rglob("*.txt"))
    json_files = list(Path(input_dir).rglob("*.json"))
    
    total_files = len(pdf_files) + len(py_files) + len(md_files) + len(txt_files) + len(json_files)
    
    logger.info("=" * 80)
    logger.info("FULL DATASET BOOTSTRAP")
    logger.info("=" * 80)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model name: {model_name}")
    logger.info(f"Log file: {log_file}")
    logger.info("")
    logger.info("File types to process:")
    logger.info(f"  📄 PDF files: {len(pdf_files)}")
    logger.info(f"  🐍 Python files: {len(py_files)}")
    logger.info(f"  📝 Markdown files: {len(md_files)}")
    logger.info(f"  📄 Text files: {len(txt_files)}")
    logger.info(f"  📊 JSON files: {len(json_files)}")
    logger.info(f"  📊 Total files: {total_files}")
    logger.info("=" * 80)
    
    # Create bootstrap configuration
    config = BootstrapConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        pipeline_name="full_dataset_bootstrap",
        enable_monitoring=True,
        save_intermediate_results=True
    )
    
    # Enable W&B for this important run
    config.wandb.enabled = True
    config.wandb.project = "hades-full-dataset"
    config.wandb.tags = ["full-dataset", "production", "hybrid-processing"]
    config.wandb.notes = f"Complete bootstrap on {total_files} files from test-data3"
    
    # Configure for hybrid processing (AST for Python, text extraction for PDFs)
    config.document_processing.use_hybrid_processing = True
    config.document_processing.python_use_ast = True
    config.document_processing.pdf_extraction_method = "docling"
    
    # Optimize chunking for diverse content
    config.chunking.chunk_size = 1024  # Larger chunks for academic content
    config.chunking.overlap = 100
    config.chunking.code_chunk_size = 512  # Smaller chunks for code
    
    # Configure embedding for production
    config.embedding.model_name = "sentence-transformers/all-MiniLM-L6-v2"
    config.embedding.batch_size = 64
    config.embedding.use_gpu = True
    
    # Configure graph construction for large scale
    config.graph_construction.similarity_threshold = 0.4  # Lower threshold for diverse content
    config.graph_construction.max_edges_per_node = 50
    config.graph_construction.use_gpu = True
    
    # Configure ISNE training for production
    config.isne_training.epochs = 100  # More epochs for production
    config.isne_training.learning_rate = 0.001
    config.isne_training.hidden_dim = 256  # Larger model
    config.isne_training.num_layers = 4
    config.isne_training.batch_size = 128
    config.isne_training.patience = 20
    config.isne_training.device = "cuda"
    
    # Configure evaluation
    config.model_evaluation.sample_size_for_visualization = 2000
    config.model_evaluation.target_relative_performance = 0.85  # High target
    
    # Adjust memory alert thresholds for large dataset (40k+ documents)
    config.monitoring.alert_thresholds = {
        "max_memory_mb": 20000,  # 20 GB threshold (vs default 4 GB)
        "memory_growth_threshold_mb": 5000  # 5 GB growth threshold (vs default 500 MB)
    }
    
    try:
        logger.info("🚀 Starting full dataset bootstrap pipeline...")
        logger.info(f"⚠️  This may take 6-12 hours depending on hardware!")
        logger.info(f"💾 Intermediate results will be saved to: {output_dir}")
        logger.info(f"📊 W&B tracking: https://wandb.ai (project: hades-full-dataset)")
        
        # Get all files to process
        input_path = Path(input_dir)
        input_files = []
        
        # Collect all supported file types
        supported_extensions = {'.pdf', '.py', '.md', '.txt', '.json', '.html', '.yaml', '.yml'}
        for file_path in input_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                input_files.append(file_path)
        
        logger.info(f"Collected {len(input_files)} files for processing")
        
        # Initialize and run pipeline
        pipeline = ISNEBootstrapPipeline(config)
        result = pipeline.run(input_files, output_dir, model_name)
        
        if result.success:
            logger.info("=" * 80)
            logger.info("✅ FULL DATASET BOOTSTRAP COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            logger.info(f"Model saved to: {result.model_path}")
            logger.info(f"Output directory: {result.output_directory}")
            logger.info(f"Total time: {result.total_time_seconds / 3600:.2f} hours")
            logger.info(f"Final stats: {result.final_stats}")
            
            # Log dataset scale
            if 'data_flow' in result.final_stats:
                data_flow = result.final_stats['data_flow']
                logger.info("")
                logger.info("📊 DATASET SCALE:")
                logger.info(f"  Documents processed: {data_flow.get('documents_generated', 0):,}")
                logger.info(f"  Chunks generated: {data_flow.get('chunks_generated', 0):,}")
                logger.info(f"  Embeddings created: {data_flow.get('embeddings_generated', 0):,}")
                logger.info(f"  Graph nodes: {data_flow.get('graph_nodes', 0):,}")
                logger.info(f"  Graph edges: {data_flow.get('graph_edges', 0):,}")
                
                if data_flow.get('evaluation_completed'):
                    logger.info("")
                    logger.info("🎯 MODEL PERFORMANCE:")
                    logger.info(f"  Inductive performance: {data_flow.get('inductive_performance', 0):.2%}")
                    logger.info(f"  Target achieved: {'✅' if data_flow.get('achieves_90_percent_target') else '❌'}")
            
            logger.info("")
            logger.info("🔄 NEXT STEPS:")
            logger.info("1. ✅ Full ISNE model ready for production")
            logger.info("2. 🔄 Implement ArangoDB incremental updates")
            logger.info("3. 🚀 Deploy production RAG system")
            
            return True
            
        else:
            logger.error("=" * 80)
            logger.error("❌ BOOTSTRAP FAILED")
            logger.error("=" * 80)
            logger.error(f"Error: {result.error_message}")
            logger.error(f"Failed at stage: {result.error_stage}")
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
    
    success = bootstrap_full_dataset()
    
    if success:
        logger.info("🎉 Full dataset bootstrap completed successfully!")
        logger.info("The production ISNE model is ready for deployment.")
    else:
        logger.error("❌ Bootstrap did not complete successfully.")
        sys.exit(1)