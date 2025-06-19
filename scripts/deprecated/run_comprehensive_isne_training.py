#!/usr/bin/env python3
"""
Comprehensive ISNE Training Script
Trains ISNE model using all available data:
- 382 PDF documents from test-data3/
- 17 Python files from ladon/
- 22,847 Python files from HADES/
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Configuration
OLYMPUS_ROOT = Path("/home/todd/ML-Lab/Olympus")
HADES_ROOT = OLYMPUS_ROOT / "HADES"
TEST_DATA_DIR = OLYMPUS_ROOT / "test-data3"
LADON_DIR = OLYMPUS_ROOT / "ladon"
OUTPUT_DIR = HADES_ROOT / "output" / "isne_comprehensive_training"
LOGS_DIR = Path(os.environ.get("ML_LOGS_DIR", "/home/todd/ML-Lab/logs"))

# ISNE pipeline script path - using the monitored complete pipeline
ISNE_BOOTSTRAP_SCRIPT = HADES_ROOT / "scripts" / "monitored_complete_isne_bootstrap_pipeline.py"

def collect_data_sources() -> Dict[str, List[str]]:
    """Collect all data sources for comprehensive training."""
    logger.info("Collecting data sources...")
    
    sources = {
        "pdfs": [],
        "python_files": []
    }
    
    # Collect PDFs from test-data3
    pdf_files = list(TEST_DATA_DIR.glob("*.pdf"))
    sources["pdfs"] = [str(f) for f in pdf_files]
    logger.info(f"Found {len(sources['pdfs'])} PDF documents")
    
    # Collect Python files from ladon
    ladon_py_files = list(LADON_DIR.rglob("*.py"))
    sources["python_files"].extend([str(f) for f in ladon_py_files])
    logger.info(f"Found {len(ladon_py_files)} Python files in ladon")
    
    # Collect Python files from HADES
    hades_py_files = list(HADES_ROOT.rglob("*.py"))
    # Filter out __pycache__ and other generated files
    hades_py_files = [
        f for f in hades_py_files 
        if "__pycache__" not in str(f) and 
        ".pyc" not in str(f) and
        "output/" not in str(f)
    ]
    sources["python_files"].extend([str(f) for f in hades_py_files])
    logger.info(f"Found {len(hades_py_files)} Python files in HADES")
    
    logger.info(f"Total data sources: {len(sources['pdfs'])} PDFs, {len(sources['python_files'])} Python files")
    
    return sources

def create_training_manifest(sources: Dict[str, List[str]]) -> Path:
    """Create a manifest file for training data."""
    manifest_path = OUTPUT_DIR / "training_manifest.json"
    
    manifest = {
        "created_at": datetime.now().isoformat(),
        "statistics": {
            "total_pdfs": len(sources["pdfs"]),
            "total_python_files": len(sources["python_files"]),
            "total_files": len(sources["pdfs"]) + len(sources["python_files"])
        },
        "sources": sources
    }
    
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Created training manifest: {manifest_path}")
    return manifest_path

def prepare_input_directory(sources: Dict[str, List[str]]) -> Path:
    """Prepare input directory with symlinks to all source files."""
    input_dir = OUTPUT_DIR / "input_data"
    input_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    pdf_dir = input_dir / "pdfs"
    pdf_dir.mkdir(exist_ok=True)
    
    python_dir = input_dir / "python"
    python_dir.mkdir(exist_ok=True)
    
    # Create symlinks to PDFs
    for pdf_path in sources["pdfs"]:
        pdf_file = Path(pdf_path)
        link_path = pdf_dir / pdf_file.name
        if link_path.exists():
            link_path.unlink()
        link_path.symlink_to(pdf_file)
    
    # Create symlinks to Python files (preserving directory structure)
    for py_path in sources["python_files"]:
        py_file = Path(py_path)
        # Create relative path structure
        if "ladon" in str(py_file):
            rel_path = py_file.relative_to(LADON_DIR)
            link_dir = python_dir / "ladon"
        else:
            rel_path = py_file.relative_to(HADES_ROOT)
            link_dir = python_dir / "hades"
        
        # Create parent directory
        (link_dir / rel_path.parent).mkdir(parents=True, exist_ok=True)
        
        # Create symlink
        link_path = link_dir / rel_path
        if link_path.exists():
            link_path.unlink()
        link_path.symlink_to(py_file)
    
    logger.info(f"Prepared input directory: {input_dir}")
    return input_dir

def run_comprehensive_pipeline(sources: Dict[str, List[str]]) -> None:
    """Run the comprehensive ISNE training pipeline."""
    start_time = time.time()
    
    # Prepare input directory
    input_dir = prepare_input_directory(sources)
    
    # Count total files
    total_files = len(sources["pdfs"]) + len(sources["python_files"])
    logger.info(f"Processing {total_files} total files")
    
    # Configuration for the pipeline
    pipeline_config = {
        "docproc": {
            "implementation": "unified",
            "config": {
                "pdf_processor": "marker",
                "code_processor": "python_ast",
                "batch_size": 10
            }
        },
        "chunking": {
            "implementation": "structural",
            "config": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "min_chunk_size": 100
            }
        },
        "embedding": {
            "implementation": "modernbert",
            "config": {
                "model_name": "nomic-ai/modernbert-embed-base",
                "batch_size": 32,
                "device": "cuda",
                "normalize_embeddings": True
            }
        },
        "graph": {
            "similarity_threshold": 0.8,
            "max_neighbors": 10,
            "batch_size": 1000,
            "use_gpu": True
        },
        "isne": {
            "embedding_dim": 768,
            "isne_dim": 128,
            "num_samples": 10,
            "learning_rate": 0.01,
            "num_epochs": 100,
            "batch_size": 256,
            "device": "cuda",
            "checkpoint_interval": 10
        }
    }
    
    # Save configuration
    config_path = OUTPUT_DIR / "pipeline_config.json"
    with open(config_path, 'w') as f:
        json.dump(pipeline_config, f, indent=2)
    
    logger.info("Starting comprehensive ISNE bootstrap pipeline...")
    
    try:
        # Run the full ISNE bootstrap pipeline
        import subprocess
        
        cmd = [
            sys.executable,
            str(ISNE_BOOTSTRAP_SCRIPT),
            "--input-dir", str(input_dir),
            "--output-dir", str(OUTPUT_DIR),
            "--log-level", "INFO"
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Run with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output
        for line in iter(process.stdout.readline, ''):
            if line:
                print(line.rstrip())
        
        # Wait for completion
        return_code = process.wait()
        
        if return_code == 0:
            logger.info("ISNE bootstrap pipeline completed successfully!")
        else:
            logger.error(f"Pipeline failed with return code: {return_code}")
            
    except Exception as e:
        logger.error(f"Failed to run pipeline: {e}")
        raise
    
    # Calculate total time
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    logger.info(f"Total training time: {hours}h {minutes}m {seconds}s")
    
    # Save summary
    summary = {
        "completed_at": datetime.now().isoformat(),
        "total_time_seconds": total_time,
        "total_time_formatted": f"{hours}h {minutes}m {seconds}s",
        "files_processed": len(all_files),
        "output_directory": str(OUTPUT_DIR),
        "model_path": str(OUTPUT_DIR / "isne_model_final.pt")
    }
    
    summary_path = OUTPUT_DIR / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Training summary saved to: {summary_path}")

def main():
    """Main function for comprehensive ISNE training."""
    logger.info("=" * 80)
    logger.info("Starting Comprehensive ISNE Model Training")
    logger.info("=" * 80)
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Collect all data sources
    sources = collect_data_sources()
    
    # Create training manifest
    manifest_path = create_training_manifest(sources)
    
    # Run comprehensive pipeline
    run_comprehensive_pipeline(sources)
    
    logger.info("=" * 80)
    logger.info("Comprehensive ISNE training completed!")
    logger.info(f"Results saved to: {OUTPUT_DIR}")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()