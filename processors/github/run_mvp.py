#!/usr/bin/env python3
"""
CLI script to run Word2Vec MVP processing.
Only run this after ArXiv rebuild is complete and GPUs are available.

Usage:
    python3 run_mvp.py --check        # Check if ready to run
    python3 run_mvp.py --clone-only   # Clone repos without embedding
    python3 run_mvp.py --embed-only   # Embed already cloned repos
    python3 run_mvp.py --full         # Full MVP processing
"""

import argparse
import os
import sys
import torch
from pathlib import Path
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from github_processor import GitHubProcessor, get_db_config, process_word2vec_mvp

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_prerequisites():
    """Check if system is ready for processing."""
    issues = []
    
    # Check GPU availability
    if not torch.cuda.is_available():
        issues.append("GPU not available - required for embeddings")
    else:
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
    
    # Check database password
    if not os.environ.get('ARANGO_PASSWORD'):
        issues.append("ARANGO_PASSWORD environment variable not set")
    
    # Check disk space
    import shutil
    stat = shutil.disk_usage("/data" if Path("/data").exists() else "/")
    free_gb = stat.free / (1024**3)
    if free_gb < 1:
        issues.append(f"Insufficient disk space: {free_gb:.2f} GB free (need 1 GB)")
    else:
        logger.info(f"Disk space available: {free_gb:.2f} GB")
    
    # Check if ArXiv rebuild is still running
    import subprocess
    try:
        result = subprocess.run(['pgrep', '-f', 'rebuild_dual_gpu'], 
                              capture_output=True, text=True)
        if result.stdout.strip():
            issues.append("ArXiv rebuild still running - wait for completion")
            logger.warning("Found ArXiv rebuild process: " + result.stdout.strip())
    except:
        pass  # pgrep might not be available
    
    return issues


def clone_only():
    """Clone repositories without embedding."""
    logger.info("Starting clone-only mode")
    
    WORD2VEC_REPOS = [
        "https://github.com/dav/word2vec",
        "https://github.com/tmikolov/word2vec", 
        "https://github.com/danielfrg/word2vec",
        "https://github.com/RaRe-Technologies/gensim"
    ]
    
    processor = GitHubProcessor(get_db_config())
    
    for repo_url in WORD2VEC_REPOS:
        logger.info(f"Cloning {repo_url}")
        result = processor.clone_repository(repo_url)
        
        if result['status'] == 'success':
            logger.info(f"✓ Cloned {result['key']}: {result['size_mb']:.2f} MB")
        elif result['status'] == 'exists':
            logger.info(f"✓ Already exists: {result['key']}")
        else:
            logger.error(f"✗ Failed to clone {repo_url}: {result.get('error')}")


def embed_only():
    """Embed already cloned repositories."""
    logger.info("Starting embed-only mode")
    
    if not torch.cuda.is_available():
        logger.error("GPU required for embedding. Exiting.")
        return
    
    processor = GitHubProcessor(get_db_config())
    
    # Process each repository
    repos = ['dav_word2vec', 'tmikolov_word2vec', 'danielfrg_word2vec', 'RaRe-Technologies_gensim']
    
    for repo_key in repos:
        logger.info(f"Processing {repo_key}")
        
        # Get files to embed
        if 'gensim' in repo_key:
            files = ['gensim/models/word2vec.py']
        else:
            files = processor.list_code_files(repo_key)[:10]
        
        if not files:
            logger.warning(f"No files found for {repo_key}")
            continue
        
        # Embed files
        result = processor.embed_files_batch(repo_key, files)
        logger.info(f"✓ Embedded {result['successful']}/{result['total_files']} files from {repo_key}")


def main():
    parser = argparse.ArgumentParser(description='Word2Vec MVP Processing')
    parser.add_argument('--check', action='store_true', 
                       help='Check if ready to run')
    parser.add_argument('--clone-only', action='store_true',
                       help='Clone repositories without embedding')
    parser.add_argument('--embed-only', action='store_true',
                       help='Embed already cloned repositories')
    parser.add_argument('--full', action='store_true',
                       help='Run full MVP processing')
    
    args = parser.parse_args()
    
    # Check prerequisites
    if args.check or not any([args.clone_only, args.embed_only, args.full]):
        issues = check_prerequisites()
        if issues:
            logger.error("Prerequisites not met:")
            for issue in issues:
                logger.error(f"  ✗ {issue}")
            sys.exit(1)
        else:
            logger.info("✓ All prerequisites met - ready to run")
            if args.check:
                sys.exit(0)
    
    # Run requested mode
    if args.clone_only:
        clone_only()
    elif args.embed_only:
        embed_only()
    elif args.full:
        logger.info("Running full MVP processing")
        process_word2vec_mvp()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()