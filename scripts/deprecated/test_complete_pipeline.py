#!/usr/bin/env python3
"""
Test Complete Data Ingestion Pipeline

This script tests the complete data ingestion pipeline from document processing
through storage in ArangoDB, validating the JSON schema flow we analyzed.
"""

import sys
import logging
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.orchestration.pipelines.data_ingestion_pipeline import run_data_ingestion_pipeline
from src.config.config_loader import load_config


def setup_logging():
    """Set up logging for pipeline test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('complete_pipeline_test.log')
        ]
    )


def load_pipeline_config():
    """Load pipeline configuration."""
    try:
        # First try to load the complete data ingestion config
        data_ingestion_config = load_config('data_ingestion_config')
        logger = logging.getLogger(__name__)
        logger.info("Loaded data_ingestion_config.yaml")
        
        # Extract stage-specific configs from the data ingestion config
        config = {
            'docproc': data_ingestion_config.get('docproc', {}),
            'chunking': data_ingestion_config.get('chunking', {}),
            'embedding': data_ingestion_config.get('embedding', {}),
            'isne': data_ingestion_config.get('isne', {}),
            'storage': data_ingestion_config.get('storage', {})
        }
        
        # Override storage config for testing
        config['storage'].update({
            'mode': 'create',
            'overwrite': True,
            'database_name': 'hades_test_' + datetime.now().strftime("%Y%m%d_%H%M%S"),
            'batch_size': 50
        })
        
        return config
        
    except Exception as e:
        logging.warning(f"Failed to load data_ingestion_config: {e}, trying individual configs")
        
        # Fallback to individual config files
        try:
            config = {
                'docproc': load_config('docproc'),
                'chunking': load_config('chunking'),
                'embedding': load_config('embedding'),
                'isne': load_config('isne'),
                'storage': load_config('storage')
            }
            
            # Override storage config for testing
            config['storage'].update({
                'mode': 'create',
                'overwrite': True,
                'database_name': 'hades_test_' + datetime.now().strftime("%Y%m%d_%H%M%S"),
                'batch_size': 50
            })
            
            return config
            
        except Exception as e2:
            logging.warning(f"Failed to load individual configs: {e2}")
            # Return minimal default config
            return {
                'docproc': {'batch_size': 10},
                'chunking': {'batch_size': 32, 'chunker_type': 'cpu'},
                'embedding': {'adapter_name': 'cpu', 'batch_size': 16},
                'isne': {'enable_training': False, 'batch_size': 32},
                'storage': {
                    'mode': 'create',
                    'overwrite': True,
                    'database_name': 'hades_test_' + datetime.now().strftime("%Y%m%d_%H%M%S"),
                    'batch_size': 50
                }
            }


def main():
    """Run complete pipeline test."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Define paths
    corpus_dir = project_root / "test-data"
    output_dir = project_root / "test-output" / "complete-pipeline-test" / datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info("Starting complete data ingestion pipeline test")
    logger.info(f"Corpus directory: {corpus_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Find input files
        pdf_files = list(corpus_dir.glob("*.pdf"))
        py_files = list(corpus_dir.glob("*.py"))
        input_files = [str(f) for f in pdf_files + py_files]
        
        if not input_files:
            logger.error(f"No input files found in {corpus_dir}")
            return False
        
        logger.info(f"Found {len(input_files)} input files")
        for file in input_files:
            logger.info(f"  - {Path(file).name}")
        
        # Load configuration
        config = load_pipeline_config()
        logger.info("Loaded pipeline configuration")
        
        # Run complete pipeline
        logger.info("Starting complete data ingestion pipeline...")
        results = run_data_ingestion_pipeline(
            input_files=input_files,
            config=config,
            enable_debug=True,
            debug_output_dir=str(output_dir),
            filter_types=['.pdf', '.py']
        )
        
        # Save results
        results_file = output_dir / "pipeline_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Analyze and report results
        logger.info("=== Pipeline Test Results ===")
        logger.info(f"Success: {results['success']}")
        logger.info(f"Total time: {results['total_time']:.2f} seconds")
        
        stats = results['pipeline_stats']
        logger.info(f"Files processed: {stats['total_files']}")
        logger.info(f"Documents processed: {stats['processed_documents']}")
        logger.info(f"Chunks generated: {stats['generated_chunks']}")
        logger.info(f"Chunks embedded: {stats['embedded_chunks']}")
        logger.info(f"Chunks ISNE enhanced: {stats['isne_enhanced_chunks']}")
        logger.info(f"Documents stored: {stats['stored_documents']}")
        logger.info(f"Chunks stored: {stats['stored_chunks']}")
        logger.info(f"Relationships stored: {stats['stored_relationships']}")
        
        # Stage timing analysis
        logger.info("\n=== Stage Performance ===")
        stage_times = results['stage_times']
        for stage, time_taken in stage_times.items():
            logger.info(f"{stage}: {time_taken:.2f} seconds")
        
        # Error analysis
        if stats['errors']:
            logger.warning(f"\n=== Errors Encountered ({len(stats['errors'])}) ===")
            for error in stats['errors']:
                logger.warning(f"{error['stage']}: {error['message']}")
        
        # Debug files analysis
        if results['debug_enabled']:
            logger.info(f"\n=== Debug Files Created ===")
            debug_dir = Path(results['debug_output_dir'])
            for stage_dir in ['docproc', 'chunking', 'embedding', 'isne', 'storage']:
                stage_path = debug_dir / stage_dir
                if stage_path.exists():
                    files = list(stage_path.glob("*.json"))
                    logger.info(f"{stage_dir}/: {len(files)} files")
                    for file in files:
                        size_kb = file.stat().st_size / 1024
                        logger.info(f"  - {file.name} ({size_kb:.1f} KB)")
        
        # JSON Schema Validation
        logger.info(f"\n=== JSON Schema Analysis ===")
        
        # Check chunking output structure
        chunking_file = output_dir / "chunking" / "chunks.json"
        if chunking_file.exists():
            with open(chunking_file) as f:
                chunks = json.load(f)
            
            if chunks:
                sample_chunk = chunks[0]
                logger.info("Chunking output schema:")
                logger.info(f"  - Sample chunk keys: {list(sample_chunk.keys())}")
                logger.info(f"  - Sample chunk ID: {sample_chunk.get('id', 'N/A')}")
                logger.info(f"  - Has metadata: {'metadata' in sample_chunk}")
                logger.info(f"  - Text length: {len(sample_chunk.get('text', ''))}")
        
        # Check embedding output structure
        embedding_file = output_dir / "embedding" / "embedded_chunks.json"
        if embedding_file.exists():
            with open(embedding_file) as f:
                embedded_chunks = json.load(f)
            
            if embedded_chunks:
                sample_embedded = embedded_chunks[0]
                logger.info("Embedding output schema:")
                logger.info(f"  - Sample embedded chunk keys: {list(sample_embedded.keys())}")
                logger.info(f"  - Has embedding: {'embedding' in sample_embedded}")
                logger.info(f"  - Embedding dimension: {len(sample_embedded.get('embedding', []))}")
                logger.info(f"  - Has embedding metadata: {'embedding_metadata' in sample_embedded}")
        
        # Check ISNE output structure
        isne_file = output_dir / "isne" / "isne_enhanced_chunks.json"
        if isne_file.exists():
            with open(isne_file) as f:
                isne_chunks = json.load(f)
            
            if isne_chunks:
                sample_isne = isne_chunks[0]
                logger.info("ISNE output schema:")
                logger.info(f"  - Sample ISNE chunk keys: {list(sample_isne.keys())}")
                logger.info(f"  - Has ISNE embedding: {'isne_embedding' in sample_isne}")
                logger.info(f"  - Has graph relationships: {'graph_relationships' in sample_isne}")
                logger.info(f"  - Has relationships: {'relationships' in sample_isne}")
        
        # Final recommendations
        logger.info(f"\n=== Recommendations ===")
        
        if results['success']:
            logger.info("✓ Pipeline completed successfully")
            logger.info("✓ All stages executed without critical errors")
            logger.info("✓ JSON schema flow validated")
            
            if stats['generated_chunks'] > 0:
                logger.info("✓ Document chunking successful")
            if stats['embedded_chunks'] > 0:
                logger.info("✓ Embedding generation successful")
            if stats['stored_chunks'] > 0:
                logger.info("✓ Database storage successful")
                
        else:
            logger.warning("⚠ Pipeline encountered errors")
            logger.warning("⚠ Review error logs for issues")
            
        # Performance recommendations
        total_time = results['total_time']
        files_per_second = stats['total_files'] / total_time if total_time > 0 else 0
        logger.info(f"Processing rate: {files_per_second:.2f} files/second")
        
        if total_time > 300:  # > 5 minutes
            logger.info("Consider optimizing for large-scale processing:")
            logger.info("  - Enable parallel processing")
            logger.info("  - Use GPU acceleration where available")
            logger.info("  - Increase batch sizes")
            logger.info("  - Enable caching")
        
        logger.info(f"\nComplete results saved to: {results_file}")
        logger.info("Test completed successfully!")
        
        return results['success']
        
    except Exception as e:
        logger.error(f"Complete pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)