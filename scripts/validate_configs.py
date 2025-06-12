#!/usr/bin/env python3
"""
Validate Configuration Files

This script validates that all configuration files are properly set up
and accessible for the HADES pipeline system.
"""

import sys
import logging
import json
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.config_loader import load_config


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def validate_config_file(config_name: str, required_sections: List[str] = None) -> Dict[str, Any]:
    """Validate a single configuration file."""
    logger = logging.getLogger(__name__)
    
    try:
        config = load_config(config_name)
        logger.info(f"✓ Successfully loaded {config_name}")
        
        if required_sections:
            missing_sections = []
            for section in required_sections:
                if section not in config:
                    missing_sections.append(section)
            
            if missing_sections:
                logger.warning(f"⚠ Missing sections in {config_name}: {missing_sections}")
                return {'status': 'warning', 'config': config, 'missing': missing_sections}
            else:
                logger.info(f"✓ All required sections present in {config_name}")
        
        return {'status': 'success', 'config': config}
        
    except Exception as e:
        logger.error(f"✗ Failed to load {config_name}: {e}")
        return {'status': 'error', 'error': str(e)}


def validate_data_ingestion_config():
    """Validate the complete data ingestion configuration."""
    logger = logging.getLogger(__name__)
    logger.info("Validating data_ingestion_config.yaml...")
    
    required_sections = [
        'pipeline', 'global', 'docproc', 'chunking', 
        'embedding', 'isne', 'storage', 'debug', 'monitoring'
    ]
    
    result = validate_config_file('data_ingestion_config', required_sections)
    
    if result['status'] == 'success':
        config = result['config']
        
        # Validate specific subsections
        validation_results = []
        
        # Check docproc configuration
        docproc = config.get('docproc', {})
        if 'supported_formats' in docproc and 'processors' in docproc:
            logger.info("✓ DocProc configuration looks good")
        else:
            logger.warning("⚠ DocProc configuration incomplete")
            validation_results.append("DocProc configuration incomplete")
        
        # Check chunking configuration  
        chunking = config.get('chunking', {})
        if 'strategies' in chunking and 'default_strategy' in chunking:
            logger.info("✓ Chunking configuration looks good")
        else:
            logger.warning("⚠ Chunking configuration incomplete")
            validation_results.append("Chunking configuration incomplete")
        
        # Check embedding configuration
        embedding = config.get('embedding', {})
        if 'adapter_name' in embedding and 'model' in embedding:
            logger.info("✓ Embedding configuration looks good")
        else:
            logger.warning("⚠ Embedding configuration incomplete")
            validation_results.append("Embedding configuration incomplete")
        
        # Check ISNE configuration
        isne = config.get('isne', {})
        if 'model' in isne and 'graph' in isne:
            logger.info("✓ ISNE configuration looks good")
        else:
            logger.warning("⚠ ISNE configuration incomplete")
            validation_results.append("ISNE configuration incomplete")
        
        # Check storage configuration
        storage = config.get('storage', {})
        if 'database' in storage and 'collections' in storage:
            logger.info("✓ Storage configuration looks good")
        else:
            logger.warning("⚠ Storage configuration incomplete")
            validation_results.append("Storage configuration incomplete")
        
        return {
            'status': 'success' if not validation_results else 'warning',
            'validations': validation_results,
            'config': config
        }
    
    return result


def validate_bootstrap_config():
    """Validate the bootstrap configuration."""
    logger = logging.getLogger(__name__)
    logger.info("Validating bootstrap_config.yaml...")
    
    required_sections = [
        'bootstrap', 'processing', 'logging', 'debug', 'integration'
    ]
    
    result = validate_config_file('bootstrap_config', required_sections)
    
    if result['status'] == 'success':
        config = result['config']
        
        # Validate bootstrap-specific sections
        bootstrap = config.get('bootstrap', {})
        required_bootstrap_sections = [
            'corpus', 'output', 'initial_graph', 'training', 'device'
        ]
        
        missing_bootstrap = []
        for section in required_bootstrap_sections:
            if section not in bootstrap:
                missing_bootstrap.append(section)
        
        if missing_bootstrap:
            logger.warning(f"⚠ Missing bootstrap subsections: {missing_bootstrap}")
            return {
                'status': 'warning',
                'missing_subsections': missing_bootstrap,
                'config': config
            }
        else:
            logger.info("✓ Bootstrap configuration structure is complete")
    
    return result


def validate_individual_configs():
    """Validate individual stage configuration files."""
    logger = logging.getLogger(__name__)
    logger.info("Validating individual stage configuration files...")
    
    config_files = [
        ('chunker_config', ['chunkers', 'defaults']),
        ('embedding_config', ['adapters', 'models']),
        ('database_config', ['connection', 'collections']),
        ('isne_config', ['model', 'training']),
        ('vllm_config', ['server', 'models'])
    ]
    
    results = {}
    
    for config_name, required_sections in config_files:
        logger.info(f"Checking {config_name}...")
        result = validate_config_file(config_name, required_sections)
        results[config_name] = result
    
    return results


def check_config_directory():
    """Check that configuration directory and files exist."""
    logger = logging.getLogger(__name__)
    
    config_dir = project_root / "src" / "config"
    
    if not config_dir.exists():
        logger.error(f"✗ Configuration directory not found: {config_dir}")
        return False
    
    logger.info(f"✓ Configuration directory found: {config_dir}")
    
    # List all config files
    config_files = list(config_dir.glob("*.yaml"))
    logger.info(f"Found {len(config_files)} YAML configuration files:")
    
    for config_file in config_files:
        logger.info(f"  - {config_file.name}")
    
    # Check for our key configuration files
    key_configs = [
        'data_ingestion_config.yaml',
        'bootstrap_config.yaml',
        'chunker_config.yaml',
        'embedding_config.yaml',
        'database_config.yaml',
        'isne_config.yaml'
    ]
    
    missing_configs = []
    for key_config in key_configs:
        config_path = config_dir / key_config
        if config_path.exists():
            logger.info(f"✓ {key_config} exists")
        else:
            logger.warning(f"⚠ {key_config} missing")
            missing_configs.append(key_config)
    
    return len(missing_configs) == 0


def main():
    """Main validation function."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("="*60)
    print("HADES Configuration Validation")
    print("="*60)
    
    validation_success = True
    
    # 1. Check configuration directory
    logger.info("\n1. Checking configuration directory...")
    if not check_config_directory():
        validation_success = False
    
    # 2. Validate main data ingestion config
    logger.info("\n2. Validating data ingestion configuration...")
    data_ingestion_result = validate_data_ingestion_config()
    if data_ingestion_result['status'] == 'error':
        validation_success = False
    
    # 3. Validate bootstrap config
    logger.info("\n3. Validating bootstrap configuration...")
    bootstrap_result = validate_bootstrap_config()
    if bootstrap_result['status'] == 'error':
        validation_success = False
    
    # 4. Validate individual stage configs
    logger.info("\n4. Validating individual stage configurations...")
    individual_results = validate_individual_configs()
    
    for config_name, result in individual_results.items():
        if result['status'] == 'error':
            validation_success = False
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    if validation_success:
        print("✓ All configuration files validated successfully!")
        print("✓ Pipeline scripts can use configuration files")
        print("✓ Ready for production use")
    else:
        print("✗ Configuration validation failed")
        print("✗ Some configuration files are missing or invalid")
        print("✗ Please fix configuration issues before running pipelines")
    
    # Detailed status
    print(f"\nDetailed Status:")
    print(f"Data Ingestion Config: {data_ingestion_result['status']}")
    print(f"Bootstrap Config: {bootstrap_result['status']}")
    
    for config_name, result in individual_results.items():
        print(f"{config_name}: {result['status']}")
    
    # Recommendations
    print(f"\nRecommendations:")
    
    if validation_success:
        print("1. Use run_data_ingestion_with_config.py for production data ingestion")
        print("2. Use run_bootstrap_with_config.py for ISNE bootstrap")
        print("3. Use test_complete_pipeline.py for testing with configs")
        print("4. Configuration files will be automatically loaded by scripts")
    else:
        print("1. Check missing configuration files and create them")
        print("2. Verify YAML syntax in existing configuration files")
        print("3. Ensure all required sections are present")
        print("4. Run this validation script again after fixes")
    
    return 0 if validation_success else 1


if __name__ == "__main__":
    sys.exit(main())