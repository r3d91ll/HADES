#!/usr/bin/env python3
"""
Test script for the file type detection and content categorization functionality.

This script verifies that:
1. Files are correctly detected based on their extensions
2. Content categories (code vs. text) are properly assigned
3. The document metadata contains the correct file type and category information
"""

import os
import sys
import logging
from pathlib import Path
from pprint import pprint

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.docproc.core import process_document
from src.docproc.utils.format_detector import (
    detect_format_from_path,
    get_content_category
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_file_type_detection(test_files):
    """
    Test the file type detection and content categorization functionality.
    
    Args:
        test_files: Dictionary mapping file path to expected format and category
    """
    logger.info("=== Testing File Type Detection and Content Categorization ===")
    
    results = []
    
    for file_path, expected in test_files.items():
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"Skipping non-existent file: {file_path}")
            continue
        
        # Test format detection
        detected_format = detect_format_from_path(path)
        format_match = detected_format == expected['format']
        
        # Test category detection
        detected_category = get_content_category(detected_format)
        category_match = detected_category == expected['category']
        
        results.append({
            'file': file_path,
            'expected_format': expected['format'],
            'detected_format': detected_format,
            'format_match': format_match,
            'expected_category': expected['category'],
            'detected_category': detected_category,
            'category_match': category_match
        })
    
    # Print results
    logger.info("=== Detection Results ===")
    for result in results:
        format_status = "✅" if result['format_match'] else "❌"
        category_status = "✅" if result['category_match'] else "❌"
        logger.info(f"{result['file']}: Format: {format_status} ({result['detected_format']}), Category: {category_status} ({result['detected_category']})")
    
    # Calculate success rates
    format_success = sum(1 for r in results if r['format_match']) / len(results) if results else 0
    category_success = sum(1 for r in results if r['category_match']) / len(results) if results else 0
    
    logger.info(f"Format detection success rate: {format_success:.2%}")
    logger.info(f"Category detection success rate: {category_success:.2%}")
    
    return results

def test_document_processing(test_files):
    """
    Test that processed documents contain the correct file type and category metadata.
    
    Args:
        test_files: Dictionary mapping file path to expected format and category
    """
    logger.info("\n=== Testing Document Processing with Content Categories ===")
    
    results = []
    
    for file_path, expected in test_files.items():
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"Skipping non-existent file: {file_path}")
            continue
        
        # Process the document
        try:
            processed_doc = process_document(path)
            
            # Check root level content_category field
            has_root_category = 'content_category' in processed_doc
            root_category_match = processed_doc.get('content_category') == expected['category'] if has_root_category else False
            
            # Check metadata content_category field
            has_meta_category = 'content_category' in processed_doc.get('metadata', {})
            meta_category_match = processed_doc.get('metadata', {}).get('content_category') == expected['category'] if has_meta_category else False
            
            # Check metadata file_type field
            has_file_type = 'file_type' in processed_doc.get('metadata', {})
            file_type_match = processed_doc.get('metadata', {}).get('file_type') == expected['format'] if has_file_type else False
            
            results.append({
                'file': file_path,
                'has_root_category': has_root_category,
                'root_category_match': root_category_match,
                'has_meta_category': has_meta_category,
                'meta_category_match': meta_category_match,
                'has_file_type': has_file_type,
                'file_type_match': file_type_match,
                'processed_doc': processed_doc
            })
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            results.append({
                'file': file_path,
                'error': str(e)
            })
    
    # Print results
    logger.info("=== Processing Results ===")
    for result in results:
        if 'error' in result:
            logger.info(f"{result['file']}: ❌ Error: {result['error']}")
            continue
            
        root_status = "✅" if result['root_category_match'] else "❌"
        meta_status = "✅" if result['meta_category_match'] else "❌"
        type_status = "✅" if result['file_type_match'] else "❌"
        
        logger.info(f"{result['file']}: Root Category: {root_status}, Metadata Category: {meta_status}, File Type: {type_status}")
    
    # Print a sample processed document
    if results and 'processed_doc' in results[0]:
        logger.info("\n=== Sample Processed Document ===")
        sample_doc = results[0]['processed_doc']
        logger.info(f"ID: {sample_doc.get('id')}")
        logger.info(f"Format: {sample_doc.get('format')}")
        logger.info(f"Content Category: {sample_doc.get('content_category')}")
        logger.info("Metadata:")
        for key, value in sample_doc.get('metadata', {}).items():
            logger.info(f"  {key}: {value}")
    
    return results

def main():
    """Main test function."""
    # Define test files with expected formats and categories
    test_files = {
        # Code files
        'test-data/app.py': {'format': 'python', 'category': 'code'},
        'test-data/file_batcher.py': {'format': 'python', 'category': 'code'},
        'test-data/embedding_config.yaml': {'format': 'yaml', 'category': 'code'},
        'test-data/embedding_report.json': {'format': 'json', 'category': 'code'},
        
        # Text files
        'test-data/docproc.md': {'format': 'markdown', 'category': 'text'},
        'test-data/CG-RAG_ Research Question Answering by Citation Graph Retrieval-Augmented LLMs_2501.15067v1.pdf': {'format': 'pdf', 'category': 'text'},
    }
    
    # Run tests
    detection_results = test_file_type_detection(test_files)
    processing_results = test_document_processing(test_files)
    
    logger.info("\n=== Test Summary ===")
    detection_success = all(r['format_match'] and r['category_match'] for r in detection_results)
    processing_success = all(r.get('root_category_match', False) and r.get('meta_category_match', False) 
                            and r.get('file_type_match', False) for r in processing_results 
                            if 'error' not in r)
    
    if detection_success and processing_success:
        logger.info("✅ All tests passed!")
        return 0
    else:
        logger.error("❌ Some tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
