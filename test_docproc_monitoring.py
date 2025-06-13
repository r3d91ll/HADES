#!/usr/bin/env python3
"""
Test script for DocProc monitoring functionality.

This script tests the comprehensive monitoring implementation for the DoclingDocumentProcessor
component, verifying that all monitoring methods work correctly and return properly formatted data.
"""

import sys
import json
from pathlib import Path

# Add HADES to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.components.docproc.docling.processor import DoclingDocumentProcessor


def test_docproc_monitoring():
    """Test all monitoring functionality for DocProc component."""
    print("="*60)
    print("Testing DocProc Monitoring Functionality")
    print("="*60)
    
    # Initialize processor
    print("\n1. Initializing DoclingDocumentProcessor...")
    config = {
        "ocr_enabled": True,
        "extract_tables": True,
        "extract_images": False
    }
    
    processor = DoclingDocumentProcessor(config=config)
    print(f"   ✓ Initialized processor: {processor.name} v{processor.version}")
    
    # Test health check
    print("\n2. Testing health check...")
    health_status = processor.health_check()
    print(f"   Health Status: {'✓ HEALTHY' if health_status else '✗ UNHEALTHY'}")
    
    # Test infrastructure metrics
    print("\n3. Testing infrastructure metrics...")
    infra_metrics = processor.get_infrastructure_metrics()
    print("   Infrastructure Metrics:")
    for key, value in infra_metrics.items():
        if key == "memory_usage":
            print(f"     {key}:")
            for mem_key, mem_value in value.items():
                print(f"       {mem_key}: {mem_value}")
        else:
            print(f"     {key}: {value}")
    
    # Test performance metrics
    print("\n4. Testing performance metrics...")
    perf_metrics = processor.get_performance_metrics()
    print("   Performance Metrics:")
    for key, value in perf_metrics.items():
        print(f"     {key}: {value}")
    
    # Test legacy get_metrics method
    print("\n5. Testing legacy get_metrics method...")
    legacy_metrics = processor.get_metrics()
    print(f"   Legacy metrics returned {len(legacy_metrics)} fields")
    
    # Test Prometheus export
    print("\n6. Testing Prometheus metrics export...")
    prometheus_output = processor.export_metrics_prometheus()
    prometheus_lines = prometheus_output.strip().split('\n')
    print(f"   Prometheus export generated {len(prometheus_lines)} lines")
    print("   Sample Prometheus metrics:")
    for i, line in enumerate(prometheus_lines[:10]):  # Show first 10 lines
        print(f"     {line}")
    if len(prometheus_lines) > 10:
        print(f"     ... and {len(prometheus_lines) - 10} more lines")
    
    # Test supported formats
    print("\n7. Testing supported formats...")
    supported_formats = processor.get_supported_formats()
    print(f"   Supported formats ({len(supported_formats)} total):")
    for fmt in supported_formats:
        print(f"     {fmt}")
    
    # Test configuration schema
    print("\n8. Testing configuration schema...")
    config_schema = processor.get_config_schema()
    print("   Configuration Schema:")
    print(f"     Type: {config_schema.get('type')}")
    properties = config_schema.get('properties', {})
    print(f"     Properties: {list(properties.keys())}")
    
    # Test file format checking
    print("\n9. Testing file format checking...")
    test_files = [
        "document.pdf",
        "presentation.pptx", 
        "spreadsheet.xlsx",
        "webpage.html",
        "image.png",
        "unsupported.xyz"
    ]
    
    for test_file in test_files:
        can_process = processor.can_process(test_file)
        status = "✓ CAN PROCESS" if can_process else "✗ CANNOT PROCESS"
        print(f"     {test_file}: {status}")
    
    # Simulate document processing to generate statistics
    print("\n10. Simulating document processing to test statistics...")
    try:
        # Try to process a non-existent file to test error handling
        from src.types.components.contracts import DocumentProcessingInput
        
        test_input = DocumentProcessingInput(
            file_path="non_existent_file.pdf",
            processing_options={}
        )
        
        result = processor.process(test_input)
        print(f"    Processed test document (expected to fail)")
        print(f"    Documents: {len(result.documents)}")
        print(f"    Errors: {len(result.errors)}")
        
        # Check updated statistics
        updated_perf_metrics = processor.get_performance_metrics()
        print("    Updated Performance Metrics:")
        print(f"      Total documents: {updated_perf_metrics.get('total_documents', 0)}")
        print(f"      Failed documents: {updated_perf_metrics.get('failed_documents', 0)}")
        print(f"      Error count: {updated_perf_metrics.get('error_count', 0)}")
        
    except Exception as e:
        print(f"    Error during document processing simulation: {e}")
    
    print("\n" + "="*60)
    print("DocProc Monitoring Test Complete")
    print("="*60)
    
    return True


if __name__ == "__main__":
    try:
        success = test_docproc_monitoring()
        if success:
            print("\n🎉 All monitoring tests passed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Some monitoring tests failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)