#!/usr/bin/env python3
"""
Simple test script for DocProc monitoring functionality.

This script tests only the core docproc monitoring without importing 
other components that may have dependency issues.
"""

import sys
import json
from pathlib import Path

# Add HADES to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_docproc_direct():
    """Test docproc component directly."""
    print("="*50)
    print("Testing DocProc Monitoring (Direct Import)")
    print("="*50)
    
    try:
        # Direct import to avoid registry issues
        from src.components.docproc.docling.processor import DoclingDocumentProcessor
        
        print("\n✓ Successfully imported DoclingDocumentProcessor")
        
        # Initialize processor
        config = {
            "ocr_enabled": True,
            "extract_tables": True,
            "extract_images": False
        }
        
        processor = DoclingDocumentProcessor(config=config)
        print(f"✓ Initialized processor: {processor.name} v{processor.version}")
        
        # Test health check
        health_status = processor.health_check()
        print(f"Health Status: {'✓ HEALTHY' if health_status else '✗ UNHEALTHY'}")
        
        # Test infrastructure metrics
        print("\nTesting infrastructure metrics...")
        infra_metrics = processor.get_infrastructure_metrics()
        print(f"✓ Infrastructure metrics returned {len(infra_metrics)} fields")
        
        # Test performance metrics
        print("Testing performance metrics...")
        perf_metrics = processor.get_performance_metrics()
        print(f"✓ Performance metrics returned {len(perf_metrics)} fields")
        
        # Test Prometheus export
        print("Testing Prometheus metrics export...")
        prometheus_output = processor.export_metrics_prometheus()
        prometheus_lines = prometheus_output.strip().split('\n')
        print(f"✓ Prometheus export generated {len(prometheus_lines)} lines")
        
        # Show key metrics
        print("\nKey Infrastructure Metrics:")
        for key in ["component_name", "component_version", "docling_available", "supported_format_count"]:
            if key in infra_metrics:
                print(f"  {key}: {infra_metrics[key]}")
        
        print("\nKey Performance Metrics:")
        for key in ["total_documents", "successful_documents", "failed_documents", "success_rate_percent"]:
            if key in perf_metrics:
                print(f"  {key}: {perf_metrics[key]}")
        
        print("\n" + "="*50)
        print("✅ DocProc monitoring test PASSED")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_docproc_direct()
    sys.exit(0 if success else 1)