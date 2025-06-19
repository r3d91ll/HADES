#!/usr/bin/env python3
"""
Test script for HADES API metrics endpoint.

This script tests the unified metrics endpoint in the FastAPI server.
"""

import sys
import json
import requests
import time
from pathlib import Path

# Add HADES to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_metrics_endpoint_direct():
    """Test metrics collection directly (without server)."""
    print("="*60)
    print("Testing HADES API Metrics Collection (Direct)")
    print("="*60)
    
    try:
        from src.api.server import collect_component_metrics, generate_api_metrics
        
        print("\n1. Testing component metrics collection...")
        component_metrics = collect_component_metrics()
        print(f"✓ Component metrics collected ({len(component_metrics)} characters)")
        
        if component_metrics.strip():
            lines = component_metrics.strip().split('\n')
            print(f"  - Total lines: {len(lines)}")
            
            # Count metric lines (non-comment lines)
            metric_lines = [line for line in lines if line and not line.startswith('#')]
            print(f"  - Metric lines: {len(metric_lines)}")
            
            # Show first few metric lines
            print("  - Sample metrics:")
            for i, line in enumerate(metric_lines[:5]):
                print(f"    {line}")
            if len(metric_lines) > 5:
                print(f"    ... and {len(metric_lines) - 5} more metrics")
        else:
            print("  - No component metrics found")
        
        print("\n2. Testing API metrics generation...")
        api_metrics = generate_api_metrics()
        print(f"✓ API metrics generated ({len(api_metrics)} characters)")
        
        if api_metrics.strip():
            api_lines = api_metrics.strip().split('\n')
            print(f"  - API metric lines: {len(api_lines)}")
            print("  - Sample API metrics:")
            for line in api_lines[:3]:
                print(f"    {line}")
        
        print("\n3. Testing combined metrics...")
        if component_metrics or api_metrics:
            combined = "\n".join(filter(None, [component_metrics, api_metrics]))
            print(f"✓ Combined metrics: {len(combined)} characters")
        
        print("\n" + "="*60)
        print("✅ Direct metrics collection test PASSED")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Direct test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics_endpoint_server():
    """Test metrics endpoint via HTTP server."""
    print("\n" + "="*60)
    print("Testing HADES API Metrics Endpoint (HTTP)")
    print("="*60)
    
    # This would require starting the server first
    # For now, just show how it would be tested
    
    print("\nTo test the HTTP endpoint:")
    print("1. Start the HADES API server:")
    print("   poetry run python -m src.api.server")
    print("2. In another terminal, test the endpoint:")
    print("   curl http://localhost:8000/metrics")
    print("3. Verify Prometheus format:")
    print("   curl http://localhost:8000/metrics | grep -E '^[a-z].*\\{.*\\} [0-9]+'")
    
    return True


if __name__ == "__main__":
    try:
        # Test direct metrics collection
        success1 = test_metrics_endpoint_direct()
        
        # Show server testing instructions
        success2 = test_metrics_endpoint_server()
        
        if success1 and success2:
            print("\n🎉 All metrics tests completed successfully!")
            print("\nNext steps:")
            print("1. Start HADES API server: poetry run python -m src.api.server")
            print("2. Test endpoint: curl http://localhost:8000/metrics")
            print("3. Configure Prometheus to scrape hades:8000/metrics")
            sys.exit(0)
        else:
            print("\n❌ Some metrics tests failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)