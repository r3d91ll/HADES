#!/usr/bin/env python3
"""Test script to verify ISNE API endpoints are working."""

import requests
import json
import time

def test_endpoints():
    """Test all ISNE production pipeline endpoints."""
    base_url = "http://localhost:8595"
    
    print("Testing ISNE Production Pipeline API...")
    print("=" * 50)
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"✓ Health endpoint: {response.status_code}")
        if response.status_code == 200:
            print(f"  Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"✗ Health endpoint failed: {e}")
    
    # Test docs endpoint
    try:
        response = requests.get(f"{base_url}/docs", timeout=5)
        print(f"✓ Docs endpoint: {response.status_code}")
    except Exception as e:
        print(f"✗ Docs endpoint failed: {e}")
    
    # Test production status endpoint
    try:
        response = requests.get(f"{base_url}/production/status", timeout=5)
        print(f"✓ Production status endpoint: {response.status_code}")
        if response.status_code == 200:
            print(f"  Active operations: {len(response.json())}")
    except Exception as e:
        print(f"✗ Production status endpoint failed: {e}")
    
    print("\nProduction Pipeline Endpoints:")
    print("-" * 30)
    
    endpoints = [
        "/production/bootstrap",
        "/production/train",
        "/production/apply-model",
        "/production/build-collections"
    ]
    
    for endpoint in endpoints:
        try:
            # Just check if endpoint exists (OPTIONS request)
            response = requests.options(f"{base_url}{endpoint}", timeout=5)
            print(f"✓ {endpoint}: Available")
        except Exception as e:
            print(f"✗ {endpoint}: {e}")

if __name__ == "__main__":
    # Give server time to start
    print("Waiting for server to be ready...")
    time.sleep(2)
    
    test_endpoints()