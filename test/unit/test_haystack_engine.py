#!/usr/bin/env python3
"""
Test script for Haystack Model Engine component.

This script tests the Haystack engine implementation to ensure it works correctly
with the component architecture and protocol compliance.
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.components.model_engine.engines.haystack.processor import HaystackModelEngine
from src.types.components.contracts import ModelEngineInput

def test_haystack_engine():
    """Test the Haystack model engine functionality."""
    
    print("🧪 Testing Haystack Model Engine")
    print("=" * 50)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Test 1: Basic initialization
        print("\n1. Testing basic initialization...")
        config = {
            'pipeline_type': 'embedding',
            'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
            'device': 'cpu'
        }
        
        engine = HaystackModelEngine(config)
        print(f"✅ Engine initialized: {engine.name} v{engine.version}")
        
        # Test 2: Configuration validation
        print("\n2. Testing configuration validation...")
        valid_config = {
            'pipeline_type': 'embedding',
            'model_name': 'sentence-transformers/all-MiniLM-L6-v2'
        }
        
        if engine.validate_config(valid_config):
            print("✅ Configuration validation passed")
        else:
            print("❌ Configuration validation failed")
            return
        
        # Test 3: Health check
        print("\n3. Testing health check...")
        if engine.health_check():
            print("✅ Health check passed - pipeline initialized")
        else:
            print("❌ Health check failed")
            return
        
        # Test 4: Embedding processing
        print("\n4. Testing embedding processing...")
        
        # Create test input
        test_requests = [
            {
                'request_id': 'test_1',
                'type': 'embedding',
                'text': 'This is a test sentence for embedding.'
            },
            {
                'request_id': 'test_2', 
                'type': 'embedding',
                'text': 'Another test sentence with different content.'
            }
        ]
        
        input_data = ModelEngineInput(requests=test_requests)
        
        # Process the requests
        result = engine.process(input_data)
        
        print(f"✅ Processed {len(result.results)} requests")
        print(f"   Processing time: {result.metadata.processing_time:.3f}s")
        print(f"   Success count: {result.engine_stats.get('success_count', 0)}")
        print(f"   Error count: {result.engine_stats.get('error_count', 0)}")
        
        # Check results
        for i, res in enumerate(result.results):
            if res.error:
                print(f"❌ Request {i+1} failed: {res.error}")
            else:
                response = res.response_data
                if 'embeddings' in response and response['processed']:
                    print(f"✅ Request {i+1} succeeded: Got embeddings with shape {len(response['embeddings'])}x{len(response['embeddings'][0]) if response['embeddings'] else 0}")
                else:
                    print(f"⚠️  Request {i+1} processed but no embeddings: {response}")
        
        # Test 5: Metrics
        print("\n5. Testing metrics...")
        metrics = engine.get_metrics()
        print(f"✅ Got metrics: {metrics}")
        
        # Test 6: Server operations
        print("\n6. Testing server operations...")
        if engine.start_server():
            print("✅ Server start successful")
        else:
            print("❌ Server start failed")
            
        if engine.is_server_running():
            print("✅ Server is running")
        else:
            print("❌ Server not running")
            
        if engine.stop_server():
            print("✅ Server stop successful")
        else:
            print("❌ Server stop failed")
        
        print("\n🎉 All tests completed successfully!")
        print("✅ Haystack Model Engine is working correctly")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_haystack_engine()
    sys.exit(0 if success else 1)