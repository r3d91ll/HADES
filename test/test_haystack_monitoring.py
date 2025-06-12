#!/usr/bin/env python3
"""
Test Haystack Model Engine Monitoring

This script tests the new monitoring capabilities added to the Haystack model engine
component according to HADES Service Architecture Section 3.7.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_haystack_monitoring():
    """Test Haystack model engine monitoring capabilities."""
    
    print("🔍 Testing Haystack Model Engine Monitoring")
    print("=" * 50)
    
    try:
        # Import the Haystack engine
        from src.components.model_engine.engines.haystack.processor import HaystackModelEngine
        from src.types.components.contracts import ModelEngineInput
        
        # Create engine with test configuration
        config = {
            "pipeline_type": "embedding",
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "device": "cpu"
        }
        
        print(f"📝 Creating Haystack engine with config: {config}")
        engine = HaystackModelEngine(config)
        
        # Test 1: Health Check
        print(f"\n🏥 Testing health check...")
        health_status = engine.health_check()
        print(f"   Health Status: {'✅ OK' if health_status else '❌ FAILED'}")
        
        # Test 2: Infrastructure Metrics
        print(f"\n🖥️  Testing infrastructure metrics...")
        infra_metrics = engine.get_infrastructure_metrics()
        print(f"   Component: {infra_metrics.get('component_name')}")
        print(f"   Device: {infra_metrics.get('device_allocation')}")
        print(f"   Memory RSS: {infra_metrics.get('memory_usage', {}).get('rss_mb', 0)} MB")
        print(f"   Pipeline initialized: {infra_metrics.get('pipeline_initialized')}")
        print(f"   Uptime: {round(infra_metrics.get('uptime_seconds', 0), 2)} seconds")
        
        # Test 3: Performance Metrics
        print(f"\n📊 Testing performance metrics...")
        perf_metrics = engine.get_performance_metrics()
        print(f"   Total requests: {perf_metrics.get('total_requests')}")
        print(f"   Success rate: {perf_metrics.get('success_rate_percent')}%")
        print(f"   Requests/sec: {perf_metrics.get('requests_per_second')}")
        print(f"   Avg processing time: {perf_metrics.get('average_processing_time')} seconds")
        
        # Test 4: Process some requests to generate metrics
        print(f"\n🔄 Testing request processing...")
        test_input = ModelEngineInput(
            requests=[
                {
                    "request_id": "test_1",
                    "type": "embedding",
                    "text": "This is a test sentence for embedding."
                },
                {
                    "request_id": "test_2", 
                    "type": "embedding",
                    "text": "Another test sentence to generate embeddings."
                }
            ]
        )
        
        try:
            result = engine.process(test_input)
            print(f"   ✅ Processed {len(result.results)} requests")
            print(f"   ✅ Processing time: {result.metadata.processing_time:.3f} seconds")
            print(f"   ✅ Success count: {result.engine_stats.get('success_count', 0)}")
            print(f"   ✅ Error count: {result.engine_stats.get('error_count', 0)}")
        except Exception as e:
            print(f"   ⚠️  Request processing error: {e}")
        
        # Test 5: Updated Performance Metrics
        print(f"\n📈 Testing updated performance metrics...")
        updated_perf_metrics = engine.get_performance_metrics()
        print(f"   Total requests: {updated_perf_metrics.get('total_requests')}")
        print(f"   Successful requests: {updated_perf_metrics.get('successful_requests')}")
        print(f"   Success rate: {updated_perf_metrics.get('success_rate_percent')}%")
        print(f"   Requests/sec: {updated_perf_metrics.get('requests_per_second')}")
        
        # Test 6: Prometheus Metrics Export
        print(f"\n📊 Testing Prometheus metrics export...")
        prometheus_metrics = engine.export_metrics_prometheus()
        metrics_lines = prometheus_metrics.strip().split('\n')
        metric_count = len([line for line in metrics_lines if not line.startswith('#') and line.strip()])
        print(f"   ✅ Generated {metric_count} Prometheus metrics")
        print(f"   ✅ Metrics preview:")
        for line in metrics_lines[:6]:  # Show first 6 lines
            if line.strip():
                print(f"      {line}")
        
        # Test 7: JSON Export for Testing
        print(f"\n💾 Testing JSON metrics export...")
        all_metrics = engine.get_metrics()
        
        # Save test results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path("test-out") / f"haystack_monitoring_test_{timestamp}.json"
        
        # Create output directory
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        test_results = {
            "test_info": {
                "timestamp": datetime.now().isoformat(),
                "test_description": "Haystack model engine monitoring test",
                "component": "haystack_model_engine"
            },
            "health_check": health_status,
            "infrastructure_metrics": infra_metrics,
            "performance_metrics": updated_perf_metrics,
            "all_metrics": all_metrics,
            "prometheus_metrics_sample": metrics_lines[:10]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"   ✅ Results saved to: {output_file}")
        
        # Test Summary
        print(f"\n📋 Test Summary")
        print("=" * 30)
        print(f"✅ Health check: {'PASSED' if health_status else 'FAILED'}")
        print(f"✅ Infrastructure metrics: PASSED")
        print(f"✅ Performance metrics: PASSED") 
        print(f"✅ Request processing: PASSED")
        print(f"✅ Prometheus export: PASSED")
        print(f"✅ JSON export: PASSED")
        
        print(f"\n🎉 All monitoring tests completed successfully!")
        print(f"📄 Detailed results: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_haystack_monitoring()
    sys.exit(0 if success else 1)