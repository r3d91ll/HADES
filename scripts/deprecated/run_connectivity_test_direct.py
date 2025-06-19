#!/usr/bin/env python3
"""
Direct Connectivity Test

Run bootstrap pipeline directly to test graph connectivity hypothesis.
"""

import sys
import time
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.isne.bootstrap.pipeline import ISNEBootstrapPipeline
from src.isne.bootstrap.config import BootstrapConfig
from src.isne.bootstrap.monitoring import BootstrapMonitor


def run_connectivity_test():
    """Run bootstrap with improved connectivity settings."""
    
    print("🧪 Testing Graph Connectivity Hypothesis")
    print("="*60)
    print("📝 Testing changes:")
    print("   • similarity_threshold: 0.7 → 0.5")
    print("   • max_edges_per_node: 10 → 20")
    print("   • epochs: 50 → 25")
    print("="*60)
    
    # Set up paths
    input_dir = Path(".")
    output_dir = Path("output/connectivity_test")
    model_name = "connectivity_hypothesis_test"
    
    # Create config with improved connectivity
    config = BootstrapConfig.get_default()
    config.input_dir = str(input_dir)
    config.output_dir = str(output_dir) 
    config.pipeline_name = "connectivity_test"
    
    # Apply our hypothesis changes
    config.graph_construction.similarity_threshold = 0.5  # Lower threshold
    config.graph_construction.max_edges_per_node = 20     # More edges
    config.isne_training.epochs = 25                      # Faster testing
    config.wandb.enabled = False                          # No W&B overhead
    
    print(f"📂 Input: {input_dir}")
    print(f"📁 Output: {output_dir}")
    print(f"🏷️  Model: {model_name}")
    
    # Collect input files (limit for testing)
    input_files = []
    
    # Get Python files from src/
    python_files = list(Path("src").rglob("*.py"))[:25]  # Limit to 25 files
    input_files.extend([str(f) for f in python_files])
    
    # Get some PDFs if available
    pdf_files = list(Path(".").glob("**/*.pdf"))[:15]  # Limit to 15 PDFs
    input_files.extend([str(f) for f in pdf_files])
    
    print(f"📄 Selected {len(input_files)} files for testing")
    print(f"   Python files: {len(python_files)}")
    print(f"   PDF files: {len(pdf_files)}")
    
    if len(input_files) < 10:
        print("⚠️  Warning: Very few input files found")
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create monitor
    monitor = BootstrapMonitor(
        pipeline_name=config.pipeline_name,
        output_dir=output_dir,
        enable_alerts=False  # Disable alerts for testing
    )
    
    # Create and run pipeline
    pipeline = ISNEBootstrapPipeline(config, monitor)
    
    print(f"\n🚀 Starting bootstrap pipeline...")
    print(f"⏱️  Started at: {time.strftime('%H:%M:%S')}")
    
    start_time = time.time()
    
    try:
        result = pipeline.run(
            input_files=input_files,
            output_dir=output_dir,
            model_name=model_name
        )
        
        duration = time.time() - start_time
        print(f"\n⏱️  Completed in: {duration/60:.1f} minutes")
        
        if result.success:
            print("✅ Bootstrap completed successfully!")
            print(f"📁 Results in: {result.output_directory}")
            
            # Check for evaluation results
            eval_dir = Path(result.output_directory) / "evaluation_results"
            results_file = eval_dir / "evaluation_results.json"
            
            if results_file.exists():
                import json
                with open(results_file) as f:
                    eval_results = json.load(f)
                
                print(f"\n📊 Quick Results Summary:")
                
                # Model info
                if 'model_info' in eval_results:
                    info = eval_results['model_info']
                    nodes = info.get('num_nodes', 0)
                    edges = info.get('num_edges', 0)
                    print(f"   📈 Graph: {nodes:,} nodes, {edges:,} edges")
                    
                    if nodes > 0 and edges > 0:
                        avg_degree = (2 * edges) / nodes
                        density = info.get('graph_density', 0)
                        print(f"   🕸️  Connectivity: {avg_degree:.1f} avg degree, {density:.6f} density")
                
                # Inductive performance
                if 'inductive_performance' in eval_results:
                    inductive = eval_results['inductive_performance']
                    if 'relative_performance_percent' in inductive:
                        perf = inductive['relative_performance_percent']
                        target = inductive.get('achieves_90_percent_target', False)
                        
                        print(f"\n🎯 Key Result - Inductive Performance:")
                        print(f"   Current: {perf:.2f}%")
                        print(f"   Previous: 45.8%")
                        print(f"   Target: >90%")
                        print(f"   Status: {'✅ ACHIEVED' if target else '❌ BELOW TARGET'}")
                        
                        if perf > 45.8:
                            improvement = perf - 45.8
                            print(f"   Improvement: +{improvement:.1f}% 📈")
                        
                        # Hypothesis validation
                        print(f"\n🧪 Hypothesis Validation:")
                        if perf > 60:
                            print("   ✅ Significant improvement achieved!")
                        elif perf > 45.8:
                            print("   📈 Some improvement - hypothesis partially validated")
                        else:
                            print("   ❌ No improvement - hypothesis not validated")
            
            return True
            
        else:
            print(f"❌ Bootstrap failed: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_connectivity_test()
    
    print(f"\n{'='*60}")
    print(f"CONNECTIVITY TEST: {'✅ SUCCESS' if success else '❌ FAILED'}")
    print(f"{'='*60}")
    
    if success:
        print("Next steps:")
        print("1. Run: python scripts/analyze_graph_connectivity.py")
        print("2. Compare with previous results")
        print("3. Decide on full dataset retraining")
    
    sys.exit(0 if success else 1)