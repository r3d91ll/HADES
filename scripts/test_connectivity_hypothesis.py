#!/usr/bin/env python3
"""
Test Graph Connectivity Hypothesis

Quick bootstrap run with improved graph connectivity settings to test
if lower similarity threshold and higher edge density improve inductive performance.

Key Changes:
- similarity_threshold: 0.7 -> 0.5 (more connections)
- max_edges_per_node: 10 -> 20 (richer neighborhoods)
- epochs: 50 -> 25 (faster iteration)
- W&B disabled (faster startup)
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.isne.bootstrap.cli import main as bootstrap_main


def run_connectivity_test():
    """Run bootstrap with improved connectivity settings."""
    
    print("🧪 Testing Graph Connectivity Hypothesis")
    print("="*60)
    print("📝 Changes from previous run:")
    print("   • similarity_threshold: 0.7 → 0.5 (lower threshold = more edges)")
    print("   • max_edges_per_node: 10 → 20 (richer neighborhoods)")
    print("   • epochs: 50 → 25 (faster testing)")
    print("   • W&B: disabled (faster startup)")
    print()
    print("🎯 Hypothesis: Better graph connectivity will improve inductive performance")
    print("   Target: >90% relative performance on unseen nodes")
    print("   Previous: 45.8% (with sparse graph)")
    print("="*60)
    
    # Use a focused subset for faster testing
    input_dir = "."  # Current directory - will pick up ladon and HADES files
    output_dir = "output/connectivity_test"
    model_name = "connectivity_test_model"
    
    print(f"📂 Input: {input_dir}")
    print(f"📁 Output: {output_dir}")
    print(f"🏷️  Model: {model_name}")
    print()
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set up command line args for bootstrap
    sys.argv = [
        "bootstrap_cli.py",
        "--input-dir", input_dir,
        "--output-dir", output_dir,
        "--model-name", model_name,
        "--file-limit", "50",  # Limit to 50 files for faster testing
        "--config-override", "graph_construction.similarity_threshold=0.5",
        "--config-override", "graph_construction.max_edges_per_node=20",
        "--config-override", "isne_training.epochs=25",
        "--config-override", "wandb.enabled=false"
    ]
    
    print("🚀 Starting bootstrap pipeline...")
    print(f"⏱️  Started at: {time.strftime('%H:%M:%S')}")
    
    start_time = time.time()
    
    try:
        # Run bootstrap
        result = bootstrap_main()
        
        duration = time.time() - start_time
        print(f"\n⏱️  Completed in: {duration/60:.1f} minutes")
        
        if result and result.success:
            print("✅ Bootstrap completed successfully!")
            
            # Check if evaluation results are available
            eval_dir = Path(output_dir) / "evaluation_results"
            if eval_dir.exists():
                results_file = eval_dir / "evaluation_results.json"
                if results_file.exists():
                    import json
                    with open(results_file) as f:
                        eval_results = json.load(f)
                    
                    print(f"\n📊 Evaluation Results:")
                    
                    # Model info
                    if 'model_info' in eval_results:
                        info = eval_results['model_info']
                        print(f"   Nodes: {info.get('num_nodes', 0):,}")
                        print(f"   Edges: {info.get('num_edges', 0):,}")
                        if 'graph_density' in info:
                            density = info['graph_density']
                            print(f"   Graph density: {density:.6f}")
                    
                    # Key metric: inductive performance
                    if 'inductive_performance' in eval_results:
                        inductive = eval_results['inductive_performance']
                        if 'relative_performance_percent' in inductive:
                            perf = inductive['relative_performance_percent']
                            target = inductive.get('achieves_90_percent_target', False)
                            
                            print(f"\n🎯 Inductive Performance Results:")
                            print(f"   Current run: {perf:.2f}%")
                            print(f"   Previous run: 45.8%")
                            print(f"   Target: >90%")
                            print(f"   Achieves target: {'✅ YES' if target else '❌ NO'}")
                            
                            # Calculate improvement
                            if perf > 45.8:
                                improvement = perf - 45.8
                                print(f"   Improvement: +{improvement:.1f}% 📈")
                            else:
                                decline = 45.8 - perf
                                print(f"   Change: -{decline:.1f}% 📉")
                    
                    # Graph connectivity analysis
                    print(f"\n🕸️  Graph Connectivity Analysis:")
                    if 'model_info' in eval_results:
                        info = eval_results['model_info']
                        nodes = info.get('num_nodes', 0)
                        edges = info.get('num_edges', 0)
                        if nodes > 0:
                            avg_degree = (2 * edges) / nodes
                            print(f"   Average degree: {avg_degree:.2f}")
                            print(f"   Density: {info.get('graph_density', 0):.6f}")
                            
                            # Compare with previous run
                            print(f"   Previous avg degree: ~15.8")
                            print(f"   Previous density: 0.0002")
                
                else:
                    print("⚠️  Evaluation results file not found")
            else:
                print("⚠️  Evaluation directory not found")
            
            print(f"\n📁 Full results in: {output_dir}")
            return True
            
        else:
            print("❌ Bootstrap failed")
            if result:
                print(f"   Error: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_hypothesis_summary():
    """Print summary of what we're testing."""
    print(f"\n{'='*60}")
    print(f"CONNECTIVITY HYPOTHESIS TEST SUMMARY")
    print(f"{'='*60}")
    print(f"📋 What we're testing:")
    print(f"   • Lower similarity threshold creates more semantic connections")
    print(f"   • Higher edge density provides richer neighborhood context") 
    print(f"   • Better graph connectivity improves inductive performance")
    print()
    print(f"📊 Expected outcomes:")
    print(f"   • Higher graph density (current: 0.0002)")
    print(f"   • More edges per node (current: ~15.8)")
    print(f"   • Better inductive performance (current: 45.8%)")
    print()
    print(f"🎯 Success criteria:")
    print(f"   • Inductive performance >60% (significant improvement)")
    print(f"   • Ideally approaching 90% target from ISNE paper")
    print(f"   • Model still maintains good embedding quality")
    print(f"{'='*60}")


if __name__ == "__main__":
    print_hypothesis_summary()
    
    print("\n🚀 Starting connectivity test automatically...")
    
    success = run_connectivity_test()
    
    print(f"\n{'='*60}")
    print(f"TEST COMPLETED: {'✅ SUCCESS' if success else '❌ FAILED'}")
    print(f"{'='*60}")
    
    sys.exit(0 if success else 1)