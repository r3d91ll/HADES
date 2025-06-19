#!/usr/bin/env python3
"""
Quick Win #1 Isolated Test: Improved Graph Connectivity

Test ONLY the connectivity improvements with same dataset size as original run.
Compare connectivity settings in isolation.

Original Settings:
- similarity_threshold: 0.7
- max_edges_per_node: 10
- Result: 79,987 nodes, 630,851 edges, density 0.0002, 45.8% inductive

New Settings:
- similarity_threshold: 0.5  
- max_edges_per_node: 20
- Expected: Similar nodes, MORE edges, HIGHER density, ?% inductive
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


def collect_100_files():
    """Collect ~100 files similar to original run."""
    
    input_files = []
    
    # Get Python files from HADES codebase (like original)
    python_files = list(Path("src").rglob("*.py"))
    
    # Get PDFs (like original) 
    pdf_files = list(Path(".").glob("**/*.pdf"))
    
    # Aim for ~100 total files (similar to original successful run)
    target_total = 100
    
    # Balance: more Python files, some PDFs
    python_count = min(60, len(python_files))  # Up to 60 Python files
    pdf_count = min(40, len(pdf_files))        # Up to 40 PDFs
    
    selected_python = python_files[:python_count]
    selected_pdfs = pdf_files[:pdf_count]
    
    input_files.extend([str(f) for f in selected_python])
    input_files.extend([str(f) for f in selected_pdfs])
    
    return input_files


def run_quick_win_1_test():
    """Test improved connectivity with same dataset size."""
    
    print("🧪 Quick Win #1: Improved Graph Connectivity Test")
    print("="*60)
    print("📋 Testing ONLY connectivity improvements:")
    print("   • similarity_threshold: 0.7 → 0.5")
    print("   • max_edges_per_node: 10 → 20")
    print("   • Same dataset size: ~100 files")
    print("   • Same training: 25 epochs")
    print()
    print("🎯 Goal: Isolate connectivity effect on inductive performance")
    print("   Original: 45.8% inductive performance")
    print("   Target: >60% (significant improvement)")
    print("="*60)
    
    # Set up paths
    output_dir = Path("output/quick_win_1_connectivity")
    model_name = "quick_win_1_connectivity_test"
    
    # Collect same scale dataset  
    input_files = collect_100_files()
    
    print(f"📄 Dataset: {len(input_files)} files")
    print(f"   Python files: {len([f for f in input_files if f.endswith('.py')])}")
    print(f"   PDF files: {len([f for f in input_files if f.endswith('.pdf')])}")
    print(f"📂 Output: {output_dir}")
    
    # Create config with ONLY connectivity improvements
    config = BootstrapConfig.get_default()
    config.input_dir = "."
    config.output_dir = str(output_dir)
    config.pipeline_name = "quick_win_1_test"
    
    # Apply ONLY Quick Win #1 changes
    config.graph_construction.similarity_threshold = 0.5  # Changed from 0.7
    config.graph_construction.max_edges_per_node = 20     # Changed from 10
    config.isne_training.epochs = 25                      # Same as before
    config.wandb.enabled = False                          # Keep simple for now
    
    print(f"\n⚙️  Configuration:")
    print(f"   similarity_threshold: {config.graph_construction.similarity_threshold}")
    print(f"   max_edges_per_node: {config.graph_construction.max_edges_per_node}")
    print(f"   epochs: {config.isne_training.epochs}")
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create monitor
    monitor = BootstrapMonitor(
        pipeline_name=config.pipeline_name,
        output_dir=output_dir,
        enable_alerts=False
    )
    
    # Run pipeline
    pipeline = ISNEBootstrapPipeline(config, monitor)
    
    print(f"\n🚀 Starting Quick Win #1 test...")
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
            print("✅ Quick Win #1 test completed successfully!")
            
            # Analyze results
            analyze_quick_win_1_results(result.output_directory)
            
            return True
            
        else:
            print(f"❌ Quick Win #1 test failed: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def analyze_quick_win_1_results(output_dir: str):
    """Analyze Quick Win #1 results and compare to baseline."""
    
    print(f"\n📊 Quick Win #1 Results Analysis")
    print("="*50)
    
    output_path = Path(output_dir)
    
    # Check for evaluation results
    eval_dir = output_path / "evaluation_results"
    results_file = eval_dir / "evaluation_results.json"
    
    if results_file.exists():
        import json
        with open(results_file) as f:
            eval_results = json.load(f)
        
        print("🎯 Connectivity Improvements:")
        
        # Model/graph info
        if 'model_info' in eval_results:
            info = eval_results['model_info']
            nodes = info.get('num_nodes', 0)
            edges = info.get('num_edges', 0)
            density = info.get('graph_density', 0)
            
            print(f"   📈 Graph: {nodes:,} nodes, {edges:,} edges")
            print(f"   🕸️  Connectivity: {(2*edges)/nodes:.1f} avg degree, {density:.6f} density")
            
            # Compare to baseline
            print(f"\n📊 Comparison to Original:")
            print(f"   Original: 79,987 nodes, 630,851 edges")
            print(f"   Original: 15.8 avg degree, 0.0002 density")
            
            if nodes > 0:
                new_avg_degree = (2 * edges) / nodes
                print(f"   New: {new_avg_degree:.1f} avg degree, {density:.6f} density")
                
                if density > 0.0002:
                    improvement = density / 0.0002
                    print(f"   🎉 Density improvement: {improvement:.1f}x")
        
        # Inductive performance
        if 'inductive_performance' in eval_results:
            inductive = eval_results['inductive_performance']
            if 'relative_performance_percent' in inductive:
                perf = inductive['relative_performance_percent']
                target = inductive.get('achieves_90_percent_target', False)
                
                print(f"\n🎯 Inductive Performance Results:")
                print(f"   Quick Win #1: {perf:.2f}%")
                print(f"   Original: 45.8%")
                print(f"   Target: >60% (significant improvement)")
                print(f"   Status: {'✅ IMPROVEMENT' if perf > 45.8 else '❌ NO IMPROVEMENT'}")
                
                if perf > 45.8:
                    improvement = perf - 45.8
                    print(f"   📈 Improvement: +{improvement:.1f}%")
                
                # Assessment
                print(f"\n📋 Quick Win #1 Assessment:")
                if perf > 60:
                    print("   🎉 SIGNIFICANT IMPROVEMENT - Connectivity hypothesis confirmed!")
                elif perf > 50:
                    print("   📈 MODERATE IMPROVEMENT - Connectivity helps but not enough")
                elif perf > 45.8:
                    print("   📈 SMALL IMPROVEMENT - Connectivity has minor effect")
                else:
                    print("   ❌ NO IMPROVEMENT - Connectivity not the main factor")
    
    else:
        print("⚠️  Evaluation results not found")
        print("   Check if evaluation stage completed successfully")


if __name__ == "__main__":
    success = run_quick_win_1_test()
    
    print(f"\n{'='*60}")
    print(f"QUICK WIN #1 TEST: {'✅ SUCCESS' if success else '❌ FAILED'}")
    print(f"{'='*60}")
    
    if success:
        print("🔬 Isolated connectivity effect measured!")
        print("📊 Ready for Quick Win #2: Full Dataset Scale Test")
    
    sys.exit(0 if success else 1)