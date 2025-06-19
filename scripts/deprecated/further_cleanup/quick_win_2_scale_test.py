#!/usr/bin/env python3
"""
Quick Win #2: Full Dataset Scale Test

Test with larger dataset using improved connectivity settings from Quick Win #1.
This isolates the effect of dataset scale on ISNE performance.

Quick Win #1 Results:
- 3,411 nodes with improved connectivity (0.5 threshold, 20 max_edges)
- Achieved 45.9% inductive performance (same as original)
- Connectivity improvements confirmed but not sufficient alone

Quick Win #2 Goal:
- Use 100+ PDFs + Python files (larger scale than Quick Win #1)
- Keep improved connectivity settings (0.5 threshold, 20 max_edges)
- Test if larger dataset improves inductive performance
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


def collect_large_dataset():
    """Collect larger dataset: 100+ PDFs and Python files."""
    
    input_files = []
    
    # Get all Python files from HADES codebase
    python_files = list(Path("src").rglob("*.py"))
    
    # Get all PDFs from available sources
    pdf_files = []
    
    # Look in common PDF locations
    pdf_locations = [
        Path("."),  # Current directory
        Path("../"),  # Parent directory
        Path("../../"),  # Grandparent directory
        Path("/home/todd/Documents"),  # User documents
        Path("/home/todd/Downloads"),  # Downloads folder
    ]
    
    for location in pdf_locations:
        if location.exists():
            found_pdfs = list(location.rglob("*.pdf"))
            pdf_files.extend(found_pdfs)
    
    # Remove duplicates
    pdf_files = list(set(pdf_files))
    
    # Target: 100+ files total with good balance
    target_total = 120
    
    # Use more files than Quick Win #1
    python_count = min(80, len(python_files))  # Up to 80 Python files  
    pdf_count = min(40, len(pdf_files))        # Up to 40 PDFs
    
    selected_python = python_files[:python_count]
    selected_pdfs = pdf_files[:pdf_count]
    
    input_files.extend([str(f) for f in selected_python])
    input_files.extend([str(f) for f in selected_pdfs])
    
    return input_files


def run_quick_win_2_test():
    """Test larger dataset scale with improved connectivity."""
    
    print("🧪 Quick Win #2: Full Dataset Scale Test")
    print("="*60)
    print("📋 Testing dataset scale effect:")
    print("   • Keep Quick Win #1 connectivity: threshold=0.5, max_edges=20")
    print("   • Increase dataset: ~120 files (vs ~100 in Quick Win #1)")
    print("   • Same training: 25 epochs")
    print("   • Research evaluation enabled")
    print()
    print("🎯 Goal: Isolate dataset scale effect on inductive performance")
    print("   Quick Win #1: 45.9% with 3,411 nodes")
    print("   Target: >60% with larger graph")
    print("="*60)
    
    # Set up paths
    output_dir = Path("output/quick_win_2_scale")
    model_name = "quick_win_2_scale_test"
    
    # Collect larger dataset
    input_files = collect_large_dataset()
    
    print(f"📄 Dataset: {len(input_files)} files")
    print(f"   Python files: {len([f for f in input_files if f.endswith('.py')])}")
    print(f"   PDF files: {len([f for f in input_files if f.endswith('.pdf')])}")
    print(f"📂 Output: {output_dir}")
    
    if len(input_files) < 100:
        print(f"⚠️  Warning: Only {len(input_files)} files found (target: 100+)")
        print("   Continuing with available files...")
    
    # Create config with Quick Win #1 + Quick Win #2 improvements
    config = BootstrapConfig.get_default()
    config.input_dir = "."
    config.output_dir = str(output_dir)
    config.pipeline_name = "quick_win_2_test"
    
    # Apply Quick Win #1 improvements (keep these)
    config.graph_construction.similarity_threshold = 0.5  # From Quick Win #1
    config.graph_construction.max_edges_per_node = 20     # From Quick Win #1
    
    # Quick Win #2: Scale improvements
    config.isne_training.epochs = 25                      # Same training
    config.wandb.enabled = False                          # Keep simple
    
    print(f"\n⚙️  Configuration:")
    print(f"   similarity_threshold: {config.graph_construction.similarity_threshold}")
    print(f"   max_edges_per_node: {config.graph_construction.max_edges_per_node}")
    print(f"   epochs: {config.isne_training.epochs}")
    print(f"   input_files: {len(input_files)}")
    
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
    
    print(f"\n🚀 Starting Quick Win #2 test...")
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
            print("✅ Quick Win #2 test completed successfully!")
            
            # Analyze results
            analyze_quick_win_2_results(result.output_directory)
            
            return True
            
        else:
            print(f"❌ Quick Win #2 test failed: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def analyze_quick_win_2_results(output_dir: str):
    """Analyze Quick Win #2 results and compare to baseline and Quick Win #1."""
    
    print(f"\n📊 Quick Win #2 Results Analysis")
    print("="*50)
    
    output_path = Path(output_dir)
    
    # Check for evaluation results
    eval_dir = output_path / "evaluation_results"
    results_file = eval_dir / "evaluation_results.json"
    
    if results_file.exists():
        import json
        with open(results_file) as f:
            eval_results = json.load(f)
        
        print("🎯 Dataset Scale Impact:")
        
        # Model/graph info
        if 'model_info' in eval_results:
            info = eval_results['model_info']
            nodes = info.get('num_nodes', 0)
            edges = info.get('num_edges', 0)
            density = info.get('graph_density', 0)
            
            print(f"   📈 Graph: {nodes:,} nodes, {edges:,} edges")
            print(f"   🕸️  Connectivity: {(2*edges)/nodes:.1f} avg degree, {density:.6f} density")
            
            # Compare to baselines
            print(f"\n📊 Scale Comparison:")
            print(f"   Original (100 files): 79,987 nodes, 630,851 edges")
            print(f"   Quick Win #1 (100 files): 3,411 nodes")
            print(f"   Quick Win #2 ({len(eval_results.get('input_files', []))} files): {nodes:,} nodes")
            
            if nodes > 3411:
                scale_improvement = nodes / 3411
                print(f"   🎉 Scale improvement: {scale_improvement:.1f}x nodes vs Quick Win #1")
        
        # Inductive performance
        if 'inductive_performance' in eval_results:
            inductive = eval_results['inductive_performance']
            if 'relative_performance_percent' in inductive:
                perf = inductive['relative_performance_percent']
                target = inductive.get('achieves_90_percent_target', False)
                
                print(f"\n🎯 Inductive Performance Results:")
                print(f"   Quick Win #2: {perf:.2f}%")
                print(f"   Quick Win #1: 45.9%")
                print(f"   Original: 45.8%")
                print(f"   Target: >60% (significant improvement)")
                
                # Compare to Quick Win #1
                if perf > 45.9:
                    improvement = perf - 45.9
                    print(f"   📈 Improvement over Quick Win #1: +{improvement:.1f}%")
                    status = "✅ SCALE HELPS"
                else:
                    decline = 45.9 - perf
                    print(f"   📉 Change from Quick Win #1: -{decline:.1f}%")
                    status = "❌ SCALE DOESN'T HELP"
                
                print(f"   Status: {status}")
                
                # Assessment
                print(f"\n📋 Quick Win #2 Assessment:")
                if perf > 60:
                    print("   🎉 SIGNIFICANT IMPROVEMENT - Scale + connectivity works!")
                elif perf > 50:
                    print("   📈 MODERATE IMPROVEMENT - Scale helps but not enough")
                elif perf > 45.9:
                    print("   📈 SMALL IMPROVEMENT - Scale has minor positive effect")
                else:
                    print("   ❌ NO SCALE BENEFIT - Need different approach")
        
        # Research-specific results (if available)
        if 'research_specific' in eval_results:
            research = eval_results['research_specific']
            print(f"\n🔬 Research Evaluation Results:")
            
            if 'content_analysis' in research:
                content = research['content_analysis']
                academic_ratio = content.get('academic_content_ratio', 0)
                print(f"   📚 Academic content: {academic_ratio:.2f} ratio")
            
            if 'cross_domain_connectivity' in research:
                cross_domain = research['cross_domain_connectivity']
                cross_ratio = cross_domain.get('cross_domain_ratio', 0)
                print(f"   🔗 Cross-domain connections: {cross_ratio:.2f} ratio")
            
            if 'concept_bridging' in research:
                bridging = research['concept_bridging']
                bridges = bridging.get('total_concept_bridges', 0)
                print(f"   🌉 Concept bridges: {bridges} total")
    
    else:
        print("⚠️  Evaluation results not found")
        print("   Check if evaluation stage completed successfully")


if __name__ == "__main__":
    success = run_quick_win_2_test()
    
    print(f"\n{'='*60}")
    print(f"QUICK WIN #2 TEST: {'✅ SUCCESS' if success else '❌ FAILED'}")
    print(f"{'='*60}")
    
    if success:
        print("🔬 Dataset scale effect measured!")
        print("📊 Ready for comprehensive test with all Quick Wins!")
        print("\n🎯 Next: Run full test with 100+ PDFs and Python files")
    
    sys.exit(0 if success else 1)