#!/usr/bin/env python3
"""
Comprehensive Quick Wins Test

Combines ALL quick wins and runs comprehensive test with 100+ PDFs and Python files.

Implemented Quick Wins:
✅ Quick Win #1: Improved Connectivity (threshold=0.5, max_edges=20)
✅ Quick Win #2: Full Dataset Scale Test (100+ files)
✅ Quick Win #3: Enhanced Research Evaluation (research-specific metrics)

This test runs the FULL combination with comprehensive dataset and research evaluation.
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


def collect_comprehensive_dataset():
    """Collect comprehensive dataset: 100+ PDFs and Python files."""
    
    input_files = []
    
    # Get all Python files from HADES codebase
    python_files = list(Path("src").rglob("*.py"))
    
    # Include additional Python files from scripts and other areas
    script_files = list(Path("scripts").glob("*.py")) if Path("scripts").exists() else []
    test_files = list(Path("test").rglob("*.py")) if Path("test").exists() else []
    
    all_python = python_files + script_files + test_files
    
    # Get all PDFs from comprehensive search
    pdf_files = []
    
    # Comprehensive PDF search locations
    pdf_locations = [
        Path("."),  # Current directory
        Path("../"),  # Parent directory
        Path("../../"),  # Grandparent directory
        Path("../../../"),  # Great-grandparent
        Path("/home/todd/Documents"),  # User documents
        Path("/home/todd/Downloads"),  # Downloads folder
        Path("/home/todd/Desktop"),  # Desktop
        Path("/home/todd/ML-Lab"),  # ML-Lab directory
        Path("/home/todd/ML-Lab/Olympus"),  # Olympus workspace
    ]
    
    print("🔍 Searching for PDFs in:")
    for location in pdf_locations:
        if location.exists():
            found_pdfs = list(location.rglob("*.pdf"))
            if found_pdfs:
                print(f"   📁 {location}: {len(found_pdfs)} PDFs found")
                pdf_files.extend(found_pdfs)
            else:
                print(f"   📁 {location}: No PDFs")
        else:
            print(f"   📁 {location}: Directory not found")
    
    # Remove duplicates
    pdf_files = list(set(pdf_files))
    all_python = list(set(all_python))
    
    # Target: Comprehensive dataset with good balance
    target_total = 150  # Aim higher than previous tests
    
    # Use substantial number of files
    python_count = min(100, len(all_python))  # Up to 100 Python files
    pdf_count = min(50, len(pdf_files))       # Up to 50 PDFs
    
    selected_python = all_python[:python_count]
    selected_pdfs = pdf_files[:pdf_count]
    
    input_files.extend([str(f) for f in selected_python])
    input_files.extend([str(f) for f in selected_pdfs])
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Total Python files available: {len(all_python)}")
    print(f"   Total PDF files available: {len(pdf_files)}")
    print(f"   Selected Python files: {len(selected_python)}")
    print(f"   Selected PDF files: {len(selected_pdfs)}")
    print(f"   Total selected: {len(input_files)}")
    
    return input_files


def run_comprehensive_test():
    """Run comprehensive test with all quick wins combined."""
    
    print("🎯 COMPREHENSIVE QUICK WINS TEST")
    print("="*70)
    print("🚀 Testing ALL quick wins combined:")
    print("   ✅ Quick Win #1: Improved Connectivity (threshold=0.5, max_edges=20)")
    print("   ✅ Quick Win #2: Full Dataset Scale (100+ files)")
    print("   ✅ Quick Win #3: Research Evaluation (enhanced metrics)")
    print()
    print("🎯 Goal: Comprehensive test with research-grade performance")
    print("   Target: >70% inductive performance (substantial improvement)")
    print("   Research evaluation: Cross-domain, methodology transfer, concept bridging")
    print("="*70)
    
    # Set up paths
    output_dir = Path("output/comprehensive_quick_wins")
    model_name = "comprehensive_quick_wins_model"
    
    # Collect comprehensive dataset
    input_files = collect_comprehensive_dataset()
    
    print(f"\n📄 Final Dataset: {len(input_files)} files")
    print(f"   Python files: {len([f for f in input_files if f.endswith('.py')])}")
    print(f"   PDF files: {len([f for f in input_files if f.endswith('.pdf')])}")
    print(f"📂 Output: {output_dir}")
    
    if len(input_files) < 100:
        print(f"⚠️  Warning: Only {len(input_files)} files found")
        print("   Continuing with available files...")
        if len(input_files) < 50:
            print("❌ Insufficient files for comprehensive test")
            return False
    
    # Create config with ALL quick wins
    config = BootstrapConfig.get_default()
    config.input_dir = "."
    config.output_dir = str(output_dir)
    config.pipeline_name = "comprehensive_quick_wins"
    
    # Apply ALL Quick Win improvements
    
    # Quick Win #1: Improved Connectivity
    config.graph_construction.similarity_threshold = 0.5  # Improved from 0.7
    config.graph_construction.max_edges_per_node = 20     # Improved from 10
    
    # Quick Win #2: Scale improvements (handled by dataset)
    
    # Quick Win #3: Enhanced evaluation (will use ResearchEvaluationStage)
    config.model_evaluation.test_ratio = 0.3  # Better test/train split
    config.model_evaluation.num_test_samples = 1000  # More test samples
    
    # Training configuration
    config.isne_training.epochs = 30                      # Slightly more training
    config.isne_training.learning_rate = 0.001            # Stable learning rate
    config.wandb.enabled = True                           # Enable W&B for tracking
    config.wandb.project = "hades-comprehensive-test"
    config.wandb.experiment_name = "quick_wins_combined"
    
    print(f"\n⚙️  Configuration:")
    print(f"   similarity_threshold: {config.graph_construction.similarity_threshold}")
    print(f"   max_edges_per_node: {config.graph_construction.max_edges_per_node}")
    print(f"   epochs: {config.isne_training.epochs}")
    print(f"   learning_rate: {config.isne_training.learning_rate}")
    print(f"   test_ratio: {config.model_evaluation.test_ratio}")
    print(f"   wandb_enabled: {config.wandb.enabled}")
    print(f"   input_files: {len(input_files)}")
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create monitor with alerts enabled
    monitor = BootstrapMonitor(
        pipeline_name=config.pipeline_name,
        output_dir=output_dir,
        enable_alerts=True  # Enable alerts for comprehensive test
    )
    
    # Run pipeline
    pipeline = ISNEBootstrapPipeline(config, monitor)
    
    print(f"\n🚀 Starting comprehensive test...")
    print(f"⏱️  Started at: {time.strftime('%H:%M:%S')}")
    print(f"💡 This will take longer due to larger dataset and enhanced evaluation")
    
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
            print("✅ Comprehensive test completed successfully!")
            
            # Analyze results
            analyze_comprehensive_results(result.output_directory, len(input_files))
            
            return True
            
        else:
            print(f"❌ Comprehensive test failed: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def analyze_comprehensive_results(output_dir: str, num_input_files: int):
    """Analyze comprehensive results and compare to all baselines."""
    
    print(f"\n📊 COMPREHENSIVE RESULTS ANALYSIS")
    print("="*60)
    
    output_path = Path(output_dir)
    
    # Check for evaluation results
    eval_dir = output_path / "evaluation_results"
    results_file = eval_dir / "evaluation_results.json"
    
    if results_file.exists():
        import json
        with open(results_file) as f:
            eval_results = json.load(f)
        
        print("🎯 Comprehensive Quick Wins Results:")
        
        # Model/graph info
        if 'model_info' in eval_results:
            info = eval_results['model_info']
            nodes = info.get('num_nodes', 0)
            edges = info.get('num_edges', 0)
            
            print(f"   📈 Final Graph: {nodes:,} nodes, {edges:,} edges")
            print(f"   📁 Input files: {num_input_files}")
            print(f"   🕸️  Connectivity: {(2*edges)/nodes:.1f} avg degree")
            
            # Compare to all previous results
            print(f"\n📊 Complete Performance Comparison:")
            print(f"   Original (345 PDFs): 79,987 nodes, 45.8% inductive")
            print(f"   Quick Win #1 (connectivity): 3,411 nodes, 45.9% inductive")
            print(f"   Comprehensive ({num_input_files} files): {nodes:,} nodes, ?% inductive")
        
        # Inductive performance
        if 'inductive_performance' in eval_results:
            inductive = eval_results['inductive_performance']
            if 'relative_performance_percent' in inductive:
                perf = inductive['relative_performance_percent']
                target = inductive.get('achieves_90_percent_target', False)
                
                print(f"\n🎯 FINAL INDUCTIVE PERFORMANCE:")
                print(f"   Comprehensive Test: {perf:.2f}%")
                print(f"   Original Baseline: 45.8%")
                print(f"   Quick Win #1: 45.9%")
                print(f"   Target: >70% (substantial improvement)")
                print(f"   ISNE Paper Target: >90%")
                
                # Calculate improvements
                original_improvement = perf - 45.8
                print(f"\n📈 Performance Analysis:")
                if original_improvement > 0:
                    print(f"   Improvement over original: +{original_improvement:.1f}%")
                else:
                    print(f"   Change from original: {original_improvement:.1f}%")
                
                # Final assessment
                print(f"\n🏆 FINAL ASSESSMENT:")
                if perf >= 90:
                    print("   🎉 EXCELLENT: Meets ISNE paper standards!")
                    status = "RESEARCH_GRADE"
                elif perf >= 70:
                    print("   ✅ SUBSTANTIAL IMPROVEMENT: Quick wins successful!")
                    status = "PRODUCTION_READY"
                elif perf >= 60:
                    print("   📈 MODERATE IMPROVEMENT: Progress made, more work needed")
                    status = "GOOD_PROGRESS"
                elif perf > 45.8:
                    print("   📈 SMALL IMPROVEMENT: Minor gains, investigate further")
                    status = "MINOR_GAINS"
                else:
                    print("   ❌ NO IMPROVEMENT: Need different approach")
                    status = "NEEDS_REWORK"
                
                print(f"   Status: {status}")
        
        # Research-specific evaluation results
        if 'research_specific' in eval_results:
            research = eval_results['research_specific']
            print(f"\n🔬 RESEARCH EVALUATION RESULTS:")
            
            # Content analysis
            if 'content_analysis' in research:
                content = research['content_analysis']
                academic_ratio = content.get('academic_content_ratio', 0)
                meets_threshold = content.get('meets_academic_threshold', False)
                avg_score = content.get('avg_academic_score', 0)
                
                print(f"   📚 Academic Content Analysis:")
                print(f"      Academic ratio: {academic_ratio:.2f}")
                print(f"      Meets threshold: {'✅' if meets_threshold else '❌'}")
                print(f"      Avg academic score: {avg_score:.1f}")
            
            # Cross-domain connectivity
            if 'cross_domain_connectivity' in research:
                cross_domain = research['cross_domain_connectivity']
                cross_ratio = cross_domain.get('cross_domain_ratio', 0)
                enables_discovery = cross_domain.get('enables_interdisciplinary_discovery', False)
                domain_dist = cross_domain.get('domain_distribution', {})
                
                print(f"   🔗 Cross-Domain Connectivity:")
                print(f"      Cross-domain ratio: {cross_ratio:.2f}")
                print(f"      Enables discovery: {'✅' if enables_discovery else '❌'}")
                print(f"      Domains found: {len(domain_dist)} types")
            
            # Methodology transfer
            if 'methodology_transfer' in research:
                method = research['methodology_transfer']
                if 'methodology_coherence' in method:
                    coherence = method.get('methodology_coherence', 0)
                    supports_transfer = method.get('supports_methodology_transfer', False)
                    quality = method.get('transfer_quality', 'unknown')
                    
                    print(f"   🔄 Methodology Transfer:")
                    print(f"      Coherence: {coherence:.2f}")
                    print(f"      Supports transfer: {'✅' if supports_transfer else '❌'}")
                    print(f"      Quality: {quality}")
            
            # Concept bridging
            if 'concept_bridging' in research:
                bridging = research['concept_bridging']
                bridges = bridging.get('total_concept_bridges', 0)
                enables_bridging = bridging.get('enables_concept_bridging', False)
                strongest = bridging.get('strongest_bridge', None)
                
                print(f"   🌉 Concept Bridging:")
                print(f"      Total bridges: {bridges}")
                print(f"      Enables bridging: {'✅' if enables_bridging else '❌'}")
                if strongest:
                    print(f"      Strongest bridge: {strongest[0]} ({strongest[1]} connections)")
            
            # Research assessment
            if 'research_capability_status' in eval_results.get('stats', {}):
                research_status = eval_results['stats']['research_capability_status']
                ready_for_research = eval_results['stats'].get('ready_for_research_applications', False)
                
                print(f"\n🎓 Overall Research Assessment:")
                print(f"   Capability status: {research_status}")
                print(f"   Ready for research: {'✅' if ready_for_research else '❌'}")
    
    else:
        print("⚠️  Evaluation results not found")
        print("   Check if evaluation stage completed successfully")
    
    # Check for W&B experiment link
    wandb_dir = output_path / "wandb"
    if wandb_dir.exists():
        print(f"\n📊 W&B Experiment Tracking:")
        print(f"   Directory: {wandb_dir}")
        print("   Check W&B dashboard for detailed metrics")


if __name__ == "__main__":
    print("🔬 COMPREHENSIVE QUICK WINS TEST")
    print("Testing all improvements combined with large dataset")
    print()
    
    success = run_comprehensive_test()
    
    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE TEST: {'✅ SUCCESS' if success else '❌ FAILED'}")
    print(f"{'='*70}")
    
    if success:
        print("🎉 All quick wins tested with comprehensive dataset!")
        print("📊 Check results for final performance assessment")
        print("🔬 Research evaluation completed")
        print("📈 Ready for production or next iteration")
    else:
        print("❌ Test failed - check logs for details")
        print("🔧 May need to adjust approach or investigate issues")
    
    sys.exit(0 if success else 1)