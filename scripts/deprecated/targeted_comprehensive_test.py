#!/usr/bin/env python3
"""
Targeted Comprehensive Quick Wins Test

More focused dataset collection to avoid system-wide PDF scanning issues.
Focuses on project-relevant PDFs and validates content before training.
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


def collect_targeted_dataset():
    """Collect targeted dataset from project-relevant locations only."""
    
    input_files = []
    
    # Get Python files from HADES codebase (validated working)
    python_files = list(Path("src").rglob("*.py"))
    script_files = list(Path("scripts").glob("*.py")) if Path("scripts").exists() else []
    
    all_python = python_files + script_files
    
    # More targeted PDF search - only project-relevant locations
    pdf_files = []
    
    # Project-focused PDF locations (avoiding system-wide search)
    project_pdf_locations = [
        Path("."),  # Current HADES directory only
        Path("../ladon"),  # ladon project if exists
        Path("../limnos"),  # limnos project if exists
    ]
    
    print("🔍 Searching for PDFs in project areas:")
    for location in project_pdf_locations:
        if location.exists():
            found_pdfs = list(location.glob("*.pdf"))  # Non-recursive for control
            if found_pdfs:
                print(f"   📁 {location}: {len(found_pdfs)} PDFs found")
                pdf_files.extend(found_pdfs)
            else:
                print(f"   📁 {location}: No PDFs")
        else:
            print(f"   📁 {location}: Directory not found")
    
    # If we need more PDFs, do limited recursive search in current dir only
    if len(pdf_files) < 30:
        print("   🔍 Expanding search in current directory...")
        current_pdfs = list(Path(".").rglob("*.pdf"))
        # Limit to reasonable number to avoid problematic files
        additional_pdfs = current_pdfs[:50]  # Max 50 additional
        pdf_files.extend(additional_pdfs)
        print(f"   📁 .: {len(additional_pdfs)} additional PDFs found")
    
    # Remove duplicates
    pdf_files = list(set(pdf_files))
    all_python = list(set(all_python))
    
    # Balanced dataset targeting ~120 files
    python_count = min(80, len(all_python))  # Up to 80 Python files
    pdf_count = min(40, len(pdf_files))      # Up to 40 PDFs (more manageable)
    
    selected_python = all_python[:python_count]
    selected_pdfs = pdf_files[:pdf_count]
    
    input_files.extend([str(f) for f in selected_python])
    input_files.extend([str(f) for f in selected_pdfs])
    
    print(f"\n📊 Targeted Dataset Summary:")
    print(f"   Python files available: {len(all_python)}")
    print(f"   PDF files available: {len(pdf_files)}")
    print(f"   Selected Python files: {len(selected_python)}")
    print(f"   Selected PDF files: {len(selected_pdfs)}")
    print(f"   Total selected: {len(input_files)}")
    
    return input_files


def run_targeted_test():
    """Run comprehensive test with targeted dataset collection."""
    
    print("🎯 TARGETED COMPREHENSIVE QUICK WINS TEST")
    print("="*70)
    print("🚀 Testing ALL quick wins with focused dataset:")
    print("   ✅ Quick Win #1: Improved Connectivity (threshold=0.5, max_edges=20)")
    print("   ✅ Quick Win #2: Targeted Dataset Scale (project-focused)")
    print("   ✅ Quick Win #3: Research Evaluation (enhanced metrics)")
    print()
    print("🎯 Goal: Comprehensive test with validated dataset")
    print("   Target: >70% inductive performance")
    print("   Focus: Project-relevant PDFs + HADES codebase")
    print("="*70)
    
    # Set up paths
    output_dir = Path("output/targeted_comprehensive")
    model_name = "targeted_comprehensive_model"
    
    # Collect targeted dataset
    input_files = collect_targeted_dataset()
    
    print(f"\n📄 Final Dataset: {len(input_files)} files")
    print(f"   Python files: {len([f for f in input_files if f.endswith('.py')])}")
    print(f"   PDF files: {len([f for f in input_files if f.endswith('.pdf')])}")
    print(f"📂 Output: {output_dir}")
    
    if len(input_files) < 50:
        print(f"⚠️  Warning: Only {len(input_files)} files found")
        print("   Continuing with available files...")
    
    # Create config with ALL quick wins
    config = BootstrapConfig.get_default()
    config.input_dir = "."
    config.output_dir = str(output_dir)
    config.pipeline_name = "targeted_comprehensive"
    
    # Apply ALL Quick Win improvements
    
    # Quick Win #1: Improved Connectivity
    config.graph_construction.similarity_threshold = 0.5  # Improved from 0.7
    config.graph_construction.max_edges_per_node = 20     # Improved from 10
    
    # Quick Win #3: Enhanced evaluation
    config.model_evaluation.test_ratio = 0.3  # Better test/train split
    config.model_evaluation.num_test_samples = 1000  # More test samples
    
    # Training configuration
    config.isne_training.epochs = 25                      # Reasonable training time
    config.isne_training.learning_rate = 0.001            # Stable learning rate
    config.wandb.enabled = False                          # Disable W&B to avoid issues
    
    # Add validation to handle empty documents
    config.document_processing.skip_empty_documents = True
    config.chunking.min_chunk_length = 10  # Skip very short chunks
    
    print(f"\n⚙️  Configuration:")
    print(f"   similarity_threshold: {config.graph_construction.similarity_threshold}")
    print(f"   max_edges_per_node: {config.graph_construction.max_edges_per_node}")
    print(f"   epochs: {config.isne_training.epochs}")
    print(f"   learning_rate: {config.isne_training.learning_rate}")
    print(f"   test_ratio: {config.model_evaluation.test_ratio}")
    print(f"   skip_empty_docs: True")
    print(f"   input_files: {len(input_files)}")
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create monitor with alerts enabled
    monitor = BootstrapMonitor(
        pipeline_name=config.pipeline_name,
        output_dir=output_dir,
        enable_alerts=True
    )
    
    # Run pipeline
    pipeline = ISNEBootstrapPipeline(config, monitor)
    
    print(f"\n🚀 Starting targeted comprehensive test...")
    print(f"⏱️  Started at: {time.strftime('%H:%M:%S')}")
    print(f"💡 Using validated, project-focused dataset")
    
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
            print("✅ Targeted comprehensive test completed successfully!")
            
            # Analyze results
            analyze_targeted_results(result.output_directory, len(input_files))
            
            return True
            
        else:
            print(f"❌ Targeted comprehensive test failed: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def analyze_targeted_results(output_dir: str, num_input_files: int):
    """Analyze targeted results with focus on performance improvements."""
    
    print(f"\n📊 TARGETED COMPREHENSIVE RESULTS")
    print("="*60)
    
    output_path = Path(output_dir)
    
    # Check for evaluation results
    eval_dir = output_path / "evaluation_results"
    results_file = eval_dir / "evaluation_results.json"
    
    if results_file.exists():
        import json
        with open(results_file) as f:
            eval_results = json.load(f)
        
        print("🎯 Targeted Quick Wins Results:")
        
        # Model/graph info
        if 'model_info' in eval_results:
            info = eval_results['model_info']
            nodes = info.get('num_nodes', 0)
            edges = info.get('num_edges', 0)
            
            print(f"   📈 Graph: {nodes:,} nodes, {edges:,} edges")
            print(f"   📁 Input files: {num_input_files}")
            print(f"   🕸️  Connectivity: {(2*edges)/nodes:.1f} avg degree")
            
            # Compare to previous results
            print(f"\n📊 Performance Comparison:")
            print(f"   Original (345 PDFs): 79,987 nodes, 45.8% inductive")
            print(f"   Quick Win #1: 3,411 nodes, 45.9% inductive")
            print(f"   Targeted Test: {nodes:,} nodes, ?% inductive")
        
        # Inductive performance
        if 'inductive_performance' in eval_results:
            inductive = eval_results['inductive_performance']
            if 'relative_performance_percent' in inductive:
                perf = inductive['relative_performance_percent']
                target = inductive.get('achieves_90_percent_target', False)
                
                print(f"\n🎯 INDUCTIVE PERFORMANCE:")
                print(f"   Targeted Test: {perf:.2f}%")
                print(f"   Original: 45.8%")
                print(f"   Quick Win #1: 45.9%")
                print(f"   Target: >70%")
                
                # Calculate improvement
                improvement = perf - 45.8
                print(f"\n📈 Performance Analysis:")
                if improvement > 0:
                    print(f"   Improvement: +{improvement:.1f}%")
                else:
                    print(f"   Change: {improvement:.1f}%")
                
                # Assessment
                print(f"\n🏆 ASSESSMENT:")
                if perf >= 90:
                    print("   🎉 EXCELLENT: Meets ISNE paper target!")
                elif perf >= 70:
                    print("   ✅ SUCCESS: Substantial improvement achieved!")
                elif perf >= 60:
                    print("   📈 GOOD: Moderate improvement, on right track")
                elif perf > 46:
                    print("   📈 PROGRESS: Small improvement, investigate further")
                else:
                    print("   ❌ LIMITED: Need different approach")
        
        # Research evaluation if available
        if 'research_specific' in eval_results:
            research = eval_results['research_specific']
            print(f"\n🔬 Research Metrics:")
            
            if 'content_analysis' in research:
                content = research['content_analysis']
                academic_ratio = content.get('academic_content_ratio', 0)
                print(f"   📚 Academic content: {academic_ratio:.2f}")
            
            if 'cross_domain_connectivity' in research:
                cross_domain = research['cross_domain_connectivity']
                cross_ratio = cross_domain.get('cross_domain_ratio', 0)
                print(f"   🔗 Cross-domain: {cross_ratio:.2f}")
    
    else:
        print("⚠️  Evaluation results not found")


if __name__ == "__main__":
    print("🔬 TARGETED COMPREHENSIVE TEST")
    print("Focused dataset with validation to avoid system-wide issues")
    print()
    
    success = run_targeted_test()
    
    print(f"\n{'='*70}")
    print(f"TARGETED TEST: {'✅ SUCCESS' if success else '❌ FAILED'}")
    print(f"{'='*70}")
    
    if success:
        print("🎉 Targeted test with all quick wins completed!")
        print("📊 Check results for performance assessment")
    else:
        print("❌ Test failed - may need further adjustments")
    
    sys.exit(0 if success else 1)