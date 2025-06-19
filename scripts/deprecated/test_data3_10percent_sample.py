#!/usr/bin/env python3
"""
Test-Data3 10% Sample Test

Uses a 10% representative sample from test-data3 dataset for efficient testing.
Samples across all supported file types to maintain diversity.

Benefits:
- Faster iteration and testing
- Representative of full dataset
- Sufficient to validate improvements
- Can scale to 100% once validated
"""

import sys
import time
import logging
import random
from pathlib import Path
from typing import List, Dict

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.isne.bootstrap.pipeline import ISNEBootstrapPipeline
from src.isne.bootstrap.config import BootstrapConfig
from src.isne.bootstrap.monitoring import BootstrapMonitor


def collect_10_percent_sample():
    """Collect 10% sample from test-data3 across all file types."""
    
    test_data3_path = Path("/home/todd/ML-Lab/Olympus/test-data3")
    
    if not test_data3_path.exists():
        raise FileNotFoundError(f"test-data3 directory not found: {test_data3_path}")
    
    # Define supported file extensions and their categories
    file_categories = {
        'pdf': ['.pdf'],
        'python': ['.py'],
        'markdown': ['.md'],
        'yaml': ['.yaml', '.yml'],
        'json': ['.json'],
        'text': ['.txt'],
        'jupyter': ['.ipynb']
    }
    
    # Collect all files by category
    files_by_category: Dict[str, List[Path]] = {cat: [] for cat in file_categories}
    
    print("🔍 Scanning test-data3 for all supported files...")
    
    # Scan all files in test-data3 recursively
    all_files = list(test_data3_path.rglob("*"))
    
    for file_path in all_files:
        if file_path.is_file():
            suffix = file_path.suffix.lower()
            for category, extensions in file_categories.items():
                if suffix in extensions:
                    files_by_category[category].append(file_path)
                    break
    
    # Show what we found
    print(f"\n📊 Files Found in test-data3:")
    total_files = 0
    for category, files in files_by_category.items():
        if files:
            print(f"   {category.upper()}: {len(files)} files")
            total_files += len(files)
    print(f"   TOTAL: {total_files} files")
    
    # Sample 10% from each category
    sampled_files = []
    
    print(f"\n📊 Sampling 10% from each category:")
    for category, files in files_by_category.items():
        if files:
            # Calculate 10% (minimum 1 file if category has any)
            sample_size = max(1, int(len(files) * 0.1))
            
            # Random sample for diversity
            sampled = random.sample(files, sample_size)
            sampled_files.extend(sampled)
            
            print(f"   {category.upper()}: {sample_size} files (10% of {len(files)})")
    
    # Convert to string paths
    input_files = [str(f) for f in sampled_files]
    
    print(f"\n📊 Final 10% Sample:")
    print(f"   Total files selected: {len(input_files)}")
    print(f"   Sampling ratio: {len(input_files)/total_files*100:.1f}%")
    
    # Show breakdown by repository for Python files
    print(f"\n📦 Python Files by Repository:")
    repo_names = ['HADES', 'PathRAG', 'chonky', 'docling', 'ladon', 'inductive-shallow-node-embedding']
    for repo in repo_names:
        repo_files = [f for f in sampled_files if repo in str(f) and f.suffix == '.py']
        if repo_files:
            print(f"   {repo}: {len(repo_files)} files")
    
    return input_files


def run_10_percent_test():
    """Run test with 10% sample from test-data3."""
    
    print("🎯 TEST-DATA3 10% SAMPLE TEST")
    print("="*70)
    print("🚀 Testing with representative 10% sample:")
    print("   ✅ Quick Win #1: Improved Connectivity (threshold=0.5, max_edges=20)")
    print("   ✅ Quick Win #2: Diverse file types (PDFs, Python, Markdown, etc.)")
    print("   ✅ Quick Win #3: Research Evaluation (enhanced metrics)")
    print()
    print("🎯 Goal: Validate approach before full-scale training")
    print("   Sample: 10% of each file type for balanced representation")
    print("   Target: >60% inductive performance to validate approach")
    print("="*70)
    
    # Set up paths
    output_dir = Path("output/test_data3_10percent")
    model_name = "test_data3_10percent_model"
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Collect 10% sample
    try:
        input_files = collect_10_percent_sample()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return False
    
    print(f"\n📄 Sample Dataset: {len(input_files)} files")
    print(f"   PDF files: {len([f for f in input_files if f.endswith('.pdf')])}")
    print(f"   Python files: {len([f for f in input_files if f.endswith('.py')])}")
    print(f"   Markdown files: {len([f for f in input_files if f.endswith('.md')])}")
    print(f"   Other files: {len([f for f in input_files if not any(f.endswith(ext) for ext in ['.pdf', '.py', '.md'])])}")
    print(f"📂 Output: {output_dir}")
    
    # Create config with all quick wins
    config = BootstrapConfig.get_default()
    config.input_dir = str(Path("/home/todd/ML-Lab/Olympus/test-data3"))
    config.output_dir = str(output_dir)
    config.pipeline_name = "test_data3_10percent"
    
    # Apply quick win improvements
    
    # Quick Win #1: Improved Connectivity
    config.graph_construction.similarity_threshold = 0.5
    config.graph_construction.max_edges_per_node = 20
    
    # Quick Win #3: Enhanced evaluation
    config.model_evaluation.test_ratio = 0.3
    config.model_evaluation.num_test_samples = 500  # Scaled for smaller dataset
    
    # Faster training for 10% sample
    config.isne_training.epochs = 20  # Fewer epochs for faster iteration
    config.isne_training.learning_rate = 0.001
    config.isne_training.batch_size = 32
    
    # Validation settings
    config.document_processing.skip_empty_documents = True
    config.chunking.min_chunk_length = 20
    config.chunking.max_chunk_length = 1000
    
    # Disable W&B for faster testing
    config.wandb.enabled = False
    
    print(f"\n⚙️  Test Configuration:")
    print(f"   similarity_threshold: {config.graph_construction.similarity_threshold}")
    print(f"   max_edges_per_node: {config.graph_construction.max_edges_per_node}")
    print(f"   epochs: {config.isne_training.epochs}")
    print(f"   learning_rate: {config.isne_training.learning_rate}")
    print(f"   test_samples: {config.model_evaluation.num_test_samples}")
    print(f"   input_files: {len(input_files)}")
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create monitor
    monitor = BootstrapMonitor(
        pipeline_name=config.pipeline_name,
        output_dir=output_dir,
        enable_alerts=True
    )
    
    # Run pipeline
    pipeline = ISNEBootstrapPipeline(config, monitor)
    
    print(f"\n🚀 Starting 10% sample test...")
    print(f"⏱️  Started at: {time.strftime('%H:%M:%S')}")
    print(f"💡 This should complete much faster than full dataset")
    
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
            print("✅ 10% sample test completed successfully!")
            
            # Analyze results
            analyze_sample_results(result.output_directory, len(input_files))
            
            return True
            
        else:
            print(f"❌ 10% sample test failed: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def analyze_sample_results(output_dir: str, num_input_files: int):
    """Analyze 10% sample results and project full dataset performance."""
    
    print(f"\n📊 10% SAMPLE RESULTS ANALYSIS")
    print("="*60)
    
    output_path = Path(output_dir)
    
    # Check for evaluation results
    eval_dir = output_path / "evaluation_results"
    results_file = eval_dir / "evaluation_results.json"
    
    if results_file.exists():
        import json
        with open(results_file) as f:
            eval_results = json.load(f)
        
        print("🎯 10% Sample Results:")
        
        # Model/graph info
        if 'model_info' in eval_results:
            info = eval_results['model_info']
            nodes = info.get('num_nodes', 0)
            edges = info.get('num_edges', 0)
            
            print(f"   📈 Graph: {nodes:,} nodes, {edges:,} edges")
            print(f"   📁 Sample size: {num_input_files} files (10%)")
            print(f"   🕸️  Connectivity: {(2*edges)/nodes:.1f} avg degree")
            
            # Project full dataset scale
            projected_nodes = nodes * 10
            projected_edges = edges * 10
            print(f"\n📊 Projected Full Dataset Scale:")
            print(f"   Projected nodes: ~{projected_nodes:,}")
            print(f"   Projected edges: ~{projected_edges:,}")
        
        # Inductive performance
        if 'inductive_performance' in eval_results:
            inductive = eval_results['inductive_performance']
            if 'relative_performance_percent' in inductive:
                perf = inductive['relative_performance_percent']
                
                print(f"\n🎯 10% SAMPLE PERFORMANCE:")
                print(f"   Sample Performance: {perf:.2f}%")
                print(f"   Baseline: 45.8%")
                print(f"   Improvement: {perf - 45.8:+.1f}%")
                
                # Validation assessment
                print(f"\n📋 APPROACH VALIDATION:")
                if perf >= 60:
                    print("   ✅ APPROACH VALIDATED!")
                    print("   ✅ Quick wins show clear improvement")
                    print("   ✅ Ready for full dataset training")
                    print(f"\n🚀 RECOMMENDATION: Run full 100% training")
                elif perf >= 50:
                    print("   📈 PROMISING RESULTS")
                    print("   📈 Moderate improvement observed")
                    print("   🔧 Consider parameter tuning before full run")
                elif perf > 45.8:
                    print("   📊 SMALL IMPROVEMENT")
                    print("   📊 Some gains but not substantial")
                    print("   🔬 Investigate additional optimizations")
                else:
                    print("   ❌ NO IMPROVEMENT")
                    print("   ❌ Approach needs rethinking")
                    print("   🔧 Debug issues before scaling up")
        
        # Research metrics preview
        if 'research_specific' in eval_results:
            research = eval_results['research_specific']
            print(f"\n🔬 Research Metrics Preview:")
            
            if 'content_analysis' in research:
                content = research['content_analysis']
                academic_ratio = content.get('academic_content_ratio', 0)
                print(f"   📚 Academic content: {academic_ratio:.2f}")
            
            if 'cross_domain_connectivity' in research:
                cross_domain = research['cross_domain_connectivity']
                cross_ratio = cross_domain.get('cross_domain_ratio', 0)
                print(f"   🔗 Cross-domain: {cross_ratio:.2f}")
        
        # Next steps
        print(f"\n{'='*60}")
        print(f"NEXT STEPS")
        print(f"{'='*60}")
        
        if 'inductive_performance' in eval_results:
            perf = eval_results['inductive_performance'].get('relative_performance_percent', 0)
            
            if perf >= 60:
                print("✅ 10% sample shows strong improvement!")
                print("🚀 Run full 100% training:")
                print("   python scripts/test_data3_comprehensive_test.py")
            elif perf >= 50:
                print("📈 10% sample shows promise")
                print("🔧 Consider tuning before full run:")
                print("   - Adjust similarity threshold")
                print("   - Increase training epochs")
                print("   - Then run full training")
            else:
                print("🔬 10% sample needs investigation")
                print("📊 Debug with smaller tests first")
    
    else:
        print("⚠️  Evaluation results not found")


if __name__ == "__main__":
    print("🔬 TEST-DATA3 10% SAMPLE TEST")
    print("Efficient validation using representative sample")
    print("Testing all quick wins before full-scale training")
    print()
    
    success = run_10_percent_test()
    
    print(f"\n{'='*70}")
    print(f"10% SAMPLE TEST: {'✅ SUCCESS' if success else '❌ FAILED'}")
    print(f"{'='*70}")
    
    if success:
        print("🎉 10% sample test completed!")
        print("📊 Check results to validate approach")
        print("🚀 If results are good (>60%), run full training")
    else:
        print("❌ Sample test failed - check logs")
        print("🔧 Fix issues before attempting full training")
    
    sys.exit(0 if success else 1)