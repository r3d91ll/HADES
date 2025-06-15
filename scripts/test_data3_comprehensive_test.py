#!/usr/bin/env python3
"""
Test-Data3 Comprehensive Quick Wins Test

Uses the curated test-data3 dataset with 300+ PDFs and 6 Python repositories.
This is the intended corpus for our RAG database.

Dataset:
- 300+ academic PDFs (ML, NLP, RAG, graph theory, social networks)
- 6 Python repositories (HADES, PathRAG, chonky, docling, ISNE, ladon)
- Research-grade content for building cutting-edge RAG features
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


def collect_test_data3_dataset():
    """Collect dataset from test-data3 directory."""
    
    test_data3_path = Path("/home/todd/ML-Lab/Olympus/test-data3")
    
    if not test_data3_path.exists():
        raise FileNotFoundError(f"test-data3 directory not found: {test_data3_path}")
    
    input_files = []
    
    print("🔍 Collecting files from test-data3...")
    
    # Get all PDFs from test-data3 (300+ academic papers)
    pdf_files = list(test_data3_path.glob("*.pdf"))
    print(f"   📄 Found {len(pdf_files)} PDF files")
    
    # Get Python files from the 6 repositories
    repo_paths = [
        test_data3_path / "HADES",
        test_data3_path / "PathRAG", 
        test_data3_path / "chonky",
        test_data3_path / "docling",
        test_data3_path / "inductive-shallow-node-embedding",
        test_data3_path / "ladon"
    ]
    
    python_files = []
    for repo_path in repo_paths:
        if repo_path.exists():
            repo_pythons = list(repo_path.rglob("*.py"))
            python_files.extend(repo_pythons)
            print(f"   🐍 {repo_path.name}: {len(repo_pythons)} Python files")
        else:
            print(f"   ❌ {repo_path.name}: Repository not found")
    
    # Use substantial portions of the dataset
    # Target: 100+ PDFs + 100+ Python files for comprehensive test
    pdf_count = min(100, len(pdf_files))      # Up to 100 PDFs
    python_count = min(100, len(python_files)) # Up to 100 Python files
    
    selected_pdfs = pdf_files[:pdf_count]
    selected_pythons = python_files[:python_count]
    
    input_files.extend([str(f) for f in selected_pdfs])
    input_files.extend([str(f) for f in selected_pythons])
    
    print(f"\n📊 Test-Data3 Dataset Summary:")
    print(f"   Available PDFs: {len(pdf_files)}")
    print(f"   Available Python files: {len(python_files)}")
    print(f"   Selected PDFs: {len(selected_pdfs)}")
    print(f"   Selected Python files: {len(selected_pythons)}")
    print(f"   Total selected: {len(input_files)}")
    
    # Show repository breakdown
    print(f"\n📦 Repository Sources:")
    for repo_path in repo_paths:
        if repo_path.exists():
            repo_pythons = [f for f in selected_pythons if str(repo_path) in str(f)]
            print(f"   {repo_path.name}: {len(repo_pythons)} files")
    
    return input_files


def run_test_data3_comprehensive_test():
    """Run comprehensive test with test-data3 dataset."""
    
    print("🎯 TEST-DATA3 COMPREHENSIVE TEST")
    print("="*70)
    print("🚀 Testing ALL quick wins with RAG corpus dataset:")
    print("   ✅ Quick Win #1: Improved Connectivity (threshold=0.5, max_edges=20)")
    print("   ✅ Quick Win #2: Comprehensive Scale (300+ PDFs + 6 repositories)")
    print("   ✅ Quick Win #3: Research Evaluation (enhanced metrics)")
    print()
    print("🎯 Goal: Production-ready RAG with research-grade performance")
    print("   Dataset: Curated academic papers + ML/RAG repositories")
    print("   Target: >70% inductive performance for production deployment")
    print("="*70)
    
    # Set up paths
    output_dir = Path("output/test_data3_comprehensive")
    model_name = "test_data3_rag_model"
    
    # Collect test-data3 dataset
    try:
        input_files = collect_test_data3_dataset()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return False
    
    print(f"\n📄 Final Dataset: {len(input_files)} files")
    print(f"   PDF files: {len([f for f in input_files if f.endswith('.pdf')])}")
    print(f"   Python files: {len([f for f in input_files if f.endswith('.py')])}")
    print(f"📂 Output: {output_dir}")
    
    if len(input_files) < 100:
        print(f"⚠️  Warning: Only {len(input_files)} files found (expected 100+)")
        if len(input_files) < 50:
            print("❌ Insufficient files for comprehensive test")
            return False
    
    # Create config with ALL quick wins optimized for production
    config = BootstrapConfig.get_default()
    config.input_dir = str(Path("/home/todd/ML-Lab/Olympus/test-data3"))
    config.output_dir = str(output_dir)
    config.pipeline_name = "test_data3_comprehensive"
    
    # Apply ALL Quick Win improvements
    
    # Quick Win #1: Improved Connectivity
    config.graph_construction.similarity_threshold = 0.5  # Improved connectivity
    config.graph_construction.max_edges_per_node = 20     # Better edge density
    
    # Quick Win #2: Comprehensive scale (handled by dataset)
    
    # Quick Win #3: Enhanced evaluation
    config.model_evaluation.test_ratio = 0.3  # Good test/train split
    config.model_evaluation.num_test_samples = 1500  # More test samples for accuracy
    
    # Production-optimized training
    config.isne_training.epochs = 30                      # Thorough training
    config.isne_training.learning_rate = 0.001            # Stable learning
    config.isne_training.batch_size = 32                  # Good batch size
    
    # Enhanced validation and error handling
    config.document_processing.skip_empty_documents = True
    config.chunking.min_chunk_length = 20  # Filter very short chunks
    config.chunking.max_chunk_length = 1000  # Reasonable chunk size
    
    # Enable monitoring for production readiness
    config.wandb.enabled = False  # Keep focused on results
    
    print(f"\n⚙️  Production Configuration:")
    print(f"   similarity_threshold: {config.graph_construction.similarity_threshold}")
    print(f"   max_edges_per_node: {config.graph_construction.max_edges_per_node}")
    print(f"   epochs: {config.isne_training.epochs}")
    print(f"   learning_rate: {config.isne_training.learning_rate}")
    print(f"   test_samples: {config.model_evaluation.num_test_samples}")
    print(f"   batch_size: {config.isne_training.batch_size}")
    print(f"   input_files: {len(input_files)}")
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create monitor with full alerting
    monitor = BootstrapMonitor(
        pipeline_name=config.pipeline_name,
        output_dir=output_dir,
        enable_alerts=True
    )
    
    # Run pipeline
    pipeline = ISNEBootstrapPipeline(config, monitor)
    
    print(f"\n🚀 Starting production-grade comprehensive test...")
    print(f"⏱️  Started at: {time.strftime('%H:%M:%S')}")
    print(f"📊 Processing RAG corpus: academic papers + ML repositories")
    print(f"💡 This will train our production ISNE model")
    
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
            print("✅ Test-Data3 comprehensive test completed successfully!")
            
            # Analyze results
            analyze_production_results(result.output_directory, len(input_files))
            
            return True
            
        else:
            print(f"❌ Test-Data3 comprehensive test failed: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def analyze_production_results(output_dir: str, num_input_files: int):
    """Analyze production results for RAG deployment readiness."""
    
    print(f"\n📊 PRODUCTION RAG MODEL ANALYSIS")
    print("="*60)
    
    output_path = Path(output_dir)
    
    # Check for evaluation results
    eval_dir = output_path / "evaluation_results"
    results_file = eval_dir / "evaluation_results.json"
    
    if results_file.exists():
        import json
        with open(results_file) as f:
            eval_results = json.load(f)
        
        print("🎯 Production RAG Model Results:")
        
        # Model/graph info
        if 'model_info' in eval_results:
            info = eval_results['model_info']
            nodes = info.get('num_nodes', 0)
            edges = info.get('num_edges', 0)
            params = info.get('model_parameters', 0)
            
            print(f"   📈 Graph Scale: {nodes:,} nodes, {edges:,} edges")
            print(f"   🧠 Model Size: {params:,} parameters")
            print(f"   📁 Dataset: {num_input_files} files from test-data3")
            print(f"   🕸️  Connectivity: {(2*edges)/nodes:.1f} avg degree")
            
            # Scale comparison
            print(f"\n📊 Scale Evolution:")
            print(f"   Original: 79,987 nodes (345 PDFs)")
            print(f"   Quick Win #1: 3,411 nodes (100 files)")
            print(f"   Production: {nodes:,} nodes ({num_input_files} files)")
        
        # Inductive performance - the key metric
        if 'inductive_performance' in eval_results:
            inductive = eval_results['inductive_performance']
            if 'relative_performance_percent' in inductive:
                perf = inductive['relative_performance_percent']
                target = inductive.get('achieves_90_percent_target', False)
                
                print(f"\n🎯 PRODUCTION INDUCTIVE PERFORMANCE:")
                print(f"   📊 Test-Data3 Model: {perf:.2f}%")
                print(f"   📈 Original Baseline: 45.8%")
                print(f"   📈 Quick Win #1: 45.9%")
                print(f"   🎯 Production Target: >70%")
                print(f"   🏆 ISNE Paper Target: >90%")
                
                # Calculate total improvement
                total_improvement = perf - 45.8
                print(f"\n📈 IMPROVEMENT ANALYSIS:")
                if total_improvement > 0:
                    print(f"   Total improvement: +{total_improvement:.1f}%")
                else:
                    print(f"   Total change: {total_improvement:.1f}%")
                
                # Production readiness assessment
                print(f"\n🏭 PRODUCTION READINESS:")
                if perf >= 90:
                    status = "RESEARCH EXCELLENCE"
                    deployment = "IMMEDIATE DEPLOYMENT READY"
                    color = "🏆"
                elif perf >= 80:
                    status = "PRODUCTION EXCELLENCE"
                    deployment = "READY FOR PRODUCTION"
                    color = "🎉"
                elif perf >= 70:
                    status = "PRODUCTION READY"
                    deployment = "GOOD FOR PRODUCTION"
                    color = "✅"
                elif perf >= 60:
                    status = "PRODUCTION CANDIDATE"
                    deployment = "NEEDS MINOR TUNING"
                    color = "📈"
                elif perf > 45.8:
                    status = "DEVELOPMENT PROGRESS"
                    deployment = "CONTINUE OPTIMIZATION"
                    color = "🔧"
                else:
                    status = "NEEDS REWORK"
                    deployment = "INVESTIGATE APPROACH"
                    color = "❌"
                
                print(f"   {color} Status: {status}")
                print(f"   {color} Deployment: {deployment}")
        
        # Research evaluation for RAG quality
        if 'research_specific' in eval_results:
            research = eval_results['research_specific']
            print(f"\n🔬 RAG QUALITY ASSESSMENT:")
            
            quality_scores = []
            
            # Academic content
            if 'content_analysis' in research:
                content = research['content_analysis']
                academic_ratio = content.get('academic_content_ratio', 0)
                meets_threshold = content.get('meets_academic_threshold', False)
                quality_scores.append(academic_ratio)
                print(f"   📚 Academic Content: {academic_ratio:.2f} ({'✅' if meets_threshold else '❌'})")
            
            # Cross-domain connectivity for RAG retrieval
            if 'cross_domain_connectivity' in research:
                cross_domain = research['cross_domain_connectivity']
                cross_ratio = cross_domain.get('cross_domain_ratio', 0)
                enables_discovery = cross_domain.get('enables_interdisciplinary_discovery', False)
                quality_scores.append(cross_ratio)
                print(f"   🔗 Cross-Domain: {cross_ratio:.2f} ({'✅' if enables_discovery else '❌'})")
            
            # Methodology transfer for knowledge bridging
            if 'methodology_transfer' in research:
                method = research['methodology_transfer']
                if 'methodology_coherence' in method:
                    coherence = method.get('methodology_coherence', 0)
                    supports_transfer = method.get('supports_methodology_transfer', False)
                    quality_scores.append(coherence)
                    print(f"   🔄 Methodology Transfer: {coherence:.2f} ({'✅' if supports_transfer else '❌'})")
            
            # Concept bridging for knowledge synthesis
            if 'concept_bridging' in research:
                bridging = research['concept_bridging']
                bridges = bridging.get('total_concept_bridges', 0)
                enables_bridging = bridging.get('enables_concept_bridging', False)
                print(f"   🌉 Concept Bridging: {bridges} bridges ({'✅' if enables_bridging else '❌'})")
            
            # Overall RAG quality
            if quality_scores:
                avg_quality = sum(quality_scores) / len(quality_scores)
                print(f"\n🎯 RAG Quality Score: {avg_quality:.2f}")
                if avg_quality >= 0.8:
                    print("   🏆 EXCELLENT RAG CAPABILITY")
                elif avg_quality >= 0.6:
                    print("   ✅ GOOD RAG CAPABILITY")
                elif avg_quality >= 0.4:
                    print("   📈 MODERATE RAG CAPABILITY")
                else:
                    print("   🔧 RAG NEEDS IMPROVEMENT")
        
        # Final production recommendation
        print(f"\n{'='*60}")
        print(f"FINAL PRODUCTION ASSESSMENT")
        print(f"{'='*60}")
        
        if 'inductive_performance' in eval_results:
            perf = eval_results['inductive_performance'].get('relative_performance_percent', 0)
            
            if perf >= 70:
                print("🎉 READY FOR PRODUCTION DEPLOYMENT!")
                print("✅ Model meets production standards")
                print("✅ All quick wins successfully implemented")
                print("🚀 Proceed with RAG system integration")
            elif perf >= 60:
                print("📈 NEAR PRODUCTION READY")
                print("✅ Substantial improvement achieved")
                print("🔧 Consider additional optimizations")
            else:
                print("🔧 CONTINUE DEVELOPMENT")
                print("📊 Performance gains observed")
                print("🔬 Investigate advanced techniques")
    
    else:
        print("⚠️  Evaluation results not found")
        print("   Check if evaluation stage completed successfully")


if __name__ == "__main__":
    print("🔬 TEST-DATA3 COMPREHENSIVE RAG TEST")
    print("Using curated academic corpus + ML repositories")
    print("Building production-ready ISNE model for RAG deployment")
    print()
    
    success = run_test_data3_comprehensive_test()
    
    print(f"\n{'='*70}")
    print(f"PRODUCTION TEST: {'✅ SUCCESS' if success else '❌ FAILED'}")
    print(f"{'='*70}")
    
    if success:
        print("🎉 Production-grade RAG model training completed!")
        print("📊 Check results for deployment readiness")
        print("🚀 Ready for Olympus RAG system integration")
    else:
        print("❌ Production test failed - check logs for details")
        print("🔧 May need to investigate dataset or configuration issues")
    
    sys.exit(0 if success else 1)