#!/usr/bin/env python3
"""
Test-Data3 10% Sample Test with Hybrid Document Processing

Uses the correct processor for each file type:
- Core processor for Python files (preserves AST analysis for code chunking)
- Docling processor for PDFs (proper text extraction)
- Core processor for other text files
"""

import sys
import time
import logging
import random
from pathlib import Path
from typing import List, Dict, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.isne.bootstrap.pipeline import ISNEBootstrapPipeline
from src.isne.bootstrap.config import BootstrapConfig
from src.isne.bootstrap.monitoring import BootstrapMonitor
from src.isne.bootstrap.stages.hybrid_document_processing import HybridDocumentProcessingStage
from src.isne.bootstrap.stages.chunking import ChunkingStage
from src.isne.bootstrap.stages.embedding import EmbeddingStage
from src.isne.bootstrap.stages.graph_construction import GraphConstructionStage
from src.isne.bootstrap.stages.isne_training import ISNETrainingStage
from src.isne.bootstrap.stages.model_evaluation import ModelEvaluationStage


class HybridISNEBootstrapPipeline(ISNEBootstrapPipeline):
    """Enhanced pipeline using hybrid document processing."""
    
    def __init__(self, config: BootstrapConfig, monitor: Optional[BootstrapMonitor] = None):
        """Initialize hybrid bootstrap pipeline with hybrid document processing."""
        # Call parent init first
        super().__init__(config, monitor)
        
        # Override the document processing stage with hybrid version
        self.stages['document_processing'] = HybridDocumentProcessingStage()
        
        logging.getLogger(__name__).info("Hybrid ISNE Bootstrap Pipeline initialized with hybrid document processing")


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
            # Skip very large files that might cause processing issues
            # BUT allow conversation files even if large (they're valuable training data)
            try:
                file_size = file_path.stat().st_size
                is_conversation = 'conversation' in file_path.name.lower() or 'chat' in file_path.name.lower()
                
                if file_size > 50 * 1024 * 1024 and not is_conversation:  # Skip files > 50MB unless conversations
                    print(f"   Skipping large file: {file_path.name} ({file_size / (1024*1024):.1f}MB)")
                    continue
                elif is_conversation and file_size > 50 * 1024 * 1024:
                    print(f"   Including large conversation file: {file_path.name} ({file_size / (1024*1024):.1f}MB)")
            except OSError:
                continue
                
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


def run_10_percent_hybrid_test():
    """Run test with 10% sample using hybrid document processing."""
    
    print("🎯 TEST-DATA3 10% SAMPLE TEST (HYBRID PROCESSING)")
    print("="*70)
    print("🚀 Testing with intelligent file routing:")
    print("   ✅ Core processor for Python files → AST analysis → smart chunking")
    print("   ✅ Docling processor for PDFs → proper text extraction")
    print("   ✅ Core processor for text files → efficient processing")
    print("   ✅ Quick Win #1: Improved Connectivity")
    print("   ✅ Quick Win #2: Diverse file types")
    print("   ✅ Quick Win #3: Research Evaluation")
    print()
    print("🎯 Goal: Best of both worlds - AST for code, extraction for PDFs")
    print("   Sample: 10% of each file type")
    print("   Target: >60% inductive performance")
    print("="*70)
    
    # Set up paths
    output_dir = Path("output/test_data3_10percent_hybrid")
    model_name = "test_data3_10percent_hybrid_model"
    
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
    config.pipeline_name = "test_data3_10percent_hybrid"
    
    # No need to specify processor_type - hybrid stage handles routing
    
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
    config.isne_training.device = "cuda"  # Force GPU training for speed
    
    # Validation settings
    config.chunking.min_chunk_length = 20
    config.chunking.max_chunk_length = 1000
    
    # Disable W&B for faster testing
    config.wandb.enabled = False
    
    print(f"\n⚙️  Test Configuration:")
    print(f"   🔧 PROCESSING: Hybrid (Core for Python/AST, Docling for PDFs)")
    print(f"   similarity_threshold: {config.graph_construction.similarity_threshold}")
    print(f"   max_edges_per_node: {config.graph_construction.max_edges_per_node}")
    print(f"   epochs: {config.isne_training.epochs}")
    print(f"   learning_rate: {config.isne_training.learning_rate}")
    print(f"   device: {config.isne_training.device} 🚀")
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
    
    # Use hybrid pipeline
    pipeline = HybridISNEBootstrapPipeline(config, monitor)
    
    print(f"\n🚀 Starting 10% sample test with hybrid processing...")
    print(f"⏱️  Started at: {time.strftime('%H:%M:%S')}")
    print(f"🐍 Python files → Core processor → AST analysis")
    print(f"📄 PDF files → Docling processor → Text extraction")
    
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
            print("✅ 10% hybrid test completed successfully!")
            
            # Analyze results
            analyze_hybrid_results(result.output_directory, len(input_files))
            
            return True
            
        else:
            print(f"❌ 10% hybrid test failed: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def analyze_hybrid_results(output_dir: str, num_input_files: int):
    """Analyze 10% hybrid processing results."""
    
    print(f"\n📊 10% HYBRID PROCESSING RESULTS")
    print("="*60)
    
    output_path = Path(output_dir)
    
    # Check for evaluation results
    eval_dir = output_path / "evaluation_results"
    results_file = eval_dir / "evaluation_results.json"
    
    if results_file.exists():
        import json
        with open(results_file) as f:
            eval_results = json.load(f)
        
        print("🎯 10% Hybrid Results:")
        
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
                
                print(f"\n🎯 10% HYBRID PERFORMANCE:")
                print(f"   Performance: {perf:.2f}%")
                print(f"   Baseline: 45.8%")
                print(f"   Improvement: {perf - 45.8:+.1f}%")
                
                # Validation assessment
                print(f"\n📋 APPROACH VALIDATION:")
                if perf >= 60:
                    print("   ✅ APPROACH VALIDATED!")
                    print("   ✅ Hybrid processing works well")
                    print("   ✅ AST for Python + PDF extraction effective")
                    print(f"\n🚀 RECOMMENDATION: Run full 100% training with hybrid")
                elif perf >= 50:
                    print("   📈 PROMISING RESULTS")
                    print("   📈 Hybrid approach shows improvement")
                    print("   🔧 Consider tuning parameters")
                else:
                    print("   📊 LIMITED IMPROVEMENT")
                    print("   🔬 Check processing quality")
        
        # Document processing quality
        print(f"\n📋 Hybrid Processing Quality:")
        if 'document_processing_stats' in eval_results:
            dp_stats = eval_results['document_processing_stats']
            
            # Check processor distribution
            files_by_proc = dp_stats.get('files_by_processor', {})
            if files_by_proc:
                print(f"   Processor distribution:")
                print(f"      Core (Python/text): {files_by_proc.get('core', 0)} files")
                print(f"      Docling (PDFs): {files_by_proc.get('docling', 0)} files")
            
            # Check Python AST processing
            file_details = dp_stats.get('file_details', [])
            python_files = [f for f in file_details if f.get('has_ast', False)]
            if python_files:
                print(f"\n   🐍 Python AST Processing:")
                print(f"      Files with AST: {len(python_files)}")
                print(f"      AST-aware chunking enabled")
                
                # Show some Python files
                for pf in python_files[:3]:
                    print(f"      • {pf['filename']}: {pf['characters']:,} chars")
            
            # Check PDF processing
            pdf_files = [f for f in file_details if f.get('format') == '.pdf']
            if pdf_files:
                pdf_chars = sum(f.get('characters', 0) for f in pdf_files)
                print(f"\n   📄 PDF Processing:")
                print(f"      PDFs processed: {len(pdf_files)}")
                print(f"      Total PDF text: {pdf_chars:,} chars")
                print(f"      Avg chars/PDF: {pdf_chars//len(pdf_files) if pdf_files else 0:,}")
        
        # Research metrics
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
                domain_dist = cross_domain.get('domain_distribution', {})
                print(f"   🔗 Cross-domain: {cross_ratio:.2f}")
                if domain_dist:
                    print(f"   📊 Domains: {', '.join(f'{k}:{v}' for k,v in domain_dist.items() if v > 0)}")
        
        # Next steps
        print(f"\n{'='*60}")
        print(f"NEXT STEPS")
        print(f"{'='*60}")
        
        if 'inductive_performance' in eval_results:
            perf = eval_results['inductive_performance'].get('relative_performance_percent', 0)
            
            if perf >= 60:
                print("✅ 10% hybrid test successful!")
                print("🚀 Create comprehensive test with hybrid processing")
                print("📋 Benefits confirmed:")
                print("   • Python files get AST analysis")
                print("   • PDFs get proper text extraction")
                print("   • Best performance for RAG")
            elif perf >= 50:
                print("📈 Hybrid approach shows promise")
                print("🔧 Consider parameter tuning")
            else:
                print("🔬 Need to investigate further")
    
    else:
        print("⚠️  Evaluation results not found")


if __name__ == "__main__":
    print("🔬 TEST-DATA3 10% HYBRID PROCESSING TEST")
    print("Intelligent file routing for optimal processing:")
    print("• Python → Core (AST) → Smart code chunking")
    print("• PDFs → Docling → Proper text extraction")
    print()
    
    success = run_10_percent_hybrid_test()
    
    print(f"\n{'='*70}")
    print(f"10% HYBRID TEST: {'✅ SUCCESS' if success else '❌ FAILED'}")
    print(f"{'='*70}")
    
    if success:
        print("🎉 Hybrid processing test completed!")
        print("📊 Check results for AST and PDF quality")
        print("🚀 Ready for full dataset with hybrid approach")
    else:
        print("❌ Test failed - check logs")
        print("🔧 May need to debug processor initialization")
    
    sys.exit(0 if success else 1)