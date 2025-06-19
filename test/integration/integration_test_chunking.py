#!/usr/bin/env python3
"""
HADES Chunking Integration Test

This script performs comprehensive integration testing of the chunking components
using real docproc output and generates performance metrics and chunked output
ready for embedding component input.

Output includes:
- Performance metrics and reports
- Chunked text for embedding component input
- Processing statistics and analysis
"""

import sys
import json
import time
import traceback
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# Add HADES to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def ensure_output_directory():
    """Ensure test-out directory exists."""
    output_dir = Path("test-out")
    output_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (output_dir / "chunks").mkdir(exist_ok=True)
    (output_dir / "chunking_reports").mkdir(exist_ok=True)
    (output_dir / "chunking_metrics").mkdir(exist_ok=True)
    
    return output_dir

def load_docproc_outputs(output_dir: Path):
    """Load processed documents from docproc integration test."""
    docs_dir = output_dir / "documents"
    
    if not docs_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")
    
    documents = []
    for doc_file in docs_dir.glob("*_core.json"):
        try:
            with open(doc_file, 'r', encoding='utf-8') as f:
                doc_data = json.load(f)
                documents.append({
                    "file_name": doc_file.name,
                    "document_id": doc_data.get("id", "unknown"),
                    "content": doc_data.get("content", ""),
                    "content_type": doc_data.get("content_type", "text/plain"),
                    "format": doc_data.get("format", "unknown"),
                    "content_category": doc_data.get("content_category", "text"),
                    "source_file": doc_data.get("source_file", "unknown"),
                    "processing_time": doc_data.get("processing_time", 0)
                })
        except Exception as e:
            print(f"Failed to load {doc_file}: {e}")
    
    print(f"📂 Loaded {len(documents)} documents from docproc output")
    for doc in documents:
        content_size = len(doc["content"])
        print(f"  - {doc['document_id']}: {content_size} chars ({doc['content_category']})")
    
    return documents

def test_cpu_chunker(documents: List[Dict], output_dir: Path):
    """Test CPU chunker with comprehensive metrics."""
    print("\n" + "="*80)
    print("🔧 TESTING CPU CHUNKER")
    print("="*80)
    
    try:
        from src.components.chunking.chunkers.cpu.processor import CPUChunker
        from src.types.components.contracts import ChunkingInput
        
        # Initialize chunker
        chunker = CPUChunker(config={
            "chunking_method": "sentence_aware",
            "chunk_size": 512,
            "chunk_overlap": 50,
            "preserve_sentence_boundaries": True,
            "language": "en"
        })
        
        print(f"✓ Initialized CPU chunker: {chunker.name} v{chunker.version}")
        
        # Track metrics
        start_time = datetime.now(timezone.utc)
        results = []
        errors = []
        performance_data = []
        total_chunks_created = 0
        
        # Process each document
        for i, doc in enumerate(documents, 1):
            print(f"\n--- Processing {i}/{len(documents)}: {doc['document_id']} ---")
            
            file_start_time = time.time()
            
            try:
                # Create chunking input
                chunking_input = ChunkingInput(
                    text=doc["content"],
                    document_id=doc["document_id"],
                    chunking_strategy="sentence_aware",
                    chunk_size=512,
                    chunk_overlap=50,
                    processing_options={
                        "chunker_type": "cpu",
                        "content_type": doc["content_category"]
                    },
                    metadata={
                        "source_file": doc["source_file"],
                        "content_type": doc["content_type"],
                        "format": doc["format"]
                    }
                )
                
                # Process with chunker
                result = chunker.chunk(chunking_input)
                
                processing_time = time.time() - file_start_time
                
                # Collect performance data
                perf_data = {
                    "document_id": doc["document_id"],
                    "content_length": len(doc["content"]),
                    "content_category": doc["content_category"],
                    "processing_time_seconds": processing_time,
                    "chunks_created": len(result.chunks),
                    "success": len(result.errors) == 0,
                    "errors": result.errors,
                    "throughput_chars_per_sec": len(doc["content"]) / max(processing_time, 0.001),
                    "avg_chunk_size": sum(len(chunk.text) for chunk in result.chunks) / max(len(result.chunks), 1),
                    "chunking_method": result.processing_stats.get("chunking_method", "unknown")
                }
                performance_data.append(perf_data)
                
                total_chunks_created += len(result.chunks)
                
                if result.errors:
                    print(f"⚠️  WARNING - {processing_time:.3f}s")
                    print(f"  Errors: {len(result.errors)}")
                    for error in result.errors:
                        print(f"    - {error}")
                    errors.extend(result.errors)
                else:
                    print(f"✓ SUCCESS - {processing_time:.3f}s")
                    print(f"  Chunks: {len(result.chunks)}")
                    print(f"  Avg chunk size: {perf_data['avg_chunk_size']:.1f} chars")
                    print(f"  Throughput: {perf_data['throughput_chars_per_sec']:.1f} chars/s")
                
                # Convert chunks to serializable format
                chunks_data = []
                for chunk in result.chunks:
                    if hasattr(chunk, 'model_dump'):
                        chunk_data = chunk.model_dump()
                    elif hasattr(chunk, 'dict'):
                        chunk_data = chunk.dict()
                    else:
                        chunk_data = {
                            "id": chunk.id,
                            "text": chunk.text,
                            "start_index": chunk.start_index,
                            "end_index": chunk.end_index,
                            "chunk_index": chunk.chunk_index,
                            "metadata": chunk.metadata
                        }
                    
                    # Add test-specific metadata
                    chunk_data.update({
                        "document_id": doc["document_id"],
                        "source_file": doc["source_file"],
                        "content_category": doc["content_category"],
                        "processed_at": datetime.now(timezone.utc).isoformat()
                    })
                    chunks_data.append(chunk_data)
                
                # Save chunks for this document
                chunks_file = output_dir / "chunks" / f"{doc['document_id']}_cpu_chunks.json"
                with open(chunks_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "document_id": doc["document_id"],
                        "source_file": doc["source_file"],
                        "chunking_method": "cpu_sentence_aware",
                        "total_chunks": len(chunks_data),
                        "chunks": chunks_data,
                        "processing_stats": result.processing_stats,
                        "metadata": result.metadata.model_dump() if hasattr(result.metadata, 'model_dump') else result.metadata
                    }, f, indent=2, ensure_ascii=False, cls=DateTimeEncoder)
                
                results.append({
                    "document_id": doc["document_id"],
                    "chunks_created": len(result.chunks),
                    "processing_time": processing_time,
                    "success": True
                })
                
            except Exception as e:
                processing_time = time.time() - file_start_time
                error_msg = str(e)
                
                print(f"✗ FAILED - {processing_time:.3f}s")
                print(f"  Error: {error_msg}")
                
                errors.append(error_msg)
                performance_data.append({
                    "document_id": doc["document_id"],
                    "content_length": len(doc["content"]),
                    "processing_time_seconds": processing_time,
                    "success": False,
                    "error": error_msg
                })
        
        # Generate comprehensive performance report
        total_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        total_content_chars = sum(len(doc["content"]) for doc in documents)
        successful_docs = sum(1 for p in performance_data if p.get("success", False))
        
        # Get chunker metrics
        infra_metrics = chunker.get_infrastructure_metrics()
        perf_metrics = chunker.get_performance_metrics()
        
        performance_report = {
            "test_summary": {
                "chunker": "cpu",
                "test_date": datetime.now(timezone.utc).isoformat(),
                "total_documents": len(documents),
                "successful_documents": successful_docs,
                "failed_documents": len(documents) - successful_docs,
                "success_rate_percent": (successful_docs / len(documents)) * 100 if documents else 0,
                "total_processing_time_seconds": total_time,
                "total_content_chars": total_content_chars,
                "total_chunks_created": total_chunks_created,
                "average_throughput_chars_per_sec": total_content_chars / max(total_time, 0.001),
                "average_chunks_per_document": total_chunks_created / max(successful_docs, 1)
            },
            "infrastructure_metrics": infra_metrics,
            "performance_metrics": perf_metrics,
            "document_performance": performance_data,
            "errors": errors,
            "chunker_configuration": chunker._config
        }
        
        # Save performance report
        report_file = output_dir / "chunking_reports" / "cpu_chunker_performance.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(performance_report, f, indent=2, ensure_ascii=False, cls=DateTimeEncoder)
        
        # Save metrics in Prometheus format
        prometheus_metrics = chunker.export_metrics_prometheus()
        metrics_file = output_dir / "chunking_metrics" / "cpu_chunker_metrics.txt"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            f.write(prometheus_metrics)
        
        print(f"\n📊 CPU CHUNKER SUMMARY:")
        print(f"  Documents processed: {successful_docs}/{len(documents)}")
        print(f"  Success rate: {(successful_docs / len(documents)) * 100:.1f}%")
        print(f"  Total chunks created: {total_chunks_created}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average throughput: {total_content_chars / max(total_time, 0.001):.1f} chars/s")
        print(f"  Average chunks per doc: {total_chunks_created / max(successful_docs, 1):.1f}")
        
        return {
            "results": results,
            "performance_report": performance_report,
            "success": successful_docs > 0,
            "total_chunks": total_chunks_created
        }
        
    except Exception as e:
        print(f"✗ CPU chunker test failed: {e}")
        traceback.print_exc()
        return {"results": [], "success": False, "error": str(e)}

def test_core_chunker(documents: List[Dict], output_dir: Path):
    """Test core chunker coordination with comprehensive metrics."""
    print("\n" + "="*80)
    print("🎯 TESTING CORE CHUNKER (COORDINATION)")
    print("="*80)
    
    try:
        from src.components.chunking.core.processor import CoreChunker
        from src.types.components.contracts import ChunkingInput
        
        # Initialize core chunker
        chunker = CoreChunker(config={
            "default_chunker": "cpu",
            "cpu_config": {
                "chunking_method": "adaptive",
                "chunk_size": 768,
                "chunk_overlap": 75
            }
        })
        
        print(f"✓ Initialized core chunker: {chunker.name} v{chunker.version}")
        print(f"✓ Default chunker: {chunker._default_chunker_type}")
        print(f"✓ Available chunkers: {chunker.get_supported_chunkers()}")
        
        # Track metrics
        start_time = datetime.now(timezone.utc)
        results = []
        errors = []
        performance_data = []
        total_chunks_created = 0
        
        # Process each document with different chunker types
        chunker_types = ["cpu", "text", "code", "ast"]  # Test different chunkers including AST
        
        for i, doc in enumerate(documents, 1):
            # Select chunker based on content type
            if doc["content_category"] == "code" and doc.get("format") == "python":
                chunker_type = "ast"  # Use AST chunker for Python code
            elif doc["content_category"] in ["code"]:
                chunker_type = "code"
            elif doc["content_category"] in ["text", "markdown"]:
                chunker_type = "text"
            else:
                chunker_type = "cpu"
            
            print(f"\n--- Processing {i}/{len(documents)}: {doc['document_id']} (using {chunker_type}) ---")
            
            file_start_time = time.time()
            
            try:
                # Create chunking input
                chunking_input = ChunkingInput(
                    text=doc["content"],
                    document_id=doc["document_id"],
                    chunking_strategy="adaptive",
                    chunk_size=768,
                    chunk_overlap=75,
                    processing_options={
                        "chunker_type": chunker_type,
                        "content_type": doc["content_category"]
                    },
                    metadata={
                        "source_file": doc["source_file"],
                        "content_type": doc["content_type"],
                        "format": doc["format"]
                    }
                )
                
                # Process with core chunker
                result = chunker.chunk(chunking_input)
                
                processing_time = time.time() - file_start_time
                
                # Collect performance data
                perf_data = {
                    "document_id": doc["document_id"],
                    "content_length": len(doc["content"]),
                    "content_category": doc["content_category"],
                    "chunker_used": chunker_type,
                    "processing_time_seconds": processing_time,
                    "chunks_created": len(result.chunks),
                    "success": len(result.errors) == 0,
                    "errors": result.errors,
                    "throughput_chars_per_sec": len(doc["content"]) / max(processing_time, 0.001),
                    "avg_chunk_size": sum(len(chunk.text) for chunk in result.chunks) / max(len(result.chunks), 1),
                    "delegated_to": result.processing_stats.get("delegated_to", "unknown"),
                    "core_processing_time": result.processing_stats.get("core_processing_time", 0)
                }
                performance_data.append(perf_data)
                
                total_chunks_created += len(result.chunks)
                
                if result.errors:
                    print(f"⚠️  WARNING - {processing_time:.3f}s")
                    print(f"  Errors: {len(result.errors)}")
                    errors.extend(result.errors)
                else:
                    print(f"✓ SUCCESS - {processing_time:.3f}s")
                    print(f"  Delegated to: {perf_data['delegated_to']}")
                    print(f"  Chunks: {len(result.chunks)}")
                    print(f"  Avg chunk size: {perf_data['avg_chunk_size']:.1f} chars")
                    print(f"  Core overhead: {perf_data['core_processing_time']:.3f}s")
                
                # Save chunks for this document
                chunks_data = []
                for chunk in result.chunks:
                    if hasattr(chunk, 'model_dump'):
                        chunk_data = chunk.model_dump()
                    elif hasattr(chunk, 'dict'):
                        chunk_data = chunk.dict()
                    else:
                        chunk_data = {
                            "id": chunk.id,
                            "text": chunk.text,
                            "start_index": chunk.start_index,
                            "end_index": chunk.end_index,
                            "chunk_index": chunk.chunk_index,
                            "metadata": chunk.metadata
                        }
                    
                    chunk_data.update({
                        "document_id": doc["document_id"],
                        "source_file": doc["source_file"],
                        "content_category": doc["content_category"],
                        "chunker_used": chunker_type,
                        "processed_at": datetime.now(timezone.utc).isoformat()
                    })
                    chunks_data.append(chunk_data)
                
                chunks_file = output_dir / "chunks" / f"{doc['document_id']}_core_chunks.json"
                with open(chunks_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "document_id": doc["document_id"],
                        "source_file": doc["source_file"],
                        "chunking_method": f"core_delegated_{chunker_type}",
                        "total_chunks": len(chunks_data),
                        "chunks": chunks_data,
                        "processing_stats": result.processing_stats,
                        "metadata": result.metadata.model_dump() if hasattr(result.metadata, 'model_dump') else result.metadata
                    }, f, indent=2, ensure_ascii=False, cls=DateTimeEncoder)
                
                results.append({
                    "document_id": doc["document_id"],
                    "chunks_created": len(result.chunks),
                    "chunker_used": chunker_type,
                    "processing_time": processing_time,
                    "success": True
                })
                
            except Exception as e:
                processing_time = time.time() - file_start_time
                error_msg = str(e)
                
                print(f"✗ FAILED - {processing_time:.3f}s")
                print(f"  Error: {error_msg}")
                
                errors.append(error_msg)
                performance_data.append({
                    "document_id": doc["document_id"],
                    "content_length": len(doc["content"]),
                    "chunker_used": chunker_type,
                    "processing_time_seconds": processing_time,
                    "success": False,
                    "error": error_msg
                })
        
        # Generate comprehensive performance report
        total_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        total_content_chars = sum(len(doc["content"]) for doc in documents)
        successful_docs = sum(1 for p in performance_data if p.get("success", False))
        
        # Get chunker metrics
        infra_metrics = chunker.get_infrastructure_metrics()
        perf_metrics = chunker.get_performance_metrics()
        
        performance_report = {
            "test_summary": {
                "chunker": "core",
                "test_date": datetime.now(timezone.utc).isoformat(),
                "total_documents": len(documents),
                "successful_documents": successful_docs,
                "failed_documents": len(documents) - successful_docs,
                "success_rate_percent": (successful_docs / len(documents)) * 100 if documents else 0,
                "total_processing_time_seconds": total_time,
                "total_content_chars": total_content_chars,
                "total_chunks_created": total_chunks_created,
                "average_throughput_chars_per_sec": total_content_chars / max(total_time, 0.001),
                "average_chunks_per_document": total_chunks_created / max(successful_docs, 1)
            },
            "infrastructure_metrics": infra_metrics,
            "performance_metrics": perf_metrics,
            "document_performance": performance_data,
            "errors": errors,
            "chunker_configuration": chunker._config
        }
        
        # Save performance report
        report_file = output_dir / "chunking_reports" / "core_chunker_performance.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(performance_report, f, indent=2, ensure_ascii=False, cls=DateTimeEncoder)
        
        # Save metrics in Prometheus format
        prometheus_metrics = chunker.export_metrics_prometheus()
        metrics_file = output_dir / "chunking_metrics" / "core_chunker_metrics.txt"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            f.write(prometheus_metrics)
        
        print(f"\n📊 CORE CHUNKER SUMMARY:")
        print(f"  Documents processed: {successful_docs}/{len(documents)}")
        print(f"  Success rate: {(successful_docs / len(documents)) * 100:.1f}%")
        print(f"  Total chunks created: {total_chunks_created}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average throughput: {total_content_chars / max(total_time, 0.001):.1f} chars/s")
        print(f"  Average chunks per doc: {total_chunks_created / max(successful_docs, 1):.1f}")
        
        return {
            "results": results,
            "performance_report": performance_report,
            "success": successful_docs > 0,
            "total_chunks": total_chunks_created
        }
        
    except Exception as e:
        print(f"✗ Core chunker test failed: {e}")
        traceback.print_exc()
        return {"results": [], "success": False, "error": str(e)}

def generate_integration_report(cpu_results: Dict, core_results: Dict, output_dir: Path, documents: List[Dict]):
    """Generate comprehensive chunking integration test report."""
    print("\n" + "="*80)
    print("📈 GENERATING CHUNKING INTEGRATION REPORT")
    print("="*80)
    
    # Combined analysis
    total_docs = len(documents)
    total_content_chars = sum(len(doc["content"]) for doc in documents)
    
    cpu_success = len(cpu_results.get("results", []))
    core_success = len(core_results.get("results", []))
    
    cpu_chunks = cpu_results.get("total_chunks", 0)
    core_chunks = core_results.get("total_chunks", 0)
    
    integration_report = {
        "integration_test_summary": {
            "test_date": datetime.now(timezone.utc).isoformat(),
            "test_duration_seconds": 0,  # Will be calculated
            "total_test_documents": total_docs,
            "total_content_chars": total_content_chars,
            "chunkers_tested": ["cpu", "core"]
        },
        "chunker_comparison": {
            "cpu_chunker": {
                "documents_processed": cpu_success,
                "success_rate": (cpu_success / total_docs) * 100 if total_docs else 0,
                "total_chunks_created": cpu_chunks,
                "status": "working" if cpu_results.get("success") else "failed"
            },
            "core_chunker": {
                "documents_processed": core_success,
                "success_rate": (core_success / total_docs) * 100 if total_docs else 0,
                "total_chunks_created": core_chunks,
                "status": "working" if core_results.get("success") else "failed"
            }
        },
        "readiness_assessment": {
            "ready_for_embedding": cpu_success > 0 or core_success > 0,
            "cpu_chunker_functional": cpu_results.get("success", False),
            "core_chunker_functional": core_results.get("success", False),
            "chunk_data_generated": cpu_chunks + core_chunks > 0,
            "recommendation": "Ready for embedding component testing" if (cpu_success > 0 or core_success > 0) else "Need chunker fixes"
        },
        "output_files": {
            "chunks_generated": cpu_chunks + core_chunks,
            "cpu_chunks": cpu_chunks,
            "core_chunks": core_chunks,
            "performance_reports": 2,
            "metrics_files": 2,
            "chunk_files": cpu_success + core_success
        }
    }
    
    # Save integration report
    report_file = output_dir / "chunking_integration_test_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(integration_report, f, indent=2, ensure_ascii=False, cls=DateTimeEncoder)
    
    # Generate human-readable summary
    summary_file = output_dir / "chunking_test_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("HADES Chunking Integration Test Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test Date: {integration_report['integration_test_summary']['test_date']}\n")
        f.write(f"Total Documents: {total_docs}\n")
        f.write(f"Total Content: {total_content_chars:,} characters\n\n")
        
        f.write("Chunker Results:\n")
        f.write(f"  CPU Chunker: {cpu_success}/{total_docs} docs ({(cpu_success/total_docs)*100:.1f}%) -> {cpu_chunks} chunks\n")
        f.write(f"  Core Chunker: {core_success}/{total_docs} docs ({(core_success/total_docs)*100:.1f}%) -> {core_chunks} chunks\n\n")
        
        f.write("Readiness Assessment:\n")
        f.write(f"  Ready for embedding: {integration_report['readiness_assessment']['ready_for_embedding']}\n")
        f.write(f"  Recommendation: {integration_report['readiness_assessment']['recommendation']}\n")
    
    print(f"✓ Integration report saved to: {report_file}")
    print(f"✓ Summary saved to: {summary_file}")
    
    return integration_report

def main():
    """Run comprehensive chunking integration test."""
    print("🚀 HADES CHUNKING INTEGRATION TEST")
    print("=" * 80)
    print(f"Test started at: {datetime.now(timezone.utc).isoformat()}")
    
    try:
        # Setup
        output_dir = ensure_output_directory()
        print(f"✓ Output directory: {output_dir}")
        
        # Load docproc outputs
        documents = load_docproc_outputs(output_dir)
        if not documents:
            print("❌ No documents found from docproc output!")
            return False
        
        # Test CPU chunker
        cpu_results = test_cpu_chunker(documents, output_dir)
        
        # Test core chunker
        core_results = test_core_chunker(documents, output_dir)
        
        # Generate integration report
        integration_report = generate_integration_report(cpu_results, core_results, output_dir, documents)
        
        # Final assessment
        print("\n" + "="*80)
        print("🎯 FINAL ASSESSMENT")
        print("="*80)
        
        cpu_working = cpu_results.get("success", False)
        core_working = core_results.get("success", False)
        
        print(f"✅ CPU Chunker: {'WORKING' if cpu_working else 'FAILED'}")
        print(f"✅ Core Chunker: {'WORKING' if core_working else 'FAILED'}")
        
        if cpu_working or core_working:
            total_chunks = cpu_results.get("total_chunks", 0) + core_results.get("total_chunks", 0)
            total_files = len([f for f in (output_dir / "chunks").glob("*.json")])
            print(f"\n🎉 SUCCESS: Generated {total_chunks} chunks in {total_files} files")
            print(f"📂 Output location: {output_dir}")
            print(f"📊 Reports available:")
            print(f"   - chunking_integration_test_report.json")
            print(f"   - chunking_test_summary.txt") 
            print(f"   - chunking_reports/cpu_chunker_performance.json")
            print(f"   - chunking_reports/core_chunker_performance.json")
            print(f"\n🔄 READY FOR EMBEDDING COMPONENT TESTING")
            return True
        else:
            print(f"\n❌ FAILED: No chunkers working properly")
            return False
            
    except Exception as e:
        print(f"\n💥 Integration test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"💥 Execution failed: {e}")
        sys.exit(1)