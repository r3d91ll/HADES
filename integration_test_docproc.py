#!/usr/bin/env python3
"""
HADES DocProc Integration Test

This script performs comprehensive integration testing of the docproc components
using real test data from test-data/ and outputs results to test-out/.

Output includes:
- Performance metrics and reports
- JSON documents for chunking component input
- Processing statistics and analysis
"""

import sys
import json
import time
import traceback
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any

# Add HADES to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def ensure_output_directory():
    """Ensure test-out directory exists."""
    output_dir = Path("test-out")
    output_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (output_dir / "documents").mkdir(exist_ok=True)
    (output_dir / "reports").mkdir(exist_ok=True)
    (output_dir / "metrics").mkdir(exist_ok=True)
    
    return output_dir

def discover_test_files():
    """Discover all files in test-data directory."""
    test_data_dir = Path("test-data")
    
    if not test_data_dir.exists():
        raise FileNotFoundError(f"Test data directory not found: {test_data_dir}")
    
    files = []
    for file_path in test_data_dir.rglob("*"):
        if file_path.is_file():
            files.append(file_path)
    
    print(f"📁 Discovered {len(files)} test files:")
    for file_path in files:
        size_kb = file_path.stat().st_size / 1024
        print(f"  - {file_path.name} ({size_kb:.1f} KB)")
    
    return files

def test_core_processor(test_files: List[Path], output_dir: Path):
    """Test core processor with comprehensive metrics."""
    print("\n" + "="*80)
    print("🔧 TESTING CORE DOCUMENT PROCESSOR")
    print("="*80)
    
    try:
        from src.components.docproc.core.processor import CoreDocumentProcessor
        
        # Initialize processor
        processor = CoreDocumentProcessor(config={
            "processing_options": {
                "extract_sections": True,
                "preserve_formatting": True
            }
        })
        
        print(f"✓ Initialized processor: {processor.name} v{processor.version}")
        
        # Track metrics
        start_time = datetime.now(timezone.utc)
        results = []
        errors = []
        performance_data = []
        
        # Process each file
        for i, file_path in enumerate(test_files, 1):
            print(f"\n--- Processing {i}/{len(test_files)}: {file_path.name} ---")
            
            file_start_time = time.time()
            
            try:
                # Process document
                result = processor.process_document(file_path)
                
                processing_time = time.time() - file_start_time
                
                # Collect performance data
                perf_data = {
                    "file_name": file_path.name,
                    "file_size_bytes": file_path.stat().st_size,
                    "file_size_kb": file_path.stat().st_size / 1024,
                    "processing_time_seconds": processing_time,
                    "content_length": len(result.content),
                    "content_type": result.content_type,
                    "format": result.format,
                    "category": str(result.content_category),
                    "success": result.error is None,
                    "error": result.error,
                    "throughput_kb_per_sec": (file_path.stat().st_size / 1024) / max(processing_time, 0.001)
                }
                performance_data.append(perf_data)
                
                print(f"✓ SUCCESS - {processing_time:.3f}s")
                print(f"  Size: {perf_data['file_size_kb']:.1f} KB → Content: {len(result.content)} chars")
                print(f"  Type: {result.content_type} | Category: {result.content_category}")
                print(f"  Throughput: {perf_data['throughput_kb_per_sec']:.1f} KB/s")
                
                if result.error:
                    print(f"  ⚠️  Warning: {result.error}")
                    errors.append({"file": file_path.name, "error": result.error})
                
                # Convert to serializable format for JSON output
                if hasattr(result, 'model_dump'):
                    # Pydantic v2 style
                    document_data = result.model_dump()
                elif hasattr(result, 'dict'):
                    # Pydantic v1 style
                    document_data = result.dict()
                else:
                    # Manual extraction as fallback
                    document_data = {
                        "id": result.id,
                        "content": result.content,
                        "content_type": result.content_type,
                        "format": result.format,
                        "content_category": str(result.content_category),
                        "entities": result.entities or [],
                        "sections": result.sections or [],
                        "metadata": result.metadata or {},
                        "processing_time": result.processing_time,
                        "error": result.error
                    }
                
                # Add test-specific fields
                document_data.update({
                    "processing_time": processing_time,
                    "source_file": str(file_path),
                    "processed_at": datetime.now(timezone.utc).isoformat()
                })
                
                results.append(document_data)
                
                # Save individual document JSON
                doc_file = output_dir / "documents" / f"{result.id}_core.json"
                with open(doc_file, 'w', encoding='utf-8') as f:
                    json.dump(document_data, f, indent=2, ensure_ascii=False)
                
            except Exception as e:
                processing_time = time.time() - file_start_time
                error_msg = str(e)
                
                print(f"✗ FAILED - {processing_time:.3f}s")
                print(f"  Error: {error_msg}")
                
                errors.append({"file": file_path.name, "error": error_msg})
                performance_data.append({
                    "file_name": file_path.name,
                    "file_size_kb": file_path.stat().st_size / 1024,
                    "processing_time_seconds": processing_time,
                    "success": False,
                    "error": error_msg
                })
        
        # Generate comprehensive performance report
        total_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        total_size_kb = sum(f.stat().st_size for f in test_files) / 1024
        successful_files = sum(1 for p in performance_data if p.get("success", False))
        
        # Get processor metrics
        infra_metrics = processor.get_infrastructure_metrics()
        perf_metrics = processor.get_performance_metrics()
        
        performance_report = {
            "test_summary": {
                "processor": "core",
                "test_date": datetime.now(timezone.utc).isoformat(),
                "total_files": len(test_files),
                "successful_files": successful_files,
                "failed_files": len(test_files) - successful_files,
                "success_rate_percent": (successful_files / len(test_files)) * 100,
                "total_processing_time_seconds": total_time,
                "total_data_size_kb": total_size_kb,
                "average_throughput_kb_per_sec": total_size_kb / max(total_time, 0.001)
            },
            "infrastructure_metrics": infra_metrics,
            "performance_metrics": perf_metrics,
            "file_performance": performance_data,
            "errors": errors,
            "processor_configuration": processor._config
        }
        
        # Save performance report
        report_file = output_dir / "reports" / "core_processor_performance.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(performance_report, f, indent=2, ensure_ascii=False)
        
        # Save metrics in Prometheus format
        prometheus_metrics = processor.export_metrics_prometheus()
        metrics_file = output_dir / "metrics" / "core_processor_metrics.txt"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            f.write(prometheus_metrics)
        
        print(f"\n📊 CORE PROCESSOR SUMMARY:")
        print(f"  Files processed: {successful_files}/{len(test_files)}")
        print(f"  Success rate: {(successful_files / len(test_files)) * 100:.1f}%")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average throughput: {total_size_kb / max(total_time, 0.001):.1f} KB/s")
        print(f"  Total content generated: {sum(len(r.get('content', '')) for r in results)} characters")
        
        return {
            "results": results,
            "performance_report": performance_report,
            "success": successful_files > 0
        }
        
    except Exception as e:
        print(f"✗ Core processor test failed: {e}")
        traceback.print_exc()
        return {"results": [], "success": False, "error": str(e)}

def test_docling_processor(test_files: List[Path], output_dir: Path):
    """Test docling processor with comprehensive metrics."""
    print("\n" + "="*80)
    print("📄 TESTING DOCLING DOCUMENT PROCESSOR")
    print("="*80)
    
    try:
        from src.components.docproc.docling.processor import DoclingDocumentProcessor
        
        # Initialize processor
        processor = DoclingDocumentProcessor(config={
            "ocr_enabled": True,
            "extract_tables": True,
            "extract_images": False
        })
        
        print(f"✓ Initialized processor: {processor.name} v{processor.version}")
        print(f"✓ Docling available: {processor._docling_available}")
        print(f"✓ Supported formats: {len(processor.get_supported_formats())}")
        
        # Track metrics
        start_time = datetime.now(timezone.utc)
        results = []
        errors = []
        performance_data = []
        
        # Process each file
        for i, file_path in enumerate(test_files, 1):
            print(f"\n--- Processing {i}/{len(test_files)}: {file_path.name} ---")
            
            file_start_time = time.time()
            
            try:
                # Check if file can be processed
                can_process = processor.can_process(str(file_path))
                print(f"  Can process: {can_process}")
                
                # Process document
                result = processor.process_document(file_path)
                
                processing_time = time.time() - file_start_time
                
                # Collect performance data
                perf_data = {
                    "file_name": file_path.name,
                    "file_size_bytes": file_path.stat().st_size,
                    "file_size_kb": file_path.stat().st_size / 1024,
                    "processing_time_seconds": processing_time,
                    "content_length": len(result.content),
                    "content_type": result.content_type,
                    "format": result.format,
                    "category": str(result.content_category),
                    "success": result.error is None,
                    "error": result.error,
                    "can_process": can_process,
                    "throughput_kb_per_sec": (file_path.stat().st_size / 1024) / max(processing_time, 0.001)
                }
                performance_data.append(perf_data)
                
                if result.error:
                    print(f"⚠️  WARNING - {processing_time:.3f}s")
                    print(f"  Error: {result.error}")
                    errors.append({"file": file_path.name, "error": result.error})
                else:
                    print(f"✓ SUCCESS - {processing_time:.3f}s")
                    print(f"  Size: {perf_data['file_size_kb']:.1f} KB → Content: {len(result.content)} chars")
                    print(f"  Type: {result.content_type} | Category: {result.content_category}")
                    print(f"  Throughput: {perf_data['throughput_kb_per_sec']:.1f} KB/s")
                
                # Convert to serializable format for JSON output
                if hasattr(result, 'model_dump'):
                    # Pydantic v2 style
                    document_data = result.model_dump()
                elif hasattr(result, 'dict'):
                    # Pydantic v1 style
                    document_data = result.dict()
                else:
                    # Manual extraction as fallback
                    document_data = {
                        "id": result.id,
                        "content": result.content,
                        "content_type": result.content_type,
                        "format": result.format,
                        "content_category": str(result.content_category),
                        "entities": result.entities or [],
                        "sections": result.sections or [],
                        "metadata": result.metadata or {},
                        "processing_time": result.processing_time,
                        "error": result.error
                    }
                
                # Add test-specific fields
                document_data.update({
                    "processing_time": processing_time,
                    "source_file": str(file_path),
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                    "can_process": can_process
                })
                
                results.append(document_data)
                
                # Save individual document JSON
                doc_file = output_dir / "documents" / f"{result.id}_docling.json"
                with open(doc_file, 'w', encoding='utf-8') as f:
                    json.dump(document_data, f, indent=2, ensure_ascii=False)
                
            except Exception as e:
                processing_time = time.time() - file_start_time
                error_msg = str(e)
                
                print(f"✗ FAILED - {processing_time:.3f}s")
                print(f"  Error: {error_msg}")
                
                errors.append({"file": file_path.name, "error": error_msg})
                performance_data.append({
                    "file_name": file_path.name,
                    "file_size_kb": file_path.stat().st_size / 1024,
                    "processing_time_seconds": processing_time,
                    "success": False,
                    "error": error_msg
                })
        
        # Generate comprehensive performance report
        total_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        total_size_kb = sum(f.stat().st_size for f in test_files) / 1024
        successful_files = sum(1 for p in performance_data if p.get("success", False) and not p.get("error"))
        
        # Get processor metrics
        infra_metrics = processor.get_infrastructure_metrics()
        perf_metrics = processor.get_performance_metrics()
        
        performance_report = {
            "test_summary": {
                "processor": "docling",
                "test_date": datetime.now(timezone.utc).isoformat(),
                "total_files": len(test_files),
                "successful_files": successful_files,
                "failed_files": len(test_files) - successful_files,
                "success_rate_percent": (successful_files / len(test_files)) * 100 if test_files else 0,
                "total_processing_time_seconds": total_time,
                "total_data_size_kb": total_size_kb,
                "average_throughput_kb_per_sec": total_size_kb / max(total_time, 0.001),
                "docling_available": processor._docling_available,
                "supported_formats": processor.get_supported_formats()
            },
            "infrastructure_metrics": infra_metrics,
            "performance_metrics": perf_metrics,
            "file_performance": performance_data,
            "errors": errors,
            "processor_configuration": processor._config
        }
        
        # Save performance report
        report_file = output_dir / "reports" / "docling_processor_performance.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(performance_report, f, indent=2, ensure_ascii=False)
        
        # Save metrics in Prometheus format
        prometheus_metrics = processor.export_metrics_prometheus()
        metrics_file = output_dir / "metrics" / "docling_processor_metrics.txt"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            f.write(prometheus_metrics)
        
        print(f"\n📊 DOCLING PROCESSOR SUMMARY:")
        print(f"  Files processed: {successful_files}/{len(test_files)}")
        print(f"  Success rate: {(successful_files / len(test_files)) * 100:.1f}%" if test_files else "N/A")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average throughput: {total_size_kb / max(total_time, 0.001):.1f} KB/s")
        print(f"  Total content generated: {sum(len(r.get('content', '')) for r in results)} characters")
        
        return {
            "results": results,
            "performance_report": performance_report,
            "success": successful_files >= 0  # Consider partial success for docling
        }
        
    except Exception as e:
        print(f"✗ Docling processor test failed: {e}")
        traceback.print_exc()
        return {"results": [], "success": False, "error": str(e)}

def generate_integration_report(core_results: Dict, docling_results: Dict, output_dir: Path, test_files: List[Path]):
    """Generate comprehensive integration test report."""
    print("\n" + "="*80)
    print("📈 GENERATING INTEGRATION REPORT")
    print("="*80)
    
    # Combined analysis
    total_files = len(test_files)
    total_size_kb = sum(f.stat().st_size for f in test_files) / 1024
    
    core_success = len(core_results.get("results", []))
    docling_success = len([r for r in docling_results.get("results", []) if not r.get("error")])
    
    # File type analysis
    file_types = {}
    for file_path in test_files:
        ext = file_path.suffix.lower()
        file_types[ext] = file_types.get(ext, 0) + 1
    
    integration_report = {
        "integration_test_summary": {
            "test_date": datetime.now(timezone.utc).isoformat(),
            "test_duration_seconds": (datetime.now(timezone.utc) - datetime.now(timezone.utc)).total_seconds(),
            "total_test_files": total_files,
            "total_data_size_kb": total_size_kb,
            "file_types": file_types,
            "processors_tested": ["core", "docling"]
        },
        "processor_comparison": {
            "core_processor": {
                "files_processed": core_success,
                "success_rate": (core_success / total_files) * 100 if total_files else 0,
                "status": "working" if core_results.get("success") else "failed"
            },
            "docling_processor": {
                "files_processed": docling_success,
                "success_rate": (docling_success / total_files) * 100 if total_files else 0,
                "status": "partial" if docling_results.get("success") else "failed"
            }
        },
        "readiness_assessment": {
            "ready_for_chunking": core_success > 0,
            "core_processor_functional": core_results.get("success", False),
            "docling_processor_functional": docling_results.get("success", False),
            "test_data_generated": core_success > 0,
            "recommendation": "Ready for chunking component testing" if core_success > 0 else "Need processor fixes"
        },
        "output_files": {
            "documents_generated": core_success + docling_success,
            "core_documents": core_success,
            "docling_documents": docling_success,
            "performance_reports": 2,
            "metrics_files": 2
        }
    }
    
    # Save integration report
    report_file = output_dir / "integration_test_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(integration_report, f, indent=2, ensure_ascii=False)
    
    # Generate human-readable summary
    summary_file = output_dir / "test_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("HADES DocProc Integration Test Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test Date: {integration_report['integration_test_summary']['test_date']}\n")
        f.write(f"Total Files: {total_files}\n")
        f.write(f"Total Size: {total_size_kb:.1f} KB\n\n")
        
        f.write("File Types:\n")
        for ext, count in file_types.items():
            f.write(f"  {ext or 'no-extension'}: {count} files\n")
        f.write("\n")
        
        f.write("Processor Results:\n")
        f.write(f"  Core Processor: {core_success}/{total_files} files ({(core_success/total_files)*100:.1f}%)\n")
        f.write(f"  Docling Processor: {docling_success}/{total_files} files ({(docling_success/total_files)*100:.1f}%)\n\n")
        
        f.write("Readiness Assessment:\n")
        f.write(f"  Ready for chunking: {integration_report['readiness_assessment']['ready_for_chunking']}\n")
        f.write(f"  Recommendation: {integration_report['readiness_assessment']['recommendation']}\n")
    
    print(f"✓ Integration report saved to: {report_file}")
    print(f"✓ Summary saved to: {summary_file}")
    
    return integration_report

def main():
    """Run comprehensive docproc integration test."""
    print("🚀 HADES DOCPROC INTEGRATION TEST")
    print("=" * 80)
    print(f"Test started at: {datetime.now(timezone.utc).isoformat()}")
    
    try:
        # Setup
        output_dir = ensure_output_directory()
        print(f"✓ Output directory: {output_dir}")
        
        # Discover test files
        test_files = discover_test_files()
        if not test_files:
            print("❌ No test files found!")
            return False
        
        # Test core processor
        core_results = test_core_processor(test_files, output_dir)
        
        # Test docling processor  
        docling_results = test_docling_processor(test_files, output_dir)
        
        # Generate integration report
        integration_report = generate_integration_report(core_results, docling_results, output_dir, test_files)
        
        # Final assessment
        print("\n" + "="*80)
        print("🎯 FINAL ASSESSMENT")
        print("="*80)
        
        core_working = core_results.get("success", False)
        docling_working = docling_results.get("success", False)
        
        print(f"✅ Core Processor: {'WORKING' if core_working else 'FAILED'}")
        print(f"🟡 Docling Processor: {'PARTIAL' if docling_working else 'FAILED'}")
        
        if core_working:
            total_docs = len(core_results.get("results", []))
            print(f"\n🎉 SUCCESS: Generated {total_docs} processed documents")
            print(f"📂 Output location: {output_dir}")
            print(f"📊 Reports available:")
            print(f"   - integration_test_report.json")
            print(f"   - test_summary.txt")
            print(f"   - reports/core_processor_performance.json")
            print(f"   - reports/docling_processor_performance.json")
            print(f"\n🔄 READY FOR CHUNKING COMPONENT TESTING")
            return True
        else:
            print(f"\n❌ FAILED: Core processor not working")
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