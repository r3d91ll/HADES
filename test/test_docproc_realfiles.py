#!/usr/bin/env python3
"""
Real File Document Processing Test

This script tests the docproc components with real files from test-data/
and captures their JSON output for use in chunking component testing.
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.components.docproc.core.processor import CoreDocumentProcessor
from src.components.docproc.docling.processor import DoclingDocumentProcessor
from src.types.components.contracts import DocumentProcessingInput

def test_file_processing(file_path: Path, processor_name: str = "auto"):
    """Test processing a single file and return results."""
    
    print(f"\n📄 Processing: {file_path.name}")
    print("=" * 50)
    
    # Determine processor to use
    if processor_name == "auto":
        # Use core processor for code/text, docling for PDF
        if file_path.suffix.lower() == '.pdf':
            processor = DoclingDocumentProcessor()
            processor_used = "docling"
        else:
            processor = CoreDocumentProcessor()
            processor_used = "core"
    elif processor_name == "core":
        processor = CoreDocumentProcessor()
        processor_used = "core"
    elif processor_name == "docling":
        processor = DoclingDocumentProcessor()
        processor_used = "docling"
    else:
        raise ValueError(f"Unknown processor: {processor_name}")
    
    print(f"Using processor: {processor_used}")
    
    # Create input
    input_data = DocumentProcessingInput(file_path=str(file_path))
    
    # Process file
    start_time = datetime.now(timezone.utc)
    result = processor.process(input_data)
    end_time = datetime.now(timezone.utc)
    processing_time = (end_time - start_time).total_seconds()
    
    # Display results
    print(f"✅ Processing completed in {processing_time:.3f}s")
    print(f"   Documents: {len(result.documents)}")
    print(f"   Errors: {len(result.errors)}")
    
    if result.errors:
        for error in result.errors:
            print(f"   ❌ Error: {error}")
    
    # Check document results
    doc_data = []
    for i, doc in enumerate(result.documents):
        if doc.error:
            print(f"   ❌ Document {i+1} failed: {doc.error}")
        else:
            content_preview = doc.content[:100] + "..." if len(doc.content) > 100 else doc.content
            print(f"   ✅ Document {i+1}: {doc.content_category} - {len(doc.content)} chars")
            print(f"      Preview: {content_preview}")
            print(f"      Format: {doc.format}")
            print(f"      Metadata: {len(doc.metadata)} items")
            
            # Convert to dict for JSON serialization
            doc_dict = {
                "id": doc.id,
                "content": doc.content,
                "content_type": doc.content_type,
                "format": doc.format,
                "content_category": doc.content_category,
                "entities": doc.entities,
                "sections": doc.sections,
                "metadata": doc.metadata,
                "error": doc.error,
                "processing_time": doc.processing_time
            }
            doc_data.append(doc_dict)
    
    # Return structured result
    return {
        "file_path": str(file_path),
        "file_name": file_path.name,
        "file_type": file_path.suffix,
        "processor_used": processor_used,
        "processing_time": processing_time,
        "success": len(result.errors) == 0,
        "documents": doc_data,
        "errors": result.errors,
        "metadata": {
            "component_type": result.metadata.component_type,
            "component_name": result.metadata.component_name,
            "component_version": result.metadata.component_version,
            "processing_time": result.metadata.processing_time,
            "processed_at": result.metadata.processed_at.isoformat() if result.metadata.processed_at else None,
            "config": result.metadata.config,
            "status": result.metadata.status
        }
    }

def save_results_json(results: dict, output_file: Path):
    """Save results to JSON file for use in chunking tests."""
    
    print(f"\n💾 Saving results to: {output_file}")
    
    # Make output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save with pretty formatting
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"✅ Results saved ({output_file.stat().st_size} bytes)")

def main():
    """Test all file types and save results."""
    
    print("🧪 HADES Document Processing - Real File Tests")
    print("=" * 60)
    
    # Configure logging to reduce noise
    logging.basicConfig(level=logging.WARNING)
    
    # Test data directory
    test_data_dir = Path("test-data")
    if not test_data_dir.exists():
        print(f"❌ Test data directory not found: {test_data_dir}")
        return False
    
    # Define test files and expected processors
    test_files = [
        ("processor.py", "core", "Python source code"),
        ("config.yaml", "core", "YAML configuration"), 
        ("HADES_SERVICE_ARCHITECTURE.md", "core", "Markdown documentation"),
        ("PathRAG Pruning Graph-based Retrieval Augmented Generation with Relational Paths.pdf", "docling", "PDF research paper")
    ]
    
    all_results = {
        "test_info": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "test_description": "Real file processing test for HADES docproc components",
            "files_tested": len(test_files)
        },
        "results": {}
    }
    
    success_count = 0
    
    # Process each test file
    for file_name, preferred_processor, description in test_files:
        file_path = test_data_dir / file_name
        
        if not file_path.exists():
            print(f"\n❌ File not found: {file_path}")
            continue
        
        print(f"\n📁 Test File: {description}")
        print(f"   Path: {file_path}")
        print(f"   Size: {file_path.stat().st_size:,} bytes")
        
        try:
            # Test with preferred processor
            result = test_file_processing(file_path, preferred_processor)
            
            # Also test with auto-selection to verify it chooses correctly
            print(f"\n🔄 Testing auto processor selection...")
            auto_result = test_file_processing(file_path, "auto")
            
            # Store both results
            file_key = file_path.stem
            all_results["results"][file_key] = {
                "preferred_processor": result,
                "auto_processor": auto_result,
                "description": description,
                "processor_matches": result["processor_used"] == auto_result["processor_used"]
            }
            
            if result["success"]:
                success_count += 1
                print(f"✅ {file_name} processed successfully")
            else:
                print(f"⚠️  {file_name} processed with errors")
                
        except Exception as e:
            print(f"❌ Failed to process {file_name}: {e}")
            import traceback
            traceback.print_exc()
            
            all_results["results"][file_path.stem] = {
                "error": str(e),
                "description": description
            }
    
    # Save comprehensive results with timestamp
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_file = Path("test-out") / f"docproc_realfile_results_{timestamp}.json"
    save_results_json(all_results, output_file)
    
    # Summary
    print(f"\n📊 Test Summary")
    print("=" * 30)
    print(f"Files processed: {success_count}/{len(test_files)}")
    print(f"Success rate: {success_count/len(test_files)*100:.1f}%")
    
    if success_count == len(test_files):
        print("\n🎉 All files processed successfully!")
        print("✅ Document processing components are working correctly")
        print(f"📄 Results saved to: {output_file}")
        print("\n🔄 These results can now be used for chunking component testing")
        
        # Print sample output for chunking
        print(f"\n📋 Sample Document Structure for Chunking:")
        if all_results["results"]:
            first_result = list(all_results["results"].values())[0]
            if "preferred_processor" in first_result and first_result["preferred_processor"]["documents"]:
                sample_doc = first_result["preferred_processor"]["documents"][0]
                print(f"   - ID: {sample_doc['id']}")
                print(f"   - Content: {len(sample_doc['content'])} characters")
                print(f"   - Category: {sample_doc['content_category']}")
                print(f"   - Format: {sample_doc['format']}")
                print(f"   - Metadata: {len(sample_doc['metadata'])} items")
        
        return True
    else:
        print(f"\n⚠️  Some files failed to process")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)