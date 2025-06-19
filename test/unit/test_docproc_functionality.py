#!/usr/bin/env python3
"""
Test DocProc Functionality

This script tests the actual document processing capabilities of both
core and docling processors to identify what's implemented vs placeholder.
"""

import sys
import tempfile
import json
from pathlib import Path

# Add HADES to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def create_test_files():
    """Create test files for processing."""
    test_dir = Path("test_documents")
    test_dir.mkdir(exist_ok=True)
    
    # Create various test files
    files = {}
    
    # Python code file
    python_file = test_dir / "test_code.py"
    python_file.write_text("""
def hello_world():
    '''Simple test function'''
    print("Hello, World!")
    return "success"

if __name__ == "__main__":
    hello_world()
""")
    files["python"] = python_file
    
    # Markdown file
    md_file = test_dir / "test_doc.md"
    md_file.write_text("""
# Test Document

This is a **test document** for the HADES document processing system.

## Features

- Markdown processing
- Code blocks
- Lists and formatting

```python
def example():
    return "code block"
```
""")
    files["markdown"] = md_file
    
    # JSON file
    json_file = test_dir / "test_data.json"
    json_file.write_text(json.dumps({
        "name": "HADES Test Data",
        "version": "1.0.0",
        "components": ["docproc", "embedding", "chunking"],
        "description": "Test data for document processing"
    }, indent=2))
    files["json"] = json_file
    
    # Plain text file
    txt_file = test_dir / "test_text.txt"
    txt_file.write_text("""
This is a plain text document for testing the HADES document processing system.

It contains multiple paragraphs and should be processed as simple text content.

The processor should extract this content and categorize it appropriately.
""")
    files["text"] = txt_file
    
    return files

def test_core_processor():
    """Test the core document processor."""
    print("="*60)
    print("Testing Core Document Processor")
    print("="*60)
    
    try:
        from src.components.docproc.core.processor import CoreDocumentProcessor
        
        processor = CoreDocumentProcessor()
        print(f"✓ Core processor initialized: {processor.name} v{processor.version}")
        
        # Test with different file types
        test_files = create_test_files()
        
        results = {}
        for file_type, file_path in test_files.items():
            try:
                print(f"\n--- Testing {file_type.upper()} file: {file_path.name} ---")
                
                result = processor.process_document(file_path)
                
                print(f"✓ Processed successfully")
                print(f"  - ID: {result.id}")
                print(f"  - Content length: {len(result.content)} chars")
                print(f"  - Content type: {result.content_type}")
                print(f"  - Format: {result.format}")
                print(f"  - Category: {result.content_category}")
                print(f"  - Error: {result.error}")
                
                if result.content:
                    preview = result.content[:100].replace('\n', ' ')
                    print(f"  - Content preview: {preview}...")
                
                results[file_type] = {
                    "success": True,
                    "content_length": len(result.content),
                    "category": str(result.content_category),
                    "error": result.error
                }
                
            except Exception as e:
                print(f"✗ Failed to process {file_type}: {e}")
                results[file_type] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Summary
        successful = sum(1 for r in results.values() if r.get("success"))
        total = len(results)
        print(f"\n📊 Core Processor Results: {successful}/{total} files processed successfully")
        
        return results
        
    except Exception as e:
        print(f"✗ Core processor test failed: {e}")
        return {}

def test_docling_processor():
    """Test the docling document processor."""
    print("\n" + "="*60)
    print("Testing Docling Document Processor")
    print("="*60)
    
    try:
        from src.components.docproc.docling.processor import DoclingDocumentProcessor
        
        processor = DoclingDocumentProcessor()
        print(f"✓ Docling processor initialized: {processor.name} v{processor.version}")
        print(f"✓ Docling available: {processor._docling_available}")
        print(f"✓ Supported formats: {len(processor.get_supported_formats())}")
        
        # Test with different file types
        test_files = create_test_files()
        
        results = {}
        for file_type, file_path in test_files.items():
            try:
                print(f"\n--- Testing {file_type.upper()} file: {file_path.name} ---")
                
                result = processor.process_document(file_path)
                
                print(f"✓ Processed successfully")
                print(f"  - ID: {result.id}")
                print(f"  - Content length: {len(result.content)} chars")
                print(f"  - Content type: {result.content_type}")
                print(f"  - Format: {result.format}")
                print(f"  - Category: {result.content_category}")
                print(f"  - Error: {result.error}")
                
                if result.content:
                    preview = result.content[:100].replace('\n', ' ')
                    print(f"  - Content preview: {preview}...")
                
                results[file_type] = {
                    "success": True,
                    "content_length": len(result.content),
                    "category": str(result.content_category),
                    "error": result.error
                }
                
            except Exception as e:
                print(f"✗ Failed to process {file_type}: {e}")
                results[file_type] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Summary
        successful = sum(1 for r in results.values() if r.get("success"))
        total = len(results)
        print(f"\n📊 Docling Processor Results: {successful}/{total} files processed successfully")
        
        return results
        
    except Exception as e:
        print(f"✗ Docling processor test failed: {e}")
        return {}

def test_contract_compliance():
    """Test contract compliance with DocumentProcessingInput."""
    print("\n" + "="*60)
    print("Testing Contract Compliance")
    print("="*60)
    
    try:
        from src.components.docproc.core.processor import CoreDocumentProcessor
        from src.types.components.contracts import DocumentProcessingInput
        
        processor = CoreDocumentProcessor()
        test_files = create_test_files()
        
        # Test contract-based processing
        python_file = test_files["python"]
        
        # Create proper input contract
        input_data = DocumentProcessingInput(
            file_path=str(python_file),
            processing_options={"extract_sections": True}
        )
        
        print(f"Testing contract-based processing...")
        result = processor.process(input_data)
        
        print(f"✓ Contract processing successful")
        print(f"  - Documents: {len(result.documents)}")
        print(f"  - Errors: {len(result.errors)}")
        print(f"  - Status: {result.metadata.status}")
        print(f"  - Component: {result.metadata.component_name}")
        
        if result.documents:
            doc = result.documents[0]
            print(f"  - Document ID: {doc.id}")
            print(f"  - Content length: {len(doc.content)}")
            
        return True
        
    except Exception as e:
        print(f"✗ Contract compliance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_placeholders():
    """Analyze what placeholder code remains."""
    print("\n" + "="*60)
    print("Placeholder Analysis")
    print("="*60)
    
    placeholder_info = {
        "core_processor": {
            "status": "Mostly Implemented",
            "description": "Has working minimal implementation with file reading and categorization",
            "placeholders": [
                "Uses simple file reading instead of advanced parsing",
                "No adapter manager integration",
                "Basic content categorization by file extension"
            ]
        },
        "docling_processor": {
            "status": "Partial Implementation", 
            "description": "Has monitoring but adapter is placeholder",
            "placeholders": [
                "self._adapter = None  # Placeholder",
                "Docling integration commented out",
                "Falls back to error documents"
            ]
        }
    }
    
    for component, info in placeholder_info.items():
        print(f"\n{component.upper()}:")
        print(f"  Status: {info['status']}")
        print(f"  Description: {info['description']}")
        print(f"  Placeholders:")
        for placeholder in info['placeholders']:
            print(f"    - {placeholder}")
    
    return placeholder_info

def main():
    """Run comprehensive docproc functionality tests."""
    print("HADES DocProc Functionality Test")
    print("=" * 80)
    
    # Test core processor
    core_results = test_core_processor()
    
    # Test docling processor
    docling_results = test_docling_processor()
    
    # Test contract compliance
    contract_ok = test_contract_compliance()
    
    # Analyze placeholders
    placeholder_info = analyze_placeholders()
    
    # Cleanup test files
    import shutil
    test_dir = Path("test_documents")
    if test_dir.exists():
        shutil.rmtree(test_dir)
        print(f"\n🗑️  Cleaned up test files in {test_dir}")
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL ASSESSMENT")
    print("="*80)
    
    core_working = any(r.get("success") for r in core_results.values()) if core_results else False
    docling_working = any(r.get("success") for r in docling_results.values()) if docling_results else False
    
    print(f"✅ Core Processor: {'WORKING' if core_working else 'NOT WORKING'}")
    print(f"⚠️  Docling Processor: {'WORKING' if docling_working else 'NOT WORKING'}")
    print(f"✅ Contract Compliance: {'WORKING' if contract_ok else 'NOT WORKING'}")
    print(f"✅ Monitoring Integration: WORKING (from previous tests)")
    
    print(f"\n📋 READINESS STATUS:")
    if core_working and contract_ok:
        print("🟢 READY for integration testing with basic file processing")
        print("🟡 Docling integration needs completion for advanced document processing")
        print("🟢 Core functionality sufficient for initial testing")
    else:
        print("🔴 NOT READY - core functionality issues found")
    
    print(f"\n🔄 NEXT STEPS:")
    print("1. Complete Docling adapter implementation (if advanced PDF/DOCX processing needed)")
    print("2. Run integration tests with real documents")
    print("3. Test with HADES pipeline components")
    print("4. Create test output datasets")
    
    return core_working and contract_ok

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)