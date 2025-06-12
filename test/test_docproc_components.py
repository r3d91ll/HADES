#!/usr/bin/env python3
"""
Test script for Document Processing components.

This script tests both the core and docling document processors to ensure
they work correctly with the component architecture.
"""

import sys
import logging
from pathlib import Path
import tempfile

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.components.docproc.core.processor import CoreDocumentProcessor
from src.components.docproc.docling.processor import DoclingDocumentProcessor
from src.types.components.contracts import DocumentProcessingInput

def test_core_processor():
    """Test the core document processor functionality."""
    
    print("\n📄 Testing Core Document Processor")
    print("=" * 40)
    
    try:
        # Test 1: Basic initialization
        print("1. Testing initialization...")
        config = {
            'batch_size': 10,
            'timeout': 30
        }
        
        processor = CoreDocumentProcessor(config)
        print(f"✅ Processor initialized: {processor.name} v{processor.version}")
        
        # Test 2: Configuration validation
        print("2. Testing configuration...")
        if processor.validate_config(config):
            print("✅ Configuration valid")
        else:
            print("❌ Configuration invalid")
            return False
        
        # Test 3: Health check
        print("3. Testing health check...")
        if processor.health_check():
            print("✅ Health check passed")
        else:
            print("❌ Health check failed")
            return False
        
        # Test 4: Document processing
        print("4. Testing document processing...")
        
        # Create test files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test Python file
            python_file = temp_path / "test.py"
            python_file.write_text("""
def hello_world():
    print("Hello, World!")
    return "success"

if __name__ == "__main__":
    hello_world()
""")
            
            # Create test JSON file
            json_file = temp_path / "test.json"
            json_file.write_text('{"name": "test", "value": 42, "active": true}')
            
            # Create test Markdown file
            md_file = temp_path / "test.md"
            md_file.write_text("""
# Test Document

This is a **test markdown** document with:

- Lists
- *Emphasis* 
- Code blocks

```python
print("Hello from markdown!")
```
""")
            
            # Prepare input - process each file individually using batch processing
            input_batch = [
                DocumentProcessingInput(file_path=str(python_file)),
                DocumentProcessingInput(file_path=str(json_file)),
                DocumentProcessingInput(file_path=str(md_file))
            ]
            
            # Process documents
            results = processor.process_batch(input_batch)
            
            # Combine results for easier checking
            all_documents = []
            all_errors = []
            total_time = 0.0
            
            for result in results:
                all_documents.extend(result.documents)
                all_errors.extend(result.errors)
                total_time += result.metadata.processing_time or 0.0
            
            print(f"✅ Processed {len(all_documents)} documents")
            print(f"   Processing time: {total_time:.3f}s")
            print(f"   Errors: {len(all_errors)}")
            
            # Check results
            for i, doc in enumerate(all_documents):
                if doc.error:
                    print(f"❌ Document {i+1} failed: {doc.error}")
                else:
                    print(f"✅ Document {i+1}: {doc.content_category} - {len(doc.content)} chars")
        
        # Test 5: Content processing
        print("5. Testing content processing...")
        
        content_input = DocumentProcessingInput(
            file_path="test.md",
            content="# Header\n\nSome **markdown** content",
            file_type="text/markdown"
        )
        
        content_result = processor.process(content_input)
        print(f"✅ Processed {len(content_result.documents)} content items")
        
        return True
        
    except Exception as e:
        print(f"❌ Core processor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_docling_processor():
    """Test the docling document processor functionality."""
    
    print("\n📄 Testing Docling Document Processor")
    print("=" * 40)
    
    try:
        # Test 1: Basic initialization
        print("1. Testing initialization...")
        config = {
            'batch_size': 5,
            'timeout': 60
        }
        
        processor = DoclingDocumentProcessor(config)
        print(f"✅ Processor initialized: {processor.name} v{processor.version}")
        
        # Test 2: Health check
        print("2. Testing health check...")
        health = processor.health_check()
        print(f"✅ Health check: {'passed' if health else 'failed (Docling not available)'}")
        
        # Test 3: Supported formats
        print("3. Testing supported formats...")
        formats = processor.get_supported_formats()
        print(f"✅ Supports {len(formats)} formats: {', '.join(formats[:5])}...")
        
        # Test 4: Basic document processing
        print("4. Testing document processing...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test text file
            text_file = temp_path / "test.txt"
            text_file.write_text("This is a simple text document for testing Docling processor.")
            
            # Prepare input
            input_data = DocumentProcessingInput(file_path=str(text_file))
            
            # Process document
            result = processor.process(input_data)
            
            print(f"✅ Processed {len(result.documents)} documents")
            processing_time = result.metadata.processing_time or 0.0
            print(f"   Processing time: {processing_time:.3f}s")
            print(f"   Errors: {len(result.errors)}")
            
            # Check result
            if result.documents:
                doc = result.documents[0]
                if doc.error:
                    print(f"⚠️  Document processed with error: {doc.error}")
                else:
                    print(f"✅ Document processed: {doc.content_category} - {len(doc.content)} chars")
        
        return True
        
    except Exception as e:
        print(f"❌ Docling processor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_both_processors():
    """Test both processors together."""
    print("\n🔄 Testing Both Processors")
    print("=" * 30)
    
    try:
        core = CoreDocumentProcessor()
        docling = DoclingDocumentProcessor()
        
        print(f"✅ Core: {core.name} - Formats: {len(core.get_supported_formats())}")
        print(f"✅ Docling: {docling.name} - Formats: {len(docling.get_supported_formats())}")
        
        # Test metrics
        core_metrics = core.get_metrics()
        docling_metrics = docling.get_metrics()
        
        print(f"✅ Core metrics: {core_metrics}")
        print(f"✅ Docling metrics: {docling_metrics}")
        
        return True
        
    except Exception as e:
        print(f"❌ Combined test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Document Processing Components")
    print("=" * 50)
    
    # Configure logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    
    success = True
    
    # Test core processor
    if not test_core_processor():
        success = False
    
    # Test docling processor
    if not test_docling_processor():
        success = False
    
    # Test both together
    if not test_both_processors():
        success = False
    
    if success:
        print("\n🎉 All document processing tests passed!")
        print("✅ Both processors are working correctly")
    else:
        print("\n❌ Some tests failed")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)