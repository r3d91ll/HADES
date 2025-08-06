#!/usr/bin/env python3
"""
Simple test of Jina v4 embeddings.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def test_jina_v4_import():
    """Test that we can import and initialize Jina v4."""
    print("Testing Jina v4 import and initialization...")
    
    try:
        from core.batch_embed_jina import JinaEmbedderV3
        print("✓ Successfully imported JinaEmbedderV3")
        
        # Initialize with CPU for testing (no GPU required)
        embedder = JinaEmbedderV3(
            device='cpu',
            chunk_size=28000,  # Jina v4 context
            chunk_overlap=5600,
            use_late_chunking=False  # Start with standard mode
        )
        print(f"✓ Initialized embedder with model: {embedder.model_name}")
        print(f"✓ Expected dimension: {embedder.expected_dim}")
        print(f"✓ Chunk size: {embedder.chunk_size}")
        
        # Test simple embedding
        test_text = "Information Reconstructionism shows that information = WHERE × WHAT × CONVEYANCE"
        embeddings = embedder.embed_batch([test_text])
        
        print(f"✓ Generated embedding with shape: {embeddings.shape}")
        print(f"✓ Embedding dimension: {embeddings.shape[1]}")
        
        # Verify it's Jina v4 (2048 dimensions)
        assert embeddings.shape[1] == 2048, f"Expected 2048 dimensions, got {embeddings.shape[1]}"
        print("\n✅ Jina v4 is working correctly!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_late_chunking():
    """Test late chunking capability."""
    print("\n" + "="*60)
    print("Testing Late Chunking...")
    
    try:
        from core.batch_embed_jina import JinaEmbedderV3
        
        # Initialize with late chunking
        embedder = JinaEmbedderV3(
            device='cpu',
            chunk_size=28000,
            chunk_overlap=5600,
            use_late_chunking=True  # Enable late chunking
        )
        
        # Create a longer text that would normally be chunked
        long_text = """
        The Information Reconstructionism framework demonstrates that information 
        exists as a multiplicative function of dimensional prerequisites. When any 
        dimension equals zero, information ceases to exist. This is not merely a 
        theoretical construct but a fundamental property of information transmission 
        across observer boundaries.
        """ * 10  # Repeat to make it longer
        
        import tempfile
        from pathlib import Path
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(long_text)
            temp_path = Path(f.name)
        
        try:
            # Process as document
            result = embedder.process_document("test_doc", temp_path)
            
            if result:
                print(f"✓ Late chunking processed {len(result.chunks)} chunks")
                print(f"✓ Total tokens: {result.total_tokens}")
                print(f"✓ Model: {result.model_name}")
                print("\n✅ Late chunking is working!")
                return True
            else:
                print("❌ Late chunking returned no results")
                return False
                
        finally:
            temp_path.unlink()
            
    except Exception as e:
        print(f"\n❌ Late chunking error: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("Jina v4 Verification Tests")
    print("="*60)
    
    # Test 1: Basic import and embedding
    test1_passed = test_jina_v4_import()
    
    # Test 2: Late chunking
    test2_passed = test_late_chunking()
    
    print("\n" + "="*60)
    print("Test Results:")
    print(f"  Basic Jina v4: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"  Late Chunking: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print("="*60)
    
    return test1_passed and test2_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)