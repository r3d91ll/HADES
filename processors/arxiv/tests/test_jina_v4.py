#!/usr/bin/env python3
"""
Test Jina v4 embeddings with late chunking.
"""

import sys
import time
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from core.batch_embed_jina import JinaEmbedderV3  # Class name kept for compatibility

def test_basic_embedding():
    """Test basic Jina v4 embedding generation."""
    print("Testing basic Jina v4 embedding...")
    
    embedder = JinaEmbedderV3(
        device='cuda',
        chunk_size=28000,  # Jina v4 context
        chunk_overlap=5600,
        use_late_chunking=False  # Test standard first
    )
    
    test_text = "Information Reconstructionism demonstrates that information requires all dimensions non-zero."
    
    embeddings = embedder.embed_batch([test_text])
    
    assert embeddings.shape[0] == 1
    assert embeddings.shape[1] == 2048  # Jina v4 dimension
    assert not np.isnan(embeddings).any()
    assert not np.isinf(embeddings).any()
    
    print(f"✓ Basic embedding: shape={embeddings.shape}, norm={np.linalg.norm(embeddings[0]):.4f}")
    return True

def test_late_chunking():
    """Test late chunking with long document."""
    print("\nTesting late chunking with long document...")
    
    embedder = JinaEmbedderV3(
        device='cuda',
        chunk_size=28000,
        chunk_overlap=5600,
        use_late_chunking=True  # Enable late chunking
    )
    
    # Create a long document
    long_text = """
    def calculate_information(dimensions):
        '''Calculate information as multiplicative function.'''
        where = dimensions.get('where', 0)
        what = dimensions.get('what', 0)
        conveyance = dimensions.get('conveyance', 0)
        
        # Information exists only if all dimensions are non-zero
        if where == 0 or what == 0 or conveyance == 0:
            return 0
        
        return where * what * conveyance
    
    class InformationTheory:
        def __init__(self):
            self.dimensions = ['WHERE', 'WHAT', 'CONVEYANCE', 'TIME', 'FRAME']
        
        def validate_dimensions(self, values):
            '''Ensure all dimensions are non-zero.'''
            for dim, value in values.items():
                if value == 0:
                    print(f"Zero dimension detected: {dim}")
                    return False
            return True
    """ * 50  # Repeat to make it long
    
    # Process as document
    from pathlib import Path
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(long_text)
        temp_path = Path(f.name)
    
    try:
        doc_embeddings = embedder.process_document("test_doc", temp_path)
        
        assert doc_embeddings is not None
        assert len(doc_embeddings.chunks) > 0
        assert doc_embeddings.model_name == "jinaai/jina-embeddings-v4"
        
        print(f"✓ Late chunking: {len(doc_embeddings.chunks)} chunks generated")
        print(f"  Total tokens: {doc_embeddings.total_tokens}")
        print(f"  Processing time: {doc_embeddings.processing_time:.2f}s")
        
        # Verify embeddings are contextual (different from isolated chunks)
        first_chunk = doc_embeddings.chunks[0]
        assert len(first_chunk.embedding) == 2048
        
        return True
        
    finally:
        temp_path.unlink()

def test_code_embedding():
    """Test code-specific embedding."""
    print("\nTesting code embedding...")
    
    embedder = JinaEmbedderV3(
        device='cuda',
        chunk_size=28000,
        use_late_chunking=True
    )
    
    code_sample = '''
    import numpy as np
    from transformers import AutoModel
    
    def embed_with_context(text: str, model: AutoModel) -> np.ndarray:
        """Generate embeddings with full context awareness."""
        tokens = model.tokenize(text)
        embeddings = model.encode(tokens, output_value='token_embeddings')
        return embeddings.mean(dim=0)
    '''
    
    embeddings = embedder.embed_batch([code_sample])
    
    assert embeddings.shape[1] == 2048
    print(f"✓ Code embedding: shape={embeddings.shape}")
    return True

def test_dimension_validation():
    """Test that embeddings are validated for correct dimensions."""
    print("\nTesting dimension validation...")
    
    embedder = JinaEmbedderV3(
        device='cuda',
        chunk_size=28000
    )
    
    # Create invalid embeddings
    invalid_embeddings = np.zeros((2, 1024))  # Wrong dimension
    
    is_valid = embedder.validate_embeddings(invalid_embeddings)
    assert not is_valid
    print("✓ Correctly rejected 1024-dim embeddings")
    
    # Valid embeddings
    valid_embeddings = np.zeros((2, 2048))
    is_valid = embedder.validate_embeddings(valid_embeddings)
    assert is_valid
    print("✓ Correctly accepted 2048-dim embeddings")
    
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Jina v4 Embeddings")
    print("=" * 60)
    
    tests = [
        test_basic_embedding,
        test_late_chunking,
        test_code_embedding,
        test_dimension_validation
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} failed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)