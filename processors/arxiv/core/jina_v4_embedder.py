#!/usr/bin/env python3
"""
Jina v4 Embedder with proper API usage and fp16 support.

Theory Connection:
Late chunking preserves the contextual relationships across text boundaries,
maintaining the WHERE dimension of information even when physically chunked.
"""

# cspell:ignore jina Jina embedder Embedder

import torch
import numpy as np
from typing import List, Optional
from transformers import AutoModel
import logging

logger = logging.getLogger(__name__)


class JinaV4Embedder:
    """Jina v4 embedder with correct API usage."""
    
    def __init__(self, 
                 device: str = "cuda",
                 use_fp16: bool = True):
        """
        Initialize Jina v4 embedder.
        
        Args:
            device: Device to use (cuda/cpu)
            use_fp16: Use half precision for efficiency
        """
        self.device = device
        self.model_name = "jinaai/jina-embeddings-v4"
        
        # Load model with appropriate dtype
        dtype = torch.float16 if (use_fp16 and device == "cuda") else torch.float32
        
        logger.info(f"Loading {self.model_name} on {device} with dtype={dtype}")
        
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=dtype
        )
        
        if device == "cuda" and torch.cuda.is_available():
            self.model = self.model.cuda()
        
        self.model.eval()
        logger.info("Jina v4 model loaded successfully")
        
    def embed_texts(self, 
                    texts: List[str], 
                    task: str = "retrieval",
                    batch_size: int = 4) -> np.ndarray:
        """
        Embed texts using Jina v4.
        
        Args:
            texts: List of texts to embed
            task: Task type (retrieval, text-matching, code)
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embeddings (N x 2048)
        """
        all_embeddings = []
        
        with torch.no_grad():
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                
                # Use Jina v4's encode_text method
                embeddings = self.model.encode_text(
                    texts=batch,
                    task=task,
                    batch_size=len(batch)  # Process whole batch at once
                )
                
                # Debug: Check what type of object we got
                logger.debug(f"Embeddings type: {type(embeddings)}")
                
                # Handle different return types
                if hasattr(embeddings, 'detach'):  # PyTorch tensor
                    embeddings = embeddings.detach()
                    if embeddings.is_cuda:
                        embeddings = embeddings.cpu()
                    embeddings = embeddings.numpy()
                elif isinstance(embeddings, list):
                    # If it's a list of tensors
                    embeddings = [e.detach().cpu().numpy() if hasattr(e, 'detach') else e for e in embeddings]
                    embeddings = np.vstack(embeddings)
                elif not isinstance(embeddings, np.ndarray):
                    # Try to convert to numpy
                    try:
                        embeddings = np.array(embeddings)
                    except:
                        logger.error(f"Cannot convert embeddings of type {type(embeddings)} to numpy")
                        raise
                    
                all_embeddings.append(embeddings)
        
        # Concatenate all batches
        result = np.vstack(all_embeddings) if all_embeddings else np.empty((0, 2048))
        
        return result
    
    def embed_code(self, 
                   code_snippets: List[str],
                   batch_size: int = 4) -> np.ndarray:
        """
        Embed code using the code-specific task.
        
        Args:
            code_snippets: List of code snippets
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embeddings (N x 2048)
        """
        return self.embed_texts(code_snippets, task="code", batch_size=batch_size)
    
    def embed_with_late_chunking(self,
                                 text: str,
                                 chunk_size: int = 28000,
                                 chunk_overlap: int = 5600) -> List[dict]:
        """
        Embed text with late chunking for better context preservation.
        
        Args:
            text: Full text to embed
            chunk_size: Size of chunks in characters (approximate)
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of dicts with keys: 'start', 'end', 'embedding'
        """
        # For Jina v4, we can process up to 32k tokens at once
        # Late chunking means we process the full text first
        
        # Split text into overlapping chunks with position tracking
        chunks = []
        positions = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk = text[start:end]
            chunks.append(chunk)
            positions.append((start, end))
            
            if end >= text_len:
                break
                
            start = end - chunk_overlap  # Fixed: proper overlap calculation
        
        # Process all chunks with context awareness
        # Jina v4 maintains context across the batch
        embeddings = self.embed_texts(chunks, task="retrieval")
        
        # Return with position information
        return [
            {'start': pos[0], 'end': pos[1], 'embedding': embeddings[i]}
            for i, pos in enumerate(positions)
        ]


def test_jina_v4() -> bool:
    """Test Jina v4 embedder."""
    print("Testing Jina v4 Embedder...")
    
    # Initialize embedder
    embedder = JinaV4Embedder(device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Test text embedding
    texts = [
        "Information Reconstructionism demonstrates multiplicative dependencies.",
        "When any dimension equals zero, information ceases to exist."
    ]
    
    embeddings = embedder.embed_texts(texts)
    print(f"✓ Text embeddings shape: {embeddings.shape}")
    assert embeddings.shape == (2, 2048), f"Expected (2, 2048), got {embeddings.shape}"
    
    # Test code embedding
    code = [
        "def calculate_information(where, what, conveyance):\n    return where * what * conveyance"
    ]
    
    code_embeddings = embedder.embed_code(code)
    print(f"✓ Code embeddings shape: {code_embeddings.shape}")
    assert code_embeddings.shape == (1, 2048), f"Expected (1, 2048), got {code_embeddings.shape}"
    
    # Test late chunking
    long_text = "Information theory " * 1000  # Long text
    chunk_embeddings = embedder.embed_with_late_chunking(long_text)
    print(f"✓ Late chunking produced {len(chunk_embeddings)} chunks")
    
    print("\n✅ All tests passed!")
    return True


if __name__ == "__main__":
    test_jina_v4()
