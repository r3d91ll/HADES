"""
ModernBERT Embedding Adapter for the HADES project.

This module provides an implementation of the EmbeddingAdapter protocol
using ModernBERT for generating document embeddings.
"""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import numpy as np
import torch
from torch import nn

from src.embedding.base import EmbeddingAdapter, EmbeddingVector, register_adapter
from src.utils.device_utils import get_device_info

logger = logging.getLogger(__name__)


# Helper function for running CPU-bound operations in a thread pool
async def run_in_threadpool(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Run a function in a thread pool and return its result asynchronously."""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, lambda: func(*args, **kwargs))


class ModernBERTPipeline:
    """Pipeline for generating embeddings with ModernBERT."""
    
    def __init__(self, model: Any, tokenizer: Any, device: str = "cpu") -> None:
        """Initialize the ModernBERT pipeline.
        
        Args:
            model: ModernBERT model
            tokenizer: ModernBERT tokenizer
            device: Device to run inference on ("cpu", "cuda", "cuda:0", etc.)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Move model to specified device
        if self.model is not None:
            self.model.to(self.device)
    
    def __call__(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
            
        # Tokenize texts
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Extract embeddings from model outputs (depends on model architecture)
        if hasattr(outputs, "pooler_output"):
            embeddings = outputs.pooler_output
        else:
            # Use mean of last hidden state as fallback
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        # Convert to numpy arrays
        return [emb.cpu().numpy() for emb in embeddings]


class EncoderEmbeddingAdapter(EmbeddingAdapter):
    """Adapter for generating embeddings using encoder models like ModernBERT."""
    
    # Class-level logger
    logger = logging.getLogger(__name__)
    
    # Flag to force CPU usage even if GPU is available
    force_cpu = False
    
    def __init__(
        self,
        model_name: str = "modernbert",
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Initialize the encoder embedding adapter.
        
        Args:
            model_name: Name of the encoder model to use
            model_path: Path to the model directory (None = use default)
            device: Device to run inference on (None = auto-detect)
            **kwargs: Additional parameters for the model
        """
        self.model_name = model_name
        self.model_path = model_path
        self.requested_device = device
        self.kwargs = kwargs
        
        # Will be initialized lazily
        self._model: Any = None
        self._tokenizer: Any = None
        self._pipeline: Any = None
    
    @classmethod
    def _get_pipeline_type(cls, model_name: str) -> str:
        """Get the pipeline type for a given model name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Pipeline type name
        """
        # ModernBERT uses a custom pipeline
        if model_name.lower() in ["modernbert", "modern-bert", "modern_bert"]:
            return "modernbert"
        
        # Other models can use standard Hugging Face pipelines
        return "feature-extraction"
    
    async def _ensure_model_loaded(self) -> None:
        """Ensure the model is loaded before use."""
        if self._pipeline is not None:
            return
            
        # Determine the device to use
        device = self._get_adjusted_device()
        
        # Import here to avoid circular imports
        if self.model_name.lower() in ["modernbert", "modern-bert", "modern_bert"]:
            await self._load_modernbert(device)
        else:
            await self._load_huggingface_model(device)
    
    def _get_adjusted_device(self) -> str:
        """Get the adjusted device based on availability and settings.
        
        Returns:
            Device string for PyTorch
        """
        if self.force_cpu:
            self.logger.info("Forcing CPU usage as requested")
            return "cpu"
            
        # Use requested device if specified
        if self.requested_device:
            self.logger.info(f"Using requested device: {self.requested_device}")
            return self.requested_device
            
        # Auto-detect device
        device = get_device_info().get("device", "cpu")
        self.logger.info(f"Auto-detected device: {device}")
        return cast(str, device)
    
    async def _load_modernbert(self, device: str) -> None:
        """Load the ModernBERT model and tokenizer.
        
        Args:
            device: Device to load the model on
        """
        try:
            # Only import when needed
            from transformers import AutoModel, AutoTokenizer
            
            self.logger.info(f"Loading ModernBERT model on {device}")
            
            # Determine model path
            model_path = self.model_path
            if not model_path:
                model_path = os.environ.get(
                    "MODERNBERT_MODEL_PATH", 
                    "sentence-transformers/all-mpnet-base-v2"
                )
            
            # Load tokenizer and model
            self._tokenizer = await run_in_threadpool(
                AutoTokenizer.from_pretrained, model_path
            )
            
            self._model = await run_in_threadpool(
                AutoModel.from_pretrained, model_path
            )
            
            # Create pipeline
            self._pipeline = ModernBERTPipeline(
                model=self._model,
                tokenizer=self._tokenizer,
                device=device
            )
            
            self.logger.info(f"ModernBERT model loaded successfully on {device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load ModernBERT model: {e}")
            raise RuntimeError(f"ModernBERT model loading failed: {e}") from e
    
    async def _load_huggingface_model(self, device: str) -> None:
        """Load a Hugging Face model for embeddings.
        
        Args:
            device: Device to load the model on
        """
        try:
            # Only import when needed
            from transformers import pipeline
            
            self.logger.info(f"Loading {self.model_name} model on {device}")
            
            # Determine pipeline type
            pipeline_type = self._get_pipeline_type(self.model_name)
            
            # Create pipeline
            self._pipeline = await run_in_threadpool(
                pipeline,
                task=pipeline_type,
                model=self.model_path or self.model_name,
                device=device
            )
            
            self.logger.info(f"{self.model_name} model loaded successfully on {device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load {self.model_name} model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e
    
    async def embed(self, texts: List[str], **kwargs: Any) -> List[EmbeddingVector]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            List of embedding vectors, one for each input text
            
        Raises:
            RuntimeError: If the embedding operation fails
        """
        if not texts:
            return []
            
        # Ensure model is loaded
        await self._ensure_model_loaded()
        
        try:
            # Generate embeddings
            if self.model_name.lower() in ["modernbert", "modern-bert", "modern_bert"]:
                # Run the pipeline in a thread pool to avoid blocking
                results = await run_in_threadpool(self._pipeline, texts)
                
                # Convert ndarray to list if needed for serialization
                if kwargs.get("as_list", False):
                    results = [emb.tolist() for emb in results]
                    
                # Explicitly cast the result to match the return type
                from typing import cast, List, Union
                return cast(List[Union[List[float], np.ndarray]], results)
                
            else:
                # Other Hugging Face models
                outputs = await run_in_threadpool(
                    self._pipeline, texts, **{**self.kwargs, **kwargs}
                )
                
                # Extract embeddings based on output format
                if isinstance(outputs[0], dict) and "embedding" in outputs[0]:
                    embeddings = [output["embedding"] for output in outputs]
                else:
                    # Use the entire output as the embedding
                    embeddings = outputs
                    
                return cast(List[EmbeddingVector], embeddings)
                
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            raise RuntimeError(f"Failed to generate embeddings: {e}") from e
    
    async def embed_single(self, text: str, **kwargs: Any) -> EmbeddingVector:
        """Generate an embedding for a single text.
        
        Args:
            text: Text string to embed
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            Embedding vector for the input text
            
        Raises:
            RuntimeError: If the embedding operation fails
        """
        results = await self.embed([text], **kwargs)
        
        if not results:
            raise RuntimeError("Empty results returned from embedding model")
            
        return results[0]


# Register the adapter under various names
register_adapter("modernbert", EncoderEmbeddingAdapter)
register_adapter("encoder", EncoderEmbeddingAdapter)
