"""Encoder adapter for embedding generation using Haystack.

This module provides an implementation of the EmbeddingAdapter protocol
that uses encoder models (like ModernBERT, CodeBERT) to convert text into
vector embeddings. It supports both CPU and GPU configurations through
the embedding_config.yaml configuration file.

The adapter is designed to work with any HuggingFace encoder model and can be
configured to use different models based on the content type or project requirements.

Supported models include:
- ModernBERT: Optimized for general text (answerdotai/ModernBERT-base)
- CodeBERT: Optimized for code and technical documentation (microsoft/codebert-base)
"""

from __future__ import annotations
import asyncio
import logging
from typing import Dict, List, Optional, Union, Any, cast, Protocol, TypeVar, Literal, overload
from types import TracebackType

import logging
import os
import torch
from typing import Any, Dict, List, Optional, Tuple, Union, cast

# Local imports
from src.embedding.base import EmbeddingAdapter, EmbeddingVector
from src.embedding.registry import register_adapter
from src.model_engine.engines.haystack import HaystackModelEngine
from transformers import AutoModel, AutoTokenizer
from src.config.embedding_config import get_adapter_config, load_config

# Type variables for better type inference
T = TypeVar('T')
ResponseDict = Dict[str, Any]
PoolingStrategy = Literal["cls", "mean", "max"]

logger = logging.getLogger(__name__)

class EncoderEmbeddingAdapter(EmbeddingAdapter):
    """Adapter for generating embeddings using encoder models via Haystack.
    
    This adapter can be configured to use different encoder models like:
    - ModernBERT (answerdotai/ModernBERT-base): Good for general text content
    - CodeBERT (microsoft/codebert-base): Optimized for code and technical documentation
    
    The model is specified via configuration or constructor parameters.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        max_length: Optional[int] = None,
        pooling_strategy: Optional[PoolingStrategy] = None,
        batch_size: Optional[int] = None,
        normalize_embeddings: Optional[bool] = None,
        device: Optional[str] = None,
        adapter_name: str = "transformer",
        **kwargs: Any
    ) -> None:
        """Initialize the transformer embedding adapter using configuration.
        
        This constructor loads settings from the embedding_config.yaml file,
        while allowing parameter overrides through constructor arguments.
        
        Args:
            model_name: Override the model name from config (e.g., 'microsoft/codebert-base')
            max_length: Override the max sequence length from config
            pooling_strategy: Override the pooling strategy from config (cls, mean, max)
            batch_size: Override the batch size from config
            normalize_embeddings: Override normalization setting from config
            device: Override the device from config (cpu, cuda:0, etc.)
            adapter_name: Name of the adapter configuration to use
            **kwargs: Additional parameters for the embedding model
        """
        # Load configuration
        config = get_adapter_config(adapter_name=adapter_name)
        
        # Set parameters, allowing constructor arguments to override config
        # Default model depends on adapter_name if specified
        default_model = "microsoft/codebert-base" if adapter_name == "codebert" else "answerdotai/ModernBERT-base"
        self.model_name: str = model_name or config.get("model_name", default_model)
        self.max_length: int = max_length or config.get("max_length", 8192)
        self.pooling_strategy: PoolingStrategy = pooling_strategy or config.get("pooling_strategy", "cls")
        self.batch_size: int = batch_size or config.get("batch_size", 8)
        self.normalize_embeddings: bool = normalize_embeddings if normalize_embeddings is not None else config.get("normalize_embeddings", True)
        
        # Check for device setting in the pipeline configuration
        pipeline_device = self._get_pipeline_device('embedding')
        # Allow explicit parameter to override, then pipeline config, then adapter config
        self.device: str = device or pipeline_device or config.get("device", "cpu")
        
        # Log the device selection
        if pipeline_device:
            logger.info(f"Using device from pipeline config: {pipeline_device}")
        elif device:
            logger.info(f"Using explicitly provided device: {device}")
        else:
            logger.info(f"Using device from adapter config: {self.device}")
        
        # Adjust for CUDA_VISIBLE_DEVICES remapping
        self.adjusted_device = self._get_adjusted_device(self.device)
        
        # Engine configuration
        self.use_model_engine: bool = config.get("use_model_engine", True)
        self.engine_type: str = config.get("engine_type", "haystack")
        
        # Store the full configuration for reference
        self.config = config
        
        # Haystack engine instance
        self._engine: Optional[HaystackModelEngine] = None
        
        # Model loading state
        self._model_loaded: bool = False
        
        # Log initialization
        logger.info(
            f"Initialized ModernBERT adapter with model={self.model_name}, "
            f"device={self.adjusted_device}, pooling={self.pooling_strategy}"
        )
        
    def _get_pipeline_device(self, component_name: str) -> Optional[str]:
        """Get the device configuration for a specific component from the pipeline config.
        
        Args:
            component_name: Name of the component (e.g., 'embedding', 'chunking', 'docproc')
            
        Returns:
            The configured device string or None if not configured/available
        """
        try:
            # Import here to avoid circular imports
            from src.config.config_loader import get_component_device
            
            # Get device for the specified component
            return get_component_device(component_name)
        except Exception as e:
            logger.warning(f"Could not get component device from pipeline config: {e}")
            return None
    
    def _get_adjusted_device(self, device: str) -> str:
        """Adjust requested device based on CUDA_VISIBLE_DEVICES setting.
        
        When CUDA_VISIBLE_DEVICES is set, it remaps available devices. For example,
        with CUDA_VISIBLE_DEVICES=1, the physical GPU 1 is available as cuda:0.
        This function handles this remapping.
        
        Args:
            device: The requested device (e.g., 'cuda:1')
            
        Returns:
            Adjusted device string for current environment
        """
        if not device.startswith('cuda:'):
            return device
            
        try:
            # Check how many GPUs are actually visible to PyTorch
            import torch
            visible_count = torch.cuda.device_count()
            
            # Extract device index from cuda:N format
            device_idx = int(device.split(':')[1])
            
            # If requested index is higher than available count, use highest available
            if device_idx >= visible_count and visible_count > 0:
                logger.warning(f"Requested device {device} is not available with CUDA_VISIBLE_DEVICES setting. "
                              f"Only {visible_count} devices visible. Using cuda:{visible_count-1} instead.")
                return f"cuda:{visible_count-1}"
                
            # If no GPUs available but CUDA requested, warn and use CPU
            if visible_count == 0 and device.startswith('cuda'):
                logger.warning(f"No CUDA devices available. Falling back to CPU.")
                return "cpu"
                
            # For valid configurations, return the requested device
            return device
            
        except Exception as e:
            logger.warning(f"Error adjusting CUDA device mapping: {e}. Using requested device: {device}")
            return device
    
    @property
    async def _ensure_model_loaded(self) -> None:
        """Ensure the encoder model is loaded."""
        if self._model is None:
            try:
                # Determine model type for better logging
                model_type = "CodeBERT" if "codebert" in self.model_name.lower() else "ModernBERT"
                
                # Check available GPU memory before loading
                effective_device = self.device
                if torch.cuda.is_available() and not self.force_cpu and "cuda" in self.device:
                    try:
                        # Check memory availability on the target GPU
                        device_idx = int(self.device.split(":")[1]) if ":" in self.device else 0
                        free_memory = torch.cuda.mem_get_info(device_idx)[0] / (1024 ** 3)  # Free memory in GB
                        self.logger.info(f"Available GPU memory before loading {model_type}: {free_memory:.2f} GB")
                        if free_memory < 1.0:  # Less than 1GB available
                            self.logger.warning(f"Low GPU memory ({free_memory:.2f} GB). Forcing CPU mode for {model_type}.")
                            effective_device = "cpu"
                    except Exception as e:
                        self.logger.warning(f"Couldn't check GPU memory: {e}. Using configured device: {self.device}")
                elif not torch.cuda.is_available() and "cuda" in self.device:
                    self.logger.warning(f"CUDA not available, falling back to CPU for {model_type}")
                    effective_device = "cpu"
                
                # Load tokenizer
                self._tokenizer = await run_in_threadpool(
                    AutoTokenizer.from_pretrained, self.model_name
                )
                self.logger.info(f"{model_type} tokenizer loaded successfully")
                
                # Load model with memory efficiency options
                if effective_device == "cpu":
                    # Use lower precision for CPU to save memory
                    self._model = await run_in_threadpool(
                        lambda: AutoModel.from_pretrained(
                            self.model_name,
                            low_cpu_mem_usage=True
                        )
                    )
                else:
                    # Use half precision for GPU to save memory
                    self._model = await run_in_threadpool(
                        lambda: AutoModel.from_pretrained(
                            self.model_name,
                            torch_dtype=torch.float16,
                            low_cpu_mem_usage=True
                        )
                    )
                
                # Move to target device and set to evaluation mode
                self._model = self._model.to(effective_device)
                self._model.eval()
                self.logger.info(f"{model_type} model loaded successfully on {effective_device}")
                self._model_loaded = True
                
            except Exception as e:
                self.logger.error(f"Error loading {model_type if 'model_type' in locals() else 'encoder'} model: {e}")
                raise RuntimeError(f"Failed to load model {self.model_name}: {e}") from e
    
    @property
    def engine(self) -> HaystackModelEngine:
        """Lazy-load the Haystack engine when first needed."""
        if self._engine is None:
            try:
                from src.model_engine.engines.haystack import HaystackModelEngine
                
                logger.info("Initializing Haystack engine for ModernBERT embeddings")
                self._engine = HaystackModelEngine()
                
                # Make sure the engine is running
                status: Dict[str, Any] = self._engine.get_status()
                if not status.get("running", False):
                    logger.info("Starting Haystack engine")
                    start_result: bool = self._engine.start()
                    if not start_result:
                        raise RuntimeError("Failed to start Haystack engine")
                
                logger.info("Haystack engine initialized successfully")
            except ImportError as e:
                logger.error(f"Failed to import HaystackModelEngine: {e}")
                raise
            except Exception as e:
                logger.error(f"Failed to initialize Haystack engine: {e}")
                raise
        
        if self._engine is None:  # This is for type checker only
            raise RuntimeError("Engine initialization failed but no exception was raised")
            
        return self._engine
    
    async def _ensure_model_loaded(self) -> None:
        """Ensure the ModernBERT model is loaded and available.
        
        This method loads the model directly using the HuggingFace transformers library
        for CPU inference to avoid issues with the Haystack caching system.
        
        Raises:
            RuntimeError: If model loading fails
        """
        if self._model_loaded:
            return
        
        try:
            # For CPU-based inference, we'll load the model directly rather than using Haystack
            import torch
            
            # Use the adjusted device that accounts for CUDA_VISIBLE_DEVICES settings
            effective_device = self.adjusted_device
            
            # Log the actual device being used
            logger.info(f"Loading ModernBERT model directly: {self.model_name} on {effective_device}")
            
            # Load tokenizer first
            if not hasattr(self, '_tokenizer') or self._tokenizer is None:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                logger.info("ModernBERT tokenizer loaded successfully")

            # Load model if not already loaded
            if not hasattr(self, '_model') or self._model is None:
                self._model = AutoModel.from_pretrained(self.model_name)
                self._model.to(effective_device)
                self._model.eval()  # Set to evaluation mode
                logger.info("ModernBERT model loaded successfully")
            
            self._model_loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load ModernBERT model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e
            
    # This commented-out code appears to be from initialization and should be in __init__
    # It's removed here to fix the syntax error
    
    # Helper method to load model when needed
    async def _load_model(self):
        """Load the encoder model and tokenizer.
        
        This helper method ensures proper loading of the model with appropriate
        error handling.
        """
        await self._ensure_model_loaded()
        logger.info(
            f"Initialized ModernBERT adapter with model={self.model_name}, "
            f"device={self.adjusted_device}, pooling={self.pooling_strategy}"
        )
        
    # Define a property to check if we need to unload the model
    @property
    def should_unload_model(self) -> bool:
        """Check if the model should be unloaded to free memory."""
        return getattr(self, "_model_unloaded", True)
    
    def _get_pipeline_device(self, component_name: str) -> Optional[str]:
        """Get the device configuration for a specific component from the pipeline config.
        
        Args:
            component_name: Name of the component (e.g., 'embedding', 'chunking', 'docproc')
            
        Returns:
            The configured device string or None if not configured/available
        """
        try:
            # Import here to avoid circular imports
            from src.config.config_loader import get_component_device
            return get_component_device(component_name)
        except (ImportError, AttributeError) as e:
            logger.debug(f"Could not get pipeline device: {e}")
            return None
        
    async def _get_embeddings_from_model(self, texts: List[str]) -> List[EmbeddingVector]:
        """Generate embeddings directly using the transformer model.
        
        This method directly interfaces with the transformer model to generate embeddings,
        rather than using the Haystack embedding calculation function. This approach ensures
        we have full control over the embedding process.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            RuntimeError: If the embedding generation fails
        """

        if not self._model_loaded:
            await self._ensure_model_loaded()
        
        try:
            # Process the texts in batches to avoid OOM issues
            all_embeddings = []
            
            # Create mini-batches to process
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                logger.debug(f"Processing batch {i//self.batch_size + 1} with {len(batch_texts)} texts")
                
                with torch.no_grad():
                    # Tokenize inputs with explicit truncation
                    # Most BERT models have a max sequence length of 512
                    max_length = 512
                    if 'bert-base' in self.model_name.lower():
                        max_length = 512
                    elif 'codebert' in self.model_name.lower():
                        max_length = 512
                    
                    # Log the truncation happening
                    if any(len(self._tokenizer.encode(text)) > max_length for text in batch_texts):
                        logger.warning(f"Some texts exceed the maximum sequence length of {max_length} tokens for model {self.model_name}. Truncating.")
                    
                    # Apply truncation and padding
                    encoding = self._tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt"
                    )
                    
                    # Move tensors to the correct device
                    input_ids = encoding["input_ids"].to(self._model.device)
                    attention_mask = encoding["attention_mask"].to(self._model.device)
                    
                    # Forward pass
                    outputs = self._model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True
                    )
                    
                    # Apply pooling based on strategy
                    if self.pooling_strategy == "cls":
                        # CLS token embedding (first token)
                        batch_embeddings = outputs.last_hidden_state[:, 0, :]
                    elif self.pooling_strategy == "mean":
                        # Mean pooling - average token embeddings
                        token_embeddings = outputs.last_hidden_state
                        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                        batch_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    elif self.pooling_strategy == "max":
                        # Max pooling - take maximum values
                        token_embeddings = outputs.last_hidden_state
                        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                        batch_embeddings = torch.max(token_embeddings * input_mask_expanded, dim=1)[0]
                    else:
                        raise ValueError(f"Invalid pooling strategy: {self.pooling_strategy}")
                    
                    # Normalize if requested
                    if self.normalize_embeddings:
                        batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                    
                    # Convert to list and add to results
                    batch_embeddings_np = batch_embeddings.cpu().numpy()
                    all_embeddings.extend(batch_embeddings_np.tolist())
            
            logger.info(f"Generated {len(all_embeddings)} embeddings with dimension {len(all_embeddings[0]) if all_embeddings else 0}")
            return cast(List[EmbeddingVector], all_embeddings)
        
        except Exception as e:
            logger.error(f"ModernBERT embedding generation failed: {e}")
            raise RuntimeError(f"Failed to generate embeddings: {e}") from e

    async def embed(self, texts: List[str], **kwargs: Any) -> List[EmbeddingVector]:
        """Generate embeddings for a list of texts using ModernBERT.
        
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
        
        # Log information about the chunking
        token_estimates: List[float] = [len(text.split()) * 1.3 for text in texts]  # Rough token estimate
        for i, (text, tokens) in enumerate(zip(texts, token_estimates)):
            if tokens > self.max_length:
                logger.warning(f"Text {i} may exceed the model's context length: {tokens:.0f} tokens (approx)")
        
        batch_size: int = kwargs.get("batch_size", self.batch_size)
        logger.info(f"Processing {len(texts)} texts with batch size {batch_size}")
        
        results: List[EmbeddingVector] = []
        # Process in batches to avoid memory issues
        for i in range(0, len(texts), batch_size):
            batch: List[str] = texts[i:i + batch_size]
            logger.debug(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            batch_results: List[EmbeddingVector] = await self._get_embeddings_from_model(batch)
            results.extend(batch_results)
        
        return results

    async def embed_single(self, text: str, **kwargs: Any) -> EmbeddingVector:
        """Generate an embedding for a single text using ModernBERT.
        
        Args:
            text: Text string to embed
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            Embedding vector for the input text
            
        Raises:
            RuntimeError: If the embedding operation fails
        """
        results: List[EmbeddingVector] = await self.embed([text], **kwargs)
        if not results:
            raise RuntimeError("Empty results from ModernBERT embedding model")
        return results[0]


# Register the adapter for different model types
register_adapter("modernbert", EncoderEmbeddingAdapter) # For general text documents
register_adapter("codebert", EncoderEmbeddingAdapter)  # For code repositories
register_adapter("python_code_bert", EncoderEmbeddingAdapter)  # For Python code specifically
register_adapter("encoder", EncoderEmbeddingAdapter)  # Generic name
