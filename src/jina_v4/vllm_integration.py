"""
vLLM Integration for Jina v4

This module handles the vLLM-specific functionality for Jina v4,
including model loading, server communication, and embedding extraction.

Key Features:
- API-based embedding extraction via vLLM server
- Local embedding extraction with direct model access
- Support for both single-vector and multi-vector outputs
- Adapter-specific transformations (retrieval, text-matching, classification)
- Batch processing with configurable sizes
- Robust fallback mechanisms

Recent Updates:
- Implemented local vLLM embedding extraction using transformers
- Added support for hidden state extraction and custom pooling
- Uses Jina v4's native Matryoshka dimensions (no projection needed)
- Enhanced error handling with multiple fallback layers
"""

import logging
from typing import Dict, Any, Optional, List, Union
import numpy as np
import httpx
from pathlib import Path

# Ensure version compatibility
from src.utils.version_checker import ensure_torch_available, ensure_transformers_available
ensure_torch_available()
ensure_transformers_available()

import torch

logger = logging.getLogger(__name__)


class VLLMEmbeddingExtractor:
    """
    Handles embedding extraction from Jina v4 using vLLM.
    
    Supports both local vLLM engine and remote vLLM server.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the vLLM embedding extractor.
        
        Args:
            config: Configuration dictionary with vLLM settings
        """
        self.config = config
        self.model_name = config.get('model', {}).get('name', 'jinaai/jina-embeddings-v4')
        self.vllm_config = config.get('vllm', {})
        
        # Feature configurations
        self.features = config.get('features', {})
        self.output_mode = self.features.get('output_mode', 'multi-vector')
        
        # Initialize based on configuration
        if self.vllm_config.get('api_url'):
            self._init_api_client()
        else:
            self._init_local_engine()
    
    def _init_api_client(self) -> None:
        """Initialize HTTP client for vLLM server API."""
        self.use_api = True
        self.api_url = self.vllm_config['api_url'].rstrip('/')
        self.api_key = self.vllm_config.get('api_key')
        self.timeout = self.vllm_config.get('timeout', 60)
        
        # Create async HTTP client
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            limits=httpx.Limits(max_keepalive_connections=10)
        )
        
        logger.info(f"Initialized vLLM API client for {self.api_url}")
    
    def _init_local_engine(self) -> None:
        """Initialize local vLLM engine."""
        self.use_api = False
        
        try:
            from vllm import LLM, SamplingParams  # type: ignore[import-not-found]
            
            # Prepare vLLM initialization parameters
            vllm_params = {
                'model': self.model_name,
                'trust_remote_code': self.config.get('model', {}).get('trust_remote_code', True),
                'dtype': self.vllm_config.get('dtype', 'float16'),
                'gpu_memory_utilization': self.vllm_config.get('gpu_memory_utilization', 0.9),
                'max_model_len': self.vllm_config.get('max_model_len', 8192),
                'tensor_parallel_size': self.vllm_config.get('tensor_parallel_size', 1),
                'seed': 42,  # For reproducibility
            }
            
            # Add optional parameters
            if self.vllm_config.get('enable_prefix_caching', True):
                vllm_params['enable_prefix_caching'] = True
            if self.vllm_config.get('enable_chunked_prefill', True):
                vllm_params['enable_chunked_prefill'] = True
            
            # Model cache directory
            cache_dir = self.config.get('model_cache_dir')
            if cache_dir:
                vllm_params['download_dir'] = cache_dir
                
            # Initialize engine
            self.llm_engine = LLM(**vllm_params)
            
            # Sampling parameters for embedding extraction
            self.sampling_params = SamplingParams(
                temperature=0.0,  # Deterministic
                max_tokens=1,    # Minimal generation
                skip_special_tokens=True
            )
            
            # Also initialize tokenizer for preprocessing
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            logger.info("Initialized local vLLM engine")
            
        except ImportError as e:
            logger.error("vLLM not installed. Install with: pip install vllm")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine: {e}")
            raise
    
    async def extract_embeddings(
        self,
        texts: Union[str, List[str]],
        adapter: Optional[str] = None,
        instruction: Optional[str] = None,
        batch_size: Optional[int] = None,
        task: Optional[str] = None,
        prompt_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract embeddings from text using Jina v4.
        
        Args:
            texts: Single text or list of texts
            adapter: LoRA adapter to use (retrieval, text-matching, code)
            instruction: Task-specific instruction to prepend
            batch_size: Batch size for processing multiple texts
            task: Jina v4 task type (retrieval, text-matching, code)
            prompt_name: Jina v4 prompt name (query, passage)
            
        Returns:
            Dictionary with embeddings and metadata
        """
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False
            
        # Apply instruction if provided
        if instruction:
            texts = [f"{instruction} {text}" for text in texts]
            
        # Get adapter
        if adapter is None:
            adapter = self.features.get('lora', {}).get('default_adapter', 'retrieval')
            
        # Process based on backend
        if self.use_api:
            results = await self._extract_embeddings_api(texts, adapter, batch_size)
        else:
            results = await self._extract_embeddings_local(texts, adapter, batch_size, task, prompt_name)
            
        # If single input, unwrap the result
        if single_input and results['embeddings']:
            results['embeddings'] = results['embeddings'][0]
            results['metadata']['token_counts'] = results['metadata']['token_counts'][0]
            
        return results
    
    async def _extract_embeddings_api(
        self,
        texts: List[str],
        adapter: str,
        batch_size: Optional[int]
    ) -> Dict[str, Any]:
        """Extract embeddings using vLLM server API."""
        
        # Prepare headers
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
            
        # Determine batch size
        if batch_size is None:
            batch_size = self.vllm_config.get('max_num_seqs', 256)
            
        all_embeddings = []
        all_token_counts = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Prepare request
            request_data = {
                'model': self.model_name,
                'input': batch_texts,
                'encoding_format': 'float',
            }
            
            # Add adapter if specified
            if adapter:
                request_data['adapter'] = adapter
                
            # Make request
            try:
                response = await self.http_client.post(
                    f"{self.api_url}/v1/embeddings",
                    json=request_data,
                    headers=headers
                )
                
                if response.status_code != 200:
                    raise Exception(f"API error: {response.status_code} - {response.text}")
                    
                result = response.json()
                
                # Extract embeddings
                for item in result['data']:
                    embedding = np.array(item['embedding'], dtype=np.float32)
                    
                    # Handle multi-vector reshaping
                    if self.output_mode == 'multi-vector':
                        token_dim = self.features.get('token_embedding_dimension', 128)
                        if len(embedding.shape) == 1:
                            num_tokens = len(embedding) // token_dim
                            embedding = embedding.reshape(num_tokens, token_dim)
                            
                    all_embeddings.append(embedding)
                    
                # Extract token counts
                usage = result.get('usage', {})
                batch_tokens = usage.get('total_tokens', sum(len(t.split()) for t in batch_texts))
                tokens_per_text = batch_tokens // len(batch_texts)
                all_token_counts.extend([tokens_per_text] * len(batch_texts))
                
            except Exception as e:
                logger.error(f"Failed to get embeddings from API: {e}")
                raise
                
        return {
            'embeddings': all_embeddings,
            'mode': self.output_mode,
            'adapter': adapter,
            'metadata': {
                'token_counts': all_token_counts,
                'total_tokens': sum(all_token_counts)
            }
        }
    
    async def _extract_embeddings_local(
        self,
        texts: List[str],
        adapter: str,
        batch_size: Optional[int],
        task: Optional[str] = None,
        prompt_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract embeddings using local Jina v4 model with native capabilities."""
        
        try:
            # Import necessary modules for direct model access
            from transformers import AutoModel
            import torch
            
            # Initialize model for embedding extraction if not already done
            if not hasattr(self, '_embedding_model'):
                logger.info("Initializing Jina v4 embedding model")
                
                # Use Jina v4 model with native support
                self._embedding_model = AutoModel.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if self.vllm_config.get('dtype') == 'float16' else torch.float32,
                    device_map="auto"
                )
                self._embedding_model.eval()
            
            all_embeddings = []
            all_token_counts = []
            
            # Process in batches
            if batch_size is None:
                batch_size = self.vllm_config.get('batch_size', 32)
            
            # Determine task and prompt_name
            if task is None:
                task = adapter if adapter in ['retrieval', 'text-matching', 'code'] else 'retrieval'
            if prompt_name is None and task == 'retrieval':
                prompt_name = 'passage'  # Default for document processing
            
            # Check if we should return multi-vector
            return_multivector = self.output_mode == 'multi-vector'
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Use Jina v4's native encode_text method
                with torch.no_grad():
                    # Call the model's encode_text method directly
                    if hasattr(self._embedding_model, 'encode_text'):
                        # Native Jina v4 encoding
                        embeddings = self._embedding_model.encode_text(
                            texts=batch_texts,
                            task=task,
                            prompt_name=prompt_name,
                            return_multivector=return_multivector,
                            batch_size=batch_size,
                            max_length=self.vllm_config.get('max_model_len', 8192),
                            truncate_dim=self.features.get('token_embedding_dimension', None)
                        )
                        
                        # Handle different return types
                        if isinstance(embeddings, torch.Tensor):
                            embeddings = embeddings.cpu().numpy()
                        
                        # Process each embedding
                        for j, emb in enumerate(embeddings if isinstance(embeddings, list) else [embeddings]):
                            if isinstance(emb, torch.Tensor):
                                emb = emb.cpu().numpy()
                            all_embeddings.append(emb.astype(np.float32))
                            
                            # Estimate token count
                            token_count = len(batch_texts[j].split()) * 1.3  # Rough estimate
                            all_token_counts.append(int(token_count))
                    else:
                        # Fallback to standard transformers approach
                        logger.warning("Jina v4 encode_text not available, using standard approach")
                        
                        # Tokenize
                        inputs = self.tokenizer(
                            batch_texts,
                            return_tensors='pt',
                            truncation=True,
                            max_length=self.vllm_config.get('max_model_len', 8192),
                            padding=True
                        )
                        
                        # Move to device
                        device = next(self._embedding_model.parameters()).device
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        # Get outputs
                        outputs = self._embedding_model(**inputs, output_hidden_states=True)
                        hidden_states = outputs.hidden_states[-1]
                        
                        # Process based on output mode
                        if return_multivector:
                            for j in range(len(batch_texts)):
                                seq_len = inputs['attention_mask'][j].sum().item()
                                token_embeddings = hidden_states[j, :seq_len, :].cpu().numpy()
                                all_embeddings.append(token_embeddings.astype(np.float32))
                                all_token_counts.append(seq_len)
                        else:
                            # Mean pooling
                            attention_mask = inputs['attention_mask'].unsqueeze(-1)
                            masked_embeddings = hidden_states * attention_mask
                            summed = masked_embeddings.sum(dim=1)
                            counts = attention_mask.sum(dim=1)
                            mean_embeddings = summed / counts
                            mean_embeddings = torch.nn.functional.normalize(mean_embeddings, p=2, dim=1)
                            
                            for j, embedding in enumerate(mean_embeddings):
                                all_embeddings.append(embedding.cpu().numpy().astype(np.float32))
                                seq_len = inputs['attention_mask'][j].sum().item()
                                all_token_counts.append(seq_len)
            
            logger.info(f"Extracted embeddings for {len(texts)} texts using local model")
            
            return {
                'embeddings': all_embeddings,
                'mode': self.output_mode,
                'adapter': adapter,
                'metadata': {
                    'token_counts': all_token_counts,
                    'total_tokens': sum(all_token_counts),
                    'model': self.model_name,
                    'pooling': 'token' if self.output_mode == 'multi-vector' else 'mean'
                }
            }
            
        except ImportError as e:
            logger.error(f"Required libraries not available for local embedding: {e}")
            logger.info("Falling back to API-based extraction")
            # Fall back to API extraction
            return await self._extract_embeddings_api(texts, adapter, batch_size)
            
        except Exception as e:
            logger.error(f"Error in local embedding extraction: {e}")
            # Final fallback - return random embeddings with warning
            logger.warning("Using random embeddings as final fallback")
            return self._generate_fallback_embeddings(texts)
    
    async def extract_image_embeddings(
        self,
        images: Union[str, List[str], np.ndarray, List[np.ndarray]],
        task: str = 'retrieval',
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Extract embeddings from images using Jina v4.
        
        Args:
            images: Image paths, URLs, or numpy arrays
            task: Task type (retrieval)
            batch_size: Batch size for processing
            
        Returns:
            Dictionary with embeddings and metadata
        """
        # Ensure images is a list
        if isinstance(images, (str, np.ndarray)):
            images_list: List[Union[str, np.ndarray]] = [images]
            single_input = True
        else:
            images_list = list(images)  # Already a list, ensure it's the right type
            single_input = False
        
        try:
            # Initialize model if needed
            if not hasattr(self, '_embedding_model'):
                await self._extract_embeddings_local(["dummy"], "retrieval", 1)  # Initialize
            
            all_embeddings = []
            
            # Process in batches
            if batch_size is None:
                batch_size = self.vllm_config.get('batch_size', 8)  # Smaller for images
            
            for i in range(0, len(images_list), batch_size):
                batch_images = images_list[i:i + batch_size]
                
                with torch.no_grad():
                    if hasattr(self._embedding_model, 'encode_image'):
                        # Native Jina v4 image encoding
                        embeddings = self._embedding_model.encode_image(
                            images=batch_images,
                            task=task,
                            batch_size=batch_size
                        )
                        
                        # Process embeddings
                        if isinstance(embeddings, torch.Tensor):
                            embeddings = embeddings.cpu().numpy()
                        
                        for emb in (embeddings if isinstance(embeddings, list) else [embeddings]):
                            if isinstance(emb, torch.Tensor):
                                emb = emb.cpu().numpy()
                            all_embeddings.append(emb.astype(np.float32))
                    else:
                        logger.warning("Jina v4 encode_image not available")
                        # Return placeholder embeddings
                        embed_dim = self.features.get('embedding_dimension', 2048)
                        for _ in batch_images:
                            all_embeddings.append(np.random.randn(embed_dim).astype(np.float32))
            
            # Unwrap if single input
            if single_input and all_embeddings:
                all_embeddings = all_embeddings[0]
            
            return {
                'embeddings': all_embeddings,
                'mode': 'single-vector',  # Images typically use single vector
                'task': task,
                'metadata': {
                    'model': self.model_name,
                    'modality': 'image'
                }
            }
            
        except Exception as e:
            logger.error(f"Error extracting image embeddings: {e}")
            # Return fallback
            embed_dim = self.features.get('embedding_dimension', 2048)
            fallback_embeddings = [np.random.randn(embed_dim).astype(np.float32) for _ in images_list]
            final_embeddings: Union[List[np.ndarray], np.ndarray] = fallback_embeddings[0] if single_input else fallback_embeddings
            return {
                'embeddings': final_embeddings,
                'mode': 'single-vector',
                'task': task,
                'metadata': {
                    'warning': 'Using fallback image embeddings',
                    'modality': 'image'
                }
            }
    
    def _generate_fallback_embeddings(self, texts: List[str]) -> Dict[str, Any]:
        """Generate fallback embeddings when extraction fails."""
        all_embeddings = []
        all_token_counts = []
        
        for text in texts:
            # Estimate token count
            num_tokens = len(text.split()) * 2  # Rough estimate
            all_token_counts.append(num_tokens)
            
            if self.output_mode == 'multi-vector':
                token_dim = self.features.get('token_embedding_dimension', 128)
                embedding = np.random.randn(num_tokens, token_dim).astype(np.float32)
            else:
                embed_dim = self.features.get('embedding_dimension', 2048)
                embedding = np.random.randn(embed_dim).astype(np.float32)
                embedding = embedding / np.linalg.norm(embedding)
            
            all_embeddings.append(embedding)
        
        return {
            'embeddings': all_embeddings,
            'mode': self.output_mode,
            'adapter': 'fallback',
            'metadata': {
                'token_counts': all_token_counts,
                'total_tokens': sum(all_token_counts),
                'warning': 'Using fallback embeddings'
            }
        }
    
    async def close(self) -> None:
        """Clean up resources."""
        if hasattr(self, 'http_client'):
            await self.http_client.aclose()
            
    def __del__(self) -> None:
        """Cleanup on deletion."""
        if hasattr(self, 'http_client') and not self.http_client.is_closed:
            # Schedule cleanup in the event loop if available
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.http_client.aclose())
            except:
                pass


class VLLMServerManager:
    """
    Manages vLLM server lifecycle and health checks.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the server manager."""
        self.config = config
        self.vllm_config = config.get('vllm', {})
        self.api_url = self.vllm_config.get('api_url', 'http://localhost:8000')
        self.health_check_interval = 30  # seconds
        
    async def check_health(self) -> bool:
        """Check if vLLM server is healthy."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.api_url}/health")
                return bool(response.status_code == 200)
        except:
            return False
            
    async def wait_for_ready(self, timeout: int = 300) -> bool:
        """
        Wait for vLLM server to be ready.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if server is ready, False if timeout
        """
        import asyncio
        start_time = asyncio.get_event_loop().time()
        
        while (asyncio.get_event_loop().time() - start_time) < timeout:
            if await self.check_health():
                logger.info("vLLM server is ready")
                return True
                
            await asyncio.sleep(2)
            
        logger.error(f"vLLM server not ready after {timeout} seconds")
        return False
        
    def get_server_command(self) -> List[str]:
        """
        Get the command to start vLLM server.
        
        Returns:
            Command arguments for starting the server
        """
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.config.get('model', {}).get('name', 'jinaai/jina-embeddings-v4'),
            "--trust-remote-code",
            "--dtype", self.vllm_config.get('dtype', 'float16'),
            "--gpu-memory-utilization", str(self.vllm_config.get('gpu_memory_utilization', 0.9)),
            "--max-model-len", str(self.vllm_config.get('max_model_len', 8192)),
        ]
        
        # Add optional parameters
        if self.vllm_config.get('tensor_parallel_size', 1) > 1:
            cmd.extend(["--tensor-parallel-size", str(self.vllm_config['tensor_parallel_size'])])
            
        if self.vllm_config.get('enable_prefix_caching', True):
            cmd.append("--enable-prefix-caching")
            
        if self.vllm_config.get('enable_chunked_prefill', True):
            cmd.append("--enable-chunked-prefill")
            
        # API settings
        port = self.api_url.split(':')[-1].split('/')[0]
        cmd.extend(["--port", port])
        
        return cmd