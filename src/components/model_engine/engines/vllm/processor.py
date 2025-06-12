"""
vLLM Model Engine Component

This module provides the vLLM model engine component that implements the
ModelEngine protocol. It wraps the existing HADES vLLM engine functionality
with the new component contract system and provides optimized GPU acceleration.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import asyncio

# Import existing HADES vLLM engine functionality
from src.components.model_engine.engines.vllm.vllm_engine import VLLMModelEngine as LegacyVLLMEngine

# Import component contracts and protocols
from src.types.components.contracts import (
    ComponentType,
    ComponentMetadata,
    ModelEngineInput,
    ModelEngineOutput,
    ModelInferenceResult,
    ProcessingStatus
)
from src.types.components.protocols import ModelEngine

# Import types
from src.types.components.model_engine.inference import InferenceRequest


class VLLMModelEngine(ModelEngine):
    """
    vLLM model engine component implementing ModelEngine protocol.
    
    This component uses the existing HADES vLLM engine to provide
    high-performance GPU-accelerated model serving with optimized
    batching and memory management.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize vLLM model engine.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self._config = config or {}
        
        # Component metadata
        self._metadata = ComponentMetadata(
            component_type=ComponentType.MODEL_ENGINE,
            component_name="vllm",
            component_version="1.0.0",
            config=self._config
        )
        
        # Initialize vLLM engine
        self._engine: Optional[LegacyVLLMEngine] = None
        
        # Configuration
        self._server_url = self._config.get('server_url', 'http://localhost:8000')
        self._device = self._config.get('device', 'cuda')
        self._max_retries = self._config.get('max_retries', 3)
        self._timeout = self._config.get('timeout', 60)
        
        self.logger.info(f"Initialized vLLM model engine with server: {self._server_url}")
    
    @property
    def name(self) -> str:
        """Component name for identification."""
        return "vllm"
    
    @property
    def version(self) -> str:
        """Component version string."""
        return "1.0.0"
    
    @property
    def component_type(self) -> ComponentType:
        """Type of component."""
        return ComponentType.MODEL_ENGINE
    
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure component with parameters.
        
        Args:
            config: Configuration dictionary containing component parameters
            
        Raises:
            ValueError: If configuration is invalid
        """
        if not self.validate_config(config):
            raise ValueError("Invalid configuration provided")
        
        self._config.update(config)
        self._metadata.config = self._config
        self._metadata.processed_at = datetime.utcnow()
        
        # Update configuration
        if 'server_url' in config:
            self._server_url = config['server_url']
        
        if 'device' in config:
            self._device = config['device']
        
        if 'max_retries' in config:
            self._max_retries = config['max_retries']
        
        if 'timeout' in config:
            self._timeout = config['timeout']
        
        # Reset engine to force re-initialization
        self._engine = None
        
        self.logger.info(f"Updated vLLM model engine configuration")
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        if not isinstance(config, dict):
            return False
        
        # Validate server URL
        if 'server_url' in config:
            if not isinstance(config['server_url'], str):
                return False
        
        # Validate device
        if 'device' in config:
            valid_devices = ['cuda', 'cpu']
            if config['device'] not in valid_devices:
                return False
        
        # Validate numeric parameters
        if 'max_retries' in config:
            if not isinstance(config['max_retries'], int) or config['max_retries'] < 1:
                return False
        
        if 'timeout' in config:
            if not isinstance(config['timeout'], (int, float)) or config['timeout'] <= 0:
                return False
        
        return True
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get the JSON schema for component configuration.
        
        Returns:
            JSON schema dictionary describing valid configuration
        """
        return {
            "type": "object",
            "properties": {
                "server_url": {
                    "type": "string",
                    "description": "URL of the vLLM server",
                    "default": "http://localhost:8000"
                },
                "device": {
                    "type": "string",
                    "enum": ["cuda", "cpu"],
                    "description": "Device to run models on",
                    "default": "cuda"
                },
                "max_retries": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum number of API call retries",
                    "default": 3
                },
                "timeout": {
                    "type": "number",
                    "minimum": 1,
                    "description": "API call timeout in seconds",
                    "default": 60
                },
                "gpu_memory_utilization": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "GPU memory utilization fraction",
                    "default": 0.8
                },
                "max_model_len": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum model sequence length",
                    "default": 4096
                },
                "batch_size": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Default batch size for processing",
                    "default": 32
                }
            }
        }
    
    def health_check(self) -> bool:
        """
        Check if component is healthy and ready to process data.
        
        Returns:
            True if component is healthy, False otherwise
        """
        try:
            # Initialize engine if needed
            if not self._engine:
                self._initialize_engine()
            
            # Check if engine is available and server is running
            if not self._engine:
                return False
            
            # Try to check server status
            return self._engine.running
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get component performance metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        metrics = {
            "component_name": self.name,
            "component_version": self.version,
            "server_url": self._server_url,
            "device": self._device,
            "engine_initialized": self._engine is not None,
            "server_running": self._engine.running if self._engine else False,
            "loaded_models": self._engine.list_loaded_models() if self._engine else [],
            "last_health_check": datetime.utcnow().isoformat()
        }
        
        # Add GPU metrics if available
        if self._device == 'cuda':
            try:
                import torch
                if torch.cuda.is_available():
                    metrics.update({
                        "gpu_available": True,
                        "gpu_count": torch.cuda.device_count(),
                        "gpu_memory_allocated": torch.cuda.memory_allocated(),
                        "gpu_memory_cached": torch.cuda.memory_reserved()
                    })
            except ImportError:
                pass
        
        return metrics
    
    def process(self, input_data: ModelEngineInput) -> ModelEngineOutput:
        """
        Process model inference requests according to the contract.
        
        Args:
            input_data: Input data conforming to ModelEngineInput contract
            
        Returns:
            Output data conforming to ModelEngineOutput contract
        """
        errors = []
        processing_stats = {}
        
        try:
            start_time = datetime.utcnow()
            
            # Initialize engine if needed
            if not self._engine:
                self._initialize_engine()
            
            if not self._engine:
                raise ValueError("Could not initialize vLLM engine")
            
            # Start server if not running
            if not self._engine.running:
                if not self._engine.start():
                    raise ValueError("Could not start vLLM server")
            
            # Process inference requests
            results = asyncio.run(self._process_requests_async(input_data.requests))
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Update metadata
            metadata = ComponentMetadata(
                component_type=self.component_type,
                component_name=self.name,
                component_version=self.version,
                processing_time=processing_time,
                processed_at=datetime.utcnow(),
                config=self._config,
                status=ProcessingStatus.SUCCESS if not errors else ProcessingStatus.ERROR
            )
            
            return ModelEngineOutput(
                results=results,
                metadata=metadata,
                engine_stats={
                    "processing_time": processing_time,
                    "request_count": len(input_data.requests),
                    "success_count": len([r for r in results if not r.error]),
                    "error_count": len([r for r in results if r.error]),
                    "server_url": self._server_url,
                    "device": self._device,
                    "batch_processing": True
                },
                errors=errors
            )
            
        except Exception as e:
            error_msg = f"vLLM engine processing failed: {str(e)}"
            errors.append(error_msg)
            self.logger.error(error_msg)
            
            metadata = ComponentMetadata(
                component_type=self.component_type,
                component_name=self.name,
                component_version=self.version,
                processed_at=datetime.utcnow(),
                config=self._config,
                status=ProcessingStatus.ERROR
            )
            
            return ModelEngineOutput(
                results=[],
                metadata=metadata,
                engine_stats=processing_stats,
                errors=errors
            )
    
    def start_server(self) -> bool:
        """
        Start the vLLM server.
        
        Returns:
            True if server started successfully, False otherwise
        """
        try:
            if not self._engine:
                self._initialize_engine()
            
            if self._engine:
                return self._engine.start()
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to start vLLM server: {e}")
            return False
    
    def stop_server(self) -> bool:
        """
        Stop the vLLM server.
        
        Returns:
            True if server stopped successfully, False otherwise
        """
        try:
            if self._engine:
                return self._engine.stop()
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to stop vLLM server: {e}")
            return False
    
    def is_server_running(self) -> bool:
        """
        Check if the vLLM server is running.
        
        Returns:
            True if server is running, False otherwise
        """
        try:
            if not self._engine:
                return False
            
            return self._engine.running
            
        except Exception:
            return False
    
    def get_supported_models(self) -> List[str]:
        """
        Get list of supported model names.
        
        Returns:
            List of supported model names
        """
        # vLLM supports these common models
        return [
            "BAAI/bge-large-en-v1.5",
            "BAAI/bge-base-en-v1.5",
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "microsoft/DialoGPT-medium",
            "microsoft/DialoGPT-large",
            "codellama/CodeLlama-7b-Python-hf"
        ]
    
    def estimate_processing_time(self, input_data: ModelEngineInput) -> float:
        """
        Estimate processing time for given input.
        
        Args:
            input_data: Input data to estimate for
            
        Returns:
            Estimated processing time in seconds
        """
        try:
            num_requests = len(input_data.requests)
            batch_size = self._config.get('batch_size', 32)
            
            # vLLM is optimized for batching
            num_batches = (num_requests + batch_size - 1) // batch_size
            base_time = num_batches * 0.05  # 50ms per batch
            
            return max(0.5, base_time)
            
        except Exception:
            return 2.0  # Default estimate
    
    def _initialize_engine(self) -> None:
        """Initialize the vLLM engine."""
        try:
            self._engine = LegacyVLLMEngine(
                server_url=self._server_url,
                device=self._device,
                max_retries=self._max_retries,
                timeout=self._timeout
            )
            
            self.logger.info("Initialized vLLM engine successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vLLM engine: {e}")
            self._engine = None
    
    async def _process_requests_async(self, requests: List[Dict[str, Any]]) -> List[ModelInferenceResult]:
        """
        Process inference requests asynchronously.
        
        Args:
            requests: List of request data
            
        Returns:
            List of ModelInferenceResult objects
        """
        results = []
        
        for request in requests:
            start_time = datetime.utcnow()
            
            try:
                request_id = request.get('request_id', 'unknown')
                request_type = request.get('type', 'completion')
                
                if request_type == 'embedding':
                    # Handle embedding request
                    texts = request.get('input', [])
                    model_id = request.get('model', 'BAAI/bge-large-en-v1.5')
                    
                    # Load model if needed
                    if not self._engine.is_model_loaded(model_id):
                        self._engine.load_model(model_id)
                    
                    # Generate embeddings
                    embeddings = await self._engine.generate_embeddings(
                        texts=texts,
                        model_id=model_id,
                        **request.get('parameters', {})
                    )
                    
                    response_data = {
                        "embeddings": embeddings,
                        "model": model_id,
                        "embedding_count": len(embeddings)
                    }
                    
                elif request_type == 'completion':
                    # Handle completion request
                    prompt = request.get('prompt', '')
                    model_id = request.get('model', 'meta-llama/Llama-2-7b-chat-hf')
                    
                    # Load model if needed
                    if not self._engine.is_model_loaded(model_id):
                        self._engine.load_model(model_id)
                    
                    # Generate completion
                    completion = await self._engine.generate_completion(
                        prompt=prompt,
                        model_id=model_id,
                        **request.get('parameters', {})
                    )
                    
                    response_data = {
                        "completion": completion,
                        "model": model_id,
                        "prompt_length": len(prompt)
                    }
                    
                elif request_type == 'chat':
                    # Handle chat completion request
                    messages = request.get('messages', [])
                    model_id = request.get('model', 'meta-llama/Llama-2-7b-chat-hf')
                    
                    # Load model if needed
                    if not self._engine.is_model_loaded(model_id):
                        self._engine.load_model(model_id)
                    
                    # Generate chat completion
                    chat_response = await self._engine.generate_chat_completion(
                        messages=messages,
                        model_id=model_id,
                        **request.get('parameters', {})
                    )
                    
                    response_data = chat_response
                    
                else:
                    raise ValueError(f"Unsupported request type: {request_type}")
                
                # Calculate processing time
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                
                result = ModelInferenceResult(
                    request_id=request_id,
                    response_data=response_data,
                    processing_time=processing_time
                )
                
            except Exception as e:
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                
                result = ModelInferenceResult(
                    request_id=request.get('request_id', 'unknown'),
                    response_data={},
                    processing_time=processing_time,
                    error=str(e)
                )
                
                self.logger.error(f"Request processing failed: {e}")
            
            results.append(result)
        
        return results