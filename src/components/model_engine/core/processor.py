"""
Core Model Engine Component

This module provides the core model engine component that implements the
ModelEngine protocol. It wraps the existing HADES model engine functionality
with the new component contract system.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path

# Import existing HADES model engine functionality
# from src.components.model_engine.factory import ModelEngineFactory
# from src.components.model_engine.server_manager import ServerManager
# from src.components.model_engine.base import BaseModelEngine

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
from src.types.components.model_engine.inference import CompletionRequest
# from src.types.components.model_engine.config import ModelEngineConfig
# from src.types.components.model_engine.inference import InferenceRequest, InferenceResponse


class CoreModelEngine(ModelEngine):
    """
    Core model engine component implementing ModelEngine protocol.
    
    This component uses the existing HADES model engine infrastructure to provide
    model serving capabilities including vLLM engines, server management, and
    inference coordination.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize core model engine.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self._config = config or {}
        
        # Component metadata
        self._metadata = ComponentMetadata(
            component_type=ComponentType.MODEL_ENGINE,
            component_name="core",
            component_version="1.0.0",
            config=self._config
        )
        
        # Initialize model engine components (TODO: implement these classes)
        self._factory: Optional[Any] = None  # ModelEngineFactory when implemented
        self._server_manager: Optional[Any] = None  # ServerManager when implemented
        self._engine: Optional[Any] = None  # BaseModelEngine when implemented
        
        # Configuration
        self._engine_type = self._config.get('engine_type', 'vllm')
        self._model_name = self._config.get('model_name', 'BAAI/bge-large-en-v1.5')
        
        self.logger.info(f"Initialized core model engine with type: {self._engine_type}")
    
    @property
    def name(self) -> str:
        """Component name for identification."""
        return "core"
    
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
        
        # Update engine configuration
        if 'engine_type' in config:
            self._engine_type = config['engine_type']
        
        if 'model_name' in config:
            self._model_name = config['model_name']
        
        # Reset components to force re-initialization
        self._factory = None
        self._server_manager = None
        self._engine = None
        
        self.logger.info(f"Updated core model engine configuration")
    
    def validate_config(self, config: Any) -> bool:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        if not isinstance(config, dict):
            return False
        
        # Validate engine type
        if 'engine_type' in config:
            valid_types = ['vllm', 'haystack']
            if config['engine_type'] not in valid_types:
                return False
        
        # Validate model name
        if 'model_name' in config:
            if not isinstance(config['model_name'], str):
                return False
        
        # Validate server config
        if 'server_config' in config:
            if not isinstance(config['server_config'], dict):
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
                "engine_type": {
                    "type": "string",
                    "enum": ["vllm", "haystack"],
                    "default": "vllm",
                    "description": "Type of model engine to use"
                },
                "model_name": {
                    "type": "string",
                    "description": "Name of the model to serve",
                    "default": "BAAI/bge-large-en-v1.5"
                },
                "server_config": {
                    "type": "object",
                    "description": "Server configuration parameters",
                    "properties": {
                        "host": {
                            "type": "string",
                            "default": "localhost"
                        },
                        "port": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 65535,
                            "default": 8000
                        },
                        "gpu_memory_utilization": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": 0.8
                        },
                        "max_model_len": {
                            "type": "integer",
                            "minimum": 1,
                            "default": 4096
                        }
                    }
                },
                "inference_config": {
                    "type": "object",
                    "description": "Inference configuration parameters",
                    "properties": {
                        "max_tokens": {
                            "type": "integer",
                            "minimum": 1,
                            "default": 1024
                        },
                        "temperature": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 2.0,
                            "default": 0.7
                        },
                        "batch_size": {
                            "type": "integer",
                            "minimum": 1,
                            "default": 32
                        }
                    }
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
            # Initialize components if needed
            if not self._engine:
                self._initialize_engine()
            
            # Check if engine is available
            if not self._engine:
                return False
            
            # Try a simple operation
            if hasattr(self._engine, 'is_ready'):
                return bool(self._engine.is_ready())
            
            return True
            
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
            "engine_type": self._engine_type,
            "model_name": self._model_name,
            "engine_initialized": self._engine is not None,
            "server_manager_initialized": self._server_manager is not None,
            "last_health_check": datetime.utcnow().isoformat()
        }
        
        # Add engine-specific metrics if available
        if self._engine and hasattr(self._engine, 'get_stats'):
            try:
                engine_stats = self._engine.get_stats()
                metrics.update({"engine_stats": engine_stats})
            except Exception as e:
                self.logger.warning(f"Could not get engine stats: {e}")
        
        # Add server metrics if available
        if self._server_manager and hasattr(self._server_manager, 'get_metrics'):
            try:
                server_metrics = self._server_manager.get_metrics()
                metrics.update({"server_metrics": server_metrics})
            except Exception as e:
                self.logger.warning(f"Could not get server metrics: {e}")
        
        return metrics
    
    def process(self, input_data: ModelEngineInput) -> ModelEngineOutput:
        """
        Process model inference requests according to the contract.
        
        Args:
            input_data: Input data conforming to ModelEngineInput contract
            
        Returns:
            Output data conforming to ModelEngineOutput contract
        """
        errors: List[str] = []
        processing_stats: Dict[str, Any] = {}
        
        try:
            start_time = datetime.utcnow()
            
            # Initialize engine if needed
            if not self._engine:
                self._initialize_engine()
            
            if not self._engine:
                raise ValueError("Could not initialize model engine")
            
            # Process inference requests
            results = []
            for request in input_data.requests:
                try:
                    # Convert to engine format and process
                    result = self._process_single_request(request)
                    results.append(result)
                except Exception as e:
                    error_msg = f"Request processing failed: {str(e)}"
                    errors.append(error_msg)
                    self.logger.error(error_msg)
                    
                    # Create error result
                    error_result = ModelInferenceResult(
                        request_id=request.get('request_id', 'unknown'),
                        response_data={},
                        processing_time=0.0,
                        error=str(e)
                    )
                    results.append(error_result)
            
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
                    "error_count": len(errors),
                    "engine_type": self._engine_type,
                    "model_name": self._model_name
                },
                errors=errors
            )
            
        except Exception as e:
            error_msg = f"Model engine processing failed: {str(e)}"
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
        Start the model server.
        
        Returns:
            True if server started successfully, False otherwise
        """
        try:
            if not self._server_manager:
                self._initialize_server_manager()
            
            if self._server_manager:
                self._server_manager.start()
                self.logger.info("Model server started successfully")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to start server: {e}")
            return False
    
    def stop_server(self) -> bool:
        """
        Stop the model server.
        
        Returns:
            True if server stopped successfully, False otherwise
        """
        try:
            if self._server_manager:
                self._server_manager.stop()
                self.logger.info("Model server stopped successfully")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to stop server: {e}")
            return False
    
    def is_server_running(self) -> bool:
        """
        Check if the model server is running.
        
        Returns:
            True if server is running, False otherwise
        """
        try:
            if not self._server_manager:
                return False
            
            return bool(self._server_manager.is_running())
            
        except Exception:
            return False
    
    def get_supported_models(self) -> List[str]:
        """
        Get list of supported model names.
        
        Returns:
            List of supported model names
        """
        # Common models that are typically supported
        return [
            "BAAI/bge-large-en-v1.5",
            "BAAI/bge-base-en-v1.5",
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "microsoft/DialoGPT-medium",
            "microsoft/DialoGPT-large"
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
            
            # Estimate based on request count and engine type
            if self._engine_type == 'vllm':
                # vLLM is faster due to batching
                base_time = num_requests * 0.1  # 100ms per request
            else:
                # Other engines may be slower
                base_time = num_requests * 0.5  # 500ms per request
            
            return max(1.0, base_time)
            
        except Exception:
            return 5.0  # Default estimate
    
    def _initialize_engine(self) -> None:
        """Initialize the model engine."""
        try:
            if not self._factory:
                # TODO: Implement ModelEngineFactory
                self._factory = None  # ModelEngineFactory()
            
            # Create engine configuration
            # TODO: Implement ModelEngineConfig
            engine_config = {
                "engine_type": self._engine_type,
                "model_name": self._model_name,
                **self._config.get('server_config', {})
            }
            
            # Create engine instance
            # TODO: Implement factory.create_engine
            self._engine = None  # self._factory.create_engine(engine_config)
            
            self.logger.info(f"Initialized {self._engine_type} engine successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize engine: {e}")
            self._engine = None
    
    def _initialize_server_manager(self) -> None:
        """Initialize the server manager."""
        try:
            server_config = self._config.get('server_config', {})
            # TODO: Implement ServerManager
            self._server_manager = None  # ServerManager(config=server_config)
            
            self.logger.info("Initialized server manager successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize server manager: {e}")
            self._server_manager = None
    
    def _process_single_request(self, request: Dict[str, Any]) -> ModelInferenceResult:
        """
        Process a single inference request.
        
        Args:
            request: Request data
            
        Returns:
            ModelInferenceResult with response data
        """
        start_time = datetime.utcnow()
        
        try:
            # Convert request to engine format
            inference_request: CompletionRequest = {
                'model': request.get('model', 'default'),
                'prompt': request.get('input_text', ''),
                'max_tokens': request.get('max_tokens', 100),
                'temperature': request.get('temperature', 0.7)
            }
            
            # Process with engine
            if self._engine is None:
                raise RuntimeError("Model engine not initialized. Call initialize() first.")
            response = self._engine.generate(inference_request)
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return ModelInferenceResult(
                request_id=request.get('request_id', 'unknown'),
                response_data={
                    "output_text": response.output_text if hasattr(response, 'output_text') else "",
                    "tokens": response.tokens if hasattr(response, 'tokens') else [],
                    "metadata": response.metadata if hasattr(response, 'metadata') else {}
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return ModelInferenceResult(
                request_id=request.get('request_id', 'unknown'),
                response_data={},
                processing_time=processing_time,
                error=str(e)
            )