"""
Haystack Model Engine Component

This module provides the Haystack model engine component that implements the
ModelEngine protocol. It wraps the existing HADES Haystack engine functionality
with the new component contract system for document processing workflows.
"""

import logging
import psutil
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

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


class HaystackModelEngine(ModelEngine):
    """
    Haystack model engine component implementing ModelEngine protocol.
    
    This component provides Haystack-based model serving for document
    processing, question answering, and retrieval workflows.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Haystack model engine.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self._config = config or {}
        
        # Component metadata
        self._metadata = ComponentMetadata(
            component_type=ComponentType.MODEL_ENGINE,
            component_name="haystack",
            component_version="1.0.0",
            config=self._config
        )
        
        # Configuration
        self._pipeline_type = self._config.get('pipeline_type', 'embedding')
        self._model_name = self._config.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
        
        # Haystack components (to be initialized)
        self._document_store = None
        self._retriever = None
        self._reader = None
        self._pipeline = None
        
        # Monitoring and metrics tracking
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time": 0.0,
            "last_request_time": None,
            "pipeline_initializations": 0,
            "errors": []
        }
        self._startup_time = datetime.now(timezone.utc)
        self._gpu_available = self._check_gpu_availability()
        
        self.logger.info(f"Initialized Haystack model engine with pipeline: {self._pipeline_type}")
    
    @property
    def name(self) -> str:
        """Component name for identification."""
        return "haystack"
    
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
        self._metadata.processed_at = datetime.now(timezone.utc)
        
        # Update configuration
        if 'pipeline_type' in config:
            self._pipeline_type = config['pipeline_type']
        
        if 'model_name' in config:
            self._model_name = config['model_name']
        
        # Reset pipeline to force re-initialization
        self._pipeline = None
        self._document_store = None
        self._retriever = None
        self._reader = None
        
        self.logger.info(f"Updated Haystack model engine configuration")
    
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
        
        # Validate pipeline type
        if 'pipeline_type' in config:
            valid_types = ['embedding', 'qa', 'summarization', 'generation']
            if config['pipeline_type'] not in valid_types:
                return False
        
        # Validate model name
        if 'model_name' in config:
            if not isinstance(config['model_name'], str):
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
                "pipeline_type": {
                    "type": "string",
                    "enum": ["embedding", "qa", "summarization", "generation"],
                    "default": "embedding",
                    "description": "Type of Haystack pipeline to use"
                },
                "model_name": {
                    "type": "string",
                    "description": "Name of the model to use",
                    "default": "sentence-transformers/all-MiniLM-L6-v2"
                },
                "document_store_config": {
                    "type": "object",
                    "description": "Configuration for document store",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["memory", "elasticsearch", "faiss"],
                            "default": "memory"
                        },
                        "embedding_dim": {
                            "type": "integer",
                            "minimum": 1,
                            "default": 384
                        }
                    }
                },
                "retriever_config": {
                    "type": "object",
                    "description": "Configuration for retriever",
                    "properties": {
                        "top_k": {
                            "type": "integer",
                            "minimum": 1,
                            "default": 10
                        },
                        "embedding_model": {
                            "type": "string",
                            "default": "sentence-transformers/all-MiniLM-L6-v2"
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
            # Check if Haystack is available
            try:
                import haystack
                self.logger.debug(f"Haystack version: {haystack.__version__}")
            except ImportError:
                self.logger.error("Haystack not available")
                return False
            
            # Initialize pipeline if needed
            if not self._pipeline:
                self._initialize_pipeline()
            
            return self._pipeline is not None
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def get_infrastructure_metrics(self) -> Dict[str, Any]:
        """
        Get infrastructure and resource inventory metrics.
        
        Returns:
            Dictionary containing infrastructure metrics
        """
        try:
            # Get memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            
            # Get GPU info if available
            gpu_info = self._get_gpu_info() if self._gpu_available else None
            
            # Determine device allocation
            device_config = self._config.get('device', 'cpu')
            
            return {
                "component_name": self.name,
                "component_version": self.version,
                "device_allocation": device_config,
                "loaded_models": [self._model_name] if self._pipeline else [],
                "pipeline_type": self._pipeline_type,
                "memory_usage": {
                    "rss_mb": round(memory_info.rss / 1024 / 1024, 2),
                    "vms_mb": round(memory_info.vms / 1024 / 1024, 2)
                },
                "gpu_info": gpu_info,
                "pipeline_initialized": self._pipeline is not None,
                "document_store_initialized": self._document_store is not None,
                "startup_time": self._startup_time.isoformat(),
                "uptime_seconds": (datetime.now(timezone.utc) - self._startup_time).total_seconds()
            }
        except Exception as e:
            self.logger.error(f"Failed to get infrastructure metrics: {e}")
            return {"error": str(e)}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get runtime performance statistics.
        
        Returns:
            Dictionary containing performance metrics
        """
        try:
            current_time = datetime.now(timezone.utc)
            uptime = (current_time - self._startup_time).total_seconds()
            
            # Calculate rates
            requests_per_second = self._stats["total_requests"] / max(uptime, 1.0)
            avg_processing_time = (
                self._stats["total_processing_time"] / max(self._stats["total_requests"], 1)
            )
            success_rate = (
                self._stats["successful_requests"] / max(self._stats["total_requests"], 1) * 100
            )
            
            return {
                "component_name": self.name,
                "total_requests": self._stats["total_requests"],
                "successful_requests": self._stats["successful_requests"],
                "failed_requests": self._stats["failed_requests"],
                "success_rate_percent": round(success_rate, 2),
                "requests_per_second": round(requests_per_second, 3),
                "average_processing_time": round(avg_processing_time, 3),
                "total_processing_time": round(self._stats["total_processing_time"], 3),
                "last_request_time": self._stats["last_request_time"],
                "pipeline_initializations": self._stats["pipeline_initializations"],
                "recent_errors": self._stats["errors"][-5:],  # Last 5 errors
                "error_count": len(self._stats["errors"]),
                "uptime_seconds": round(uptime, 2)
            }
        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {e}")
            return {"error": str(e)}

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get component performance metrics (legacy method for compatibility).
        
        Returns:
            Dictionary containing performance metrics
        """
        # Combine infrastructure and performance metrics for compatibility
        infra_metrics = self.get_infrastructure_metrics()
        perf_metrics = self.get_performance_metrics()
        
        return {
            **infra_metrics,
            **perf_metrics,
            "last_health_check": datetime.now(timezone.utc).isoformat()
        }
    
    def export_metrics_prometheus(self) -> str:
        """
        Export metrics in Prometheus format.
        
        Returns:
            Prometheus-compatible metrics string
        """
        try:
            infra_metrics = self.get_infrastructure_metrics()
            perf_metrics = self.get_performance_metrics()
            
            metrics_lines = []
            
            # Infrastructure metrics
            metrics_lines.append(f"# HELP hades_component_uptime_seconds Component uptime in seconds")
            metrics_lines.append(f"# TYPE hades_component_uptime_seconds gauge")
            metrics_lines.append(f'hades_component_uptime_seconds{{component="haystack"}} {infra_metrics.get("uptime_seconds", 0)}')
            
            metrics_lines.append(f"# HELP hades_component_memory_rss_mb Memory RSS usage in MB")
            metrics_lines.append(f"# TYPE hades_component_memory_rss_mb gauge")
            memory_rss = infra_metrics.get("memory_usage", {}).get("rss_mb", 0)
            metrics_lines.append(f'hades_component_memory_rss_mb{{component="haystack"}} {memory_rss}')
            
            metrics_lines.append(f"# HELP hades_component_loaded_models Number of loaded models")
            metrics_lines.append(f"# TYPE hades_component_loaded_models gauge")
            loaded_models_count = len(infra_metrics.get("loaded_models", []))
            metrics_lines.append(f'hades_component_loaded_models{{component="haystack"}} {loaded_models_count}')
            
            # Performance metrics
            metrics_lines.append(f"# HELP hades_component_requests_total Total number of requests")
            metrics_lines.append(f"# TYPE hades_component_requests_total counter")
            metrics_lines.append(f'hades_component_requests_total{{component="haystack"}} {perf_metrics.get("total_requests", 0)}')
            
            metrics_lines.append(f"# HELP hades_component_requests_successful_total Total number of successful requests")
            metrics_lines.append(f"# TYPE hades_component_requests_successful_total counter")
            metrics_lines.append(f'hades_component_requests_successful_total{{component="haystack"}} {perf_metrics.get("successful_requests", 0)}')
            
            metrics_lines.append(f"# HELP hades_component_requests_failed_total Total number of failed requests")
            metrics_lines.append(f"# TYPE hades_component_requests_failed_total counter")
            metrics_lines.append(f'hades_component_requests_failed_total{{component="haystack"}} {perf_metrics.get("failed_requests", 0)}')
            
            metrics_lines.append(f"# HELP hades_component_success_rate_percent Success rate percentage")
            metrics_lines.append(f"# TYPE hades_component_success_rate_percent gauge")
            metrics_lines.append(f'hades_component_success_rate_percent{{component="haystack"}} {perf_metrics.get("success_rate_percent", 0)}')
            
            metrics_lines.append(f"# HELP hades_component_requests_per_second Requests per second")
            metrics_lines.append(f"# TYPE hades_component_requests_per_second gauge")
            metrics_lines.append(f'hades_component_requests_per_second{{component="haystack"}} {perf_metrics.get("requests_per_second", 0)}')
            
            metrics_lines.append(f"# HELP hades_component_avg_processing_time_seconds Average processing time in seconds")
            metrics_lines.append(f"# TYPE hades_component_avg_processing_time_seconds gauge")
            metrics_lines.append(f'hades_component_avg_processing_time_seconds{{component="haystack"}} {perf_metrics.get("average_processing_time", 0)}')
            
            # GPU metrics if available
            gpu_info = infra_metrics.get("gpu_info")
            if gpu_info and isinstance(gpu_info, list):
                for i, gpu in enumerate(gpu_info):
                    if isinstance(gpu, dict):
                        metrics_lines.append(f"# HELP hades_component_gpu_utilization_percent GPU utilization percentage")
                        metrics_lines.append(f"# TYPE hades_component_gpu_utilization_percent gauge")
                        utilization = gpu.get("utilization", {}).get("gpu", 0)
                        metrics_lines.append(f'hades_component_gpu_utilization_percent{{component="haystack",gpu="{i}"}} {utilization}')
                        
                        metrics_lines.append(f"# HELP hades_component_gpu_memory_used_mb GPU memory used in MB")
                        metrics_lines.append(f"# TYPE hades_component_gpu_memory_used_mb gauge")
                        memory_used = gpu.get("memory", {}).get("used", 0)
                        metrics_lines.append(f'hades_component_gpu_memory_used_mb{{component="haystack",gpu="{i}"}} {memory_used}')
            
            return "\n".join(metrics_lines) + "\n"
            
        except Exception as e:
            self.logger.error(f"Failed to export Prometheus metrics: {e}")
            return f"# Error exporting metrics: {str(e)}\n"
    
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
            start_time = datetime.now(timezone.utc)
            
            # Update request statistics
            self._stats["total_requests"] += len(input_data.requests)
            self._stats["last_request_time"] = start_time.isoformat()
            
            # Initialize pipeline if needed
            if not self._pipeline:
                self._initialize_pipeline()
            
            if not self._pipeline:
                raise ValueError("Could not initialize Haystack pipeline")
            
            # Process requests
            results = []
            successful_count = 0
            for request in input_data.requests:
                try:
                    result = self._process_single_request(request)
                    results.append(result)
                    if not result.error:
                        successful_count += 1
                except Exception as e:
                    error_msg = f"Request processing failed: {str(e)}"
                    errors.append(error_msg)
                    self.logger.error(error_msg)
                    
                    # Track error
                    self._stats["errors"].append({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "error": error_msg
                    })
                    
                    # Create error result
                    error_result = ModelInferenceResult(
                        request_id=request.get('request_id', 'unknown'),
                        response_data={},
                        processing_time=0.0,
                        error=str(e)
                    )
                    results.append(error_result)
            
            # Calculate processing time
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Update statistics
            self._stats["successful_requests"] += successful_count
            self._stats["failed_requests"] += (len(input_data.requests) - successful_count)
            self._stats["total_processing_time"] += processing_time
            
            # Update metadata
            metadata = ComponentMetadata(
                component_type=self.component_type,
                component_name=self.name,
                component_version=self.version,
                processing_time=processing_time,
                processed_at=datetime.now(timezone.utc),
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
                    "pipeline_type": self._pipeline_type,
                    "model_name": self._model_name
                },
                errors=errors
            )
            
        except Exception as e:
            error_msg = f"Haystack engine processing failed: {str(e)}"
            errors.append(error_msg)
            self.logger.error(error_msg)
            
            metadata = ComponentMetadata(
                component_type=self.component_type,
                component_name=self.name,
                component_version=self.version,
                processed_at=datetime.now(timezone.utc),
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
        Start the Haystack pipeline (no server needed).
        
        Returns:
            True if pipeline initialized successfully, False otherwise
        """
        try:
            if not self._pipeline:
                self._initialize_pipeline()
            
            return self._pipeline is not None
            
        except Exception as e:
            self.logger.error(f"Failed to start Haystack pipeline: {e}")
            return False
    
    def stop_server(self) -> bool:
        """
        Stop the Haystack pipeline.
        
        Returns:
            True always (no server to stop)
        """
        # Haystack doesn't require server stopping
        return True
    
    def is_server_running(self) -> bool:
        """
        Check if the Haystack pipeline is ready.
        
        Returns:
            True if pipeline is initialized, False otherwise
        """
        return self._pipeline is not None
    
    def get_supported_models(self) -> List[str]:
        """
        Get list of supported model names.
        
        Returns:
            List of supported model names
        """
        return [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
            "sentence-transformers/paraphrase-MiniLM-L6-v2",
            "deepset/roberta-base-squad2",
            "deepset/bert-large-uncased-whole-word-masking-squad2"
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
            
            # Haystack processing time varies by pipeline type
            if self._pipeline_type == 'embedding':
                base_time = num_requests * 0.1  # 100ms per request
            elif self._pipeline_type == 'qa':
                base_time = num_requests * 0.5  # 500ms per QA request
            else:
                base_time = num_requests * 0.3  # 300ms per other request
            
            return max(1.0, base_time)
            
        except Exception:
            return 3.0  # Default estimate
    
    def _initialize_pipeline(self) -> None:
        """Initialize the Haystack pipeline."""
        try:
            from haystack import Pipeline
            from haystack.components.embedders import SentenceTransformersTextEmbedder
            from haystack.components.retrievers import InMemoryEmbeddingRetriever
            from haystack.document_stores.in_memory import InMemoryDocumentStore
            
            self.logger.info(f"Initializing Haystack {self._pipeline_type} pipeline")
            
            if self._pipeline_type == 'embedding':
                # Initialize document store for embeddings
                if not self._document_store:
                    self._document_store = InMemoryDocumentStore()
                
                # Initialize embedder component  
                device_config = self._config.get('device', 'cpu')
                # Convert string device to ComponentDevice if needed
                embedder_kwargs = {"model": self._model_name}
                if device_config and device_config != 'auto':
                    from haystack.utils import ComponentDevice
                    if isinstance(device_config, str):
                        embedder_kwargs["device"] = ComponentDevice.from_str(device_config)
                    else:
                        embedder_kwargs["device"] = device_config
                
                embedder = SentenceTransformersTextEmbedder(**embedder_kwargs)
                
                # Create simple embedding pipeline
                pipeline = Pipeline()
                pipeline.add_component("embedder", embedder)
                
                self._pipeline = pipeline
                self.logger.info(f"Initialized Haystack embedding pipeline with model: {self._model_name}")
                
            elif self._pipeline_type == 'qa':
                # Initialize components for QA pipeline
                if not self._document_store:
                    self._document_store = InMemoryDocumentStore()
                
                # Create QA pipeline with embedder and retriever
                device_config = self._config.get('device', 'cpu')
                # Convert string device to ComponentDevice if needed
                embedder_kwargs = {"model": self._model_name}
                if device_config and device_config != 'auto':
                    from haystack.utils import ComponentDevice
                    if isinstance(device_config, str):
                        embedder_kwargs["device"] = ComponentDevice.from_str(device_config)
                    else:
                        embedder_kwargs["device"] = device_config
                
                embedder = SentenceTransformersTextEmbedder(**embedder_kwargs)
                retriever = InMemoryEmbeddingRetriever(document_store=self._document_store)
                
                pipeline = Pipeline()
                pipeline.add_component("embedder", embedder)
                pipeline.add_component("retriever", retriever)
                
                # Connect components
                pipeline.connect("embedder.embedding", "retriever.query_embedding")
                
                self._pipeline = pipeline
                self._retriever = retriever
                self.logger.info(f"Initialized Haystack QA pipeline with model: {self._model_name}")
                
            else:
                # Default to embedding pipeline for other types
                self.logger.warning(f"Pipeline type '{self._pipeline_type}' not fully implemented, using embedding pipeline")
                self._pipeline_type = 'embedding'
                self._initialize_pipeline()
                return
            
            # Track successful pipeline initialization
            self._stats["pipeline_initializations"] += 1
            
        except ImportError as e:
            self.logger.error(f"Haystack components not available: {e}")
            self._pipeline = None
        except Exception as e:
            self.logger.error(f"Failed to initialize Haystack pipeline: {e}")
            self._pipeline = None
    
    def _process_single_request(self, request: Dict[str, Any]) -> ModelInferenceResult:
        """
        Process a single inference request.
        
        Args:
            request: Request data
            
        Returns:
            ModelInferenceResult with response data
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            request_id = request.get('request_id', 'unknown')
            request_type = request.get('type', self._pipeline_type)
            
            if not self._pipeline:
                raise ValueError("Pipeline not initialized")
            
            # Process based on request type and available data
            if request_type == 'embedding' and 'text' in request:
                # Handle embedding requests
                text = request['text']
                if isinstance(text, list):
                    # Process each text separately as Haystack SentenceTransformersTextEmbedder expects string input
                    all_embeddings = []
                    for single_text in text:
                        result = self._pipeline.run({"embedder": {"text": single_text}})
                        embedding = result["embedder"]["embedding"]
                        all_embeddings.append(embedding)
                    embeddings = all_embeddings
                else:
                    # Single string input
                    result = self._pipeline.run({"embedder": {"text": text}})
                    embeddings = [result["embedder"]["embedding"]]
                
                response_data = {
                    "embeddings": [emb.tolist() if hasattr(emb, 'tolist') else emb for emb in embeddings],
                    "model": self._model_name,
                    "pipeline_type": "embedding",
                    "processed": True
                }
                
            elif request_type == 'qa' and 'query' in request:
                # Handle QA requests  
                query = request['query']
                
                # Run QA pipeline (single string input for embedder)
                result = self._pipeline.run({
                    "embedder": {"text": query},
                    "retriever": {"top_k": request.get('top_k', 5)}
                })
                
                # Extract retrieved documents
                documents = result.get("retriever", {}).get("documents", [])
                
                response_data = {
                    "query": query,
                    "documents": [{"content": doc.content, "score": getattr(doc, 'score', 0.0)} for doc in documents],
                    "model": self._model_name,
                    "pipeline_type": "qa",
                    "processed": True
                }
                
            else:
                # Fallback for unsupported request types
                self.logger.warning(f"Unsupported request type '{request_type}' or missing required fields")
                response_data = {
                    "pipeline_type": request_type,
                    "model": self._model_name,
                    "processed": False,
                    "warning": f"Request type '{request_type}' not fully supported or missing required fields"
                }
            
            # Calculate processing time
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            return ModelInferenceResult(
                request_id=request_id,
                response_data=response_data,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            return ModelInferenceResult(
                request_id=request.get('request_id', 'unknown'),
                response_data={},
                processing_time=processing_time,
                error=str(e)
            )
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU monitoring is available."""
        try:
            import pynvml
            pynvml.nvmlInit()
            return True
        except ImportError:
            return False
        except Exception:
            return False
    
    def _get_gpu_info(self) -> Optional[List[Dict[str, Any]]]:
        """Get GPU information if available."""
        if not self._gpu_available:
            return None
        
        try:
            import pynvml
            
            device_count = pynvml.nvmlDeviceGetCount()
            gpu_info = []
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Get GPU name
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                
                # Get memory info
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_total = memory_info.total // 1024 // 1024  # Convert to MB
                memory_used = memory_info.used // 1024 // 1024   # Convert to MB
                memory_free = memory_info.free // 1024 // 1024   # Convert to MB
                
                # Get utilization
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                # Get temperature
                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    temperature = None
                
                gpu_info.append({
                    "index": i,
                    "name": name,
                    "memory": {
                        "total": memory_total,
                        "used": memory_used,
                        "free": memory_free,
                        "utilization_percent": round((memory_used / memory_total) * 100, 2)
                    },
                    "utilization": {
                        "gpu": utilization.gpu,
                        "memory": utilization.memory
                    },
                    "temperature": temperature
                })
            
            return gpu_info
            
        except Exception as e:
            self.logger.warning(f"Failed to get GPU info: {e}")
            return None