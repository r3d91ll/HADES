"""
Multi-Engine RAG Platform for HADES.

This package provides a comprehensive platform for comparing different RAG strategies
and running controlled experiments. It includes:

- Base engine interfaces and protocols
- PathRAG engine implementation
- A/B testing and comparison framework
- FastAPI endpoints for engine management
- Performance monitoring and metrics collection

The platform is designed to be extensible, allowing easy addition of new RAG engines
like SageGraph while maintaining consistent interfaces and comparison capabilities.
"""

from .base import (
    EngineType,
    RetrievalMode,
    EngineRetrievalRequest,
    EngineRetrievalResponse,
    EngineConfigRequest,
    EngineConfigResponse,
    ExperimentRequest,
    ExperimentResponse,
    ExperimentStatus,
    RAGEngine,
    BaseRAGEngine,
    EngineRegistry,
    get_engine_registry
)

from .pathrag_engine import PathRAGEngine
from .comparison import (
    ComparisonMetric,
    ComparisonResult,
    ExperimentManager,
    get_experiment_manager
)
from .router import router

__all__ = [
    # Base types
    "EngineType",
    "RetrievalMode",
    "EngineRetrievalRequest",
    "EngineRetrievalResponse",
    "EngineConfigRequest",
    "EngineConfigResponse",
    "ExperimentRequest",
    "ExperimentResponse",
    "ExperimentStatus",
    
    # Engine interfaces
    "RAGEngine",
    "BaseRAGEngine",
    "EngineRegistry",
    "get_engine_registry",
    
    # Specific engines
    "PathRAGEngine",
    
    # Comparison framework
    "ComparisonMetric",
    "ComparisonResult",
    "ExperimentManager",
    "get_experiment_manager",
    
    # FastAPI router
    "router"
]