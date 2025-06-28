"""
Base interfaces and protocols for the multi-engine RAG platform.

This module defines the common interfaces that all RAG engines must implement
to enable A/B testing, comparison, and experimentation.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Protocol
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

from src.types.components.contracts import RAGMode


class EngineType(str, Enum):
    """Supported RAG engine types."""
    PATHRAG = "pathrag"
    SAGEGRAPH = "sagegraph"
    NAIVE = "naive"
    HYBRID = "hybrid"


class RetrievalMode(str, Enum):
    """Retrieval modes supported across engines."""
    NAIVE = "naive"
    LOCAL = "local"
    GLOBAL = "global"
    HYBRID = "hybrid"
    FULL_ENGINE = "full_engine"  # Engine's primary algorithm


class ExperimentStatus(str, Enum):
    """Status of A/B testing experiments."""
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Request/Response Models
class EngineRetrievalRequest(BaseModel):
    """Standard retrieval request for any RAG engine."""
    query: str = Field(..., description="Natural language query")
    mode: RetrievalMode = Field(default=RetrievalMode.FULL_ENGINE, description="Retrieval mode to use")
    top_k: int = Field(default=10, description="Maximum number of results")
    max_token_for_context: int = Field(default=4000, description="Maximum tokens in context")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Query filters")
    engine_params: Optional[Dict[str, Any]] = Field(default=None, description="Engine-specific parameters")
    experiment_id: Optional[str] = Field(default=None, description="Optional experiment tracking ID")


class EngineRetrievalResult(BaseModel):
    """Standard result from any RAG engine."""
    content: str = Field(..., description="Retrieved content")
    score: float = Field(..., description="Relevance score (0.0-1.0)")
    source: str = Field(..., description="Source identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional result metadata")
    engine_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Engine-specific metadata")


class EngineRetrievalResponse(BaseModel):
    """Standard response from any RAG engine."""
    engine_type: EngineType = Field(..., description="Type of engine that processed the request")
    mode: RetrievalMode = Field(..., description="Retrieval mode used")
    query: str = Field(..., description="Original query")
    results: List[EngineRetrievalResult] = Field(..., description="Retrieved results")
    execution_time_ms: float = Field(..., description="Execution time in milliseconds")
    total_candidates: int = Field(default=0, description="Total candidates considered")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")
    experiment_id: Optional[str] = Field(default=None, description="Experiment tracking ID")


class EngineConfigRequest(BaseModel):
    """Request to update engine configuration."""
    engine_type: EngineType = Field(..., description="Target engine type")
    config_updates: Dict[str, Any] = Field(..., description="Configuration parameters to update")
    temporary: bool = Field(default=False, description="Whether changes are temporary (session-only)")


class EngineConfigResponse(BaseModel):
    """Response with current engine configuration."""
    engine_type: EngineType = Field(..., description="Engine type")
    current_config: Dict[str, Any] = Field(..., description="Current configuration")
    config_schema: Dict[str, Any] = Field(..., description="Configuration schema")
    last_updated: datetime = Field(..., description="Last configuration update time")


class ExperimentRequest(BaseModel):
    """Request to start an A/B testing experiment."""
    name: str = Field(..., description="Experiment name")
    description: str = Field(..., description="Experiment description")
    engines: List[EngineType] = Field(..., description="Engines to compare")
    test_queries: List[str] = Field(..., description="Queries to test")
    engine_configs: Optional[Dict[EngineType, Dict[str, Any]]] = Field(default=None, description="Engine-specific configs")
    metrics: List[str] = Field(default=["relevance", "speed", "diversity"], description="Metrics to collect")
    iterations: int = Field(default=1, description="Number of iterations per query")


class ExperimentResult(BaseModel):
    """Result from a single experiment run."""
    experiment_id: str
    engine_type: EngineType
    query: str
    iteration: int
    execution_time_ms: float
    num_results: int
    avg_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExperimentResponse(BaseModel):
    """Response with experiment results and analysis."""
    experiment_id: str
    name: str
    status: ExperimentStatus
    engines_tested: List[EngineType]
    total_queries: int
    results: List[ExperimentResult]
    summary_metrics: Dict[str, Any] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    created_at: datetime
    completed_at: Optional[datetime] = None


# Engine Protocol
class RAGEngine(Protocol):
    """Protocol that all RAG engines must implement."""
    
    @property
    def engine_type(self) -> EngineType:
        """Get the engine type identifier."""
        ...
    
    @property
    def supported_modes(self) -> List[RetrievalMode]:
        """Get list of supported retrieval modes."""
        ...
    
    async def retrieve(self, request: EngineRetrievalRequest) -> EngineRetrievalResponse:
        """Perform retrieval using this engine."""
        ...
    
    async def configure(self, config: Dict[str, Any]) -> bool:
        """Update engine configuration."""
        ...
    
    async def get_config(self) -> Dict[str, Any]:
        """Get current engine configuration."""
        ...
    
    async def get_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema for this engine."""
        ...
    
    async def health_check(self) -> bool:
        """Check if engine is healthy and ready."""
        ...
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get engine performance metrics."""
        ...


# Abstract Base Engine
class BaseRAGEngine(ABC):
    """Abstract base class for RAG engines with common functionality."""
    
    def __init__(self, engine_type: EngineType, config: Optional[Dict[str, Any]] = None):
        """Initialize base engine."""
        self._engine_type = engine_type
        self._config = config or {}
        self._last_config_update = datetime.utcnow()
        self._metrics: Dict[str, Union[int, float, Optional[str]]] = {
            "total_queries": 0,
            "avg_execution_time_ms": 0.0,
            "total_execution_time_ms": 0.0,
            "last_query_time": None,
            "error_count": 0
        }
    
    @property
    def engine_type(self) -> EngineType:
        """Get the engine type identifier."""
        return self._engine_type
    
    @property
    @abstractmethod
    def supported_modes(self) -> List[RetrievalMode]:
        """Get list of supported retrieval modes."""
        pass
    
    @abstractmethod
    async def _execute_retrieval(self, request: EngineRetrievalRequest) -> List[EngineRetrievalResult]:
        """Execute the actual retrieval logic. Must be implemented by subclasses."""
        pass
    
    async def retrieve(self, request: EngineRetrievalRequest) -> EngineRetrievalResponse:
        """Perform retrieval with common wrapper logic."""
        start_time = time.time()
        
        try:
            # Validate mode support
            if request.mode not in self.supported_modes:
                raise ValueError(f"Mode {request.mode} not supported by {self.engine_type}")
            
            # Execute retrieval
            results = await self._execute_retrieval(request)
            
            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Update metrics
            self._update_metrics(execution_time_ms, len(results), success=True)
            
            # Build response
            return EngineRetrievalResponse(
                engine_type=self.engine_type,
                mode=request.mode,
                query=request.query,
                results=results,
                execution_time_ms=execution_time_ms,
                total_candidates=len(results),
                metadata={
                    "engine_config": self._config,
                    "timestamp": datetime.utcnow().isoformat()
                },
                experiment_id=request.experiment_id
            )
            
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            self._update_metrics(execution_time_ms, 0, success=False)
            
            # Return error response
            return EngineRetrievalResponse(
                engine_type=self.engine_type,
                mode=request.mode,
                query=request.query,
                results=[],
                execution_time_ms=execution_time_ms,
                total_candidates=0,
                metadata={
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                },
                experiment_id=request.experiment_id
            )
    
    async def configure(self, config: Dict[str, Any]) -> bool:
        """Update engine configuration."""
        try:
            # Validate config against schema
            schema = await self.get_config_schema()
            # TODO: Add proper validation
            
            self._config.update(config)
            self._last_config_update = datetime.utcnow()
            return True
        except Exception:
            return False
    
    async def get_config(self) -> Dict[str, Any]:
        """Get current engine configuration."""
        return self._config.copy()
    
    @abstractmethod
    async def get_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema for this engine."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if engine is healthy and ready."""
        pass
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get engine performance metrics."""
        return {
            **self._metrics,
            "engine_type": self.engine_type.value,
            "last_config_update": self._last_config_update.isoformat(),
            "supported_modes": [mode.value for mode in self.supported_modes]
        }
    
    def _update_metrics(self, execution_time_ms: float, num_results: int, success: bool) -> None:
        """Update internal metrics."""
        if success:
            # Type-safe metric updates with default values
            current_total = self._metrics.get("total_queries", 0)
            total_queries = int(current_total if current_total is not None else 0) + 1
            self._metrics["total_queries"] = total_queries
            
            current_exec_time = self._metrics.get("total_execution_time_ms", 0.0)
            total_exec_time = float(current_exec_time if current_exec_time is not None else 0.0) + execution_time_ms
            self._metrics["total_execution_time_ms"] = total_exec_time
            
            self._metrics["avg_execution_time_ms"] = total_exec_time / total_queries
            self._metrics["last_query_time"] = datetime.utcnow().isoformat()
        else:
            current_errors = self._metrics.get("error_count", 0)
            error_count = int(current_errors if current_errors is not None else 0) + 1
            self._metrics["error_count"] = error_count


# Engine Registry
class EngineRegistry:
    """Registry for managing multiple RAG engines."""
    
    def __init__(self) -> None:
        self._engines: Dict[EngineType, RAGEngine] = {}
        self._default_engine: Optional[EngineType] = None
    
    def register_engine(self, engine: RAGEngine, is_default: bool = False) -> None:
        """Register a new engine."""
        self._engines[engine.engine_type] = engine
        if is_default or self._default_engine is None:
            self._default_engine = engine.engine_type
    
    def get_engine(self, engine_type: EngineType) -> Optional[RAGEngine]:
        """Get engine by type."""
        return self._engines.get(engine_type)
    
    def get_default_engine(self) -> Optional[RAGEngine]:
        """Get the default engine."""
        if self._default_engine:
            return self._engines.get(self._default_engine)
        return None
    
    def list_engines(self) -> List[EngineType]:
        """List all registered engines."""
        return list(self._engines.keys())
    
    def list_supported_modes(self, engine_type: EngineType) -> List[RetrievalMode]:
        """List supported modes for an engine."""
        engine = self.get_engine(engine_type)
        return engine.supported_modes if engine else []
    
    async def health_check_all(self) -> Dict[EngineType, bool]:
        """Health check all engines."""
        results = {}
        for engine_type, engine in self._engines.items():
            try:
                results[engine_type] = await engine.health_check()
            except Exception:
                results[engine_type] = False
        return results


# Global registry instance
_engine_registry = EngineRegistry()


def get_engine_registry() -> EngineRegistry:
    """Get the global engine registry."""
    return _engine_registry