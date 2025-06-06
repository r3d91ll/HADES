"""
Base type definitions for orchestration components.

This module provides core type definitions for the orchestration system,
including enums, protocols, and type aliases.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Union
from pathlib import Path

# Type aliases
PathType = Union[str, Path]
ConfigDict = Dict[str, Any]
MetricsDict = Dict[str, Any]
ResultDict = Dict[str, Any]


class PipelineState(str, Enum):
    """Pipeline execution states."""
    
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ComponentState(str, Enum):
    """Component states for workers, queues, monitors."""
    
    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class PipelineType(str, Enum):
    """Types of pipelines available."""
    
    TEXT_PROCESSING = "text_processing"
    ISNE_TRAINING = "isne_training"
    EMBEDDING = "embedding"
    INGESTION = "ingestion"
    PARALLEL = "parallel"
    CUSTOM = "custom"


class Priority(str, Enum):
    """Task priority levels."""
    
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class WorkerType(str, Enum):
    """Types of workers in the system."""
    
    CPU = "cpu"
    GPU = "gpu"
    IO = "io"
    NETWORK = "network"
    MIXED = "mixed"


# Protocol definitions

class MonitorProtocol(Protocol):
    """Protocol for monitoring components."""
    
    def register_component(self, component_type: str, component_name: str, component: Any) -> None:
        """Register a component for monitoring."""
        ...
    
    def get_all_metrics(self) -> MetricsDict:
        """Get all monitoring metrics."""
        ...
    
    def get_component_metrics(self, component_name: str) -> Optional[MetricsDict]:
        """Get metrics for a specific component."""
        ...


class WorkerProtocol(Protocol):
    """Protocol for worker components."""
    
    def start(self) -> bool:
        """Start the worker."""
        ...
    
    def stop(self) -> bool:
        """Stop the worker."""
        ...
    
    def is_running(self) -> bool:
        """Check if worker is running."""
        ...
    
    def get_metrics(self) -> MetricsDict:
        """Get worker metrics."""
        ...


class QueueProtocol(Protocol):
    """Protocol for queue components."""
    
    def put(self, item: Any, priority: Priority = Priority.NORMAL) -> bool:
        """Add item to queue."""
        ...
    
    def get(self, timeout: Optional[float] = None) -> Any:
        """Get item from queue."""
        ...
    
    def size(self) -> int:
        """Get queue size."""
        ...
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        ...


class PipelineProtocol(Protocol):
    """Protocol for pipeline components."""
    
    def run(self, **kwargs: Any) -> ResultDict:
        """Run the pipeline."""
        ...
    
    def get_state(self) -> PipelineState:
        """Get current pipeline state."""
        ...
    
    def pause(self) -> bool:
        """Pause pipeline execution."""
        ...
    
    def resume(self) -> bool:
        """Resume pipeline execution."""
        ...
    
    def cancel(self) -> bool:
        """Cancel pipeline execution."""
        ...


# Abstract base classes

class OrchestrationComponent(ABC):
    """Base class for all orchestration components."""
    
    def __init__(self, name: str, config: Optional[ConfigDict] = None):
        self.name = name
        self.config = config or {}
        self.state = ComponentState.INITIALIZED
    
    @abstractmethod
    def start(self) -> bool:
        """Start the component."""
        pass
    
    @abstractmethod
    def stop(self) -> bool:
        """Stop the component."""
        pass
    
    @abstractmethod
    def get_metrics(self) -> MetricsDict:
        """Get component metrics."""
        pass
    
    def get_state(self) -> ComponentState:
        """Get current component state."""
        return self.state


class BasePipeline(OrchestrationComponent):
    """Base class for all pipeline implementations."""
    
    def __init__(self, name: str, pipeline_type: PipelineType, config: Optional[ConfigDict] = None):
        super().__init__(name, config)
        self.pipeline_type = pipeline_type
        self.pipeline_state = PipelineState.IDLE
    
    @abstractmethod
    def run(self, **kwargs: Any) -> ResultDict:
        """Run the pipeline."""
        pass
    
    def get_pipeline_state(self) -> PipelineState:
        """Get current pipeline state."""
        return self.pipeline_state
    
    def pause(self) -> bool:
        """Pause pipeline execution (base implementation)."""
        if self.pipeline_state == PipelineState.RUNNING:
            self.pipeline_state = PipelineState.PAUSED
            return True
        return False
    
    def resume(self) -> bool:
        """Resume pipeline execution (base implementation)."""
        if self.pipeline_state == PipelineState.PAUSED:
            self.pipeline_state = PipelineState.RUNNING
            return True
        return False
    
    def cancel(self) -> bool:
        """Cancel pipeline execution (base implementation)."""
        if self.pipeline_state in (PipelineState.RUNNING, PipelineState.PAUSED):
            self.pipeline_state = PipelineState.CANCELLED
            return True
        return False