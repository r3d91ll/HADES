"""Base model engine interface for HADES.

This module provides the core ModelEngine abstract base class that all
concrete engine implementations should inherit from. It imports the actual
class definition from the centralized types module.
"""

# Import the ModelEngine base class from types
from src.types.components.protocols import ModelEngine
from src.types.components.model_engine.engine import (
    ModelEngineProtocol,
    EngineType,
    ModelType,
    ModelStatus,
    EngineStatus,
    ModelInfo,
    HealthStatus,
    EngineRegistry
)

__all__ = [
    "ModelEngine",
    "ModelEngineProtocol", 
    "EngineType",
    "ModelType",
    "ModelStatus",
    "EngineStatus",
    "ModelInfo",
    "HealthStatus",
    "EngineRegistry"
]