"""
Type definitions for the ISNE module.

This package contains type definitions, data models, and related utilities
for the ISNE (Inductive Shallow Node Embedding) implementation.

MIGRATION NOTE: This module now re-exports types from the centralized type system
at src.types.isne for backward compatibility. New code should import directly
from src.types.isne instead.
"""

# Re-export from centralized type system for backward compatibility
from src.types.isne.base import (
    DocumentType,
    RelationType,
    EmbeddingVector,
    ISNEModelType,
    ActivationFunction,
    AttentionType,
    OptimizerType,
    LossType,
    SamplingStrategy,
    ISNEModelProtocol,
    ISNELoaderProtocol,
    ISNETrainerProtocol,
    RELATIONSHIP_WEIGHTS,
)

from src.types.isne.documents import (
    IngestDocument,
    DocumentRelation,
    LoaderResult,
    IngestDocumentDict,
    DocumentRelationDict,
    LoaderResultDict,
    PydanticIngestDocument,
    PydanticDocumentRelation,
    AnyIngestDocument,
    AnyDocumentRelation,
    AnyLoaderResult,
)

from src.types.isne.models import (
    ISNEModelConfig,
    ISNETrainingConfig,
    ISNEGraphConfig,
    ISNEDirectoriesConfig,
    ISNEConfig,
)

# Keep original __all__ for backward compatibility
__all__ = [
    # Original exports
    'DocumentType',
    'RelationType',
    'IngestDocument',
    'DocumentRelation',
    'LoaderResult',
    'EmbeddingVector',
    # Additional exports from new types
    'ISNEModelType',
    'ActivationFunction',
    'AttentionType',
    'OptimizerType',
    'LossType',
    'SamplingStrategy',
    'ISNEModelProtocol',
    'ISNELoaderProtocol',
    'ISNETrainerProtocol',
    'RELATIONSHIP_WEIGHTS',
    'IngestDocumentDict',
    'DocumentRelationDict',
    'LoaderResultDict',
    'PydanticIngestDocument',
    'PydanticDocumentRelation',
    'AnyIngestDocument',
    'AnyDocumentRelation',
    'AnyLoaderResult',
    'ISNEModelConfig',
    'ISNETrainingConfig',
    'ISNEGraphConfig',
    'ISNEDirectoriesConfig',
    'ISNEConfig',
]
