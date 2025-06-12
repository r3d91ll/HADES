"""ISNE type definitions.

This module provides type definitions for the ISNE (Inductive Shallow Node Embedding) system.
"""

# Export base types
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

# Export document types
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

# Export model config types
from src.types.isne.models import (
    ISNEModelConfig,
    ISNETrainingConfig,
    ISNEGraphConfig,
    ISNEDirectoriesConfig,
    ISNEConfig,
)

__all__ = [
    # Base types
    "DocumentType",
    "RelationType",
    "EmbeddingVector",
    "ISNEModelType",
    "ActivationFunction",
    "AttentionType",
    "OptimizerType",
    "LossType",
    "SamplingStrategy",
    "ISNEModelProtocol",
    "ISNELoaderProtocol",
    "ISNETrainerProtocol",
    "RELATIONSHIP_WEIGHTS",
    # Document types
    "IngestDocument",
    "DocumentRelation",
    "LoaderResult",
    "IngestDocumentDict",
    "DocumentRelationDict",
    "LoaderResultDict",
    "PydanticIngestDocument",
    "PydanticDocumentRelation",
    "AnyIngestDocument",
    "AnyDocumentRelation",
    "AnyLoaderResult",
    # Model config types
    "ISNEModelConfig",
    "ISNETrainingConfig",
    "ISNEGraphConfig",
    "ISNEDirectoriesConfig",
    "ISNEConfig",
]
