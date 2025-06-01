"""Document processing type definitions.

This module provides centralized type definitions for the document processing system,
including document types, metadata, adapters, and processing-related types.
"""

# Import all types from submodules
from src.types.docproc.base import *
from src.types.docproc.schema import *
from src.types.docproc.adapter import *
from src.types.docproc.processing import *
from src.types.docproc.document import *
from src.types.docproc.metadata import *

__all__ = [
    # Base document types
    "BaseEntity",
    "BaseMetadata", 
    "BaseDocument",
    
    # Schema types
    "DocumentSchema",
    "ChunkSchema",
    "MetadataSchema",
    "EntitySchema",
    
    # Adapter types
    "AdapterConfig",
    "AdapterOptions",
    "ProcessedDocument",
    "AdapterResult",
    "DocumentProcessorType",
    
    # Processing types
    "ProcessingOptions",
    "ProcessingResult",
    "BatchProcessingResult",
    "ProcessingStats",
    "FormatDetectionResult",
    "SuccessCallback",
    "ErrorCallback",
    
    # Document types
    "DocumentDict",
    "DocumentRelationDict",
    "DocumentCollectionDict",
    
    # Metadata types
    "EntityDict",
    "MetadataDict"
]
