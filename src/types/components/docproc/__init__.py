"""
Type definitions for the document processing (docproc) module.

This package contains centralized type definitions for the document processing
components of the HADES system, including document models, adapter protocols,
and specialized schemas for different document formats.
"""

# Import enum types
from src.types.components.docproc.enums import (
    RelationshipType,
    AccessLevel,
    ImportSourceType,
    ContentCategory
)

# Import Pydantic model types
from src.types.components.docproc.python import (
    PythonMetadata,
    PythonEntity,
    CodeRelationship as PydanticCodeRelationship,
    CodeElement,
    FunctionElement as PydanticFunctionElement,
    MethodElement as PydanticMethodElement,
    ClassElement as PydanticClassElement,
    ImportElement as PydanticImportElement,
    SymbolTable,
    PythonDocument as PydanticPythonDocument,
    # Import validator helpers
    typed_field_validator,
    typed_model_validator
)

# Import TypedDict definitions for code elements
from src.types.components.docproc.code_elements import (
    CodeRelationship as TypedDictCodeRelationship,
    ElementRelationship,
    LineRange,
    Annotation,
    FunctionElement as TypedDictFunctionElement,
    MethodElement as TypedDictMethodElement,
    ClassElement as TypedDictClassElement,
    ImportElement as TypedDictImportElement,
    ModuleElement,
    PySymbolTable,
    PythonDocument as TypedDictPythonDocument
)

# Import adapter types
from src.types.components.docproc.adapter import (
    AdapterProtocol,
    AdapterRegistry,
    ExtractorOptions,
    MetadataExtractionConfig,
    EntityExtractionConfig,
    ChunkingPreparationConfig,
    ProcessorConfig,
    DocumentProcessingError,
    FormatDetectionResult,
    ProcessSuccessCallback,
    ProcessErrorCallback
)

# Import document types
from src.types.components.docproc.document import (
    # Core TypedDict definitions
    DocumentEntity,
    DocumentMetadata,
    DocumentSection,
    ProcessedDocument,
    ChunkPreparationMarker,
    ChunkPreparedDocument,
    BatchProcessingStatistics,
    DocumentProcessingResult,
    ChunkSchema,
    DocumentSchema,
    
    # Pydantic model definitions
    PydanticDocumentEntity,
    PydanticDocumentMetadata,
    PydanticProcessedDocument,
    PydanticChunkSchema,
    
    # Legacy compatibility classes (consolidated from src.types.documents)
    BaseEntity,
    BaseMetadata,
    BaseDocument,
    
    # Aliases for backward compatibility
    EntitySchema,
    MetadataSchema,
    
    # Helper types
    DocumentSource,
    DocumentFilter,
    DocumentSortOptions,
    ProcessingMode,
    
    # Consolidated types from src.types.documents (main system-wide aliases)
    ChunkMetadata,  # System-wide alias for ConsolidatedChunkMetadata
    DocumentSchema  # System-wide alias for ConsolidatedDocumentSchema
)

# Import configuration types
from src.types.components.docproc.config import (
    FormatDetectionConfig,
    TextCleaningConfig,
    AdapterConfig,
    MarkdownConfig,
    HtmlConfig,
    PdfConfig,
    PythonConfig,
    JsonConfig,
    YamlConfig,
    DocumentProcessorConfig
)

# Import format-specific types
# JSON format types
from src.types.components.docproc.formats.json import (
    JSONNodeInfo,
    JSONQueryResult,
    JSONRelationship,
    JSONPathSegment,
    JSONSchemaValidationResult
)

# Python format types
from src.types.components.docproc.formats.python import (
    FunctionInfo,
    ClassInfo,
    ImportInfo,
    RelationshipInfo,
    PythonParserResult,
    ASTNodeInfo
)

# YAML format types
from src.types.components.docproc.formats.yaml import (
    YAMLNodeInfo,
    YAMLValidationResult,
    YAMLRelationship,
    YAMLDocumentInfo
)
