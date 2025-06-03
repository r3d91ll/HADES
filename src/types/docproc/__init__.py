"""
Type definitions for the document processing (docproc) module.

This package contains centralized type definitions for the document processing
components of the HADES system, including document models, adapter protocols,
and specialized schemas for different document formats.
"""

# Import centralized type modules
# Import enum types
from src.types.docproc.enums import (
    RelationshipType,
    AccessLevel,
    ImportSourceType
)

# Import Pydantic model types
from src.types.docproc.python import (
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

# Import TypedDict definitions
from src.types.docproc.code_elements import (
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
