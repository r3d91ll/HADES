"""
Python document schemas for document processing validation.

This module provides Pydantic models that define the structure for Python
code documents processed in the pipeline. These models ensure proper
validation of Python-specific fields and relationships.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import Field, field_validator, model_validator, ConfigDict

from src.schemas.common.base import BaseSchema
from src.schemas.common.types import MetadataDict

from src.docproc.schemas.base import BaseDocument, BaseEntity, BaseMetadata
from src.docproc.models.python_code import RelationshipType, AccessLevel


class PythonMetadata(BaseMetadata):
    """Metadata specific to Python documents."""
    
    function_count: int = Field(0, description="Number of functions in the document")
    class_count: int = Field(0, description="Number of classes in the document")
    import_count: int = Field(0, description="Number of imports in the document")
    method_count: int = Field(0, description="Number of methods in the document")
    has_module_docstring: bool = Field(False, description="Whether the module has a docstring")
    has_syntax_errors: bool = Field(False, description="Whether the file has Python syntax errors")


class PythonEntity(BaseEntity):
    """Entity specific to Python code."""
    
    type: Literal["module", "class", "function", "method", "import", "decorator"] = Field(
        ..., description="Type of Python entity"
    )


class CodeRelationship(BaseSchema):
    """Relationship between code elements."""
    
    source: str = Field(..., description="Source element identifier")
    target: str = Field(..., description="Target element identifier")
    type: str = Field(..., description="Type of relationship")
    weight: float = Field(..., ge=0.0, le=1.0, description="Weight/strength of the relationship")
    line: Optional[int] = Field(None, description="Line number where relationship is defined")
    
    @field_validator("type")
    @classmethod
    def validate_relationship_type(cls, v: str) -> str:
        """Validate relationship type against defined values."""
        valid_types = {rel_type.value for rel_type in RelationshipType}
        if v not in valid_types:
            raise ValueError(f"Invalid relationship type: {v}. Must be one of: {valid_types}")
        return v


class CodeElement(BaseSchema):
    """Generic code element."""
    
    type: str = Field(..., description="Type of code element")
    name: str = Field(..., description="Name of the code element")
    qualified_name: Optional[str] = Field(None, description="Fully qualified name")
    docstring: Optional[str] = Field(None, description="Element docstring")
    line_range: Optional[List[int]] = Field(None, description="Line range [start, end]")
    content: Optional[str] = Field(None, description="Element source code content")
    
    model_config = ConfigDict(extra="allow")  # Allow type-specific fields


class FunctionElement(CodeElement):
    """Function code element."""
    
    type: Literal["function"] = "function"
    parameters: List[str] = Field(default_factory=list, description="Function parameters")
    returns: Optional[str] = Field(None, description="Return type annotation")
    is_async: bool = Field(False, description="Whether this is an async function")
    access: str = Field(AccessLevel.PUBLIC.value, description="Access level")
    decorators: List[str] = Field(default_factory=list, description="Function decorators")


class MethodElement(CodeElement):
    """Method code element."""
    
    type: Literal["method"] = "method"
    parameters: List[str] = Field(default_factory=list, description="Method parameters")
    returns: Optional[str] = Field(None, description="Return type annotation")
    is_async: bool = Field(False, description="Whether this is an async method")
    access: str = Field(AccessLevel.PUBLIC.value, description="Access level")
    decorators: List[str] = Field(default_factory=list, description="Method decorators")
    is_static: bool = Field(False, description="Whether this is a static method")
    is_class_method: bool = Field(False, description="Whether this is a class method")
    is_property: bool = Field(False, description="Whether this is a property")
    parent_class: str = Field(..., description="Name of the containing class")


class ClassElement(CodeElement):
    """Class code element."""
    
    type: Literal["class"] = "class"
    base_classes: List[str] = Field(default_factory=list, description="Base class names")
    access: str = Field(AccessLevel.PUBLIC.value, description="Access level")
    decorators: List[str] = Field(default_factory=list, description="Class decorators")
    elements: List[Dict[str, Any]] = Field(default_factory=list, description="Class members")


class ImportElement(CodeElement):
    """Import code element."""
    
    type: Literal["import"] = "import"
    alias: Optional[str] = Field(None, description="Import alias")
    source: str = Field(..., description="Import source type")


class SymbolTable(BaseSchema):
    """Python module symbol table."""
    
    type: Literal["module"] = "module"
    name: str = Field(..., description="Module name")
    docstring: Optional[str] = Field(None, description="Module docstring")
    path: Optional[str] = Field(None, description="File path of the module")
    module_path: Optional[str] = Field(None, description="Dotted import path")
    line_range: Optional[List[int]] = Field(None, description="Line range [start, end]")
    elements: List[Dict[str, Any]] = Field(default_factory=list, description="Module elements")
    


class PythonDocument(BaseDocument):
    """Document model for Python code."""
    
    format: Literal["python"] = "python"
    metadata: PythonMetadata = Field(..., description="Python-specific metadata")
    # Use type annotation that preserves compatibility with BaseDocument
    entities: List[BaseEntity] = Field(default_factory=list, description="Python entities")
    relationships: Optional[List[CodeRelationship]] = Field(None, description="Code relationships")
    symbol_table: Optional[SymbolTable] = Field(None, description="Python symbol table")
    
    @model_validator(mode='after')
    def validate_code_consistency(self) -> 'PythonDocument':
        """Ensure code-specific fields are consistent."""
        # If there are syntax errors, symbol_table might be missing
        if self.metadata.has_syntax_errors and not self.symbol_table:
            return self
            
        # Check if entity counts are consistent with metadata when there's a symbol table
        if self.symbol_table and not self.metadata.has_errors:
            # Count entities by type
            entity_counts = {
                "function": 0, 
                "class": 0, 
                "method": 0,
                "import": 0
            }
            
            for entity in self.entities:
                if entity.type in entity_counts:
                    entity_counts[entity.type] += 1
            
            # Verify metadata counts (just log warnings, don't fail validation)
            if self.metadata.function_count != entity_counts["function"]:
                print(f"Warning: Metadata function count ({self.metadata.function_count}) " 
                      f"doesn't match entities count ({entity_counts['function']})")
                
            if self.metadata.class_count != entity_counts["class"]:
                print(f"Warning: Metadata class count ({self.metadata.class_count}) " 
                      f"doesn't match entities count ({entity_counts['class']})")
                
            if self.metadata.import_count != entity_counts["import"]:
                print(f"Warning: Metadata import count ({self.metadata.import_count}) " 
                      f"doesn't match entities count ({entity_counts['import']})")
                
        return self
