"""
TypedDict definitions for code elements in document processing.

This module provides type definitions for representing Python code elements
in a structured format, suitable for static type checking and validation.
These types are used throughout the docproc module for Python code analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, TypedDict, Union

# Import enums but with an alias to avoid namespace conflicts
from src.types.components.docproc.enums import RelationshipType as RelType
from src.types.components.docproc.enums import AccessLevel, ImportSourceType


class CodeRelationship(TypedDict):
    """A relationship between code elements."""
    
    source: str  # Source element ID or fully qualified name
    target: str  # Target element ID or fully qualified name
    type: str  # Relationship type (from RelationshipType enum)
    weight: float  # Relationship weight (0.0-1.0)
    line: int  # Line number where relationship occurs


class ElementRelationship(TypedDict):
    """A relationship from the current element to another element."""
    
    type: str  # Relationship type (from RelationshipType enum)
    target: str  # Target element ID or fully qualified name
    line: int  # Line number where relationship occurs
    weight: float  # Relationship weight (0.0-1.0)


class LineRange(TypedDict):
    """A range of lines in the source code."""
    
    start: int  # Start line (inclusive)
    end: int  # End line (inclusive)


class Annotation(TypedDict, total=False):
    """Type annotation for a parameter or return value."""
    
    raw: str  # Raw annotation string as it appears in code
    resolved: str  # Resolved annotation (if possible)


class ImportElement(TypedDict):
    """An import statement in Python code."""
    
    type: Literal["import"]  # Element type
    name: str  # Module name
    alias: Optional[str]  # Import alias (e.g., 'np' for 'import numpy as np')
    source: str  # Source type (from ImportSourceType enum)
    line_range: List[int]  # [start_line, end_line]
    content: str  # Raw import statement


class FunctionElement(TypedDict, total=False):
    """A function definition in Python code."""
    
    type: Literal["function"]  # Element type
    name: str  # Function name
    qualified_name: str  # Fully qualified name (e.g., module.function)
    docstring: Optional[str]  # Function docstring
    parameters: List[str]  # Parameter names
    returns: Optional[str]  # Return type hint as string
    is_async: bool  # Whether function is async
    access: str  # Access level (from AccessLevel enum)
    line_range: List[int]  # [start_line, end_line]
    content: str  # Raw function code
    decorators: List[str]  # List of decorator names
    annotations: Dict[str, Annotation]  # Parameter and return annotations
    raises: List[str]  # Exceptions that may be raised
    doc_tags: List[str]  # Documentation tags (@param, @return, etc.)
    relationships: List[ElementRelationship]  # Relationships to other elements
    complexity: Optional[int]  # Cyclomatic complexity


class MethodElement(TypedDict, total=False):
    """A method definition in a class."""
    
    # Base fields from FunctionElement
    type: Literal["method"]  # Element type
    name: str  # Method name
    qualified_name: str  # Fully qualified name (e.g., module.class.method)
    docstring: Optional[str]  # Method docstring
    parameters: List[str]  # Parameter names
    returns: Optional[str]  # Return type hint as string
    is_async: bool  # Whether function is async
    access: str  # Access level (from AccessLevel enum)
    line_range: List[int]  # [start_line, end_line]
    content: str  # Raw function code
    decorators: List[str]  # List of decorator names
    annotations: Dict[str, Annotation]  # Parameter and return annotations
    raises: List[str]  # Exceptions that may be raised
    doc_tags: List[str]  # Documentation tags (@param, @return, etc.)
    relationships: List[ElementRelationship]  # Relationships to other elements
    complexity: Optional[int]  # Cyclomatic complexity
    
    # Method-specific fields
    is_static: bool  # Whether method is a static method
    is_class_method: bool  # Whether method is a class method
    is_property: bool  # Whether method is a property
    parent_class: str  # Containing class (ID or fully qualified name)


class ClassElement(TypedDict, total=False):
    """A class definition in Python code."""
    
    type: Literal["class"]  # Element type
    name: str  # Class name
    qualified_name: str  # Fully qualified name (e.g., module.class)
    docstring: Optional[str]  # Class docstring
    base_classes: List[str]  # Base class names
    access: str  # Access level (from AccessLevel enum)
    line_range: List[int]  # [start_line, end_line]
    content: str  # Raw class code
    decorators: List[str]  # List of decorator names
    elements: List[Union[Dict[str, Any]]]  # Class members
    relationships: List[ElementRelationship]  # Relationships to other elements


class ModuleElement(TypedDict, total=False):
    """A Python module."""
    
    type: Literal["module"]  # Element type
    name: str  # Module name
    path: str  # File path
    docstring: Optional[str]  # Module docstring
    module_path: Optional[str]  # Dotted import path
    line_range: List[int]  # [start_line, end_line]
    elements: List[Dict[str, Any]]  # Module elements
    relationships: List[ElementRelationship]  # Relationships to other elements


class PySymbolTable(TypedDict, total=False):
    """The symbol table for a Python module."""
    
    type: Literal["module"]
    name: str
    docstring: Optional[str]
    path: str  # File path of the module
    module_path: str  # Dotted import path
    line_range: List[int]  # [start_line, end_line]
    elements: List[Dict[str, Any]]  # Can contain any element type


class PythonDocument(TypedDict, total=False):
    """The complete representation of a Python document."""
    
    id: str  # Unique document ID
    source: str  # Source file path
    content: str  # Markdown-formatted content
    content_type: str  # Content format type (e.g., "markdown")
    format: str  # Document format (e.g., "python")
    raw_content: str  # Original source code
    
    metadata: Dict[str, Any]  # Document metadata
    entities: List[Dict[str, Any]]  # Flat list of entities for quick reference
    symbol_table: PySymbolTable  # Hierarchical symbol table
    relationships: List[CodeRelationship]  # Cross-references between symbols
    error: Optional[str]  # Error message if parsing failed
