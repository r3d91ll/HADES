"""
Type definitions for Python document processing.

This module provides type definitions specific to Python code processing,
including function information, class structures, and import relationships.
"""

from typing import Dict, Any, List, Optional, Union, TypedDict, Set

from src.types.docproc.enums import AccessLevel, RelationshipType


class FunctionInfo(TypedDict, total=False):
    """Information about a function in Python code."""
    
    name: str  # Function name
    qualified_name: str  # Fully qualified function name
    docstring: Optional[str]  # Function docstring
    parameters: List[str]  # List of parameter names
    parameter_types: Dict[str, str]  # Parameter name to type hint mapping
    returns: Optional[str]  # Return type hint as string
    is_async: bool  # Whether the function is async
    access_level: AccessLevel  # Public, protected, private
    line_range: List[int]  # [start_line, end_line]
    decorators: List[str]  # List of decorator names
    complexity: Optional[int]  # Cyclomatic complexity
    raises: List[str]  # Exceptions the function may raise
    calls: List[str]  # Functions this function calls
    parent: Optional[str]  # Parent class or module
    body_hash: Optional[str]  # Hash of function body for change detection


class ClassInfo(TypedDict, total=False):
    """Information about a class in Python code."""
    
    name: str  # Class name
    qualified_name: str  # Fully qualified class name
    docstring: Optional[str]  # Class docstring
    base_classes: List[str]  # List of base class names
    access_level: AccessLevel  # Public, protected, private
    line_range: List[int]  # [start_line, end_line]
    decorators: List[str]  # List of decorator names
    methods: List[str]  # Method qualified names
    attributes: List[Dict[str, Any]]  # Class attributes
    inner_classes: List[str]  # Nested class qualified names
    parent: Optional[str]  # Parent module or class


class ImportInfo(TypedDict, total=False):
    """Information about an import statement."""
    
    module: str  # Imported module name
    names: List[str]  # Names imported from module
    alias: Optional[str]  # Module alias if any
    is_from: bool  # Whether this is a from import
    line: int  # Line number of import statement
    is_relative: bool  # Whether this is a relative import
    level: int  # Level of relative import
    source_type: str  # stdlib, third-party, local, unknown
    resolved_path: Optional[str]  # Resolved filesystem path if local


class RelationshipInfo(TypedDict, total=False):
    """Information about a relationship between Python code elements."""
    
    source: str  # Source element qualified name
    target: str  # Target element qualified name
    type: RelationshipType  # Type of relationship
    line: int  # Line number where relationship occurs
    weight: float  # Relationship weight (0.0-1.0)
    description: Optional[str]  # Textual description of relationship


class PythonParserResult(TypedDict):
    """Result of parsing a Python file."""
    
    path: str  # File path
    module_name: str  # Module name
    imports: List[ImportInfo]  # Import statements
    functions: List[FunctionInfo]  # Functions
    classes: List[ClassInfo]  # Classes
    relationships: List[RelationshipInfo]  # Relationships
    errors: List[str]  # Parsing errors


class ASTNodeInfo(TypedDict, total=False):
    """Information about an AST node."""
    
    node_type: str  # Type of node
    line_start: int  # Starting line
    line_end: int  # Ending line
    col_start: int  # Starting column
    col_end: int  # Ending column
    parent_type: Optional[str]  # Type of parent node
    children_types: List[str]  # Types of child nodes
    source: Optional[str]  # Source code for this node
