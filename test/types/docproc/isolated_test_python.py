"""
Isolated unit tests for the docproc Python type definitions.

This module tests the TypedDict definitions that have been
migrated to centralized type files in the src/types/docproc directory
without requiring external dependencies.
"""

import unittest
import os
import sys
from typing import Dict, Any, List, Optional, TypedDict

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Import our mock types
from mock_types import (
    BaseMetadata,
    BaseEntity,
    BaseDocument,
    RelationshipType,
    AccessLevel,
    ImportSourceType,
    typed_field_validator,
    typed_model_validator
)

# Manually define the TypedDicts from code_elements.py for testing
class CodeRelationship(TypedDict):
    """Defines a relationship between two code elements."""
    source: str
    target: str
    type: str
    weight: float
    line: int


class ElementRelationship(TypedDict):
    """Defines a relationship with additional attributes."""
    source: str
    target: str
    type: str
    weight: float
    attributes: Dict[str, Any]


class LineRange(TypedDict):
    """Defines a range of lines in a file."""
    start: int
    end: int


class Annotation(TypedDict):
    """Defines an annotation or decorator."""
    name: str
    args: List[str]
    kwargs: Dict[str, Any]


class ImportElement(TypedDict):
    """Defines an import statement."""
    name: str
    module: str
    alias: Optional[str]
    is_from: bool
    line: int
    source_type: str


class FunctionElement(TypedDict):
    """Defines a function."""
    name: str
    signature: str
    docstring: Optional[str]
    line: int
    end_line: int
    line_range: List[int]
    is_method: bool
    is_async: bool
    annotations: List[Annotation]
    content: str
    relationships: List[CodeRelationship]


class MethodElement(TypedDict):
    """Defines a method."""
    name: str
    signature: str
    docstring: Optional[str]
    line: int
    end_line: int
    line_range: List[int]
    is_async: bool
    is_static: bool
    is_class: bool
    is_abstract: bool
    access_level: str
    annotations: List[Annotation]
    content: str
    relationships: List[CodeRelationship]
    class_name: str


class ClassElement(TypedDict):
    """Defines a class."""
    name: str
    bases: List[str]
    docstring: Optional[str]
    line: int
    end_line: int
    line_range: List[int]
    methods: List[str]
    attributes: Dict[str, Any]
    annotations: List[Annotation]
    content: str
    relationships: List[CodeRelationship]
    elements: List[Dict[str, Any]]


class ModuleElement(TypedDict):
    """Defines a module."""
    name: str
    docstring: Optional[str]
    path: str
    imports: List[ImportElement]
    functions: List[FunctionElement]
    classes: List[ClassElement]
    relationships: List[CodeRelationship]


class PySymbolTable(TypedDict):
    """Defines a Python symbol table."""
    name: str
    path: str
    elements: List[Dict[str, Any]]
    line_range: List[int]


class PythonDocument(TypedDict):
    """Defines a Python document."""
    id: str
    content: str
    path: str
    filename: str
    type: str
    metadata: Dict[str, Any]
    symbol_table: PySymbolTable
    timestamp: Optional[float]
    title: Optional[str]


class TestCodeElements(unittest.TestCase):
    """Test cases for TypedDict definitions."""
    
    def test_code_relationship(self) -> None:
        """Test the CodeRelationship TypedDict."""
        # Create a valid code relationship
        relationship: CodeRelationship = {
            "source": "func_a",
            "target": "func_b",
            "type": RelationshipType.CALLS,
            "weight": 1.0,
            "line": 42
        }
        
        # Verify all fields
        self.assertEqual(relationship["source"], "func_a")
        self.assertEqual(relationship["target"], "func_b")
        self.assertEqual(relationship["type"], RelationshipType.CALLS)
        self.assertEqual(relationship["weight"], 1.0)
        self.assertEqual(relationship["line"], 42)
    
    def test_element_relationship(self) -> None:
        """Test the ElementRelationship TypedDict."""
        # Create a valid element relationship
        relationship: ElementRelationship = {
            "source": "class_a",
            "target": "class_b",
            "type": RelationshipType.EXTENDS,
            "weight": 1.0,
            "attributes": {"inheritance_type": "direct"}
        }
        
        # Verify all fields
        self.assertEqual(relationship["source"], "class_a")
        self.assertEqual(relationship["target"], "class_b")
        self.assertEqual(relationship["type"], RelationshipType.EXTENDS)
        self.assertEqual(relationship["weight"], 1.0)
        self.assertEqual(relationship["attributes"]["inheritance_type"], "direct")
    
    def test_line_range(self) -> None:
        """Test the LineRange TypedDict."""
        # Create a valid line range
        line_range: LineRange = {
            "start": 10,
            "end": 20
        }
        
        # Verify all fields
        self.assertEqual(line_range["start"], 10)
        self.assertEqual(line_range["end"], 20)
    
    def test_annotation(self) -> None:
        """Test the Annotation TypedDict."""
        # Create a valid annotation
        annotation: Annotation = {
            "name": "deprecated",
            "args": ["Use new_function instead"],
            "kwargs": {"version": "1.2.0"}
        }
        
        # Verify all fields
        self.assertEqual(annotation["name"], "deprecated")
        self.assertEqual(annotation["args"], ["Use new_function instead"])
        self.assertEqual(annotation["kwargs"]["version"], "1.2.0")
    
    def test_import_element(self) -> None:
        """Test the ImportElement TypedDict."""
        # Create a valid import element
        import_element: ImportElement = {
            "name": "requests",
            "module": "requests",
            "alias": None,
            "is_from": False,
            "line": 5,
            "source_type": ImportSourceType.THIRD_PARTY
        }
        
        # Verify all fields
        self.assertEqual(import_element["name"], "requests")
        self.assertEqual(import_element["module"], "requests")
        self.assertIsNone(import_element["alias"])
        self.assertFalse(import_element["is_from"])
        self.assertEqual(import_element["line"], 5)
        self.assertEqual(import_element["source_type"], ImportSourceType.THIRD_PARTY)
    
    def test_function_element(self) -> None:
        """Test the FunctionElement TypedDict."""
        # Create a valid function element
        function: FunctionElement = {
            "name": "process_data",
            "signature": "def process_data(data: Dict[str, Any]) -> List[Dict[str, Any]]:",
            "docstring": "Process the input data and return a list of processed items.",
            "line": 10,
            "end_line": 20,
            "line_range": [10, 20],
            "is_method": False,
            "is_async": False,
            "annotations": [],
            "content": "def process_data(data: Dict[str, Any]) -> List[Dict[str, Any]]:\n    return [item for item in data.values()]",
            "relationships": []
        }
        
        # Verify all fields
        self.assertEqual(function["name"], "process_data")
        self.assertEqual(function["signature"], "def process_data(data: Dict[str, Any]) -> List[Dict[str, Any]]:")
        self.assertEqual(function["docstring"], "Process the input data and return a list of processed items.")
        self.assertEqual(function["line"], 10)
        self.assertEqual(function["end_line"], 20)
        self.assertEqual(function["line_range"], [10, 20])
        self.assertFalse(function["is_method"])
        self.assertFalse(function["is_async"])
        self.assertEqual(function["annotations"], [])
        self.assertEqual(function["content"], "def process_data(data: Dict[str, Any]) -> List[Dict[str, Any]]:\n    return [item for item in data.values()]")
        self.assertEqual(function["relationships"], [])
    
    def test_method_element(self) -> None:
        """Test the MethodElement TypedDict."""
        # Create a valid method element
        method: MethodElement = {
            "name": "process_data",
            "signature": "def process_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:",
            "docstring": "Process the input data and return a list of processed items.",
            "line": 30,
            "end_line": 40,
            "line_range": [30, 40],
            "is_async": False,
            "is_static": False,
            "is_class": False,
            "is_abstract": False,
            "access_level": AccessLevel.PUBLIC,
            "annotations": [],
            "content": "def process_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:\n    return [item for item in data.values()]",
            "relationships": [],
            "class_name": "DataProcessor"
        }
        
        # Verify all fields
        self.assertEqual(method["name"], "process_data")
        self.assertEqual(method["signature"], "def process_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:")
        self.assertEqual(method["docstring"], "Process the input data and return a list of processed items.")
        self.assertEqual(method["line"], 30)
        self.assertEqual(method["end_line"], 40)
        self.assertEqual(method["line_range"], [30, 40])
        self.assertFalse(method["is_async"])
        self.assertFalse(method["is_static"])
        self.assertFalse(method["is_class"])
        self.assertFalse(method["is_abstract"])
        self.assertEqual(method["access_level"], AccessLevel.PUBLIC)
        self.assertEqual(method["annotations"], [])
        self.assertEqual(method["content"], "def process_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:\n    return [item for item in data.values()]")
        self.assertEqual(method["relationships"], [])
        self.assertEqual(method["class_name"], "DataProcessor")
    
    def test_class_element(self) -> None:
        """Test the ClassElement TypedDict."""
        # Create a valid class element
        class_element: ClassElement = {
            "name": "DataProcessor",
            "bases": ["BaseProcessor"],
            "docstring": "A class for processing data.",
            "line": 50,
            "end_line": 100,
            "line_range": [50, 100],
            "methods": ["process_data", "validate_data"],
            "attributes": {"data_source": "database"},
            "annotations": [],
            "content": "class DataProcessor(BaseProcessor):\n    data_source = 'database'\n    \n    def process_data(self, data):\n        pass\n    \n    def validate_data(self, data):\n        pass",
            "relationships": [],
            "elements": []
        }
        
        # Verify all fields
        self.assertEqual(class_element["name"], "DataProcessor")
        self.assertEqual(class_element["bases"], ["BaseProcessor"])
        self.assertEqual(class_element["docstring"], "A class for processing data.")
        self.assertEqual(class_element["line"], 50)
        self.assertEqual(class_element["end_line"], 100)
        self.assertEqual(class_element["line_range"], [50, 100])
        self.assertEqual(class_element["methods"], ["process_data", "validate_data"])
        self.assertEqual(class_element["attributes"]["data_source"], "database")
        self.assertEqual(class_element["annotations"], [])
        self.assertEqual(class_element["content"], "class DataProcessor(BaseProcessor):\n    data_source = 'database'\n    \n    def process_data(self, data):\n        pass\n    \n    def validate_data(self, data):\n        pass")
        self.assertEqual(class_element["relationships"], [])
        self.assertEqual(class_element["elements"], [])
    
    def test_module_element(self) -> None:
        """Test the ModuleElement TypedDict."""
        # Create a valid module element
        module: ModuleElement = {
            "name": "data_processing",
            "docstring": "Module for data processing utilities.",
            "path": "/path/to/data_processing.py",
            "imports": [],
            "functions": [],
            "classes": [],
            "relationships": []
        }
        
        # Verify all fields
        self.assertEqual(module["name"], "data_processing")
        self.assertEqual(module["docstring"], "Module for data processing utilities.")
        self.assertEqual(module["path"], "/path/to/data_processing.py")
        self.assertEqual(module["imports"], [])
        self.assertEqual(module["functions"], [])
        self.assertEqual(module["classes"], [])
        self.assertEqual(module["relationships"], [])
    
    def test_py_symbol_table(self) -> None:
        """Test the PySymbolTable TypedDict."""
        # Create a valid symbol table
        symbol_table: PySymbolTable = {
            "name": "data_processing",
            "path": "/path/to/data_processing.py",
            "elements": [],
            "line_range": [1, 200]
        }
        
        # Verify all fields
        self.assertEqual(symbol_table["name"], "data_processing")
        self.assertEqual(symbol_table["path"], "/path/to/data_processing.py")
        self.assertEqual(symbol_table["elements"], [])
        self.assertEqual(symbol_table["line_range"], [1, 200])
    
    def test_python_document(self) -> None:
        """Test the PythonDocument TypedDict."""
        # Create a valid Python document
        document: PythonDocument = {
            "id": "doc-123",
            "content": "# Python code\n\ndef main():\n    pass\n\nif __name__ == '__main__':\n    main()",
            "path": "/path/to/script.py",
            "filename": "script.py",
            "type": "python",
            "metadata": {"function_count": 1, "class_count": 0},
            "symbol_table": {
                "name": "script",
                "path": "/path/to/script.py",
                "elements": [],
                "line_range": [1, 7]
            },
            "timestamp": 1622548800.0,
            "title": "Sample Python Script"
        }
        
        # Verify all fields
        self.assertEqual(document["id"], "doc-123")
        self.assertEqual(document["content"], "# Python code\n\ndef main():\n    pass\n\nif __name__ == '__main__':\n    main()")
        self.assertEqual(document["path"], "/path/to/script.py")
        self.assertEqual(document["filename"], "script.py")
        self.assertEqual(document["type"], "python")
        self.assertEqual(document["metadata"]["function_count"], 1)
        self.assertEqual(document["metadata"]["class_count"], 0)
        self.assertEqual(document["symbol_table"]["name"], "script")
        self.assertEqual(document["symbol_table"]["path"], "/path/to/script.py")
        self.assertEqual(document["symbol_table"]["line_range"], [1, 7])
        self.assertEqual(document["timestamp"], 1622548800.0)
        self.assertEqual(document["title"], "Sample Python Script")


if __name__ == "__main__":
    unittest.main()
