"""
Unit tests for the docproc code element type definitions.

This module tests the TypedDict definitions that have been migrated to 
centralized type files in the src/types/docproc directory.
"""

import os
import sys
import unittest
from typing import Dict, Any, List, Optional, Union, cast

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.types.docproc.code_elements import (
    CodeRelationship,
    ElementRelationship,
    LineRange,
    Annotation,
    ImportElement,
    FunctionElement,
    MethodElement,
    ClassElement,
    ModuleElement,
    PySymbolTable,
    PythonDocument
)
from src.types.docproc.enums import RelationshipType, AccessLevel, ImportSourceType


class TestCodeRelationship(unittest.TestCase):
    """Test cases for the CodeRelationship TypedDict."""
    
    def test_code_relationship_creation(self) -> None:
        """Test creating a CodeRelationship."""
        relationship: CodeRelationship = {
            "source": "source_id",
            "target": "target_id",
            "type": RelationshipType.CALLS.value,
            "weight": 0.9,
            "line": 42
        }
        
        self.assertEqual(relationship["source"], "source_id")
        self.assertEqual(relationship["target"], "target_id")
        self.assertEqual(relationship["type"], RelationshipType.CALLS.value)
        self.assertEqual(relationship["weight"], 0.9)
        self.assertEqual(relationship["line"], 42)


class TestElementRelationship(unittest.TestCase):
    """Test cases for the ElementRelationship TypedDict."""
    
    def test_element_relationship_creation(self) -> None:
        """Test creating an ElementRelationship."""
        relationship: ElementRelationship = {
            "type": RelationshipType.CONTAINS.value,
            "target": "target_id",
            "line": 10,
            "weight": 0.8
        }
        
        self.assertEqual(relationship["type"], RelationshipType.CONTAINS.value)
        self.assertEqual(relationship["target"], "target_id")
        self.assertEqual(relationship["line"], 10)
        self.assertEqual(relationship["weight"], 0.8)


class TestLineRange(unittest.TestCase):
    """Test cases for the LineRange TypedDict."""
    
    def test_line_range_creation(self) -> None:
        """Test creating a LineRange."""
        line_range: LineRange = {
            "start": 10,
            "end": 20
        }
        
        self.assertEqual(line_range["start"], 10)
        self.assertEqual(line_range["end"], 20)


class TestAnnotation(unittest.TestCase):
    """Test cases for the Annotation TypedDict."""
    
    def test_annotation_creation_with_all_fields(self) -> None:
        """Test creating an Annotation with all fields."""
        annotation: Annotation = {
            "raw": "List[str]",
            "resolved": "typing.List[str]"
        }
        
        self.assertEqual(annotation["raw"], "List[str]")
        self.assertEqual(annotation["resolved"], "typing.List[str]")
    
    def test_annotation_creation_with_raw_only(self) -> None:
        """Test creating an Annotation with raw only."""
        annotation: Annotation = {
            "raw": "Dict[str, Any]"
        }
        
        self.assertEqual(annotation["raw"], "Dict[str, Any]")
        self.assertNotIn("resolved", annotation)


class TestImportElement(unittest.TestCase):
    """Test cases for the ImportElement TypedDict."""
    
    def test_import_element_creation(self) -> None:
        """Test creating an ImportElement."""
        import_element: ImportElement = {
            "type": "import",
            "name": "numpy",
            "alias": "np",
            "source": ImportSourceType.THIRD_PARTY.value,
            "line_range": [1, 1],
            "content": "import numpy as np"
        }
        
        self.assertEqual(import_element["type"], "import")
        self.assertEqual(import_element["name"], "numpy")
        self.assertEqual(import_element["alias"], "np")
        self.assertEqual(import_element["source"], ImportSourceType.THIRD_PARTY.value)
        self.assertEqual(import_element["line_range"], [1, 1])
        self.assertEqual(import_element["content"], "import numpy as np")
    
    def test_import_element_without_alias(self) -> None:
        """Test creating an ImportElement without an alias."""
        import_element: ImportElement = {
            "type": "import",
            "name": "sys",
            "source": ImportSourceType.STDLIB.value,
            "line_range": [2, 2],
            "content": "import sys"
        }
        
        self.assertEqual(import_element["type"], "import")
        self.assertEqual(import_element["name"], "sys")
        self.assertEqual(import_element["source"], ImportSourceType.STDLIB.value)
        self.assertEqual(import_element["line_range"], [2, 2])
        self.assertEqual(import_element["content"], "import sys")
        self.assertNotIn("alias", import_element)


class TestFunctionElement(unittest.TestCase):
    """Test cases for the FunctionElement TypedDict."""
    
    def test_function_element_creation(self) -> None:
        """Test creating a FunctionElement."""
        function: FunctionElement = {
            "type": "function",
            "name": "test_function",
            "qualified_name": "module.test_function",
            "docstring": "Test function docstring",
            "parameters": ["param1", "param2"],
            "returns": "str",
            "is_async": False,
            "access": AccessLevel.PUBLIC.value,
            "line_range": [10, 20],
            "content": "def test_function():\n    pass",
            "decorators": ["decorator1"]
        }
        
        self.assertEqual(function["type"], "function")
        self.assertEqual(function["name"], "test_function")
        self.assertEqual(function["qualified_name"], "module.test_function")
        self.assertEqual(function["docstring"], "Test function docstring")
        self.assertEqual(function["parameters"], ["param1", "param2"])
        self.assertEqual(function["returns"], "str")
        self.assertEqual(function["is_async"], False)
        self.assertEqual(function["access"], AccessLevel.PUBLIC.value)
        self.assertEqual(function["line_range"], [10, 20])
        self.assertEqual(function["content"], "def test_function():\n    pass")
        self.assertEqual(function["decorators"], ["decorator1"])
    
    def test_function_element_with_optional_fields(self) -> None:
        """Test creating a FunctionElement with only required fields."""
        function: FunctionElement = {
            "type": "function",
            "name": "minimal_function"
        }
        
        self.assertEqual(function["type"], "function")
        self.assertEqual(function["name"], "minimal_function")
        self.assertNotIn("qualified_name", function)
        self.assertNotIn("docstring", function)
        self.assertNotIn("parameters", function)
        self.assertNotIn("returns", function)


class TestMethodElement(unittest.TestCase):
    """Test cases for the MethodElement TypedDict."""
    
    def test_method_element_creation(self) -> None:
        """Test creating a MethodElement."""
        method: MethodElement = {
            "type": "method",
            "name": "test_method",
            "qualified_name": "module.TestClass.test_method",
            "docstring": "Test method docstring",
            "parameters": ["self", "param1"],
            "returns": "str",
            "is_async": True,
            "access": AccessLevel.PROTECTED.value,
            "decorators": ["decorator1"],
            "is_static": False,
            "is_class_method": False,
            "is_property": False,
            "parent_class": "TestClass",
            "line_range": [30, 40],
            "content": "def test_method(self):\n    pass"
        }
        
        self.assertEqual(method["type"], "method")
        self.assertEqual(method["name"], "test_method")
        self.assertEqual(method["qualified_name"], "module.TestClass.test_method")
        self.assertEqual(method["docstring"], "Test method docstring")
        self.assertEqual(method["parameters"], ["self", "param1"])
        self.assertEqual(method["returns"], "str")
        self.assertEqual(method["is_async"], True)
        self.assertEqual(method["access"], AccessLevel.PROTECTED.value)
        self.assertEqual(method["decorators"], ["decorator1"])
        self.assertEqual(method["is_static"], False)
        self.assertEqual(method["is_class_method"], False)
        self.assertEqual(method["is_property"], False)
        self.assertEqual(method["parent_class"], "TestClass")
        self.assertEqual(method["line_range"], [30, 40])
        self.assertEqual(method["content"], "def test_method(self):\n    pass")


class TestClassElement(unittest.TestCase):
    """Test cases for the ClassElement TypedDict."""
    
    def test_class_element_creation(self) -> None:
        """Test creating a ClassElement."""
        class_element: ClassElement = {
            "type": "class",
            "name": "TestClass",
            "qualified_name": "module.TestClass",
            "docstring": "Test class docstring",
            "base_classes": ["BaseClass"],
            "access": AccessLevel.PUBLIC.value,
            "decorators": ["decorator1"],
            "line_range": [50, 100],
            "content": "class TestClass:\n    pass",
            "elements": []
        }
        
        self.assertEqual(class_element["type"], "class")
        self.assertEqual(class_element["name"], "TestClass")
        self.assertEqual(class_element["qualified_name"], "module.TestClass")
        self.assertEqual(class_element["docstring"], "Test class docstring")
        self.assertEqual(class_element["base_classes"], ["BaseClass"])
        self.assertEqual(class_element["access"], AccessLevel.PUBLIC.value)
        self.assertEqual(class_element["decorators"], ["decorator1"])
        self.assertEqual(class_element["line_range"], [50, 100])
        self.assertEqual(class_element["content"], "class TestClass:\n    pass")
        self.assertEqual(class_element["elements"], [])
    
    def test_class_element_with_methods(self) -> None:
        """Test creating a ClassElement with method elements."""
        method: MethodElement = {
            "type": "method",
            "name": "test_method",
            "parent_class": "TestClass"
        }
        
        class_element: ClassElement = {
            "type": "class",
            "name": "TestClass",
            "elements": [method]
        }
        
        self.assertEqual(class_element["type"], "class")
        self.assertEqual(class_element["name"], "TestClass")
        self.assertEqual(len(class_element["elements"]), 1)
        self.assertEqual(class_element["elements"][0]["type"], "method")
        self.assertEqual(class_element["elements"][0]["name"], "test_method")
        self.assertEqual(class_element["elements"][0]["parent_class"], "TestClass")


class TestModuleElement(unittest.TestCase):
    """Test cases for the ModuleElement TypedDict."""
    
    def test_module_element_creation(self) -> None:
        """Test creating a ModuleElement."""
        module: ModuleElement = {
            "type": "module",
            "name": "test_module",
            "path": "/path/to/test_module.py",
            "docstring": "Test module docstring",
            "module_path": "package.test_module",
            "line_range": [1, 200],
            "elements": []
        }
        
        self.assertEqual(module["type"], "module")
        self.assertEqual(module["name"], "test_module")
        self.assertEqual(module["path"], "/path/to/test_module.py")
        self.assertEqual(module["docstring"], "Test module docstring")
        self.assertEqual(module["module_path"], "package.test_module")
        self.assertEqual(module["line_range"], [1, 200])
        self.assertEqual(module["elements"], [])
    
    def test_module_element_with_child_elements(self) -> None:
        """Test creating a ModuleElement with child elements."""
        import_element: ImportElement = {
            "type": "import",
            "name": "sys",
            "source": ImportSourceType.STDLIB.value
        }
        
        function: FunctionElement = {
            "type": "function",
            "name": "test_function"
        }
        
        module: ModuleElement = {
            "type": "module",
            "name": "test_module",
            "elements": [import_element, function]
        }
        
        self.assertEqual(module["type"], "module")
        self.assertEqual(module["name"], "test_module")
        self.assertEqual(len(module["elements"]), 2)
        self.assertEqual(module["elements"][0]["type"], "import")
        self.assertEqual(module["elements"][0]["name"], "sys")
        self.assertEqual(module["elements"][1]["type"], "function")
        self.assertEqual(module["elements"][1]["name"], "test_function")


class TestPythonDocument(unittest.TestCase):
    """Test cases for the PythonDocument TypedDict."""
    
    def test_python_document_creation(self) -> None:
        """Test creating a PythonDocument."""
        document: PythonDocument = {
            "id": "test-doc-123",
            "source": "/path/to/test_module.py",
            "content": "Processed content",
            "content_type": "markdown",
            "format": "python",
            "raw_content": "# Original source code"
        }
        
        self.assertEqual(document["id"], "test-doc-123")
        self.assertEqual(document["source"], "/path/to/test_module.py")
        self.assertEqual(document["content"], "Processed content")
        self.assertEqual(document["content_type"], "markdown")
        self.assertEqual(document["format"], "python")
        self.assertEqual(document["raw_content"], "# Original source code")
    
    def test_python_document_with_metadata_and_symbol_table(self) -> None:
        """Test creating a PythonDocument with metadata and symbol table."""
        document: PythonDocument = {
            "id": "test-doc-123",
            "source": "/path/to/test_module.py",
            "format": "python",
            "metadata": {
                "function_count": 1,
                "class_count": 1,
                "import_count": 2
            },
            "symbol_table": {
                "type": "module",
                "name": "test_module",
                "elements": []
            }
        }
        
        self.assertEqual(document["id"], "test-doc-123")
        self.assertEqual(document["source"], "/path/to/test_module.py")
        self.assertEqual(document["format"], "python")
        self.assertEqual(document["metadata"]["function_count"], 1)
        self.assertEqual(document["metadata"]["class_count"], 1)
        self.assertEqual(document["metadata"]["import_count"], 2)
        self.assertEqual(document["symbol_table"]["type"], "module")
        self.assertEqual(document["symbol_table"]["name"], "test_module")
        self.assertEqual(document["symbol_table"]["elements"], [])


if __name__ == "__main__":
    unittest.main()
