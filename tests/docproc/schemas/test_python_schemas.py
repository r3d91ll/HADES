"""
Unit tests for the Python document schemas.

This module contains tests for the Python-specific document schemas 
used to validate Python code documents.
"""

import unittest
from typing import Dict, Any, List
from pydantic import ValidationError

from src.docproc.schemas.python_document import (
    PythonMetadata, 
    PythonEntity, 
    CodeRelationship, 
    CodeElement,
    FunctionElement,
    MethodElement,
    ClassElement,
    ImportElement,
    SymbolTable,
    PythonDocument
)
from src.docproc.models.python_code import RelationshipType, AccessLevel


class TestPythonMetadata(unittest.TestCase):
    """Test the PythonMetadata model."""
    
    def test_valid_metadata(self):
        """Test that valid Python metadata passes validation."""
        metadata = PythonMetadata(
            language="python",
            format="python",
            content_type="code",
            function_count=5,
            class_count=2,
            import_count=10,
            method_count=15,
            has_module_docstring=True,
            has_syntax_errors=False
        )
        
        self.assertEqual(metadata.language, "python")
        self.assertEqual(metadata.format, "python")
        self.assertEqual(metadata.content_type, "code")
        self.assertEqual(metadata.function_count, 5)
        self.assertEqual(metadata.class_count, 2)
        self.assertEqual(metadata.import_count, 10)
        self.assertEqual(metadata.method_count, 15)
        self.assertTrue(metadata.has_module_docstring)
        self.assertFalse(metadata.has_syntax_errors)
    
    def test_minimal_metadata(self):
        """Test that metadata with only required fields passes validation."""
        metadata = PythonMetadata(
            language="python",
            format="python",
            content_type="code"
        )
        
        self.assertEqual(metadata.language, "python")
        self.assertEqual(metadata.format, "python")
        self.assertEqual(metadata.content_type, "code")
        self.assertEqual(metadata.function_count, 0)  # Default value
        self.assertEqual(metadata.class_count, 0)  # Default value
        self.assertEqual(metadata.import_count, 0)  # Default value
        self.assertEqual(metadata.method_count, 0)  # Default value
        self.assertFalse(metadata.has_module_docstring)  # Default value
        self.assertFalse(metadata.has_syntax_errors)  # Default value


class TestPythonEntity(unittest.TestCase):
    """Test the PythonEntity model."""
    
    def test_valid_entities(self):
        """Test that valid Python entities pass validation."""
        entity_types = ["module", "class", "function", "method", "import", "decorator"]
        
        for entity_type in entity_types:
            entity = PythonEntity(
                type=entity_type,
                value=f"test_{entity_type}",
                line=10,
                confidence=0.9
            )
            
            self.assertEqual(entity.type, entity_type)
            self.assertEqual(entity.value, f"test_{entity_type}")
    
    def test_invalid_entity_type(self):
        """Test that an entity with invalid type fails validation."""
        with self.assertRaises(ValidationError):
            PythonEntity(
                type="invalid_type",  # Not in the allowed types
                value="test_value"
            )


class TestCodeRelationship(unittest.TestCase):
    """Test the CodeRelationship model."""
    
    def test_valid_relationships(self):
        """Test that valid code relationships pass validation."""
        # Test all relationship types
        for rel_type in RelationshipType:
            relationship = CodeRelationship(
                source="source_element",
                target="target_element",
                type=rel_type.value,
                weight=0.8,
                line=42
            )
            
            self.assertEqual(relationship.source, "source_element")
            self.assertEqual(relationship.target, "target_element")
            self.assertEqual(relationship.type, rel_type.value)
            self.assertEqual(relationship.weight, 0.8)
            self.assertEqual(relationship.line, 42)
    
    def test_invalid_relationship_type(self):
        """Test that a relationship with invalid type fails validation."""
        with self.assertRaises(ValidationError):
            CodeRelationship(
                source="source_element",
                target="target_element",
                type="INVALID_TYPE",  # Not a valid relationship type
                weight=0.8
            )
    
    def test_invalid_weight(self):
        """Test that a relationship with invalid weight fails validation."""
        with self.assertRaises(ValidationError):
            CodeRelationship(
                source="source_element",
                target="target_element",
                type=RelationshipType.CALLS.value,
                weight=1.5  # Greater than 1.0
            )
        
        with self.assertRaises(ValidationError):
            CodeRelationship(
                source="source_element",
                target="target_element",
                type=RelationshipType.CALLS.value,
                weight=-0.5  # Less than 0.0
            )


class TestCodeElements(unittest.TestCase):
    """Test the various code element models."""
    
    def test_base_code_element(self):
        """Test the base CodeElement model."""
        element = CodeElement(
            type="generic",
            name="test_element",
            qualified_name="module.test_element",
            docstring="Test docstring",
            line_range=[10, 20],
            content="def test_element():\n    pass"
        )
        
        self.assertEqual(element.type, "generic")
        self.assertEqual(element.name, "test_element")
        self.assertEqual(element.qualified_name, "module.test_element")
        self.assertEqual(element.docstring, "Test docstring")
        self.assertEqual(element.line_range, [10, 20])
        self.assertEqual(element.content, "def test_element():\n    pass")
        
        # Test with minimal fields
        minimal_element = CodeElement(
            type="generic",
            name="test_element"
        )
        
        self.assertEqual(minimal_element.type, "generic")
        self.assertEqual(minimal_element.name, "test_element")
        self.assertIsNone(minimal_element.qualified_name)
        self.assertIsNone(minimal_element.docstring)
        self.assertIsNone(minimal_element.line_range)
        self.assertIsNone(minimal_element.content)
    
    def test_function_element(self):
        """Test the FunctionElement model."""
        function = FunctionElement(
            name="test_function",
            qualified_name="module.test_function",
            docstring="Test function docstring",
            parameters=["arg1", "arg2"],
            returns="int",
            is_async=True,
            access=AccessLevel.PRIVATE.value,
            decorators=["decorator1", "decorator2"],
            line_range=[10, 20],
            content="async def test_function(arg1, arg2) -> int:\n    return 42"
        )
        
        self.assertEqual(function.type, "function")  # Fixed type
        self.assertEqual(function.name, "test_function")
        self.assertEqual(function.qualified_name, "module.test_function")
        self.assertEqual(function.docstring, "Test function docstring")
        self.assertEqual(function.parameters, ["arg1", "arg2"])
        self.assertEqual(function.returns, "int")
        self.assertTrue(function.is_async)
        self.assertEqual(function.access, AccessLevel.PRIVATE.value)
        self.assertEqual(function.decorators, ["decorator1", "decorator2"])
        
        # Test with minimal fields
        minimal_function = FunctionElement(
            name="minimal_function"
        )
        
        self.assertEqual(minimal_function.type, "function")  # Fixed type
        self.assertEqual(minimal_function.name, "minimal_function")
        self.assertEqual(minimal_function.parameters, [])  # Default empty list
        self.assertFalse(minimal_function.is_async)  # Default False
        self.assertEqual(minimal_function.access, AccessLevel.PUBLIC.value)  # Default PUBLIC
        self.assertEqual(minimal_function.decorators, [])  # Default empty list
    
    def test_method_element(self):
        """Test the MethodElement model."""
        method = MethodElement(
            name="test_method",
            qualified_name="TestClass.test_method",
            docstring="Test method docstring",
            parameters=["self", "arg1", "arg2"],
            returns="str",
            is_async=True,
            access=AccessLevel.PROTECTED.value,
            decorators=["staticmethod", "custom_decorator"],
            is_static=True,
            is_class_method=False,
            is_property=False,
            parent_class="TestClass",
            line_range=[30, 40],
            content="@staticmethod\n@custom_decorator\nasync def test_method(arg1, arg2) -> str:\n    return 'test'"
        )
        
        self.assertEqual(method.type, "method")  # Fixed type
        self.assertEqual(method.name, "test_method")
        self.assertEqual(method.qualified_name, "TestClass.test_method")
        self.assertEqual(method.docstring, "Test method docstring")
        self.assertEqual(method.parameters, ["self", "arg1", "arg2"])
        self.assertEqual(method.returns, "str")
        self.assertTrue(method.is_async)
        self.assertEqual(method.access, AccessLevel.PROTECTED.value)
        self.assertEqual(method.decorators, ["staticmethod", "custom_decorator"])
        self.assertTrue(method.is_static)
        self.assertFalse(method.is_class_method)
        self.assertFalse(method.is_property)
        self.assertEqual(method.parent_class, "TestClass")
        
        # Test missing required fields
        with self.assertRaises(ValidationError):
            MethodElement(
                name="invalid_method"
                # Missing parent_class (required)
            )
    
    def test_class_element(self):
        """Test the ClassElement model."""
        class_element = ClassElement(
            name="TestClass",
            qualified_name="module.TestClass",
            docstring="Test class docstring",
            base_classes=["BaseClass", "Interface"],
            access=AccessLevel.PUBLIC.value,
            decorators=["dataclass", "custom_decorator"],
            line_range=[50, 100],
            content="@dataclass\n@custom_decorator\nclass TestClass(BaseClass, Interface):\n    ...",
            elements=[
                {"type": "method", "name": "test_method", "parent_class": "TestClass"}
            ]
        )
        
        self.assertEqual(class_element.type, "class")  # Fixed type
        self.assertEqual(class_element.name, "TestClass")
        self.assertEqual(class_element.qualified_name, "module.TestClass")
        self.assertEqual(class_element.docstring, "Test class docstring")
        self.assertEqual(class_element.base_classes, ["BaseClass", "Interface"])
        self.assertEqual(class_element.access, AccessLevel.PUBLIC.value)
        self.assertEqual(class_element.decorators, ["dataclass", "custom_decorator"])
        self.assertEqual(len(class_element.elements), 1)
        self.assertEqual(class_element.elements[0]["type"], "method")
        self.assertEqual(class_element.elements[0]["name"], "test_method")
        
        # Test with minimal fields
        minimal_class = ClassElement(
            name="MinimalClass"
        )
        
        self.assertEqual(minimal_class.type, "class")  # Fixed type
        self.assertEqual(minimal_class.name, "MinimalClass")
        self.assertEqual(minimal_class.base_classes, [])  # Default empty list
        self.assertEqual(minimal_class.access, AccessLevel.PUBLIC.value)  # Default PUBLIC
        self.assertEqual(minimal_class.decorators, [])  # Default empty list
        self.assertEqual(minimal_class.elements, [])  # Default empty list
    
    def test_import_element(self):
        """Test the ImportElement model."""
        import_element = ImportElement(
            name="module.submodule",
            alias="submodule",
            source="THIRD_PARTY",
            line_range=[5, 5],
            content="import module.submodule as submodule"
        )
        
        self.assertEqual(import_element.type, "import")  # Fixed type
        self.assertEqual(import_element.name, "module.submodule")
        self.assertEqual(import_element.alias, "submodule")
        self.assertEqual(import_element.source, "THIRD_PARTY")
        
        # Test missing required fields
        with self.assertRaises(ValidationError):
            ImportElement(
                name="module.submodule",
                # Missing source (required)
            )


class TestSymbolTable(unittest.TestCase):
    """Test the SymbolTable model."""
    
    def test_valid_symbol_table(self):
        """Test that a valid symbol table passes validation."""
        symbol_table = SymbolTable(
            name="test_module",
            docstring="Test module docstring",
            path="/path/to/test_module.py",
            module_path="package.test_module",
            line_range=[1, 100],
            elements=[
                {
                    "type": "class",
                    "name": "TestClass",
                    "elements": [
                        {"type": "method", "name": "test_method", "parent_class": "TestClass"}
                    ]
                },
                {
                    "type": "function",
                    "name": "test_function"
                }
            ]
        )
        
        self.assertEqual(symbol_table.type, "module")  # Fixed type
        self.assertEqual(symbol_table.name, "test_module")
        self.assertEqual(symbol_table.docstring, "Test module docstring")
        self.assertEqual(symbol_table.path, "/path/to/test_module.py")
        self.assertEqual(symbol_table.module_path, "package.test_module")
        self.assertEqual(symbol_table.line_range, [1, 100])
        self.assertEqual(len(symbol_table.elements), 2)
        self.assertEqual(symbol_table.elements[0]["type"], "class")
        self.assertEqual(symbol_table.elements[0]["name"], "TestClass")
        self.assertEqual(symbol_table.elements[1]["type"], "function")
        self.assertEqual(symbol_table.elements[1]["name"], "test_function")
    
    def test_minimal_symbol_table(self):
        """Test that a symbol table with only required fields passes validation."""
        symbol_table = SymbolTable(
            name="minimal_module"
        )
        
        self.assertEqual(symbol_table.type, "module")  # Fixed type
        self.assertEqual(symbol_table.name, "minimal_module")
        self.assertIsNone(symbol_table.docstring)
        self.assertIsNone(symbol_table.path)
        self.assertIsNone(symbol_table.module_path)
        self.assertIsNone(symbol_table.line_range)
        self.assertEqual(symbol_table.elements, [])  # Default empty list


class TestPythonDocument(unittest.TestCase):
    """Test the PythonDocument model."""
    
    def setUp(self):
        """Set up test data."""
        self.valid_metadata = PythonMetadata(
            language="python",
            format="python",
            content_type="code",
            function_count=2,
            class_count=1,
            import_count=3
        )
        
        self.valid_entity = PythonEntity(
            type="function",
            value="test_function",
            line=10
        )
        
        self.valid_relationship = CodeRelationship(
            source="TestClass",
            target="BaseClass",
            type=RelationshipType.EXTENDS.value,
            weight=0.9,
            line=5
        )
        
        self.valid_symbol_table = SymbolTable(
            name="test_module",
            docstring="Test module docstring",
            path="/path/to/test_module.py",
            elements=[
                {
                    "type": "class",
                    "name": "TestClass",
                    "base_classes": ["BaseClass"],
                    "elements": [
                        {"type": "method", "name": "test_method", "parent_class": "TestClass"}
                    ]
                },
                {
                    "type": "function",
                    "name": "test_function"
                }
            ]
        )
        
        self.valid_document_data = {
            "id": "python_12345",
            "source": "/path/to/test_module.py",
            "content": "```python\nclass TestClass(BaseClass):\n    def test_method(self):\n        pass\n\ndef test_function():\n    pass\n```",
            "content_type": "markdown",
            "format": "python",
            "raw_content": "class TestClass(BaseClass):\n    def test_method(self):\n        pass\n\ndef test_function():\n    pass",
            "metadata": self.valid_metadata.model_dump(),
            "entities": [self.valid_entity.model_dump()],
            "relationships": [self.valid_relationship.model_dump()],
            "symbol_table": self.valid_symbol_table.model_dump()
        }
    
    def test_valid_document(self):
        """Test that a valid Python document passes validation."""
        document = PythonDocument(**self.valid_document_data)
        
        self.assertEqual(document.id, "python_12345")
        self.assertEqual(document.source, "/path/to/test_module.py")
        self.assertEqual(document.format, "python")
        self.assertEqual(document.metadata.function_count, 2)
        self.assertEqual(document.metadata.class_count, 1)
        self.assertEqual(len(document.entities), 1)
        self.assertEqual(document.entities[0].type, "function")
        self.assertEqual(len(document.relationships), 1)
        self.assertEqual(document.relationships[0].type, RelationshipType.EXTENDS.value)
        self.assertEqual(document.symbol_table.name, "test_module")
    
    def test_document_without_optional_fields(self):
        """Test that a document without optional fields passes validation."""
        # Remove optional fields
        data_without_optional = self.valid_document_data.copy()
        del data_without_optional["relationships"]
        del data_without_optional["symbol_table"]
        
        document = PythonDocument(**data_without_optional)
        
        self.assertEqual(document.id, "python_12345")
        self.assertEqual(document.format, "python")
        self.assertIsNone(document.relationships)
        self.assertIsNone(document.symbol_table)
    
    def test_document_with_syntax_errors(self):
        """Test that a document with syntax errors passes validation without symbol table."""
        # Document with syntax errors but no symbol table
        data_with_errors = self.valid_document_data.copy()
        data_with_errors["metadata"]["has_syntax_errors"] = True
        del data_with_errors["symbol_table"]
        
        document = PythonDocument(**data_with_errors)
        
        self.assertEqual(document.id, "python_12345")
        self.assertEqual(document.format, "python")
        self.assertTrue(document.metadata.has_syntax_errors)
        self.assertIsNone(document.symbol_table)  # No error about missing symbol table
    
    def test_model_validation(self):
        """Test model validation with different inputs."""
        # Valid document from model_validate
        document = PythonDocument.model_validate(self.valid_document_data)
        self.assertEqual(document.id, "python_12345")
        
        # Invalid document (wrong format)
        invalid_data = self.valid_document_data.copy()
        invalid_data["format"] = "java"  # Not "python"
        
        with self.assertRaises(ValidationError):
            PythonDocument.model_validate(invalid_data)


if __name__ == "__main__":
    unittest.main()
