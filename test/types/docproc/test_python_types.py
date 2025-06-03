"""
Unit tests for the docproc Python type definitions.

This module tests the Pydantic models and TypedDict definitions that have been
migrated to centralized type files in the src/types/docproc directory.
"""

import os
import sys
import unittest
from typing import Dict, Any, List, Optional, cast
import json

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from pydantic import ValidationError

# Define mock base classes to avoid importing dependencies
class BaseMetadata:
    """Mock base class for metadata models."""
    pass

class BaseEntity:
    """Mock base class for entity models."""
    pass

class BaseDocument:
    """Mock base class for document models."""
    pass

# Import our actual centralized types
from src.types.docproc.python import (
    PythonMetadata,
    PythonEntity,
    CodeRelationship,
    CodeElement,
    FunctionElement,
    MethodElement,
    ClassElement,
    ImportElement,
    SymbolTable,
    PythonDocument,
    typed_field_validator,
    typed_model_validator
)
from src.types.docproc.enums import RelationshipType, AccessLevel


class TestPythonMetadata(unittest.TestCase):
    """Test cases for the PythonMetadata Pydantic model."""
    
    def test_init_default_values(self) -> None:
        """Test initializing PythonMetadata with default values."""
        metadata = PythonMetadata()
        self.assertEqual(metadata.function_count, 0)
        self.assertEqual(metadata.class_count, 0)
        self.assertEqual(metadata.import_count, 0)
        self.assertEqual(metadata.method_count, 0)
        self.assertFalse(metadata.has_module_docstring)
        self.assertFalse(metadata.has_syntax_errors)
        
    def test_init_custom_values(self) -> None:
        """Test initializing PythonMetadata with custom values."""
        metadata = PythonMetadata(
            function_count=5,
            class_count=2,
            import_count=10,
            method_count=8,
            has_module_docstring=True,
            has_syntax_errors=False
        )
        self.assertEqual(metadata.function_count, 5)
        self.assertEqual(metadata.class_count, 2)
        self.assertEqual(metadata.import_count, 10)
        self.assertEqual(metadata.method_count, 8)
        self.assertTrue(metadata.has_module_docstring)
        self.assertFalse(metadata.has_syntax_errors)
        
    def test_inheritance(self) -> None:
        """Test that PythonMetadata inherits from BaseMetadata."""
        metadata = PythonMetadata()
        self.assertIsInstance(metadata, BaseMetadata)


class TestPythonEntity(unittest.TestCase):
    """Test cases for the PythonEntity Pydantic model."""
    
    def test_init_with_valid_type(self) -> None:
        """Test initializing PythonEntity with valid type values."""
        valid_types = ["module", "class", "function", "method", "import", "decorator"]
        
        for entity_type in valid_types:
            entity = PythonEntity(type=entity_type, name="test", id="test-id")
            self.assertEqual(entity.type, entity_type)
            self.assertEqual(entity.name, "test")
            self.assertEqual(entity.id, "test-id")
            
    def test_inheritance(self) -> None:
        """Test that PythonEntity inherits from BaseEntity."""
        entity = PythonEntity(type="module", name="test", id="test-id")
        self.assertIsInstance(entity, BaseEntity)


class TestCodeRelationship(unittest.TestCase):
    """Test cases for the CodeRelationship Pydantic model."""
    
    def test_init_with_valid_data(self) -> None:
        """Test initializing CodeRelationship with valid data."""
        relationship = CodeRelationship(
            source="source_id",
            target="target_id",
            type=RelationshipType.CALLS.value,
            weight=0.8,
            line=42
        )
        self.assertEqual(relationship.source, "source_id")
        self.assertEqual(relationship.target, "target_id")
        self.assertEqual(relationship.type, RelationshipType.CALLS.value)
        self.assertEqual(relationship.weight, 0.8)
        self.assertEqual(relationship.line, 42)
        
    def test_relationship_type_validation(self) -> None:
        """Test validation of relationship type."""
        valid_types = [rel_type.value for rel_type in RelationshipType]
        
        # Test with valid types
        for valid_type in valid_types:
            relationship = CodeRelationship(
                source="source_id",
                target="target_id",
                type=valid_type,
                weight=0.8,
                line=42
            )
            self.assertEqual(relationship.type, valid_type)
        
        # Test with invalid type (should raise ValidationError)
        with self.assertRaises(ValidationError):
            CodeRelationship(
                source="source_id",
                target="target_id",
                type="INVALID_TYPE",
                weight=0.8,
                line=42
            )
            
    def test_weight_validation(self) -> None:
        """Test validation of relationship weight."""
        # Test with valid weights
        valid_weights = [0.0, 0.1, 0.5, 0.99, 1.0]
        for weight in valid_weights:
            relationship = CodeRelationship(
                source="source_id",
                target="target_id",
                type=RelationshipType.CALLS.value,
                weight=weight,
                line=42
            )
            self.assertEqual(relationship.weight, weight)
        
        # Test with invalid weights (should raise ValidationError)
        invalid_weights = [-0.1, 1.1, 2.0]
        for weight in invalid_weights:
            with self.assertRaises(ValidationError):
                CodeRelationship(
                    source="source_id",
                    target="target_id",
                    type=RelationshipType.CALLS.value,
                    weight=weight,
                    line=42
                )


class TestCodeElements(unittest.TestCase):
    """Test cases for the code element Pydantic models."""
    
    def test_function_element(self) -> None:
        """Test the FunctionElement Pydantic model."""
        function = FunctionElement(
            name="test_function",
            qualified_name="module.test_function",
            docstring="Test function docstring",
            parameters=["param1", "param2"],
            returns="str",
            is_async=False,
            access=AccessLevel.PUBLIC.value,
            decorators=["decorator1"],
            line_range=[10, 20],
            content="def test_function():\n    pass"
        )
        
        self.assertEqual(function.type, "function")
        self.assertEqual(function.name, "test_function")
        self.assertEqual(function.qualified_name, "module.test_function")
        self.assertEqual(function.docstring, "Test function docstring")
        self.assertEqual(function.parameters, ["param1", "param2"])
        self.assertEqual(function.returns, "str")
        self.assertFalse(function.is_async)
        self.assertEqual(function.access, AccessLevel.PUBLIC.value)
        self.assertEqual(function.decorators, ["decorator1"])
        self.assertEqual(function.line_range, [10, 20])
        self.assertEqual(function.content, "def test_function():\n    pass")
        
    def test_method_element(self) -> None:
        """Test the MethodElement Pydantic model."""
        method = MethodElement(
            name="test_method",
            qualified_name="module.TestClass.test_method",
            docstring="Test method docstring",
            parameters=["self", "param1"],
            returns="str",
            is_async=True,
            access=AccessLevel.PROTECTED.value,
            decorators=["decorator1"],
            is_static=False,
            is_class_method=False,
            is_property=False,
            parent_class="TestClass",
            line_range=[30, 40],
            content="def test_method(self):\n    pass"
        )
        
        self.assertEqual(method.type, "method")
        self.assertEqual(method.name, "test_method")
        self.assertEqual(method.qualified_name, "module.TestClass.test_method")
        self.assertEqual(method.docstring, "Test method docstring")
        self.assertEqual(method.parameters, ["self", "param1"])
        self.assertEqual(method.returns, "str")
        self.assertTrue(method.is_async)
        self.assertEqual(method.access, AccessLevel.PROTECTED.value)
        self.assertEqual(method.decorators, ["decorator1"])
        self.assertFalse(method.is_static)
        self.assertFalse(method.is_class_method)
        self.assertFalse(method.is_property)
        self.assertEqual(method.parent_class, "TestClass")
        self.assertEqual(method.line_range, [30, 40])
        self.assertEqual(method.content, "def test_method(self):\n    pass")
        
    def test_class_element(self) -> None:
        """Test the ClassElement Pydantic model."""
        class_element = ClassElement(
            name="TestClass",
            qualified_name="module.TestClass",
            docstring="Test class docstring",
            base_classes=["BaseClass"],
            access=AccessLevel.PUBLIC.value,
            decorators=["decorator1"],
            line_range=[50, 100],
            content="class TestClass:\n    pass",
            elements=[]
        )
        
        self.assertEqual(class_element.type, "class")
        self.assertEqual(class_element.name, "TestClass")
        self.assertEqual(class_element.qualified_name, "module.TestClass")
        self.assertEqual(class_element.docstring, "Test class docstring")
        self.assertEqual(class_element.base_classes, ["BaseClass"])
        self.assertEqual(class_element.access, AccessLevel.PUBLIC.value)
        self.assertEqual(class_element.decorators, ["decorator1"])
        self.assertEqual(class_element.line_range, [50, 100])
        self.assertEqual(class_element.content, "class TestClass:\n    pass")
        self.assertEqual(class_element.elements, [])
        
    def test_import_element(self) -> None:
        """Test the ImportElement Pydantic model."""
        import_element = ImportElement(
            name="module",
            alias="m",
            source="third-party",
            line_range=[1, 1],
            content="import module as m"
        )
        
        self.assertEqual(import_element.type, "import")
        self.assertEqual(import_element.name, "module")
        self.assertEqual(import_element.alias, "m")
        self.assertEqual(import_element.source, "third-party")
        self.assertEqual(import_element.line_range, [1, 1])
        self.assertEqual(import_element.content, "import module as m")


class TestSymbolTable(unittest.TestCase):
    """Test cases for the SymbolTable Pydantic model."""
    
    def test_init_with_valid_data(self) -> None:
        """Test initializing SymbolTable with valid data."""
        symbol_table = SymbolTable(
            name="test_module",
            docstring="Test module docstring",
            path="/path/to/test_module.py",
            module_path="package.test_module",
            line_range=[1, 100],
            elements=[]
        )
        
        self.assertEqual(symbol_table.type, "module")
        self.assertEqual(symbol_table.name, "test_module")
        self.assertEqual(symbol_table.docstring, "Test module docstring")
        self.assertEqual(symbol_table.path, "/path/to/test_module.py")
        self.assertEqual(symbol_table.module_path, "package.test_module")
        self.assertEqual(symbol_table.line_range, [1, 100])
        self.assertEqual(symbol_table.elements, [])
        
    def test_with_nested_elements(self) -> None:
        """Test SymbolTable with nested elements."""
        symbol_table = SymbolTable(
            name="test_module",
            path="/path/to/test_module.py",
            elements=[
                FunctionElement(
                    name="test_function",
                    parameters=[],
                    returns=None,
                    is_async=False,
                    access=AccessLevel.PUBLIC.value,
                    decorators=[],
                    line_range=[10, 20],
                    content="def test_function():\n    pass"
                ).dict(),
                ClassElement(
                    name="TestClass",
                    base_classes=[],
                    access=AccessLevel.PUBLIC.value,
                    decorators=[],
                    line_range=[30, 50],
                    content="class TestClass:\n    pass",
                    elements=[]
                ).dict()
            ]
        )
        
        self.assertEqual(len(symbol_table.elements), 2)
        self.assertEqual(symbol_table.elements[0]["name"], "test_function")
        self.assertEqual(symbol_table.elements[0]["type"], "function")
        self.assertEqual(symbol_table.elements[1]["name"], "TestClass")
        self.assertEqual(symbol_table.elements[1]["type"], "class")


class TestPythonDocument(unittest.TestCase):
    """Test cases for the PythonDocument Pydantic model."""
    
    def test_init_with_valid_data(self) -> None:
        """Test initializing PythonDocument with valid data."""
        document = PythonDocument(
            id="test-doc-123",
            source="/path/to/test_module.py",
            content="Processed content",
            content_type="markdown",
            raw_content="# Original source code",
            metadata=PythonMetadata(
                function_count=1,
                class_count=1,
                import_count=2,
                method_count=0,
                has_module_docstring=True,
                has_syntax_errors=False,
                language="python",
                format="python",
                content_type="code",
                file_size=1000,
                line_count=50,
                char_count=1000
            ),
            entities=[
                PythonEntity(
                    id="entity-1",
                    name="test_function",
                    type="function",
                    start_pos=10,
                    end_pos=20,
                    text="def test_function():\n    pass"
                )
            ],
            relationships=[
                CodeRelationship(
                    source="entity-1",
                    target="entity-2",
                    type=RelationshipType.CALLS.value,
                    weight=0.9,
                    line=15
                )
            ],
            symbol_table=SymbolTable(
                name="test_module",
                path="/path/to/test_module.py",
                module_path="package.test_module",
                elements=[
                    FunctionElement(
                        name="test_function",
                        parameters=[],
                        returns=None,
                        is_async=False,
                        access=AccessLevel.PUBLIC.value,
                        decorators=[],
                        line_range=[10, 20],
                        content="def test_function():\n    pass"
                    ).dict()
                ]
            )
        )
        
        self.assertEqual(document.format, "python")
        self.assertEqual(document.id, "test-doc-123")
        self.assertEqual(document.source, "/path/to/test_module.py")
        self.assertEqual(document.metadata.function_count, 1)
        self.assertEqual(document.metadata.class_count, 1)
        self.assertEqual(len(document.entities), 1)
        self.assertEqual(document.entities[0].name, "test_function")
        self.assertEqual(len(document.relationships), 1)
        self.assertEqual(document.relationships[0].type, RelationshipType.CALLS.value)
        self.assertEqual(document.symbol_table.name, "test_module")
        self.assertEqual(len(document.symbol_table.elements), 1)
        
    def test_inheritance(self) -> None:
        """Test that PythonDocument inherits from BaseDocument."""
        document = PythonDocument(
            id="test-doc-123",
            source="/path/to/test_module.py",
            content="Processed content",
            content_type="markdown",
            metadata=PythonMetadata()
        )
        self.assertIsInstance(document, BaseDocument)
        
    def test_validate_code_consistency(self) -> None:
        """Test code consistency validation."""
        # Test with consistent metadata
        document = PythonDocument(
            id="test-doc-123",
            source="/path/to/test_module.py",
            content="Processed content",
            content_type="markdown",
            metadata=PythonMetadata(
                function_count=1,
                class_count=1,
                import_count=1,
                method_count=0
            ),
            entities=[
                PythonEntity(id="entity-1", name="func1", type="function"),
                PythonEntity(id="entity-2", name="Class1", type="class"),
                PythonEntity(id="entity-3", name="import1", type="import")
            ],
            symbol_table=SymbolTable(
                name="test_module",
                path="/path/to/test_module.py",
                elements=[]
            )
        )
        # This shouldn't raise any validation errors
        validated_doc = document.validate_code_consistency()
        self.assertEqual(validated_doc, document)
        
        # Test with syntax errors
        document_with_errors = PythonDocument(
            id="test-doc-123",
            source="/path/to/test_module.py",
            content="Processed content",
            content_type="markdown",
            metadata=PythonMetadata(
                has_syntax_errors=True
            )
        )
        # This shouldn't raise any validation errors even without symbol_table
        validated_doc = document_with_errors.validate_code_consistency()
        self.assertEqual(validated_doc, document_with_errors)


class TestTypedValidators(unittest.TestCase):
    """Test cases for the typed validator wrapper functions."""
    
    def test_typed_field_validator(self) -> None:
        """Test typed_field_validator function."""
        # Since we can't directly test a decorator, we'll check it compiles
        # and returns a callable that can be used as a decorator
        validator_decorator = typed_field_validator("field_name")
        self.assertTrue(callable(validator_decorator))
        
    def test_typed_model_validator(self) -> None:
        """Test typed_model_validator function."""
        # Since we can't directly test a decorator, we'll check it compiles
        # and returns a callable that can be used as a decorator
        validator_decorator = typed_model_validator(mode="after")
        self.assertTrue(callable(validator_decorator))


if __name__ == "__main__":
    unittest.main()
