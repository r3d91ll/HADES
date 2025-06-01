"""
Tests for the python_adapter module.

This module contains improved tests for the PythonAdapter class, focusing on 
previously untested methods to improve test coverage beyond the 85% standard.
"""

import unittest
from unittest import mock
import ast
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, cast

# Mock dependencies to avoid import errors
import sys
sys.modules['src.utils'] = mock.MagicMock()
sys.modules['src.utils.file_utils'] = mock.MagicMock()
sys.modules['src.types'] = mock.MagicMock()
sys.modules['src.types.docproc'] = mock.MagicMock()
sys.modules['src.types.docproc.adapter'] = mock.MagicMock()
sys.modules['src.types.docproc.metadata'] = mock.MagicMock()
sys.modules['src.docproc.adapters.registry'] = mock.MagicMock()

# Define minimal required classes for testing
class ProcessedDocument(dict):
    pass

class EntityDict(dict):
    pass

class MetadataDict(dict):
    pass

class AdapterOptions(dict):
    pass

# Add these to the mocked modules
sys.modules['src.types.docproc.adapter'].ProcessedDocument = ProcessedDocument
sys.modules['src.types.docproc.adapter'].AdapterOptions = AdapterOptions
sys.modules['src.types.docproc.metadata'].EntityDict = EntityDict
sys.modules['src.types.docproc.metadata'].MetadataDict = MetadataDict

# Import the PythonAdapter class
from src.docproc.adapters.python_adapter import (
    PythonAdapter, EntityExtractor, CallFinder, PythonMetadataDict,
    FunctionInfo, ClassInfo, ImportInfo, RelationshipInfo
)

class TestPythonAdapter(unittest.TestCase):
    """Test cases for PythonAdapter."""

    def setUp(self):
        """Set up test fixtures."""
        self.adapter = PythonAdapter()
        
        # Sample Python code for testing
        self.sample_code = '''"""
Module docstring for test code.
"""

import os
import sys
from typing import List, Dict

def hello(name: str) -> str:
    """Say hello to someone.
    
    Args:
        name: The name to greet
        
    Returns:
        A greeting message
    """
    return f"Hello, {name}!"

class Person:
    """A simple person class."""
    
    def __init__(self, name: str, age: int):
        """Initialize a person.
        
        Args:
            name: The person's name
            age: The person's age
        """
        self.name = name
        self.age = age
        
    def greet(self) -> str:
        """Generate a greeting.
        
        Returns:
            A greeting message
        """
        return f"Hi, I'm {self.name} and I'm {self.age} years old."

def call_hello():
    """Function that calls another function."""
    person = Person("Alice", 30)
    return hello(person.name)
'''

    def test_init(self):
        """Test initialization of PythonAdapter."""
        # Test with default options
        adapter = PythonAdapter()
        self.assertEqual(False, adapter.create_symbol_table)
        self.assertEqual({}, adapter.options)
        
        # Test with custom options
        custom_options = {"create_symbol_table": True, "analyze_imports": True}
        adapter = PythonAdapter(create_symbol_table=True, options=custom_options)
        self.assertEqual(True, adapter.create_symbol_table)
        self.assertEqual(custom_options, adapter.options)

    def test_process_text(self):
        """Test process_text method."""
        result = self.adapter.process_text(self.sample_code)
        
        # Check basic structure
        self.assertIsInstance(result, dict)
        self.assertEqual("python", result["format"])
        self.assertEqual(self.sample_code, result["content"])
        
        # Check code analysis
        self.assertIn("code_analysis", result)
        code_analysis = result["code_analysis"]
        
        # Check module info
        self.assertIn("module", code_analysis)
        self.assertIn("docstring", code_analysis["module"])
        self.assertIn("Module docstring for test code", code_analysis["module"]["docstring"])
        
        # Check functions
        self.assertIn("functions", code_analysis)
        self.assertIn("hello", code_analysis["functions"])
        self.assertIn("call_hello", code_analysis["functions"])
        
        # Check classes
        self.assertIn("classes", code_analysis)
        self.assertIn("Person", code_analysis["classes"])
        
        # Check imports
        self.assertIn("imports", code_analysis["module"])
        imports = code_analysis["module"]["imports"]
        self.assertGreaterEqual(len(imports), 3)  # os, sys, typing
    
    def test_extract_entities(self):
        """Test extract_entities method."""
        entities = self.adapter.extract_entities(self.sample_code)
        
        # Check basic structure
        self.assertIsInstance(entities, list)
        self.assertGreater(len(entities), 0)
        
        # Check entity types
        entity_types = [e["type"] for e in entities]
        self.assertIn("function", entity_types)
        self.assertIn("class", entity_types)
        
        # Check specific entities
        function_entity = next((e for e in entities if e["type"] == "function" and e["name"] == "hello"), None)
        class_entity = next((e for e in entities if e["type"] == "class" and e["name"] == "Person"), None)
        
        self.assertIsNotNone(function_entity)
        self.assertIsNotNone(class_entity)
        self.assertIn("Say hello to someone", function_entity["content"])
        self.assertIn("A simple person class", class_entity["content"])
    
    def test_extract_metadata(self):
        """Test extract_metadata method."""
        metadata = self.adapter.extract_metadata(self.sample_code)
        
        # Check basic structure
        self.assertIsInstance(metadata, dict)
        self.assertEqual("python", metadata["language"])
        self.assertEqual("code", metadata["document_type"])
        
        # Check counts
        self.assertEqual("2", metadata["function_count"])
        self.assertEqual("1", metadata["class_count"])
        self.assertGreaterEqual(int(metadata["import_count"]), 3)
        
        # Check complexity metrics
        self.assertIn("cyclomatic_complexity", metadata)
        self.assertIn("average_function_length", metadata)
    
    def test_process_python_file_private(self):
        """Test _process_python_file private method."""
        # Parse the sample code to get an AST
        tree = ast.parse(self.sample_code)
        
        # Call the private method
        result = self.adapter._process_python_file(tree, self.sample_code)
        
        # Check basic structure
        self.assertIsInstance(result, dict)
        
        # Check module info
        self.assertIn("module", result)
        self.assertIn("docstring", result["module"])
        
        # Check functions
        self.assertIn("functions", result)
        self.assertGreaterEqual(len(result["functions"]), 2)
        
        # Check classes
        self.assertIn("classes", result)
        self.assertGreaterEqual(len(result["classes"]), 1)
    
    def test_extract_entities_private(self):
        """Test _extract_entities private method."""
        # Parse the sample code to get an AST
        tree = ast.parse(self.sample_code)
        
        # Process the file first to get analysis
        analysis = self.adapter._process_python_file(tree, self.sample_code)
        
        # Call the private method
        entities = self.adapter._extract_entities(analysis)
        
        # Check basic structure
        self.assertIsInstance(entities, list)
        self.assertGreater(len(entities), 0)
        
        # Check entity types
        entity_types = [e["type"] for e in entities]
        self.assertIn("function", entity_types)
        self.assertIn("class", entity_types)
    
    def test_build_entity_relationships(self):
        """Test _build_entity_relationships private method."""
        # Parse the sample code to get an AST
        tree = ast.parse(self.sample_code)
        
        # Process the file first to get analysis
        analysis = self.adapter._process_python_file(tree, self.sample_code)
        
        # Call the private method
        relationships = self.adapter._build_entity_relationships(analysis)
        
        # Check basic structure
        self.assertIsInstance(relationships, list)
        
        # Check for expected relationships
        # The call_hello function calls hello and creates a Person
        call_relationships = [r for r in relationships 
                             if r["source"] == "call_hello" and r["target"] == "hello"]
        self.assertGreaterEqual(len(call_relationships), 1)


class TestEntityExtractor(unittest.TestCase):
    """Test cases for EntityExtractor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = EntityExtractor()
        
        # Sample Python code for testing
        self.sample_code = '''
import os
import sys
from typing import List, Dict

def hello(name: str) -> str:
    return f"Hello, {name}!"

class Person:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age
        
    def greet(self):
        return f"Hi, I'm {self.name}!"
'''
        self.tree = ast.parse(self.sample_code)
    
    def test_init(self):
        """Test initialization of EntityExtractor."""
        extractor = EntityExtractor()
        self.assertEqual({}, extractor.module_info)
        self.assertEqual({}, extractor.functions)
        self.assertEqual({}, extractor.classes)
    
    def test_visit_ClassDef(self):
        """Test visit_ClassDef method."""
        # Visit the AST to extract entities
        self.extractor.visit(self.tree)
        
        # Check classes
        self.assertIn("Person", self.extractor.classes)
        person_class = self.extractor.classes["Person"]
        
        # Check class structure
        self.assertIn("methods", person_class)
        self.assertGreaterEqual(len(person_class["methods"]), 2)  # __init__ and greet
        self.assertIn("__init__", person_class["methods"])
        self.assertIn("greet", person_class["methods"])
    
    def test_visit_FunctionDef(self):
        """Test visit_FunctionDef method."""
        # Visit the AST to extract entities
        self.extractor.visit(self.tree)
        
        # Check functions
        self.assertIn("hello", self.extractor.functions)
        hello_func = self.extractor.functions["hello"]
        
        # Check function structure
        self.assertIn("args", hello_func)
        self.assertIn("returns", hello_func)
        self.assertEqual("str", hello_func["returns"])
        self.assertEqual(["name"], hello_func["args"])
    
    def test_visit_Import(self):
        """Test visit_Import method."""
        # Visit the AST to extract entities
        self.extractor.visit(self.tree)
        
        # Check imports
        self.assertIn("imports", self.extractor.module_info)
        imports = self.extractor.module_info["imports"]
        
        # Check for specific imports
        import_names = [imp["name"] for imp in imports]
        self.assertIn("os", import_names)
        self.assertIn("sys", import_names)
    
    def test_visit_ImportFrom(self):
        """Test visit_ImportFrom method."""
        # Visit the AST to extract entities
        self.extractor.visit(self.tree)
        
        # Check imports
        self.assertIn("imports", self.extractor.module_info)
        imports = self.extractor.module_info["imports"]
        
        # Check for specific imports
        from_imports = [imp for imp in imports if imp.get("from") == "typing"]
        self.assertGreaterEqual(len(from_imports), 1)
        
        # Check imported names
        imported_names = []
        for imp in from_imports:
            if "imports" in imp:
                imported_names.extend(imp["imports"])
        
        self.assertIn("List", imported_names)
        self.assertIn("Dict", imported_names)


class TestCallFinder(unittest.TestCase):
    """Test cases for CallFinder."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.finder = CallFinder()
        
        # Sample Python code with function calls
        self.sample_code = '''
def main():
    x = hello("world")
    person = Person("Alice", 30)
    greeting = person.greet()
    print(greeting)
    return x

def hello(name):
    return f"Hello, {name}!"

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
        
    def greet(self):
        return f"Hi, I'm {self.name}!"
'''
        self.tree = ast.parse(self.sample_code)
    
    def test_init(self):
        """Test initialization of CallFinder."""
        finder = CallFinder()
        self.assertEqual([], finder.calls)
    
    def test_visit_Call(self):
        """Test visit_Call method."""
        # Create a function node to visit
        main_func = next((node for node in self.tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"), None)
        self.assertIsNotNone(main_func)
        
        # Set the context for the finder
        self.finder.current_function = "main"
        
        # Visit the function to find calls
        self.finder.visit(main_func)
        
        # Check the calls that were found
        self.assertGreater(len(self.finder.calls), 0)
        
        # Check for specific calls
        call_targets = [call["target"] for call in self.finder.calls]
        self.assertIn("hello", call_targets)
        self.assertIn("Person", call_targets)
        self.assertIn("print", call_targets)


class TestTypeDefinitions(unittest.TestCase):
    """Test cases for the type definitions in python_adapter.py."""
    
    def test_python_metadata_dict(self):
        """Test PythonMetadataDict type."""
        # Create a dictionary matching the PythonMetadataDict structure
        metadata = PythonMetadataDict(
            language="python",
            document_type="code",
            function_count="2",
            class_count="1",
            import_count="3",
            line_count="50",
            cyclomatic_complexity="1.5",
            average_function_length="10"
        )
        
        # Check that the dictionary has the expected structure
        self.assertEqual("python", metadata["language"])
        self.assertEqual("code", metadata["document_type"])
        self.assertEqual("2", metadata["function_count"])
        self.assertEqual("1", metadata["class_count"])
        
    def test_function_info(self):
        """Test FunctionInfo type."""
        # Create a dictionary matching the FunctionInfo structure
        func_info = FunctionInfo(
            name="test_func",
            args=["arg1", "arg2"],
            returns="str",
            docstring="Test function docstring",
            source="def test_func(): pass",
            line_start=10,
            line_end=15,
            complexity=1.0,
            calls=["other_func"]
        )
        
        # Check that the dictionary has the expected structure
        self.assertEqual("test_func", func_info["name"])
        self.assertEqual(["arg1", "arg2"], func_info["args"])
        self.assertEqual("str", func_info["returns"])
        
    def test_class_info(self):
        """Test ClassInfo type."""
        # Create a dictionary matching the ClassInfo structure
        class_info = ClassInfo(
            name="TestClass",
            bases=["BaseClass"],
            docstring="Test class docstring",
            source="class TestClass(BaseClass): pass",
            methods=["__init__", "test_method"],
            attributes=["attr1", "attr2"],
            line_start=20,
            line_end=30
        )
        
        # Check that the dictionary has the expected structure
        self.assertEqual("TestClass", class_info["name"])
        self.assertEqual(["BaseClass"], class_info["bases"])
        self.assertEqual(["__init__", "test_method"], class_info["methods"])
        
    def test_import_info(self):
        """Test ImportInfo type."""
        # Create a dictionary matching the ImportInfo structure
        import_info = ImportInfo(
            name="os",
            from_module=None,
            imports=None,
            line=5
        )
        
        # Check that the dictionary has the expected structure
        self.assertEqual("os", import_info["name"])
        self.assertIsNone(import_info["from"])
        
        # Test from import
        from_import = ImportInfo(
            name=None,
            from_module="typing",
            imports=["List", "Dict"],
            line=6
        )
        
        self.assertEqual("typing", from_import["from"])
        self.assertEqual(["List", "Dict"], from_import["imports"])
        
    def test_relationship_info(self):
        """Test RelationshipInfo type."""
        # Create a dictionary matching the RelationshipInfo structure
        rel_info = RelationshipInfo(
            source="func_a",
            target="func_b",
            type="calls",
            weight=1
        )
        
        # Check that the dictionary has the expected structure
        self.assertEqual("func_a", rel_info["source"])
        self.assertEqual("func_b", rel_info["target"])
        self.assertEqual("calls", rel_info["type"])
        self.assertEqual(1, rel_info["weight"])


if __name__ == "__main__":
    unittest.main()
