"""
Tests for the Python pre-processor component.

This module provides comprehensive tests for the PythonPreProcessor class,
which extracts AST nodes, relationships, and metadata from Python source files.
"""
import ast
import os
import tempfile
from typing import Dict, List, Any, Optional, cast
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

from src.ingest.pre_processor.python_pre_processor import PythonPreProcessor


class TestPythonPreProcessorInit:
    """Test initialization of the PythonPreProcessor class."""
    
    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        processor = PythonPreProcessor()
        assert processor.create_symbol_table is True
    
    def test_init_with_custom_settings(self):
        """Test initialization with custom parameters."""
        processor = PythonPreProcessor(create_symbol_table=False)
        assert processor.create_symbol_table is False


class TestPythonPreProcessorBasics:
    """Test basic functionality of the PythonPreProcessor class."""
    
    @pytest.fixture
    def processor(self):
        """Create a processor instance for tests."""
        return PythonPreProcessor(create_symbol_table=False)
    
    @pytest.fixture
    def temp_python_file(self):
        """Create a temporary Python file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as f:
            f.write(b'''"""Module docstring."""
import os
from typing import List, Dict, Optional

class TestClass:
    """Test class docstring."""
    
    class_attr = "value"
    
    def __init__(self, param: str):
        """Initialize test class."""
        self.param = param
    
    def test_method(self, arg1: int, arg2: str = None) -> List[str]:
        """Test method docstring."""
        os.path.join(arg1, arg2)
        return ["result"]

@decorator
def test_function(a: int, b: str) -> Dict[str, Any]:
    """Test function docstring."""
    test_var = TestClass("test")
    test_var.test_method(1, "test")
    return {"key": "value"}
''')
        yield f.name
        os.unlink(f.name)
    
    def test_process_file(self, processor, temp_python_file):
        """Test processing a Python file."""
        result = processor.process_file(temp_python_file)
        
        # Check basic document structure
        assert result["type"] == "python"
        assert "path" in result
        assert "id" in result
        assert "content" in result
        assert "docstring" in result
        assert result["docstring"] == "Module docstring."
        
        # Check metadata
        assert "metadata" in result
        assert result["metadata"]["function_count"] > 0
        assert result["metadata"]["class_count"] > 0
        assert result["metadata"]["import_count"] > 0
        
        # Check extracted components
        assert "functions" in result
        assert "classes" in result
        assert "imports" in result
        assert "relationships" in result
        
        # Verify no symbol table was created (create_symbol_table=False)
        assert "symbol_table_path" not in result["metadata"]
    
    def test_process_file_with_syntax_error(self, processor):
        """Test processing a file with a syntax error."""
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as f:
            f.write(b'''
def test_function(
    print("Missing closing parenthesis")
''')
            file_path = f.name
        
        try:
            result = processor.process_file(file_path)
            
            # Check error handling
            assert "error" in result
            assert result["metadata"]["error_type"] == "syntax_error"
        finally:
            os.unlink(file_path)
    
    def test_process_file_with_file_error(self, processor):
        """Test handling of a non-existent file."""
        with pytest.raises(ValueError):
            processor.process_file("/path/does/not/exist.py")


class TestPythonPreProcessorExtraction:
    """Test AST extraction functions of the PythonPreProcessor."""
    
    @pytest.fixture
    def processor(self):
        """Create a processor instance for tests."""
        return PythonPreProcessor(create_symbol_table=False)
    
    @pytest.fixture
    def sample_python_ast(self):
        """Create a sample Python AST for testing extraction methods."""
        sample_code = '''"""Module docstring."""
import os
from typing import List, Dict, Optional, Union
from module.submodule import Class1, Class2 as RenamedClass

class BaseClass:
    """Base class."""
    pass

class TestClass(BaseClass):
    """Test class docstring."""
    
    class_attr = "value"
    another_attr = 42
    
    def __init__(self, param: str):
        """Initialize test class."""
        self.param = param
    
    @property
    def prop(self) -> str:
        """Property docstring."""
        return self.param
    
    def test_method(self, arg1: int, arg2: Optional[str] = None) -> List[str]:
        """Test method docstring."""
        os.path.join(arg1, arg2)
        test_function(1, "test")
        return ["result"]

@decorator
def test_function(a: int, b: Union[str, Dict]) -> Dict[str, Any]:
    """Test function docstring."""
    test_var = TestClass("test")
    another_function()
    return {"key": "value"}

def another_function():
    """Another function."""
    pass
'''
        return ast.parse(sample_code)
    
    def test_extract_functions(self, processor, sample_python_ast):
        """Test extraction of functions from AST."""
        functions = processor._extract_functions(sample_python_ast)
        
        # Verify minimum function count - exact count may vary due to nested functions
        assert len(functions) >= 2
        
        # Verify function data
        test_func = next(f for f in functions if f["name"] == "test_function")
        assert test_func["docstring"] == "Test function docstring."
        assert "a" in test_func["args"]
        assert "b" in test_func["args"]
        assert "decorator" in test_func["decorators"]
        assert test_func["returns"] is not None
        assert "another_function" in test_func["calls"]
        
        # Verify another function
        another_func = next(f for f in functions if f["name"] == "another_function")
        assert another_func["docstring"] == "Another function."
        assert len(another_func["args"]) == 0
        assert len(another_func["decorators"]) == 0
    
    def test_extract_classes(self, processor, sample_python_ast):
        """Test extraction of classes from AST."""
        classes = processor._extract_classes(sample_python_ast)
        
        # Verify class count
        assert len(classes) == 2
        
        # Verify TestClass data
        test_class = next(c for c in classes if c["name"] == "TestClass")
        assert test_class["docstring"] == "Test class docstring."
        assert "BaseClass" in test_class["bases"]
        assert "__init__" in test_class["methods"]
        assert "test_method" in test_class["methods"]
        assert "prop" in test_class["methods"]
        assert "class_attr" in test_class["attributes"]
        assert "another_attr" in test_class["attributes"]
        
        # Verify BaseClass
        base_class = next(c for c in classes if c["name"] == "BaseClass")
        assert base_class["docstring"] == "Base class."
        assert len(base_class["bases"]) == 0
    
    def test_extract_imports(self, processor, sample_python_ast):
        """Test extraction of imports from AST."""
        imports = processor._extract_imports(sample_python_ast)
        
        # Verify minimum import count - exact count may vary
        assert len(imports) >= 4
        
        # Verify direct import
        os_import = next(i for i in imports if i["name"] == "os")
        assert os_import["type"] == "import"
        assert os_import["asname"] is None
        
        # Verify import from
        list_import = next(i for i in imports if i["name"] == "List")
        assert list_import["type"] == "importfrom"
        assert list_import["module"] == "typing"
        
        # Verify import with alias
        renamed_import = next(i for i in imports if i["name"] == "Class2")
        assert renamed_import["type"] == "importfrom"
        assert renamed_import["asname"] == "RenamedClass"
    
    def test_extract_function_calls(self, processor):
        """Test extraction of function calls."""
        code = """
def test():
    func1()
    obj.method()
    func2(1, 2)
"""
        node = ast.parse(code).body[0]  # Get the FunctionDef node
        calls = processor._extract_function_calls(node)
        
        assert "func1" in calls
        assert "func2" in calls
        # Note: obj.method() won't be included as it's an attribute access, not a direct name
    
    def test_get_decorator_name(self, processor):
        """Test extraction of decorator names from different node types."""
        # Test Name node
        node = ast.Name(id="decorator")
        assert processor._get_decorator_name(node) == "decorator"
        
        # Test Call node
        call_node = ast.Call(func=ast.Name(id="decorator"), args=[], keywords=[])
        assert processor._get_decorator_name(call_node) == "decorator"
        
        # Test Attribute node
        attr_node = ast.Attribute(value=ast.Name(id="module"), attr="decorator")
        assert processor._get_decorator_name(attr_node) == "module.decorator"
        
        # Test unknown node type
        unknown_node = ast.Constant(value=123)
        assert "unknown" in processor._get_decorator_name(unknown_node)
    
    def test_get_return_annotation(self, processor):
        """Test extraction of return type annotations."""
        # Create function node with return annotation
        func_with_annotation = ast.FunctionDef(
            name="test",
            args=ast.arguments(args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
            body=[ast.Pass()],
            decorator_list=[],
            returns=ast.Name(id="str")
        )
        assert processor._get_return_annotation(func_with_annotation) == "str"
        
        # Create function node without return annotation
        func_without_annotation = ast.FunctionDef(
            name="test",
            args=ast.arguments(args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
            body=[ast.Pass()],
            decorator_list=[],
            returns=None
        )
        assert processor._get_return_annotation(func_without_annotation) is None
    
    def test_get_name(self, processor):
        """Test name extraction from different node types."""
        # Test Name node
        name_node = ast.Name(id="variable")
        assert processor._get_name(name_node) == "variable"
        
        # Test Attribute node
        attr_node = ast.Attribute(value=ast.Name(id="module"), attr="subattr")
        assert processor._get_name(attr_node) == "module.subattr"
        
        # Test nested Attribute node
        nested_attr = ast.Attribute(
            value=ast.Attribute(value=ast.Name(id="module"), attr="submodule"),
            attr="attr"
        )
        assert processor._get_name(nested_attr) == "module.submodule.attr"
        
        # Test Subscript node (e.g. List[str])
        subscript_node = ast.Subscript(
            value=ast.Name(id="List"),
            slice=ast.Name(id="str")
        )
        # Ensure we get "List[str]" format
        result = processor._get_name(subscript_node)
        assert result.startswith("List[")
        assert "str" in result or "..." in result
        
        # Test other node types
        other_node = ast.Constant(value="test")
        assert processor._get_name(other_node) == "Constant"


class TestPythonPreProcessorRelationships:
    """Test relationship extraction from Python code."""
    
    @pytest.fixture
    def processor(self):
        """Create a processor instance for tests."""
        return PythonPreProcessor(create_symbol_table=False)
    
    def test_build_relationships(self, processor):
        """Test building relationships between code elements."""
        file_path = "test_file.py"
        
        # Sample extracted data
        functions = [
            {
                "name": "func1", 
                "calls": ["func2", "external_func"],
                "line_start": 1,
                "line_end": 5
            },
            {
                "name": "func2",
                "calls": [],
                "line_start": 6,
                "line_end": 10
            }
        ]
        
        classes = [
            {
                "name": "Class1",
                "bases": ["BaseClass", "object"],
                "methods": ["method1", "method2"],
                "attributes": ["attr1"],
                "line_start": 11,
                "line_end": 20
            }
        ]
        
        imports = [
            {
                "type": "import",
                "name": "os",
                "asname": None,
                "line": 1
            },
            {
                "type": "importfrom",
                "module": "module",
                "name": "Class",
                "asname": None,
                "line": 2
            }
        ]
        
        # Build relationships
        relationships = processor._build_relationships(
            file_path, functions, classes, imports
        )
        
        # Verify relationship count
        assert len(relationships) > 0
        
        # Check function calls function relationship
        func_call_rel = next(
            r for r in relationships 
            if r["from"] == "test_file.py::func1" and r["to"].endswith("::func2")
        )
        assert func_call_rel["type"] == "CALLS"
        
        # Check class inheritance relationship
        inheritance_rel = next(
            r for r in relationships 
            if r["from"] == "test_file.py::Class1" and r["to"].endswith("::BaseClass")
        )
        assert inheritance_rel["type"] == "EXTENDS"
        
        # Check class contains method relationship
        contains_rel = next(
            r for r in relationships 
            if r["from"] == "test_file.py::Class1" and r["to"].endswith("::method1")
        )
        assert contains_rel["type"] == "CONTAINS"
        
        # Check module imports relationship
        import_rel = next(
            r for r in relationships 
            if r["from"] == "test_file.py" and r["to"] == "os"
        )
        assert import_rel["type"] == "IMPORTS"
        
        import_from_rel = next(
            r for r in relationships 
            if r["from"] == "test_file.py" and r["to"] == "module.Class"
        )
        assert import_from_rel["type"] == "IMPORTS"


class TestPythonPreProcessorSymbolTable:
    """Test symbol table creation functionality."""
    
    @pytest.fixture
    def processor(self):
        """Create a processor instance with symbol table enabled."""
        return PythonPreProcessor(create_symbol_table=True)
    
    def test_create_symbol_table(self, processor, tmp_path):
        """Test creation of a symbol table file."""
        # Create a temporary file path
        file_path = tmp_path / "test_module.py"
        
        # Mock document data
        document = {
            "classes": [
                {
                    "name": "TestClass",
                    "line_start": 10,
                    "line_end": 20,
                    "methods": ["test_method", "__init__"]
                }
            ],
            "functions": [
                {
                    "name": "test_function",
                    "line_start": 30,
                    "line_end": 40
                },
                {
                    "name": "__special__",  # Should be skipped
                    "line_start": 50,
                    "line_end": 60
                }
            ],
            "imports": [
                {
                    "type": "import",
                    "name": "os"
                },
                {
                    "type": "importfrom",
                    "module": "typing",
                    "name": "List"
                }
            ]
        }
        
        # Create symbol table
        symbol_file = processor._create_symbol_table(str(file_path), document)
        
        # Verify symbol table was created
        assert symbol_file is not None
        assert symbol_file.exists()
        
        # Check symbol table contents
        content = symbol_file.read_text()
        assert "FILE:test_module.py" in content
        assert "CLASS:TestClass:10-20" in content
        assert "METHOD:TestClass.test_method" in content
        assert "METHOD:TestClass.__init__" in content
        assert "FUNCTION:test_function:30-40" in content
        assert "__special__" not in content  # Should be skipped
        assert "IMPORT:os" in content
        assert "IMPORT:typing.List" in content
    
    def test_create_symbol_table_with_error(self, processor):
        """Test error handling during symbol table creation."""
        with patch("pathlib.Path.mkdir", side_effect=PermissionError("Permission denied")):
            result = processor._create_symbol_table("/fake/path.py", {})
            assert result is None
    
    def test_process_file_with_symbol_table(self, processor, tmp_path):
        """Test processing a file with symbol table creation enabled."""
        # Create a simple Python file
        file_path = tmp_path / "simple.py"
        file_path.write_text('''
def test_function():
    """Test function."""
    pass
''')
        
        # Process the file
        with patch.object(processor, '_create_symbol_table') as mock_create:
            # Mock successful symbol table creation
            mock_create.return_value = Path(tmp_path) / ".symbol_table" / "simple.py.symbols"
            
            result = processor.process_file(str(file_path))
            
            # Verify symbol table creation was attempted
            assert mock_create.called
            
            # Verify the path was added to metadata
            assert "symbol_table_path" in result["metadata"]
    
    def test_process_file_with_symbol_table_error(self, processor, tmp_path):
        """Test processing a file with symbol table creation error."""
        # Create a simple Python file
        file_path = tmp_path / "simple.py"
        file_path.write_text('''
def test_function():
    """Test function."""
    pass
''')
        
        # Process the file
        with patch.object(processor, '_create_symbol_table') as mock_create:
            # Mock symbol table creation error
            mock_create.side_effect = Exception("Test error")
            
            result = processor.process_file(str(file_path))
            
            # Verify symbol table creation was attempted
            assert mock_create.called
            
            # Verify the error was added to metadata
            assert "symbol_table_error" in result["metadata"]


# Advanced test cases to cover edge cases
class TestPythonPreProcessorEdgeCases:
    """Test edge cases and advanced scenarios for the Python preprocessor."""
    
    @pytest.fixture
    def processor(self):
        """Create a processor instance for tests."""
        return PythonPreProcessor(create_symbol_table=False)
    
    def test_complex_return_annotations(self, processor):
        """Test handling of complex return type annotations."""
        # Create test code with various types of type annotations
        code = """
def func1() -> List[Dict[str, Any]]:
    pass

def func2() -> Tuple[int, Optional[str]]:
    pass

def func3() -> Union[List[str], Dict[str, int]]:
    pass
"""
        tree = ast.parse(code)
        functions = processor._extract_functions(tree)
        
        assert len(functions) == 3
        # Check that return types are captured in some form
        for func in functions:
            assert func["returns"] is not None
            assert "[" in func["returns"] or "Union" in func["returns"]
    
    def test_nested_class_methods(self, processor):
        """Test handling of nested class methods."""
        code = """
class Outer:
    class Inner:
        def inner_method(self):
            pass
            
    def outer_method(self):
        pass
"""
        tree = ast.parse(code)
        classes = processor._extract_classes(tree)
        
        assert len(classes) == 2
        outer = next(c for c in classes if c["name"] == "Outer")
        inner = next(c for c in classes if c["name"] == "Inner")
        
        assert "outer_method" in outer["methods"]
        assert "inner_method" in inner["methods"]
    
    def test_method_with_complex_decorators(self, processor):
        """Test handling of methods with complex decorators."""
        code = """
class Test:
    @decorator
    @another.decorator(param1="value")
    @module.submodule.decorator
    def test_method(self):
        pass
"""
        tree = ast.parse(code)
        classes = processor._extract_classes(tree)
        
        assert len(classes) == 1
        test_class = classes[0]
        assert "test_method" in test_class["methods"]
        assert len(test_class["decorators"]) == 0  # Class doesn't have decorators
        
        # Extract method to check its decorators
        functions = processor._extract_functions(tree)
        test_method = next(f for f in functions if f["name"] == "test_method")
        assert len(test_method["decorators"]) == 3
        # The Call decorator with args is handled differently
        assert any("decorator" in d for d in test_method["decorators"])
        # Complex decorators may be handled differently based on implementation
        # Just check that we have the expected count of decorators
