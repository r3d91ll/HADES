# Document Processing (docproc) Type System

This document describes the type system for the document processing module in HADES. The types have been migrated to a centralized location in `src/types/docproc/` as part of the type system migration project.

## Type Structure

The document processing type system is organized as follows:

### Base Types

- `src/types/docproc/document.py` - Base document types
  - `BaseDocument` - Base class for all document types
  - `BaseEntity` - Base class for document entities
  - `BaseMetadata` - Base class for document metadata

### Adapter Types

- `src/types/docproc/adapter.py` - Document processor adapter protocols
  - `DocumentProcessorProtocol` - Protocol for document processors
  - `AdapterResult` - Result type for adapter processing

### Python-Specific Types

- `src/types/docproc/python.py` - Python document schemas
  - `PythonMetadata` - Metadata specific to Python documents
  - `PythonEntity` - Entity specific to Python code
  - `CodeRelationship` - Relationship between code elements
  - `CodeElement` - Generic code element
  - `FunctionElement` - Function code element
  - `MethodElement` - Method code element
  - `ClassElement` - Class code element
  - `ImportElement` - Import code element
  - `SymbolTable` - Python module symbol table
  - `PythonDocument` - Document model for Python code

### TypedDict Definitions

- `src/types/docproc/code_elements.py` - TypedDict definitions for code elements
  - `CodeRelationship` - A relationship between code elements
  - `ElementRelationship` - A relationship from the current element to another element
  - `LineRange` - A range of lines in the source code
  - `Annotation` - Type annotation for a parameter or return value
  - `ImportElement` - An import statement in Python code
  - `FunctionElement` - A function definition in Python code
  - `MethodElement` - A method definition in a class
  - `ClassElement` - A class definition in Python code
  - `ModuleElement` - A Python module
  - `PySymbolTable` - The symbol table for a Python module
  - `PythonDocument` - The complete representation of a Python document

### Enumerations

- `src/types/docproc/enums.py` - Enumeration types for document processing
  - `RelationshipType` - Type of relationship between code elements
  - `ImportSourceType` - Source type of an import
  - `AccessLevel` - Access level of a code element

## Usage Examples

### Processing Python Documents

```python
from src.types.docproc.python import PythonDocument, PythonMetadata
from src.types.docproc.code_elements import CodeRelationship

# Create a Python document with proper typing
document = PythonDocument(
    id="doc-123",
    format="python",
    content="def hello(): print('Hello, world!')",
    metadata=PythonMetadata(
        function_count=1,
        class_count=0,
        import_count=0,
        method_count=0,
        has_module_docstring=False,
        has_syntax_errors=False
    )
)
```

### Working with Code Relationships

```python
from src.types.docproc.enums import RelationshipType
from src.types.docproc.python import CodeRelationship

# Create a code relationship
relationship = CodeRelationship(
    source="function:hello",
    target="function:world",
    type=RelationshipType.CALLS.value,
    weight=0.9,
    line=42
)
```

## Migration Status

- ✅ Basic type definitions migrated to centralized location
- ✅ Python-specific TypedDict definitions in `src/types/docproc/code_elements.py`
- ✅ Python document models in `src/types/docproc/python.py`
- ✅ Enum types migrated to `src/types/docproc/enums.py`
- ✅ Unit tests implemented for code element TypedDict definitions
- ✅ Unit tests implemented for Python document types
- ✅ Unit tests implemented for enum types
- ✅ Isolated tests implemented to verify type structures without dependencies
- ✅ Utility script created in `temp-utils/fix_docproc_types.py` to update import statements

## Testing

Tests for the docproc types are located in the `/test/types/docproc/` directory and include:

- `test_code_elements.py` - Tests for the code element TypedDict definitions
- `test_python_types.py` - Tests for the Python document types
- `test_enums.py` - Tests for the enum types
- `isolated_test_python.py` - Isolated tests that verify TypedDict structures without requiring external dependencies

These tests verify that:

1. TypedDict definitions have the correct structure and field types
2. Enums have the correct values and behavior
3. Pydantic models properly inherit from base classes
4. Validation functions work correctly
5. Complex nested structures can be properly created and accessed

## Usage Notes

When updating existing code to use the centralized types:

1. Import types from the centralized location:

   ```python
   # Old way
   from src.docproc.models.python_code import CodeRelationship
   
   # New way
   from src.types.docproc.code_elements import CodeRelationship
   ```

2. Use typed validator decorators:

   ```python
   # Old way
   @field_validator('field_name')
   
   # New way
   from src.types.docproc.python import typed_field_validator
   @typed_field_validator('field_name')
   ```

3. Run the utility script to automate common fixes:

   ```bash
   python temp-utils/fix_docproc_types.py
   ```

4. Verify type correctness with mypy:

   ```bash
   mypy src/docproc --show-column-numbers --pretty
   ```

- ✅ Python-specific schemas migrated to `src/types/docproc/python.py`
- ✅ Code model enums migrated to `src/types/docproc/enums.py`
- ✅ TypedDict definitions migrated to `src/types/docproc/code_elements.py`

## Next Steps

- Update imports in the docproc module to use the centralized types
- Add unit tests for the new type definitions
- Verify type compatibility with mypy
