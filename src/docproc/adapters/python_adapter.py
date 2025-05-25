from __future__ import annotations

"""
Python code adapter for the document processing system.

This module provides an adapter for processing Python source code files,
focusing on Python code with AST parsing capabilities.
It extracts functions, classes, imports, and relationships between components.
"""

import ast
import hashlib
import os
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Set, TypedDict, cast
from collections import defaultdict

from .base import BaseAdapter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FunctionInfo(TypedDict):
    """TypedDict for function information."""
    name: str
    line_start: int
    line_end: int
    args: List[str]
    docstring: Optional[str]
    decorators: List[str]
    returns: Optional[str]
    is_method: bool
    calls: List[str]


class ClassInfo(TypedDict):
    """TypedDict for class information."""
    name: str
    line_start: int
    line_end: int
    docstring: Optional[str]
    methods: List[str]
    bases: List[str]


class ImportInfo(TypedDict):
    """TypedDict for import information."""
    type: str  # 'import' or 'importfrom'
    name: str
    module: Optional[str]  # only for importfrom
    line: int


class RelationshipInfo(TypedDict):
    """TypedDict for relationship information."""
    type: str  # 'CALLS', 'CONTAINS', 'EXTENDS', etc.
    from_entity: str
    to_entity: str
    weight: float


class PythonAdapter(BaseAdapter):
    """Adapter for Python source code files with AST parsing capabilities."""

    def __init__(self, create_symbol_table: Optional[bool] = None, options: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the code adapter with configuration options.
        
        Args:
            create_symbol_table: Whether to create a symbol table for Python files
            options: Optional configuration options
        """
        # Initialize base adapter with format type
        super().__init__(format_type="python")
        
        # Get format-specific configuration with defaults
        self.create_symbol_table = create_symbol_table if create_symbol_table is not None else self.format_config.get("create_symbol_table", True)
        self.extract_docstrings = self.format_config.get("extract_docstrings", True)
        self.analyze_imports = self.format_config.get("analyze_imports", True)
        self.analyze_calls = self.format_config.get("analyze_calls", True)
        self.extract_type_hints = self.format_config.get("extract_type_hints", True)
        self.compute_complexity = self.format_config.get("compute_complexity", False)
        
        # Merge options with defaults
        self.options = {**self.format_config, **(options or {})}
        
        logger.info(f"Initialized PythonAdapter with create_symbol_table={self.create_symbol_table}, analyze_calls={self.analyze_calls}")

    def process(self, file_path: Union[str, Path], options: Optional[Union[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Process a source-code file.

        Args:
            file_path: Path to the source code file (as string or Path object)
            options: Optional processing options

        Returns:
            Structured document with code analysis

        Raises:
            FileNotFoundError: If the specified file doesn't exist
            ValueError: If there's an error reading or parsing the file
        """
        # Handle options based on type
        process_options = dict(self.options)
        if options is not None:
            if isinstance(options, dict):
                process_options.update(options)
            elif isinstance(options, str):
                # Handle string options (e.g., format specification)
                process_options["format"] = options

        # Convert to Path object if string
        path_obj = Path(file_path) if isinstance(file_path, str) else file_path

        if not path_obj.exists():
            raise FileNotFoundError(f"Code file not found: {path_obj}")

        # Get file extension and determine language
        # Make sure language is consistently set to "python" for Python files
        if path_obj.suffix.lower() == ".py":
            language = "python"
        else:
            language = path_obj.suffix.lstrip(".").lower() or "unknown"
        
        # Read file content
        encoding = process_options.get("encoding", "utf-8")
        try:
            source = path_obj.read_text(encoding=encoding)
        except UnicodeDecodeError:
            # Fallback to binary read then decode ignoring errors
            source = path_obj.read_bytes().decode("utf-8", errors="ignore")

        # Generate a stable ID for the document
        doc_id = f"code_{hashlib.md5(str(path_obj).encode()).hexdigest()[:8]}"
        
        # Extract metadata
        metadata = self.extract_metadata(source)
        metadata["language"] = language
        metadata["file_size"] = len(source)
        metadata["file_path"] = str(path_obj)
        metadata["format"] = "python"
        
        # Extract entities
        entities = self.extract_entities(source)
        
        # We no longer convert Python code to markdown - we keep it in raw form
        
        # Ensure metadata has the required fields according to schema
        if "content_type" not in metadata:
            metadata["content_type"] = "code"  # All Python files are code
        
        # Set format in metadata to python since content is Python code
        metadata["format"] = "python"  # Python content is stored as raw Python code
        
        # Basic document structure
        document: Dict[str, Any] = {
            "id": doc_id,
            "source": str(path_obj),
            "format": "python",  # Original document format 
            "content": source,
            "content_format": "python",  # How the content is stored in this JSON
            "content_type": "code",  # Top-level content_type for primary chunking decision
            "metadata": metadata,  # metadata.format describes the content format
            "entities": entities
        }
        
        # Special handling for Python files
        if language == "python" and self.create_symbol_table:
            python_data = self._process_python_file(path_obj, source)
            
            # Update document with Python-specific data
            if python_data:
                # Copy entities and relationships to document
                if "entities" in python_data:
                    document["symbol_table"] = python_data.get("entities", {})
                if "relationships" in python_data:
                    document["relationships"] = python_data.get("relationships", [])
                if "module_id" in python_data:
                    document["module_id"] = python_data.get("module_id", "")
                
                # Add metadata counts
                entity_types: Dict[str, int] = {}
                for entity_data in python_data.get("entities", {}).values():
                    entity_type = entity_data.get("type", "unknown")
                    entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
                
                for entity_type, count in entity_types.items():
                    document["metadata"][f"{entity_type}_count"] = count
                
                # Include error if present
                if "error" in python_data:
                    document["error"] = python_data["error"]
        
        return document
    
    def process_text(self, text: str, options: Optional[Union[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Process Python text content.
        
        Args:
            text: Python source code as text
            options: Optional processing options
            
        Returns:
            Dictionary with processed content and metadata
        """
        # Handle options based on type
        process_options = dict(self.options)
        if options is not None:
            if isinstance(options, dict):
                process_options.update(options)
            elif isinstance(options, str):
                # Handle string options (e.g., format specification)
                process_options["format"] = options
        
        # Generate a stable document ID
        doc_id = f"python_text_{hashlib.md5(text[:100].encode()).hexdigest()[:8]}"
        
        try:
            # Extract metadata
            metadata = self.extract_metadata(text)
            metadata["file_path"] = "text"  # Consistent with other adapters
            metadata["format"] = "python"
            
            # Extract entities
            entities = self.extract_entities(text)
            
            # Convert to markdown (integrated from former convert_to_markdown method)
            # Escape any existing markdown formatting
            escaped_content = text.replace("```", "\\```")
            # Format as a code block
            markdown_content = f"```python\n{escaped_content}\n```"
            
            result = {
                "id": doc_id,
                "source": "text",
                "content": markdown_content,
                "content_type": "markdown",
                "format": "python",
                "metadata": metadata,
                "entities": entities,
                "raw_content": text
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing Python text: {e}", exc_info=True)
            raise ValueError(f"Error processing Python text: {str(e)}")
    
    def extract_entities(self, content: Union[str, Dict[str, Any]], options: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Extract entities from Python content.
        
        Args:
            content: Python content as string or parsed data
            
        Returns:
            List of extracted entities with metadata
        """
        entities: List[Dict[str, Any]] = []
        
        # Handle dictionary input
        if isinstance(content, dict) and "raw_content" in content:
            python_content = str(content["raw_content"])
        elif isinstance(content, str):
            python_content = content
        else:
            return entities
        
        try:
            # Parse the AST
            tree = ast.parse(python_content)
            
            # Extract functions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Extract function information
                    function_entity = {
                        "type": "function",
                        "value": node.name,
                        "line": node.lineno,
                        "confidence": 1.0
                    }
                    
                    # Add docstring if available
                    docstring = ast.get_docstring(node)
                    if docstring:
                        function_entity["docstring"] = docstring
                    
                    entities.append(function_entity)
                    
                elif isinstance(node, ast.ClassDef):
                    # Extract class information
                    class_entity = {
                        "type": "class",
                        "value": node.name,
                        "line": node.lineno,
                        "confidence": 1.0
                    }
                    
                    # Add docstring if available
                    docstring = ast.get_docstring(node)
                    if docstring:
                        class_entity["docstring"] = docstring
                    
                    entities.append(class_entity)
                    
                elif isinstance(node, ast.Import):
                    # Extract import information
                    for name in node.names:
                        entities.append({
                            "type": "import",
                            "value": name.name,
                            "line": node.lineno,
                            "confidence": 1.0
                        })
                        
                elif isinstance(node, ast.ImportFrom):
                    # Extract import from information
                    module = node.module or ""
                    for name in node.names:
                        entities.append({
                            "type": "import",
                            "value": f"{module}.{name.name}",
                            "line": node.lineno,
                            "confidence": 1.0
                        })
                        
        except SyntaxError:
            # Fall back to regex-based extraction for syntax errors
            # Extract potential function definitions
            for match in re.finditer(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', python_content):
                entities.append({
                    "type": "function",
                    "value": match.group(1),
                    "line": python_content[:match.start()].count('\n') + 1,
                    "confidence": 0.8  # Lower confidence for regex matches
                })
                
            # Extract potential class definitions
            for match in re.finditer(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\(|:)', python_content):
                entities.append({
                    "type": "class",
                    "value": match.group(1),
                    "line": python_content[:match.start()].count('\n') + 1,
                    "confidence": 0.8
                })
                
            # Extract potential imports
            for match in re.finditer(r'import\s+([a-zA-Z0-9_.]+)', python_content):
                entities.append({
                    "type": "import",
                    "value": match.group(1),
                    "line": python_content[:match.start()].count('\n') + 1,
                    "confidence": 0.8
                })
                
            for match in re.finditer(r'from\s+([a-zA-Z0-9_.]+)\s+import', python_content):
                entities.append({
                    "type": "import",
                    "value": match.group(1),
                    "line": python_content[:match.start()].count('\n') + 1,
                    "confidence": 0.8
                })
        
        return entities
    
    def extract_metadata(self, content: Union[str, Dict[str, Any]], options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract metadata from Python content.
        
        Args:
            content: Python content as string or parsed data
            
        Returns:
            Dictionary of metadata
        """
        # Always include format for consistency
        metadata: Dict[str, Any] = {
            "format": "python",
            "content_type": "code"
        }
        
        # Handle dictionary input
        if isinstance(content, dict) and "raw_content" in content:
            python_content = str(content["raw_content"])
        elif isinstance(content, str):
            python_content = content
        else:
            return metadata
        
        # Basic metrics
        metadata["line_count"] = python_content.count('\n') + 1
        metadata["char_count"] = len(python_content)
        
        try:
            # Parse the AST
            tree = ast.parse(python_content)
            
            # Get module docstring
            module_docstring = ast.get_docstring(tree)
            if module_docstring:
                metadata["has_module_docstring"] = True
                
            # Count definitions
            function_count = 0
            class_count = 0
            import_count = 0
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    function_count += 1
                elif isinstance(node, ast.ClassDef):
                    class_count += 1
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_count += 1
                    
            metadata["function_count"] = function_count
            metadata["class_count"] = class_count
            metadata["import_count"] = import_count
            
        except SyntaxError:
            # Fall back to regex-based analysis for syntax errors
            metadata["function_count"] = len(re.findall(r'def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(', python_content))
            metadata["class_count"] = len(re.findall(r'class\s+[a-zA-Z_][a-zA-Z0-9_]*\s*(?:\(|:)', python_content))
            metadata["import_count"] = len(re.findall(r'import\s+[a-zA-Z0-9_.]+|from\s+[a-zA-Z0-9_.]+\s+import', python_content))
            metadata["has_syntax_errors"] = True
            
        return metadata
    
    # convert_to_markdown and convert_to_text methods were removed as they are outside
    # the core functionality of the document processing pipeline
    
    def _process_python_file(self, file_path: Union[str, Path], source: str) -> Dict[str, Any]:
        """Process a Python file, extracting AST nodes, relationships, and metadata.
        
        Args:
            file_path: Path to the Python file
            source: Python source code
            
        Returns:
            Dict with extracted information or None if parsing fails
        """
        try:
            # Parse the AST
            tree = ast.parse(source)
            
            # Generate unique ID for module
            module_id = f"module_{hashlib.md5(str(file_path).encode()).hexdigest()[:8]}"
            
            # Generate a module ID based on file path for linking
            path_obj = Path(file_path) if isinstance(file_path, str) else file_path
            module_name = path_obj.stem
            
            # Create entities dictionary with module as the first entry
            entities: Dict[str, Dict[str, Any]] = {
                module_id: {
                    "type": "module",
                    "name": module_name,
                    "path": str(file_path),
                    "docstring": ast.get_docstring(tree),
                    "contains": []  # Will be filled with entity IDs
                }
            }
            
            # Extract all code entities (functions, classes, imports, variables)
            path_obj = Path(file_path) if isinstance(file_path, str) else file_path
            self._extract_entities(tree, path_obj, entities, module_id)
            
            # Build relationships between entities
            relationships = self._build_entity_relationships(entities)
            
            return {
                "module_id": module_id,
                "entities": entities,
                "relationships": relationships
            }
            
        except SyntaxError as e:
            logger.error(f"Syntax error parsing {file_path}: {e}")
            return {
                "error": f"Syntax error: {str(e)}",
                "entities": {},
                "relationships": []
            }
        except Exception as e:
            logger.error(f"Error processing Python file {file_path}: {e}", exc_info=True)
            return {
                "error": f"Processing error: {str(e)}",
                "entities": {},
                "relationships": []
            }

    def _extract_entities(self, tree: ast.Module, file_path: Path, entities: Dict[str, Dict[str, Any]], parent_id: str) -> None:
        """Extract all entity information from AST.
        
        Args:
            tree: AST tree
            file_path: Path to the Python file
            entities: Dictionary to populate with entity data
            parent_id: ID of the parent entity
        """
        visitor = EntityExtractor(str(file_path), entities, parent_id)
        visitor.visit(tree)
        
    def _build_entity_relationships(self, entities: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build relationships between entities.
        
        Args:
            entities: Dictionary of entities
            
        Returns:
            List of relationship dictionaries
        """
        relationships: List[Dict[str, Any]] = []
        
        # Map of function/method names to their entity IDs
        name_to_id: Dict[str, str] = {}
        for entity_id, entity in entities.items():
            if entity["type"] in ("function", "method", "class"):
                name_to_id[entity["name"]] = entity_id
                
                # Add qualified names for methods
                if entity["type"] == "method" and "parent_class" in entity:
                    parent_class_id = entity.get("parent_class")
                    if parent_class_id is not None and parent_class_id in entities:
                        class_entity = entities[parent_class_id]
                        qualified_name = f"{class_entity['name']}.{entity['name']}"
                        name_to_id[qualified_name] = entity_id
        
        # Process call relationships
        for entity_id, entity in entities.items():
            # Process CONTAINS relationships
            if "contains" in entity and entity["contains"]:
                for contained_id in entity["contains"]:
                    relationships.append({
                        "type": "CONTAINS",
                        "from_entity": entity_id,
                        "to_entity": contained_id,
                        "weight": 1.0
                    })
            
            # Process CALLS relationships
            if entity["type"] in ("function", "method") and "calls" in entity:
                for called_name in entity["calls"]:
                    if called_name in name_to_id:
                        relationships.append({
                            "type": "CALLS",
                            "from_entity": entity_id,
                            "to_entity": name_to_id[called_name],
                            "weight": 0.8
                        })
            
            # Process EXTENDS relationships
            if entity["type"] == "class" and "bases" in entity:
                for base_name in entity["bases"]:
                    if base_name in name_to_id:
                        relationships.append({
                            "type": "EXTENDS",
                            "from_entity": entity_id,
                            "to_entity": name_to_id[base_name],
                            "weight": 0.7
                        })
                        
        return relationships


class EntityExtractor(ast.NodeVisitor):
    """AST Visitor that extracts all relevant code entities."""
    
    def __init__(self, file_path: str, entities: Dict[str, Dict[str, Any]], parent_id: str):
        """Initialize the extractor.
        
        Args:
            file_path: Path to the Python file
            entities: Dictionary to populate
            parent_id: ID of the parent entity
        """
        self.file_path = file_path
        self.entities = entities
        self.parent_id = parent_id
        self.current_class: Optional[str] = None
        self.call_finder = CallFinder()
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Extract class definition.
        
        Args:
            node: Class definition node
        """
        # Generate stable ID
        name_path = f"{self.file_path}:{node.name}"
        class_id = f"class_{hashlib.md5(name_path.encode()).hexdigest()[:8]}"
        
        # Extract base classes
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(self._get_attribute_path(base))
        
        # Create class entity
        self.entities[class_id] = {
            "type": "class",
            "name": node.name,
            "path": self.file_path,
            "line_start": node.lineno,
            "line_end": self._get_end_line(node),
            "docstring": ast.get_docstring(node),
            "bases": bases,
            "methods": [],  # Will be filled by method visits
            "contains": []  # Will be filled by method visits
        }
        
        # Add class to parent's contains list
        parent_entity = self.entities.get(self.parent_id)
        if parent_entity:
            parent_entity["contains"].append(class_id)
        
        # Save previous class context and set current
        prev_class = self.current_class
        self.current_class = class_id
        
        # Visit all children with this class as parent
        prev_parent = self.parent_id
        self.parent_id = class_id
        for child in node.body:
            self.visit(child)
        
        # Restore context
        self.parent_id = prev_parent
        self.current_class = prev_class
        
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Extract function or method definition.
        
        Args:
            node: Function definition node
        """
        # Determine if this is a method or standalone function
        is_method = self.current_class is not None
        entity_type = "method" if is_method else "function"
        
        # Generate stable ID
        if is_method and self.current_class is not None:
            class_name = self.entities[self.current_class]["name"]
            name_path = f"{self.file_path}:{class_name}.{node.name}"
        else:
            name_path = f"{self.file_path}:{node.name}"
        func_id = f"{entity_type}_{hashlib.md5(name_path.encode()).hexdigest()[:8]}"
        
        # Extract arguments
        args = [arg.arg for arg in node.args.args]
        
        # Extract decorator names
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                decorators.append(self._get_attribute_path(decorator))
        
        # Extract return annotation if present
        returns = None
        if node.returns:
            if isinstance(node.returns, ast.Name):
                returns = node.returns.id
            elif isinstance(node.returns, ast.Attribute):
                returns = self._get_attribute_path(node.returns)
        
        # Find function calls
        self.call_finder.calls = []
        self.call_finder.visit(node)
        
        # Create function entity
        self.entities[func_id] = {
            "type": entity_type,
            "name": node.name,
            "path": self.file_path,
            "line_start": node.lineno,
            "line_end": self._get_end_line(node),
            "docstring": ast.get_docstring(node),
            "args": args,
            "decorators": decorators,
            "returns": returns,
            "calls": self.call_finder.calls
        }
        
        # If this is a method, add reference to parent class
        if is_method and self.current_class is not None:
            self.entities[func_id]["parent_class"] = self.current_class
            
            # Add to class's methods list
            class_entity = self.entities.get(self.current_class)
            if class_entity:
                class_entity["methods"].append(node.name)
                class_entity["contains"].append(func_id)
        else:
            # Add function to module's contains list
            parent_entity = self.entities.get(self.parent_id)
            if parent_entity:
                parent_entity["contains"].append(func_id)
    
    def visit_Import(self, node: ast.Import) -> None:
        """Extract import statement.
        
        Args:
            node: Import node
        """
        for name in node.names:
            # Generate stable ID
            import_path = f"{self.file_path}:import:{name.name}"
            import_id = f"import_{hashlib.md5(import_path.encode()).hexdigest()[:8]}"
            
            # Create import entity
            self.entities[import_id] = {
                "type": "import",
                "name": name.name,
                "asname": name.asname,
                "path": self.file_path,
                "line": node.lineno
            }
            
            # Add to parent's contains list
            parent_entity = self.entities.get(self.parent_id)
            if parent_entity:
                parent_entity["contains"].append(import_id)
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Extract import from statement.
        
        Args:
            node: ImportFrom node
        """
        module = node.module or ""
        for name in node.names:
            # Generate stable ID
            import_path = f"{self.file_path}:importfrom:{module}.{name.name}"
            import_id = f"import_{hashlib.md5(import_path.encode()).hexdigest()[:8]}"
            
            # Create import entity
            self.entities[import_id] = {
                "type": "importfrom",
                "name": name.name,
                "asname": name.asname,
                "module": module,
                "path": self.file_path,
                "line": node.lineno
            }
            
            # Add to parent's contains list
            parent_entity = self.entities.get(self.parent_id)
            if parent_entity:
                parent_entity["contains"].append(import_id)
    
    def _get_attribute_path(self, node: ast.Attribute) -> str:
        """Convert an Attribute node to a dotted path string.
        
        Args:
            node: Attribute node
            
        Returns:
            Dotted path as string
        """
        parts = []
        
        current: Any = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
            
        if isinstance(current, ast.Name):
            parts.append(current.id)
            
        return ".".join(reversed(parts))
    
    def _get_end_line(self, node: ast.AST) -> int:
        """Get the end line number of an AST node.
        
        Args:
            node: AST node
            
        Returns:
            End line number
        """
        if hasattr(node, 'end_lineno') and node.end_lineno is not None:
            return cast(int, node.end_lineno)
            
        # If end_lineno not available, use lineno of last child or current line
        if not hasattr(node, 'lineno'):
            return 0
            
        max_line = cast(int, node.lineno)
        for child in ast.iter_child_nodes(node):
            if hasattr(child, 'lineno'):
                child_end = self._get_end_line(child)
                max_line = max(max_line, child_end)
        
        return max_line


class CallFinder(ast.NodeVisitor):
    """Find all function calls in an AST node."""
    
    def __init__(self) -> None:
        """Initialize the call finder."""
        self.calls: List[str] = []
    
    def visit_Call(self, node: ast.Call) -> None:
        """Record a function call.
        
        Args:
            node: Call node
        """
        # Handle simple name calls
        if isinstance(node.func, ast.Name):
            self.calls.append(node.func.id)
        # Handle attribute calls (obj.method())
        elif isinstance(node.func, ast.Attribute):
            # Only record the method name
            self.calls.append(node.func.attr)
            # Also record the full path if it's a direct attribute
            if isinstance(node.func.value, ast.Name):
                self.calls.append(f"{node.func.value.id}.{node.func.attr}")
                
        # Visit all child nodes
        self.generic_visit(node)


# Register the Python adapter for the specific format
from .registry import register_adapter
register_adapter('python', PythonAdapter)
