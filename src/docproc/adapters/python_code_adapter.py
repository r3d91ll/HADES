"""
Python code file adapter for HADES-PathRAG.

This adapter specializes in processing Python code files, using Python's
built-in AST module to extract information about the code structure,
such as functions, classes, and their relationships.
"""

from typing import Any, Dict, List, Optional, Union, cast
from pathlib import Path
import logging
import ast
import json
import re
import hashlib

from src.docproc.adapters.base import BaseAdapter
from src.docproc.adapters.registry import register_adapter
from src.docproc.utils.code_analysis import analyze_python_code
from src.types.docproc.adapter import AdapterOptions, ProcessedDocument
from src.types.docproc.metadata import MetadataDict, EntityDict
from .python_adapter import PythonAdapter

logger = logging.getLogger(__name__)


class PythonCodeAdapter(BaseAdapter):
    """Adapter specialized for Python code files.
    
    This adapter processes Python source code files using Python's AST module,
    extracting meaningful information about the code structure and relationships.
    """
    
    def __init__(self, options: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the Python code adapter.
        
        Args:
            options: Optional configuration options for the adapter
        """
        super().__init__()
        self.options = options or {}
        
        # Create underlying Python adapter for AST processing
        self.python_adapter = PythonAdapter(
            create_symbol_table=True,
            options={
                "extract_docstrings": True,
                "analyze_imports": True,
                "analyze_calls": True,
                "extract_type_hints": True,
                "compute_complexity": True
            }
        )
        
        logger.info("PythonCodeAdapter initialized")
    
    def process(self, file_path: Union[str, Path], options: Optional[AdapterOptions] = None) -> ProcessedDocument:
        """Process a Python code file.
        
        Args:
            file_path: Path to the Python file
            options: Optional processing options
            
        Returns:
            Processed document with metadata and content
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file isn't a valid Python file
        """
        path_obj = Path(file_path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if path_obj.suffix.lower() != ".py":
            raise ValueError(f"Not a Python file: {file_path}")
        
        # Read file content
        with open(path_obj, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Process with python_adapter
        processed = cast(Dict[str, Any], self.python_adapter.process_text(content))
        
        # Create a document ID
        file_hash = hashlib.md5(str(path_obj).encode()).hexdigest()[:8]
        doc_id = f"python_{file_hash}_{path_obj.stem}"
        
        # Build document structure
        document: ProcessedDocument = {
            "id": doc_id,
            "source": str(path_obj),
            "content": content,
            "format": "python",  # Required field in ProcessedDocument
            "metadata": {
                "path": str(path_obj),
                "filename": path_obj.name,
                "extension": path_obj.suffix,
                "language": "python",
                "code_analysis": processed
            },
            "entities": [],  # Required field in ProcessedDocument
            "processing_time": 0.0,  # Required field
            "error": None  # Required field
        }
        
        return document
    
    def process_text(self, text: str, format_type: str = "python", options: Optional[AdapterOptions] = None) -> ProcessedDocument:
        """Process Python code text content.
        
        Args:
            text: Python code content
            options: Optional processing options
            
        Returns:
            Processed document with metadata and content
        """
        # Process with python_adapter
        processed = cast(Dict[str, Any], self.python_adapter.process_text(text))
        
        # Create a document ID
        content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        doc_id = f"python_{content_hash}"
        
        # Build document structure
        document: ProcessedDocument = {
            "id": doc_id,
            "source": "text_content",
            "content": text,
            "format": "python",  # Required field in ProcessedDocument
            "metadata": {
                "language": "python",
                "code_analysis": processed
            },
            "entities": [],  # Required field in ProcessedDocument
            "processing_time": 0.0,  # Required field
            "error": None  # Required field
        }
        
        return document


    def extract_entities(self, content: Union[str, Dict[str, Any]], options: Optional[AdapterOptions] = None) -> List[EntityDict]:
        """
        Extract entities from Python code content.
        
        Args:
            content: Python code content as string or parsed data
            options: Optional processing options
            
        Returns:
            List of extracted entities with metadata
        """
        # If content is already processed data, use it directly
        if isinstance(content, dict) :
            processed = content
        else:
            # Process the text content
            processed = cast(Dict[str, Any], self.python_adapter.process_text(content))
        
        # Extract entities from processed data
        code_analysis = processed.get("code_analysis", {})
        
        # Return entities
        entities: List[Any] = []
        
        # Add module entity
        module_info = code_analysis.get("module", {})
        if module_info:
            entities.append({
                "type": "module",
                "name": module_info.get("name", "unknown"),
                "content": module_info.get("docstring", ""),
                "metadata": {
                    "imports": module_info.get("imports", []),
                    "line_count": module_info.get("line_count", 0)
                }
            })
        
        # Add function entities
        for func_name, func_info in code_analysis.get("functions", {}).items():
            entities.append({
                "type": "function",
                "name": func_name,
                "content": func_info.get("docstring", "") or func_info.get("source", ""),
                "metadata": {
                    "args": func_info.get("args", []),
                    "returns": func_info.get("returns", None),
                    "line_start": func_info.get("line_start", 0),
                    "line_end": func_info.get("line_end", 0),
                    "calls": func_info.get("calls", [])
                }
            })
        
        # Add class entities
        for class_name, class_info in code_analysis.get("classes", {}).items():
            entities.append({
                "type": "class",
                "name": class_name,
                "content": class_info.get("docstring", "") or class_info.get("source", ""),
                "metadata": {
                    "bases": class_info.get("bases", []),
                    "methods": class_info.get("methods", []),
                    "line_start": class_info.get("line_start", 0),
                    "line_end": class_info.get("line_end", 0)
                }
            })
        
        return entities

    def extract_metadata(self, content: Union[str, Dict[str, Any]], options: Optional[AdapterOptions] = None) -> MetadataDict:
        """
        Extract metadata from Python code content.
        
        Args:
            content: Python code content as string or parsed data
            options: Optional processing options
            
        Returns:
            Dictionary of metadata
        """
        # If content is already processed data, use it directly
        if isinstance(content, dict) :
            processed = content
        else:
            # Process the text content
            processed = cast(Dict[str, Any], self.python_adapter.process_text(content))
        
        # Extract metadata from processed data
        code_analysis = processed.get("code_analysis", {})
        
        # Build metadata object
        metadata: MetadataDict = {
            "language": "python",
            "document_type": "code",
            "code_type": "python"
        }
        
        # Extract module info
        module_info = code_analysis.get("module", {})
        if module_info:
            metadata["module_name"] = module_info.get("name", "unknown")
            metadata["imports"] = module_info.get("imports", [])
            metadata["line_count"] = module_info.get("line_count", 0)
        
        # Count code entities
        metadata["function_count"] = str(len(code_analysis.get("functions", {})))
        metadata["class_count"] = str(len(code_analysis.get("classes", {})))
        metadata["import_count"] = str(len(module_info.get("imports", [])))
        
        return metadata


# Register the adapter for Python code files
register_adapter('python', PythonCodeAdapter)
