"""
Python code file adapter for HADES-PathRAG.

This adapter specializes in processing Python code files, using Python's
built-in AST module to extract information about the code structure,
such as functions, classes, and their relationships.
"""

import ast
import hashlib
import logging
import uuid 
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, cast
from copy import deepcopy 
from datetime import datetime 

try:
    import black
    BLACK_AVAILABLE = True
except ImportError:
    black = None  # type: ignore
    BLACK_AVAILABLE = False

from src.docproc.adapters.base import BaseAdapter
from src.types.docproc.adapter import (
    ExtractorOptions,
    MetadataExtractionConfig,
    EntityExtractionConfig
)
from src.types.docproc.document import (
    ProcessedDocument,
    DocumentEntity,
    DocumentMetadata 
)
from src.types.docproc.enums import ContentCategory
from src.types.docproc.formats.python import PythonParserResult, FunctionInfo, ClassInfo, ImportInfo
from .registry import register_adapter 
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
    
    def process(self, file_path: Union[str, Path], options: Optional[ExtractorOptions] = None) -> ProcessedDocument:  
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
            raw_python_content = f.read()
        
        # Process with the wrapped python_adapter
        inner_doc: ProcessedDocument = self.python_adapter.process_text(raw_python_content, options)

        doc_id = f"pycode_{hashlib.md5(str(path_obj).encode()).hexdigest()[:8]}_{path_obj.stem}"

        final_doc_data: Dict[str, Any] = {
            "id": doc_id,
            "content": inner_doc.get("content", raw_python_content), 
            "raw_content": raw_python_content,
            "content_type": inner_doc.get("content_type", "text/x-python"),
            "format": inner_doc.get("format", "python"),
            "content_category": ContentCategory.CODE, 
            "metadata": {
                **(inner_doc.get("metadata", {})),
                "title": inner_doc.get("metadata", {}).get("title", f"Python Code Analysis: {path_obj.name}"),
                "path": str(path_obj),
                "filename": path_obj.name,
                "extension": path_obj.suffix,
                "language": "python",
                "custom": {
                    **(inner_doc.get("metadata", {}).get("custom", {})),
                    "adapter_type": "PythonCodeAdapter", 
                    "original_doc_id_from_inner_adapter": inner_doc.get("id")
                }
            },
            "entities": inner_doc.get("entities", []),
            "sections": inner_doc.get("sections", []),
            "error": inner_doc.get("error")
        }

        if "source" in ProcessedDocument.__annotations__:
            final_doc_data["source"] = str(path_obj)
            
        return cast(ProcessedDocument, final_doc_data)
    
    def process_text(self, text: str, options: Optional[ExtractorOptions] = None) -> ProcessedDocument:  
        """Process Python code text content.
        
        Args:
            text: Python code content
            options: Optional processing options
            
        Returns:
            Processed document with metadata and content
        """
        inner_doc: ProcessedDocument = self.python_adapter.process_text(text, options)

        doc_id = f"pycode_text_{hashlib.md5(text.encode()).hexdigest()[:8]}_{str(uuid.uuid4())[:4]}"

        final_doc_data: Dict[str, Any] = {
            "id": doc_id,
            "content": inner_doc.get("content", text), 
            "raw_content": text, 
            "content_type": inner_doc.get("content_type", "text/x-python"),
            "format": inner_doc.get("format", "python"),
            "content_category": ContentCategory.CODE, 
            "metadata": {
                **(inner_doc.get("metadata", {})),
                "title": inner_doc.get("metadata", {}).get("title", f"Python Code Analysis (text input {doc_id})"),
                "language": "python", 
                "custom": {
                    **(inner_doc.get("metadata", {}).get("custom", {})),
                    "adapter_type": "PythonCodeAdapter",
                    "original_doc_id_from_inner_adapter": inner_doc.get("id"),
                    "line_count": inner_doc.get("metadata", {}).get("custom", {}).get("line_count", len(text.split('\n'))),
                    "char_count": inner_doc.get("metadata", {}).get("custom", {}).get("char_count", len(text))
                }
            },
            "entities": inner_doc.get("entities", []), 
            "sections": inner_doc.get("sections", []), 
            "error": inner_doc.get("error")     
        }
        
        if "source" in ProcessedDocument.__annotations__ and options and options.get("source_uri"):
            final_doc_data["source"] = str(options.get("source_uri"))

        return cast(ProcessedDocument, final_doc_data)


    def extract_entities(self, content: Union[str, Path, ProcessedDocument], options: Optional[EntityExtractionConfig] = None) -> List[DocumentEntity]:  # type: ignore[override]
        """
        Extract entities from Python code content.
        Relies on the entities already extracted by the wrapped python_adapter.

        Args:
            content: Python code content as string, path, or processed document.
            options: Optional extraction configuration.

        Returns:
            List of document entities.
        """
        inner_doc: Optional[ProcessedDocument] = None
        extractor_options = cast(Optional[ExtractorOptions], options)

        if isinstance(content, ProcessedDocument):
            inner_doc = content
        elif isinstance(content, str):
            try:
                inner_doc = self.python_adapter.process_text(content, extractor_options)
            except Exception as e:
                logger.error(f"Error processing text for entity extraction: {e}")
                return []
        elif isinstance(content, Path):
            try:
                inner_doc = self.python_adapter.process(content, extractor_options)
            except FileNotFoundError:
                logger.error(f"File not found for entity extraction: {content}")
                return []
            except Exception as e:
                logger.error(f"Error processing file for entity extraction {content}: {e}")
                return []
        else:
            logger.warning(f"Unsupported content type for extract_entities: {type(content)}")
            return []

        if inner_doc:
            return inner_doc.get("entities", [])
        return []

    def extract_metadata(self, content: Union[str, Path, ProcessedDocument], options: Optional[MetadataExtractionConfig] = None) -> DocumentMetadata:  # type: ignore[override]
        """
        Extract metadata from Python code content.
        Leverages metadata from the wrapped python_adapter and augments it.

        Args:
            content: Python code content as string, path, or processed document.
            options: Optional extraction configuration.

        Returns:
            Document metadata dictionary.
        """
        base_doc_metadata: Dict[str, Any] = {}
        doc_id_for_title: Optional[str] = None
        raw_text_for_counts: Optional[str] = None
        file_path_for_metadata: Optional[Path] = None

        extractor_options = cast(Optional[ExtractorOptions], options)

        if isinstance(content, ProcessedDocument):
            base_doc_metadata = deepcopy(content.get("metadata", {}))
            doc_id_for_title = content.get("id")
            raw_text_for_counts = content.get("raw_content", content.get("content"))
            if content.get("metadata", {}).get("path"):
                file_path_for_metadata = Path(content["metadata"]["path"])
        elif isinstance(content, str):
            raw_text_for_counts = content
            try:
                inner_doc = self.python_adapter.process_text(content, extractor_options)
                base_doc_metadata = deepcopy(inner_doc.get("metadata", {}))
                doc_id_for_title = inner_doc.get("id")
            except Exception as e:
                logger.error(f"Error processing text for metadata extraction: {e}")
                return cast(DocumentMetadata, {
                    "title": "Error: Processing text failed", "language": "python",
                    "content_category": ContentCategory.CODE, "custom": {"error": f"Processing error: {e}"}
                })
        elif isinstance(content, Path):
            file_path_for_metadata = content
            try:
                with open(content, "r", encoding="utf-8") as f:
                    raw_text_for_counts = f.read()
                inner_doc = self.python_adapter.process(content, extractor_options)
                base_doc_metadata = deepcopy(inner_doc.get("metadata", {}))
                doc_id_for_title = inner_doc.get("id")
            except FileNotFoundError:
                logger.error(f"File not found for metadata extraction: {content}")
                return cast(DocumentMetadata, {
                    "title": f"Error: File not found {content.name}", "language": "python",
                    "content_category": ContentCategory.CODE, "path": str(content.resolve()), "filename": content.name,
                    "custom": {"error": f"File not found: {content}"}
                })
            except Exception as e:
                logger.error(f"Error processing file for metadata extraction {content}: {e}")
                return cast(DocumentMetadata, {
                    "title": f"Error processing file {content.name}", "language": "python",
                    "content_category": ContentCategory.CODE, "path": str(content.resolve()), "filename": content.name,
                    "custom": {"error": f"Processing error: {e}"}
                })
        else:
            logger.warning(f"Unsupported content type for extract_metadata: {type(content)}")
            return cast(DocumentMetadata, {
                "title": "Error: Unsupported content type", "language": "python",
                "content_category": ContentCategory.CODE,
                "custom": {"error": f"Unsupported content type: {type(content)}"}
            })

        if "custom" not in base_doc_metadata:
            base_doc_metadata["custom"] = {}

        final_metadata: Dict[str, Any] = deepcopy(base_doc_metadata)
        
        final_metadata["language"] = "python"
        final_metadata["content_category"] = ContentCategory.CODE

        if not final_metadata.get("title"):
            if file_path_for_metadata:
                final_metadata["title"] = file_path_for_metadata.name
            elif doc_id_for_title:
                final_metadata["title"] = f"Python Analysis {doc_id_for_title}"
            else:
                final_metadata["title"] = "Python Code Analysis"

        if file_path_for_metadata:
            if "path" not in final_metadata:
                final_metadata["path"] = str(file_path_for_metadata.resolve())
            if "filename" not in final_metadata:
                final_metadata["filename"] = file_path_for_metadata.name
            if "extension" not in final_metadata:
                final_metadata["extension"] = file_path_for_metadata.suffix
        
        if "custom" not in final_metadata:
            final_metadata["custom"] = {}
            
        final_metadata["custom"]["adapter_type"] = "PythonCodeAdapter"
        final_metadata["custom"]["adapter_version"] = self.options.get("version", "0.1.0")
        final_metadata["custom"]["processing_timestamp"] = datetime.now().isoformat()
        if doc_id_for_title and not isinstance(content, ProcessedDocument): 
             final_metadata["custom"]["original_doc_id_from_inner_adapter"] = doc_id_for_title

        if raw_text_for_counts:
            if "line_count" not in final_metadata.get("custom", {}):
                 final_metadata["custom"]["line_count"] = len(raw_text_for_counts.splitlines())
            if "character_count" not in final_metadata and "character_count" not in final_metadata.get("custom", {}):
                 final_metadata["custom"]["character_count"] = len(raw_text_for_counts)

        return cast(DocumentMetadata, final_metadata)

# Register the adapter for Python code files
register_adapter('python', PythonCodeAdapter)
