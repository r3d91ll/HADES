"""
Python AST Document Processor

This module provides AST-based Python document processing that extracts
symbol tables, code elements, and structural information from Python source files.
"""

import ast
import logging
import psutil
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone
from pathlib import Path

# Import component contracts and protocols
from src.types.components.contracts import (
    ComponentType,
    ComponentMetadata,
    DocumentProcessingInput,
    DocumentProcessingOutput,
    ProcessedDocument,
    ProcessingStatus,
    ContentCategory
)
from src.types.components.protocols import DocumentProcessor

# Import Python-specific types
from src.types.components.docproc.python import (
    PythonDocument,
    PythonMetadata,
    PythonEntity,
    SymbolTable,
    FunctionElement,
    MethodElement,
    ClassElement,
    ImportElement,
    CodeRelationship
)
from src.types.components.docproc.enums import AccessLevel


class PythonASTProcessor(DocumentProcessor):
    """
    Python AST-based document processor implementing DocumentProcessor protocol.
    
    This component uses Python's AST module to parse Python source files and
    extract symbol tables, code elements, and structural information.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Python AST processor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self._config = config or {}
        
        # Component metadata
        self._metadata = ComponentMetadata(
            component_type=ComponentType.DOCPROC,
            component_name="python_ast",
            component_version="1.0.0",
            config=self._config
        )
        
        # Configuration options
        self._include_docstrings = self._config.get('include_docstrings', True)
        self._include_private_members = self._config.get('include_private_members', True)
        self._max_line_length = self._config.get('max_line_length', 10000)
        self._extract_relationships_enabled = self._config.get('extract_relationships', True)
        
        # Monitoring and metrics tracking
        self._stats: Dict[str, Any] = {
            "total_files_processed": 0,
            "successful_files": 0,
            "failed_files": 0,
            "total_processing_time": 0.0,
            "total_functions_extracted": 0,
            "total_classes_extracted": 0,
            "total_imports_extracted": 0,
            "last_processing_time": None,
            "initialization_count": 1,
            "errors": []
        }
        self._startup_time = datetime.now(timezone.utc)
        
        self.logger.info("Initialized Python AST processor")
    
    @property
    def name(self) -> str:
        """Component name for identification."""
        return "python_ast"
    
    @property
    def version(self) -> str:
        """Component version string."""
        return "1.0.0"
    
    @property
    def component_type(self) -> ComponentType:
        """Type of component."""
        return ComponentType.DOCPROC
    
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure component with parameters.
        
        Args:
            config: Configuration dictionary containing component parameters
        """
        self._config.update(config)
        self._metadata.config = self._config
        self._metadata.processed_at = datetime.now(timezone.utc)
        
        # Update configuration parameters
        self._include_docstrings = self._config.get('include_docstrings', True)
        self._include_private_members = self._config.get('include_private_members', True)
        self._max_line_length = self._config.get('max_line_length', 10000)
        self._extract_relationships_enabled = self._config.get('extract_relationships', True)
        
        self.logger.info("Updated Python AST processor configuration")
    
    def validate_config(self, config: Any) -> bool:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        if not isinstance(config, dict):
            return False
        
        # Validate boolean options
        bool_options = ['include_docstrings', 'include_private_members', 'extract_relationships']
        for option in bool_options:
            if option in config and not isinstance(config[option], bool):
                return False
        
        # Validate integer options
        if 'max_line_length' in config:
            if not isinstance(config['max_line_length'], int) or config['max_line_length'] < 1:
                return False
        
        return True
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get the JSON schema for component configuration.
        
        Returns:
            JSON schema dictionary describing valid configuration
        """
        return {
            "type": "object",
            "properties": {
                "include_docstrings": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to extract docstrings from code elements"
                },
                "include_private_members": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to include private/protected members"
                },
                "max_line_length": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 10000,
                    "description": "Maximum line length to process"
                },
                "extract_relationships": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to extract code relationships"
                }
            }
        }
    
    def health_check(self) -> bool:
        """
        Check if component is healthy and ready to process data.
        
        Returns:
            True if component is healthy, False otherwise
        """
        try:
            # Test basic AST parsing functionality
            test_code = """
def test_function():
    \"\"\"Test function.\"\"\"
    return True

class TestClass:
    \"\"\"Test class.\"\"\"
    def test_method(self):
        pass
"""
            tree = ast.parse(test_code)
            return tree is not None
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def get_infrastructure_metrics(self) -> Dict[str, Any]:
        """
        Get infrastructure and resource inventory metrics.
        
        Returns:
            Dictionary containing infrastructure metrics
        """
        try:
            # Get memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "component_name": self.name,
                "component_version": self.version,
                "include_docstrings": self._include_docstrings,
                "include_private_members": self._include_private_members,
                "max_line_length": self._max_line_length,
                "extract_relationships": self._extract_relationships_enabled,
                "memory_usage": {
                    "rss_mb": round(memory_info.rss / 1024 / 1024, 2),
                    "vms_mb": round(memory_info.vms / 1024 / 1024, 2)
                },
                "startup_time": self._startup_time.isoformat(),
                "uptime_seconds": (datetime.now(timezone.utc) - self._startup_time).total_seconds()
            }
        except Exception as e:
            self.logger.error(f"Failed to get infrastructure metrics: {e}")
            return {"error": str(e)}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get runtime performance statistics.
        
        Returns:
            Dictionary containing performance metrics
        """
        try:
            current_time = datetime.now(timezone.utc)
            uptime = (current_time - self._startup_time).total_seconds()
            
            # Calculate rates
            successful_files = int(self._stats["successful_files"])
            total_files = int(self._stats["total_files_processed"])
            total_time = float(self._stats["total_processing_time"])
            
            files_per_second = successful_files / max(uptime, 1.0)
            avg_processing_time = total_time / max(successful_files, 1)
            success_rate = successful_files / max(total_files, 1) * 100
            
            return {
                "component_name": self.name,
                "total_files_processed": self._stats["total_files_processed"],
                "successful_files": self._stats["successful_files"],
                "failed_files": self._stats["failed_files"],
                "success_rate_percent": round(success_rate, 2),
                "files_per_second": round(files_per_second, 3),
                "average_processing_time": round(avg_processing_time, 3),
                "total_processing_time": round(total_time, 3),
                "total_functions_extracted": self._stats["total_functions_extracted"],
                "total_classes_extracted": self._stats["total_classes_extracted"],
                "total_imports_extracted": self._stats["total_imports_extracted"],
                "last_processing_time": self._stats["last_processing_time"],
                "initialization_count": self._stats["initialization_count"],
                "recent_errors": list(self._stats["errors"])[-5:],  # Last 5 errors
                "error_count": len(list(self._stats["errors"])),
                "uptime_seconds": round(uptime, 2)
            }
        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {e}")
            return {"error": str(e)}
    
    def export_metrics_prometheus(self) -> str:
        """
        Export metrics in Prometheus format.
        
        Returns:
            Prometheus-compatible metrics string
        """
        try:
            infra_metrics = self.get_infrastructure_metrics()
            perf_metrics = self.get_performance_metrics()
            
            metrics_lines = []
            
            # Infrastructure metrics
            metrics_lines.append(f"# HELP hades_component_uptime_seconds Component uptime in seconds")
            metrics_lines.append(f"# TYPE hades_component_uptime_seconds gauge")
            metrics_lines.append(f'hades_component_uptime_seconds{{component="python_ast_processor"}} {infra_metrics.get("uptime_seconds", 0)}')
            
            metrics_lines.append(f"# HELP hades_component_memory_rss_mb Memory RSS usage in MB")
            metrics_lines.append(f"# TYPE hades_component_memory_rss_mb gauge")
            memory_rss = infra_metrics.get("memory_usage", {}).get("rss_mb", 0)
            metrics_lines.append(f'hades_component_memory_rss_mb{{component="python_ast_processor"}} {memory_rss}')
            
            # Performance metrics
            metrics_lines.append(f"# HELP hades_component_files_total Total number of processed files")
            metrics_lines.append(f"# TYPE hades_component_files_total counter")
            metrics_lines.append(f'hades_component_files_total{{component="python_ast_processor"}} {perf_metrics.get("total_files_processed", 0)}')
            
            metrics_lines.append(f"# HELP hades_component_files_successful_total Total number of successful files")
            metrics_lines.append(f"# TYPE hades_component_files_successful_total counter")
            metrics_lines.append(f'hades_component_files_successful_total{{component="python_ast_processor"}} {perf_metrics.get("successful_files", 0)}')
            
            metrics_lines.append(f"# HELP hades_component_files_failed_total Total number of failed files")
            metrics_lines.append(f"# TYPE hades_component_files_failed_total counter")
            metrics_lines.append(f'hades_component_files_failed_total{{component="python_ast_processor"}} {perf_metrics.get("failed_files", 0)}')
            
            metrics_lines.append(f"# HELP hades_component_success_rate_percent Success rate percentage")
            metrics_lines.append(f"# TYPE hades_component_success_rate_percent gauge")
            metrics_lines.append(f'hades_component_success_rate_percent{{component="python_ast_processor"}} {perf_metrics.get("success_rate_percent", 0)}')
            
            metrics_lines.append(f"# HELP hades_component_files_per_second Files processed per second")
            metrics_lines.append(f"# TYPE hades_component_files_per_second gauge")
            metrics_lines.append(f'hades_component_files_per_second{{component="python_ast_processor"}} {perf_metrics.get("files_per_second", 0)}')
            
            # Code element extraction metrics
            metrics_lines.append(f"# HELP hades_component_functions_extracted_total Total functions extracted")
            metrics_lines.append(f"# TYPE hades_component_functions_extracted_total counter")
            metrics_lines.append(f'hades_component_functions_extracted_total{{component="python_ast_processor"}} {perf_metrics.get("total_functions_extracted", 0)}')
            
            metrics_lines.append(f"# HELP hades_component_classes_extracted_total Total classes extracted")
            metrics_lines.append(f"# TYPE hades_component_classes_extracted_total counter")
            metrics_lines.append(f'hades_component_classes_extracted_total{{component="python_ast_processor"}} {perf_metrics.get("total_classes_extracted", 0)}')
            
            metrics_lines.append(f"# HELP hades_component_imports_extracted_total Total imports extracted")
            metrics_lines.append(f"# TYPE hades_component_imports_extracted_total counter")
            metrics_lines.append(f'hades_component_imports_extracted_total{{component="python_ast_processor"}} {perf_metrics.get("total_imports_extracted", 0)}')
            
            return "\\n".join(metrics_lines) + "\\n"
            
        except Exception as e:
            self.logger.error(f"Failed to export Prometheus metrics: {e}")
            return f"# Error exporting metrics: {str(e)}\\n"
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported file formats.
        
        Returns:
            List of supported file extensions
        """
        return ['.py', '.pyx', '.pyi']
    
    def can_process(self, file_path: str) -> bool:
        """
        Check if this processor can handle the given file.
        
        Args:
            file_path: Path to file to check
            
        Returns:
            True if file can be processed, False otherwise
        """
        file_path_obj = Path(file_path)
        return file_path_obj.suffix.lower() in self.get_supported_formats()
    
    def estimate_processing_time(self, input_data: DocumentProcessingInput) -> float:
        """
        Estimate processing time for given input.
        
        Args:
            input_data: Input data to estimate for
            
        Returns:
            Estimated processing time in seconds
        """
        try:
            # Rough estimation based on file size
            if input_data.content:
                content_size = len(input_data.content)
            else:
                file_path = Path(input_data.file_path)
                if file_path.exists():
                    content_size = file_path.stat().st_size
                else:
                    content_size = 1000  # Default estimate
            
            # Estimate ~1000 characters per second for AST processing
            estimated_time = content_size / 1000.0
            return max(0.1, estimated_time)  # Minimum 0.1 seconds
            
        except Exception:
            return 1.0  # Default estimate
    
    def process(self, input_data: DocumentProcessingInput) -> DocumentProcessingOutput:
        """
        Process documents according to the contract (alias for process_documents).
        
        Args:
            input_data: Input data conforming to DocumentProcessingInput contract
            
        Returns:
            Output data conforming to DocumentProcessingOutput contract
        """
        return self.process_documents(input_data)
    
    def process_batch(self, input_batch: List[DocumentProcessingInput]) -> List[DocumentProcessingOutput]:
        """
        Process multiple documents in batch.
        
        Args:
            input_batch: List of input data
            
        Returns:
            List of output data
        """
        results = []
        for input_data in input_batch:
            try:
                result = self.process_documents(input_data)
                results.append(result)
            except Exception as e:
                # Create error result for failed processing
                error_msg = f"Batch processing failed for {input_data.file_path}: {str(e)}"
                self.logger.error(error_msg)
                
                metadata = ComponentMetadata(
                    component_type=self.component_type,
                    component_name=self.name,
                    component_version=self.version,
                    processed_at=datetime.now(timezone.utc),
                    config=self._config,
                    status=ProcessingStatus.ERROR
                )
                
                error_result = DocumentProcessingOutput(
                    documents=[],
                    metadata=metadata,
                    processing_stats={},
                    errors=[error_msg],
                    total_processed=0,
                    total_errors=1
                )
                results.append(error_result)
        
        return results
    
    def process_documents(self, input_data: DocumentProcessingInput) -> DocumentProcessingOutput:
        """
        Process Python documents using AST analysis.
        
        Args:
            input_data: Input data conforming to DocumentProcessingInput contract
            
        Returns:
            Output data conforming to DocumentProcessingOutput contract
        """
        try:
            start_time = datetime.now(timezone.utc)
            
            # Update request statistics
            self._stats["last_processing_time"] = start_time.isoformat()
            self._stats["total_files_processed"] += 1
            
            # Read file content
            file_path = Path(input_data.file_path)
            if input_data.content:
                content = input_data.content
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            # Parse Python code
            python_doc = self._parse_python_file(file_path, content)
            
            # Convert to ProcessedDocument format
            processed_doc = self._convert_to_processed_document(python_doc, file_path, content)
            
            # Calculate processing time
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            processed_doc.processing_time = processing_time
            
            # Update statistics
            self._stats["successful_files"] += 1
            self._stats["total_processing_time"] += processing_time
            
            if python_doc.symbol_table:
                self._stats["total_functions_extracted"] += python_doc.metadata.function_count
                self._stats["total_classes_extracted"] += python_doc.metadata.class_count
                self._stats["total_imports_extracted"] += python_doc.metadata.import_count
            
            # Create output metadata
            metadata = ComponentMetadata(
                component_type=self.component_type,
                component_name=self.name,
                component_version=self.version,
                processing_time=processing_time,
                processed_at=datetime.now(timezone.utc),
                config=self._config,
                status=ProcessingStatus.SUCCESS
            )
            
            return DocumentProcessingOutput(
                documents=[processed_doc],
                metadata=metadata,
                processing_stats={
                    "processing_time": processing_time,
                    "file_size_bytes": len(content),
                    "functions_extracted": python_doc.metadata.function_count,
                    "classes_extracted": python_doc.metadata.class_count,
                    "imports_extracted": python_doc.metadata.import_count,
                    "has_syntax_errors": python_doc.metadata.has_syntax_errors,
                    "has_symbol_table": python_doc.symbol_table is not None
                },
                errors=[],
                total_processed=1,
                total_errors=0
            )
            
        except Exception as e:
            error_msg = f"Python AST processing failed for {input_data.file_path}: {str(e)}"
            self.logger.error(error_msg)
            
            # Track error statistics
            self._stats["failed_files"] += 1
            self._track_error(error_msg)
            
            metadata = ComponentMetadata(
                component_type=self.component_type,
                component_name=self.name,
                component_version=self.version,
                processed_at=datetime.now(timezone.utc),
                config=self._config,
                status=ProcessingStatus.ERROR
            )
            
            return DocumentProcessingOutput(
                documents=[],
                metadata=metadata,
                processing_stats={},
                errors=[error_msg],
                total_processed=0,
                total_errors=1
            )
    
    def _parse_python_file(self, file_path: Path, content: str) -> PythonDocument:
        """Parse Python file using AST and extract symbol table."""
        try:
            # Parse AST
            tree = ast.parse(content, filename=str(file_path))
            
            # Extract symbol table and entities
            symbol_table = self._extract_symbol_table(tree, file_path)
            entities = self._extract_entities(tree)
            relationships = self._extract_relationships(tree) if self._extract_relationships_enabled else []
            
            # Create metadata
            metadata = PythonMetadata(
                function_count=len([e for e in entities if e.type == "function"]),
                class_count=len([e for e in entities if e.type == "class"]),
                import_count=len([e for e in entities if e.type == "import"]),
                method_count=len([e for e in entities if e.type == "method"]),
                has_module_docstring=ast.get_docstring(tree) is not None,
                has_syntax_errors=False,
                has_errors=False
            )
            
            return PythonDocument(
                format="python",
                metadata=metadata,
                entities=entities,
                relationships=relationships,
                symbol_table=symbol_table
            )
            
        except SyntaxError as e:
            self.logger.warning(f"Syntax error in {file_path}: {e}")
            
            # Return document with syntax error flag
            metadata = PythonMetadata(
                function_count=0,
                class_count=0,
                import_count=0,
                method_count=0,
                has_module_docstring=False,
                has_syntax_errors=True,
                has_errors=True
            )
            
            return PythonDocument(
                format="python",
                metadata=metadata,
                entities=[],
                relationships=[],
                symbol_table=None
            )
    
    def _extract_symbol_table(self, tree: ast.AST, file_path: Path) -> SymbolTable:
        """Extract symbol table from AST."""
        # Ensure we have a Module node
        if not isinstance(tree, ast.Module):
            raise ValueError(f"Expected ast.Module, got {type(tree).__name__}")
        module_docstring = ast.get_docstring(tree)
        
        # Extract all elements from the module
        elements = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                elements.append(self._extract_function_element(node).model_dump())
            elif isinstance(node, ast.ClassDef):
                elements.append(self._extract_class_element(node).model_dump())
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                elements.extend([elem.model_dump() for elem in self._extract_import_elements(node)])
        
        return SymbolTable(
            type="module",
            name=file_path.stem,
            docstring=module_docstring,
            path=str(file_path),
            module_path=file_path.stem,
            line_range=[1, len(tree.body)] if tree.body else [1, 1],
            elements=elements
        )
    
    def _extract_entities(self, tree: ast.AST) -> List[PythonEntity]:
        """Extract Python entities from AST."""
        entities = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                entities.append(PythonEntity(
                    type="function",
                    name=node.name,
                    start_pos=getattr(node, 'lineno', 0),
                    end_pos=getattr(node, 'end_lineno', 0),
                    metadata={
                        "docstring": ast.get_docstring(node),
                        "args": [arg.arg for arg in node.args.args],
                        "decorators": [ast.unparse(d) for d in node.decorator_list],
                        "is_async": isinstance(node, ast.AsyncFunctionDef)
                    }
                ))
            elif isinstance(node, ast.ClassDef):
                entities.append(PythonEntity(
                    type="class",
                    name=node.name,
                    start_pos=getattr(node, 'lineno', 0),
                    end_pos=getattr(node, 'end_lineno', 0),
                    metadata={
                        "docstring": ast.get_docstring(node),
                        "bases": [ast.unparse(base) for base in node.bases],
                        "decorators": [ast.unparse(d) for d in node.decorator_list],
                        "methods": [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    }
                ))
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for import_entity in self._extract_import_entities(node):
                    entities.append(import_entity)
        
        return entities
    
    def _extract_function_element(self, node: ast.FunctionDef) -> FunctionElement:
        """Extract function element from AST node."""
        return FunctionElement(
            name=node.name,
            qualified_name=node.name,  # Could be improved with proper qualification
            docstring=ast.get_docstring(node),
            line_range=[getattr(node, 'lineno', 0), getattr(node, 'end_lineno', 0)],
            content=ast.unparse(node),
            parameters=[arg.arg for arg in node.args.args],
            returns=ast.unparse(node.returns) if node.returns else None,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            access=self._determine_access_level(node.name),
            decorators=[ast.unparse(d) for d in node.decorator_list]
        )
    
    def _extract_class_element(self, node: ast.ClassDef) -> ClassElement:
        """Extract class element from AST node."""
        # Extract class members
        elements = []
        for child in node.body:
            if isinstance(child, ast.FunctionDef):
                method = MethodElement(
                    name=child.name,
                    qualified_name=f"{node.name}.{child.name}",
                    docstring=ast.get_docstring(child),
                    line_range=[getattr(child, 'lineno', 0), getattr(child, 'end_lineno', 0)],
                    content=ast.unparse(child),
                    parameters=[arg.arg for arg in child.args.args],
                    returns=ast.unparse(child.returns) if child.returns else None,
                    is_async=isinstance(child, ast.AsyncFunctionDef),
                    access=self._determine_access_level(child.name),
                    decorators=[ast.unparse(d) for d in child.decorator_list],
                    is_static="staticmethod" in [ast.unparse(d) for d in child.decorator_list],
                    is_class_method="classmethod" in [ast.unparse(d) for d in child.decorator_list],
                    is_property="property" in [ast.unparse(d) for d in child.decorator_list],
                    parent_class=node.name
                )
                elements.append(method.model_dump())
        
        return ClassElement(
            name=node.name,
            qualified_name=node.name,
            docstring=ast.get_docstring(node),
            line_range=[getattr(node, 'lineno', 0), getattr(node, 'end_lineno', 0)],
            content=ast.unparse(node),
            base_classes=[ast.unparse(base) for base in node.bases],
            access=self._determine_access_level(node.name),
            decorators=[ast.unparse(d) for d in node.decorator_list],
            elements=elements
        )
    
    def _extract_import_elements(self, node: Union[ast.Import, ast.ImportFrom]) -> List[ImportElement]:
        """Extract import elements from AST node."""
        elements = []
        
        if isinstance(node, ast.Import):
            for alias in node.names:
                elements.append(ImportElement(
                    type="import",
                    name=alias.name,
                    qualified_name=alias.name,
                    docstring=None,
                    content=f"import {alias.name}",
                    alias=alias.asname,
                    source="import",
                    line_range=[getattr(node, 'lineno', 0), getattr(node, 'lineno', 0)]
                ))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                elements.append(ImportElement(
                    type="import",
                    name=alias.name,
                    qualified_name=f"{module}.{alias.name}" if module else alias.name,
                    docstring=None,
                    content=f"from {module} import {alias.name}" if module else f"import {alias.name}",
                    alias=alias.asname,
                    source="from_import",
                    line_range=[getattr(node, 'lineno', 0), getattr(node, 'lineno', 0)]
                ))
        
        return elements
    
    def _extract_import_entities(self, node: Union[ast.Import, ast.ImportFrom]) -> List[PythonEntity]:
        """Extract import entities from AST node."""
        entities = []
        
        if isinstance(node, ast.Import):
            for alias in node.names:
                entities.append(PythonEntity(
                    type="import",
                    name=alias.name,
                    start_pos=getattr(node, 'lineno', 0),
                    end_pos=getattr(node, 'lineno', 0),
                    metadata={
                        "alias": alias.asname,
                        "source": "import",
                        "module": alias.name
                    }
                ))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                entities.append(PythonEntity(
                    type="import",
                    name=alias.name,
                    start_pos=getattr(node, 'lineno', 0),
                    end_pos=getattr(node, 'lineno', 0),
                    metadata={
                        "alias": alias.asname,
                        "source": "from_import",
                        "module": module,
                        "imported_name": alias.name
                    }
                ))
        
        return entities
    
    def _extract_relationships(self, tree: ast.AST) -> List[CodeRelationship]:
        """Extract code relationships from AST."""
        relationships = []
        
        # This is a simplified implementation
        # Could be extended to detect inheritance, method calls, etc.
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Inheritance relationships
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        relationships.append(CodeRelationship(
                            source=node.name,
                            target=base.id,
                            type="inherits_from",
                            weight=1.0,
                            line=getattr(node, 'lineno', 0)
                        ))
        
        return relationships
    
    def _determine_access_level(self, name: str) -> str:
        """Determine access level based on naming convention."""
        if name.startswith('__') and name.endswith('__'):
            # Dunder methods are technically public (e.g., __init__, __str__)
            return AccessLevel.PUBLIC.value
        elif name.startswith('__'):
            return AccessLevel.PRIVATE.value
        elif name.startswith('_'):
            return AccessLevel.PROTECTED.value
        else:
            return AccessLevel.PUBLIC.value
    
    def _convert_to_processed_document(self, python_doc: PythonDocument, file_path: Path, content: str) -> ProcessedDocument:
        """Convert PythonDocument to ProcessedDocument format."""
        # Create document ID
        doc_id = str(file_path.stem)
        
        # Convert entities to dict format
        entities = [entity.to_dict() for entity in python_doc.entities]
        
        # Create sections from symbol table
        sections = []
        if python_doc.symbol_table:
            sections.append({
                "type": "module",
                "name": python_doc.symbol_table.name,
                "content": python_doc.symbol_table.docstring or "",
                "line_range": python_doc.symbol_table.line_range,
                "elements": python_doc.symbol_table.elements
            })
        
        # Create metadata
        metadata = {
            "file_path": str(file_path),
            "file_size": len(content),
            "python_metadata": python_doc.metadata.model_dump(),
            "symbol_table": python_doc.symbol_table.model_dump() if python_doc.symbol_table else None,
            "relationships": [rel.model_dump() for rel in python_doc.relationships] if python_doc.relationships else []
        }
        
        return ProcessedDocument(
            id=doc_id,
            content=content,
            content_type="text/x-python",
            format="python",
            content_category=ContentCategory.CODE,
            entities=entities,
            sections=sections,
            metadata=metadata,
            error=None,
            processing_time=None
        )
    
    def _track_error(self, error_msg: str) -> None:
        """Track an error in statistics."""
        self._stats["errors"].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": error_msg
        })
        
        # Keep only last 50 errors to prevent memory growth
        if len(self._stats["errors"]) > 50:
            self._stats["errors"] = self._stats["errors"][-50:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get processing metrics.
        
        Returns:
            Dictionary containing processing metrics
        """
        return {
            "total_files_processed": self._stats.get("total_files_processed", 0),
            "successful_files": self._stats.get("successful_files", 0),
            "failed_files": self._stats.get("failed_files", 0),
            "total_classes": self._stats.get("total_classes", 0),
            "total_functions": self._stats.get("total_functions", 0),
            "total_methods": self._stats.get("total_methods", 0),
            "total_imports": self._stats.get("total_imports", 0),
            "total_relationships": self._stats.get("total_relationships", 0),
            "errors": len(self._stats.get("errors", [])),
            "last_updated": self._stats.get("last_updated", "")
        }