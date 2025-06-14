"""
Core Document Processing Component

This module provides the core document processing component that implements the
DocumentProcessor protocol. It wraps the existing HADES document processing
adapters with the new component contract system.
"""

import logging
import psutil
import time
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime, timezone

# Clean component implementation with no legacy dependencies

# Import component contracts and protocols
from src.types.components.contracts import (
    ComponentType,
    ComponentMetadata,
    DocumentProcessingInput,
    DocumentProcessingOutput,
    ProcessedDocument,
    ContentCategory,
    ProcessingStatus
)
from src.types.components.protocols import DocumentProcessor

# Import types
# from src.types.docproc.enums import ContentCategory as LegacyContentCategory


class CoreDocumentProcessor(DocumentProcessor):
    """
    Core document processor component implementing DocumentProcessor protocol.
    
    This component uses the existing HADES document processing adapters to handle
    various file formats including Python, Markdown, JSON, YAML, and more.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize core document processor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self._config = config or {}
        
        # Component metadata
        self._metadata = ComponentMetadata(
            component_type=ComponentType.DOCPROC,
            component_name="core",
            component_version="1.0.0",
            config=self._config
        )
        
        # Monitoring and metrics tracking
        self._stats: Dict[str, Any] = {
            "total_documents": 0,
            "successful_documents": 0,
            "failed_documents": 0,
            "total_processing_time": 0.0,
            "last_processing_time": None,
            "initialization_count": 1,
            "errors": [],
            "format_counts": {}  # Track processed file formats
        }
        self._startup_time = datetime.now(timezone.utc)
        
        # Get supported formats (minimal set for now)
        self._supported_formats = self._get_supported_formats_minimal()
        
        self.logger.info(f"Initialized core document processor with {len(self._supported_formats)} supported formats")
    
    @property
    def name(self) -> str:
        """Component name for identification."""
        return "core"
    
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
            
        Raises:
            ValueError: If configuration is invalid
        """
        if not self.validate_config(config):
            raise ValueError("Invalid configuration provided")
        
        self._config.update(config)
        self._metadata.config = self._config
        self._metadata.processed_at = datetime.now(timezone.utc)
        
        # Reconfigure manager if needed (minimal implementation)
        if 'manager_config' in config:
            # TODO: Initialize real manager when available
            pass
        
        self.logger.info(f"Updated core document processor configuration")
    
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
        
        # Validate manager config if provided
        if 'manager_config' in config:
            if not isinstance(config['manager_config'], dict):
                return False
        
        # Validate processing options
        if 'processing_options' in config:
            if not isinstance(config['processing_options'], dict):
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
                "manager_config": {
                    "type": "object",
                    "description": "Configuration for the document processor manager",
                    "properties": {
                        "adapter_config": {
                            "type": "object",
                            "description": "Configuration for individual adapters"
                        },
                        "cache_enabled": {
                            "type": "boolean",
                            "description": "Whether to enable caching",
                            "default": True
                        },
                        "max_file_size": {
                            "type": "integer",
                            "description": "Maximum file size to process in bytes",
                            "minimum": 1,
                            "default": 100 * 1024 * 1024  # 100MB
                        }
                    }
                },
                "processing_options": {
                    "type": "object",
                    "description": "Default processing options",
                    "properties": {
                        "extract_metadata": {
                            "type": "boolean",
                            "default": True
                        },
                        "extract_entities": {
                            "type": "boolean",
                            "default": True
                        },
                        "preserve_formatting": {
                            "type": "boolean",
                            "default": False
                        }
                    }
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
            # Check if we have adapters available
            if not self._supported_formats:
                return False
            
            # Test with a simple text processing
            test_input = DocumentProcessingInput(
                file_path="test.txt",
                content="Test content",
                file_type="text/plain"
            )
            
            # Try to process (but don't actually do it)
            return self.can_process(str(test_input.file_path))
            
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
                "supported_formats": self._supported_formats,
                "supported_format_count": len(self._supported_formats),
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
            total_docs = int(self._stats["total_documents"])
            successful_docs = int(self._stats["successful_documents"])
            total_time = float(self._stats["total_processing_time"])
            
            documents_per_second = total_docs / max(uptime, 1.0)
            avg_processing_time = total_time / max(total_docs, 1)
            success_rate = successful_docs / max(total_docs, 1) * 100
            
            return {
                "component_name": self.name,
                "total_documents": self._stats["total_documents"],
                "successful_documents": self._stats["successful_documents"],
                "failed_documents": self._stats["failed_documents"],
                "success_rate_percent": round(success_rate, 2),
                "documents_per_second": round(documents_per_second, 3),
                "average_processing_time": round(avg_processing_time, 3),
                "total_processing_time": round(self._stats["total_processing_time"], 3),
                "last_processing_time": self._stats["last_processing_time"],
                "initialization_count": self._stats["initialization_count"],
                "format_distribution": self._stats["format_counts"],
                "recent_errors": self._stats["errors"][-5:],  # Last 5 errors
                "error_count": len(self._stats["errors"]),
                "uptime_seconds": round(uptime, 2)
            }
        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {e}")
            return {"error": str(e)}

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get component performance metrics (legacy method for compatibility).
        
        Returns:
            Dictionary containing performance metrics
        """
        # Combine infrastructure and performance metrics for compatibility
        infra_metrics = self.get_infrastructure_metrics()
        perf_metrics = self.get_performance_metrics()
        
        return {
            **infra_metrics,
            **perf_metrics,
            "last_health_check": datetime.now(timezone.utc).isoformat()
        }
    
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
            metrics_lines.append(f'hades_component_uptime_seconds{{component="core"}} {infra_metrics.get("uptime_seconds", 0)}')
            
            metrics_lines.append(f"# HELP hades_component_memory_rss_mb Memory RSS usage in MB")
            metrics_lines.append(f"# TYPE hades_component_memory_rss_mb gauge")
            memory_rss = infra_metrics.get("memory_usage", {}).get("rss_mb", 0)
            metrics_lines.append(f'hades_component_memory_rss_mb{{component="core"}} {memory_rss}')
            
            metrics_lines.append(f"# HELP hades_component_supported_formats Number of supported formats")
            metrics_lines.append(f"# TYPE hades_component_supported_formats gauge")
            supported_count = infra_metrics.get("supported_format_count", 0)
            metrics_lines.append(f'hades_component_supported_formats{{component="core"}} {supported_count}')
            
            # Performance metrics
            metrics_lines.append(f"# HELP hades_component_documents_total Total number of processed documents")
            metrics_lines.append(f"# TYPE hades_component_documents_total counter")
            metrics_lines.append(f'hades_component_documents_total{{component="core"}} {perf_metrics.get("total_documents", 0)}')
            
            metrics_lines.append(f"# HELP hades_component_documents_successful_total Total number of successful documents")
            metrics_lines.append(f"# TYPE hades_component_documents_successful_total counter")
            metrics_lines.append(f'hades_component_documents_successful_total{{component="core"}} {perf_metrics.get("successful_documents", 0)}')
            
            metrics_lines.append(f"# HELP hades_component_documents_failed_total Total number of failed documents")
            metrics_lines.append(f"# TYPE hades_component_documents_failed_total counter")
            metrics_lines.append(f'hades_component_documents_failed_total{{component="core"}} {perf_metrics.get("failed_documents", 0)}')
            
            metrics_lines.append(f"# HELP hades_component_success_rate_percent Success rate percentage")
            metrics_lines.append(f"# TYPE hades_component_success_rate_percent gauge")
            metrics_lines.append(f'hades_component_success_rate_percent{{component="core"}} {perf_metrics.get("success_rate_percent", 0)}')
            
            metrics_lines.append(f"# HELP hades_component_documents_per_second Documents processed per second")
            metrics_lines.append(f"# TYPE hades_component_documents_per_second gauge")
            metrics_lines.append(f'hades_component_documents_per_second{{component="core"}} {perf_metrics.get("documents_per_second", 0)}')
            
            metrics_lines.append(f"# HELP hades_component_avg_processing_time_seconds Average processing time in seconds")
            metrics_lines.append(f"# TYPE hades_component_avg_processing_time_seconds gauge")
            metrics_lines.append(f'hades_component_avg_processing_time_seconds{{component="core"}} {perf_metrics.get("average_processing_time", 0)}')
            
            # Format distribution metrics
            format_counts = perf_metrics.get("format_distribution", {})
            if format_counts:
                metrics_lines.append(f"# HELP hades_component_format_count Number of documents processed by format")
                metrics_lines.append(f"# TYPE hades_component_format_count counter")
                for format_name, count in format_counts.items():
                    metrics_lines.append(f'hades_component_format_count{{component="core",format="{format_name}"}} {count}')
            
            return "\n".join(metrics_lines) + "\n"
            
        except Exception as e:
            self.logger.error(f"Failed to export Prometheus metrics: {e}")
            return f"# Error exporting metrics: {str(e)}\n"
    
    def process_documents(
        self, 
        file_paths: List[Union[str, Path]], 
        **kwargs: Any
    ) -> List[ProcessedDocument]:
        """
        Process multiple documents from file paths.
        
        Args:
            file_paths: List of file paths to process
            **kwargs: Additional processing parameters
            
        Returns:
            List of processed document data
        """
        results = []
        
        for file_path in file_paths:
            try:
                # Create input for single document
                doc_input = DocumentProcessingInput(
                    file_path=file_path,
                    processing_options=kwargs.get('processing_options', {})
                )
                
                # Process single document
                result = self.process_document(doc_input.file_path, **kwargs)
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Failed to process {file_path}: {e}")
                # Create error document
                error_doc = ProcessedDocument(
                    id=f"error_{Path(file_path).name}",
                    content="",
                    content_type="text/plain",
                    format="unknown",
                    content_category=ContentCategory.UNKNOWN,
                    error=str(e)
                )
                results.append(error_doc)
        
        return results
    
    def process_document(
        self, 
        file_path: Union[str, Path], 
        **kwargs: Any
    ) -> ProcessedDocument:
        """
        Process a single document.
        
        Args:
            file_path: Path to file to process
            **kwargs: Additional processing parameters
            
        Returns:
            Processed document data
        """
        start_time = datetime.now(timezone.utc)
        
        # Update request statistics
        self._stats["total_documents"] += 1
        self._stats["last_processing_time"] = start_time.isoformat()
        
        try:
            # Convert to string path
            file_path_str = str(file_path)
            
            # Get processing options
            processing_options = kwargs.get('processing_options', self._config.get('processing_options', {}))
            
            # Process using minimal implementation
            result = self._process_document_minimal(file_path_str, **processing_options)
            
            # Convert to our contract format
            doc = self._convert_to_processed_document(result, file_path_str)
            
            # Calculate processing time and update statistics
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            doc.processing_time = processing_time
            
            if doc.error:
                self._stats["failed_documents"] += 1
                self._track_error(doc.error)
            else:
                self._stats["successful_documents"] += 1
                
            self._stats["total_processing_time"] += processing_time
            
            # Track format statistics
            format_key = doc.format
            self._stats["format_counts"][format_key] = self._stats["format_counts"].get(format_key, 0) + 1
            
            return doc
            
        except Exception as e:
            # Calculate processing time and update error statistics
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self._stats["failed_documents"] += 1
            self._stats["total_processing_time"] += processing_time
            
            error_msg = f"Document processing failed for {file_path}: {e}"
            self.logger.error(error_msg)
            self._track_error(error_msg)
            
            # Return error document
            error_doc = ProcessedDocument(
                id=f"error_{Path(file_path).name}",
                content="",
                content_type="text/plain",
                format="unknown",
                content_category=ContentCategory.UNKNOWN,
                error=str(e),
                processing_time=processing_time
            )
            return error_doc
    
    def process(self, input_data: DocumentProcessingInput) -> DocumentProcessingOutput:
        """
        Process documents according to the contract.
        
        Args:
            input_data: Input data conforming to DocumentProcessingInput contract
            
        Returns:
            Output data conforming to DocumentProcessingOutput contract
        """
        documents: List[Any] = []
        errors: List[str] = []
        processing_stats: Dict[str, Any] = {}
        
        try:
            start_time = datetime.now(timezone.utc)
            
            # Process the document
            if input_data.content:
                # Process from content
                doc = self._process_from_content(input_data)
            else:
                # Process from file
                doc = self.process_document(
                    input_data.file_path,
                    processing_options=input_data.processing_options
                )
            
            documents.append(doc)
            
            # Calculate processing time
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Update metadata
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
                documents=documents,
                metadata=metadata,
                processing_stats={
                    "processing_time": processing_time,
                    "file_size": Path(input_data.file_path).stat().st_size if Path(input_data.file_path).exists() else 0
                },
                errors=errors
            )
            
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            errors.append(error_msg)
            self.logger.error(error_msg)
            
            metadata = ComponentMetadata(
                component_type=self.component_type,
                component_name=self.name,
                component_version=self.version,
                processed_at=datetime.now(timezone.utc),
                config=self._config,
                status=ProcessingStatus.ERROR
            )
            
            return DocumentProcessingOutput(
                documents=documents,
                metadata=metadata,
                processing_stats=processing_stats,
                errors=errors
            )
    
    def process_batch(
        self, 
        input_batch: List[DocumentProcessingInput]
    ) -> List[DocumentProcessingOutput]:
        """
        Process multiple documents in batch.
        
        Args:
            input_batch: List of input data
            
        Returns:
            List of output data
        """
        results = []
        
        for input_data in input_batch:
            result = self.process(input_data)
            results.append(result)
        
        return results
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported file formats.
        
        Returns:
            List of supported file extensions (e.g., ['.pdf', '.docx'])
        """
        return self._supported_formats
    
    def can_process(self, file_path: str) -> bool:
        """
        Check if this processor can handle the given file.
        
        Args:
            file_path: Path to file to check
            
        Returns:
            True if file can be processed, False otherwise
        """
        try:
            # Check file extension
            file_ext = Path(file_path).suffix.lower()
            if file_ext in self._supported_formats:
                return True
            
            # Try minimal format detection
            try:
                # Basic format detection based on extension
                return True  # For now, assume we can process most text-based files
            except Exception:
                return False
            
        except Exception:
            return False
    
    def estimate_processing_time(self, input_data: DocumentProcessingInput) -> float:
        """
        Estimate processing time for given input.
        
        Args:
            input_data: Input data to estimate for
            
        Returns:
            Estimated processing time in seconds
        """
        try:
            # Get file size
            if Path(input_data.file_path).exists():
                file_size = Path(input_data.file_path).stat().st_size
            else:
                file_size = len(input_data.content or "")
            
            # Simple estimation: 1MB per second
            return max(0.1, file_size / (1024 * 1024))
            
        except Exception:
            return 1.0  # Default estimate
    
    def _get_supported_formats_minimal(self) -> List[str]:
        """Get supported formats (minimal implementation)."""
        # Basic set of formats we can handle
        return [
            '.py', '.pyw',  # Python
            '.md', '.markdown',  # Markdown
            '.json',  # JSON
            '.yaml', '.yml',  # YAML
            '.txt', '.text',  # Plain text
            '.js', '.ts',  # JavaScript/TypeScript
            '.java', '.cpp', '.c', '.h',  # Other code files
            '.html', '.xml',  # Markup
        ]
    
    def _convert_to_processed_document(
        self, 
        legacy_result: Dict[str, Any], 
        file_path: str
    ) -> ProcessedDocument:
        """
        Convert legacy document processor result to new contract format.
        
        Args:
            legacy_result: Result from legacy document processor
            file_path: Original file path
            
        Returns:
            ProcessedDocument conforming to contract
        """
        # Extract content category (simplified)
        content_category = legacy_result.get('content_category', ContentCategory.TEXT)
        if isinstance(content_category, str):
            # Try to convert string value
            try:
                content_category = ContentCategory(content_category)
            except ValueError:
                content_category = ContentCategory.UNKNOWN
        elif not isinstance(content_category, ContentCategory):
            content_category = ContentCategory.UNKNOWN
        
        return ProcessedDocument(
            id=legacy_result.get('id', Path(file_path).stem),
            content=legacy_result.get('content', ''),
            content_type=legacy_result.get('content_type', 'text/plain'),
            format=legacy_result.get('format', Path(file_path).suffix.lstrip('.')),
            content_category=content_category,
            entities=legacy_result.get('entities', []),
            sections=legacy_result.get('sections', []),
            metadata=legacy_result.get('metadata', {}),
            error=legacy_result.get('error'),
            processing_time=legacy_result.get('processing_time')
        )
    
    def _process_from_content(
        self, 
        input_data: DocumentProcessingInput
    ) -> ProcessedDocument:
        """
        Process document from content string instead of file.
        
        Args:
            input_data: Input data with content
            
        Returns:
            Processed document
        """
        try:
            # Determine format from file extension or type
            file_ext = Path(input_data.file_path).suffix.lower()
            
            # Get file type - handle MIME types by extracting the extension part
            if input_data.file_type:
                if input_data.file_type.startswith('text/x-'):
                    file_type = input_data.file_type.replace('text/x-', '')
                elif input_data.file_type.startswith('application/'):
                    file_type = input_data.file_type.replace('application/', '')
                else:
                    file_type = file_ext.lstrip('.')
            else:
                file_type = file_ext.lstrip('.')
            
            # Process using minimal implementation
            result = self._process_text_minimal(
                text=input_data.content or "",
                file_type=file_type,
                **input_data.processing_options
            )
            
            return self._convert_to_processed_document(result, str(input_data.file_path))
            
        except Exception as e:
            self.logger.error(f"Content processing failed: {e}")
            
            return ProcessedDocument(
                id=f"content_{Path(input_data.file_path).stem}",
                content=input_data.content or "",
                content_type=input_data.file_type or "text/plain",
                format="unknown",
                content_category=ContentCategory.UNKNOWN,
                error=str(e)
            )
    
    def _process_document_minimal(self, file_path: str, **options: Any) -> Dict[str, Any]:
        """
        Minimal document processing implementation.
        
        Args:
            file_path: Path to file to process
            **options: Processing options
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Determine content category based on extension
            file_ext = Path(file_path).suffix.lower()
            if file_ext in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h']:
                category = ContentCategory.CODE
            elif file_ext in ['.json', '.yaml', '.yml', '.xml']:
                category = ContentCategory.DATA
            elif file_ext in ['.md', '.markdown']:
                category = ContentCategory.MARKDOWN
            else:
                category = ContentCategory.TEXT
            
            return {
                'id': Path(file_path).stem,
                'content': content,
                'content_type': self._get_content_type(file_ext),
                'format': file_ext.lstrip('.'),
                'content_category': category,
                'entities': [],
                'sections': [],
                'metadata': {
                    'file_path': file_path,
                    'file_size': len(content),
                    'processed_at': datetime.now(timezone.utc).isoformat()
                },
                'error': None
            }
            
        except Exception as e:
            return {
                'id': f'error_{Path(file_path).stem}',
                'content': '',
                'content_type': 'text/plain',
                'format': 'unknown',
                'content_category': ContentCategory.UNKNOWN,
                'entities': [],
                'sections': [],
                'metadata': {},
                'error': str(e)
            }
    
    def _process_text_minimal(self, text: str, file_type: str, **options: Any) -> Dict[str, Any]:
        """
        Minimal text processing implementation.
        
        Args:
            text: Text content to process
            file_type: Type/extension of the content
            **options: Processing options
            
        Returns:
            Dictionary with processing results
        """
        # Determine content category
        if file_type in ['py', 'python', 'js', 'javascript', 'ts', 'typescript', 'java', 'cpp', 'c', 'h']:
            category = ContentCategory.CODE
        elif file_type in ['json', 'yaml', 'yml', 'xml']:
            category = ContentCategory.DATA
        elif file_type in ['md', 'markdown']:
            category = ContentCategory.MARKDOWN
        else:
            category = ContentCategory.TEXT
        
        return {
            'id': f'text_{hash(text) % 10000}',
            'content': text,
            'content_type': self._get_content_type(f'.{file_type}'),
            'format': file_type,
            'content_category': category,
            'entities': [],
            'sections': [],
            'metadata': {
                'content_length': len(text),
                'processed_at': datetime.now(timezone.utc).isoformat()
            },
            'error': None
        }
    
    def _get_content_type(self, file_ext: str) -> str:
        """Get MIME type for file extension."""
        type_map = {
            '.txt': 'text/plain',
            '.py': 'text/x-python',
            '.js': 'application/javascript',
            '.ts': 'application/typescript',
            '.json': 'application/json',
            '.yaml': 'application/x-yaml',
            '.yml': 'application/x-yaml',
            '.md': 'text/markdown',
            '.html': 'text/html',
            '.xml': 'application/xml',
        }
        return type_map.get(file_ext, 'text/plain')
    
    def _track_error(self, error_msg: str) -> None:
        """Track an error in statistics."""
        self._stats["errors"].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": error_msg
        })
        
        # Keep only last 50 errors to prevent memory growth
        if len(self._stats["errors"]) > 50:
            self._stats["errors"] = self._stats["errors"][-50:]