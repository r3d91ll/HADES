"""
Docling Document Processing Component

This module provides a Docling-based document processing component that implements
the DocumentProcessor protocol. Docling specializes in processing complex documents
like PDFs, Word documents, and other structured formats.
"""

import logging
import psutil
import time
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime, timezone

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


class DoclingDocumentProcessor(DocumentProcessor):
    """
    Docling-based document processor component implementing DocumentProcessor protocol.
    
    This component uses the Docling library for advanced document processing,
    particularly for PDFs and complex document formats.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Docling document processor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self._config = config or {}
        
        # Component metadata
        self._metadata = ComponentMetadata(
            component_type=ComponentType.DOCPROC,
            component_name="docling",
            component_version="1.0.0",
            config=self._config
        )
        
        # Try to import Docling
        self._docling_available = self._check_docling_availability()
        
        # Monitoring and metrics tracking
        self._stats: Dict[str, Any] = {
            "total_documents": 0,
            "successful_documents": 0,
            "failed_documents": 0,
            "total_processing_time": 0.0,
            "last_processing_time": None,
            "initialization_count": 0,
            "errors": [],
            "format_counts": {}  # Track processed file formats
        }
        self._startup_time = datetime.now(timezone.utc)
        
        if self._docling_available:
            # Initialize Docling adapter (placeholder for minimal implementation)
            # In a full implementation, would initialize a Docling adapter here
            self._adapter = None  # Placeholder for minimal implementation
            self._stats["initialization_count"] += 1
            self.logger.info("Initialized Docling document processor (minimal)")
        else:
            self._adapter = None
            self.logger.warning("Docling not available - processor will return errors")
    
    @property
    def name(self) -> str:
        """Component name for identification."""
        return "docling"
    
    @property
    def version(self) -> str:
        """Component version string."""
        return "1.0.0"
    
    @property
    def component_type(self) -> ComponentType:
        """Type of component."""
        return ComponentType.DOCPROC
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure component with parameters."""
        self._config.update(config)
        self._metadata.config = self._config
        self._metadata.processed_at = datetime.now(timezone.utc)
    
    def validate_config(self, config: Any) -> bool:
        """Validate configuration parameters."""
        return isinstance(config, dict)
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for component configuration."""
        return {
            "type": "object",
            "properties": {
                "ocr_enabled": {
                    "type": "boolean",
                    "description": "Enable OCR for scanned documents",
                    "default": True
                },
                "extract_tables": {
                    "type": "boolean",
                    "description": "Extract tables from documents",
                    "default": True
                },
                "extract_images": {
                    "type": "boolean",
                    "description": "Extract images from documents",
                    "default": False
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
            # Check Docling availability
            if not self._docling_available:
                return False
            
            # Check if we can create a DocumentConverter (basic functionality test)
            if self._docling_available:
                try:
                    from docling.document_converter import DocumentConverter
                    # Don't actually create it to avoid overhead, just check import
                    return True
                except ImportError:
                    return False
            else:
                return True
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
                "docling_available": self._docling_available,
                "supported_formats": self.get_supported_formats(),
                "supported_format_count": len(self.get_supported_formats()),
                "memory_usage": {
                    "rss_mb": round(memory_info.rss / 1024 / 1024, 2),
                    "vms_mb": round(memory_info.vms / 1024 / 1024, 2)
                },
                "adapter_initialized": self._adapter is not None,
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
            documents_per_second = self._stats["total_documents"] / max(uptime, 1.0)
            total_processing_time = self._stats.get("total_processing_time", 0.0)
            total_documents = self._stats.get("total_documents", 0)
            successful_documents = self._stats.get("successful_documents", 0)
            
            if isinstance(total_processing_time, (int, float)) and isinstance(total_documents, (int, float)):
                avg_processing_time = float(total_processing_time) / max(int(total_documents), 1)
            else:
                avg_processing_time = 0.0
                
            if isinstance(successful_documents, (int, float)) and isinstance(total_documents, (int, float)):
                success_rate = float(successful_documents) / max(int(total_documents), 1) * 100
            else:
                success_rate = 0.0
            
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
                "recent_errors": self._get_recent_errors(5),  # Last 5 errors
                "error_count": self._get_error_count(),
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
            metrics_lines.append(f'hades_component_uptime_seconds{{component="docling"}} {infra_metrics.get("uptime_seconds", 0)}')
            
            metrics_lines.append(f"# HELP hades_component_memory_rss_mb Memory RSS usage in MB")
            metrics_lines.append(f"# TYPE hades_component_memory_rss_mb gauge")
            memory_rss = infra_metrics.get("memory_usage", {}).get("rss_mb", 0)
            metrics_lines.append(f'hades_component_memory_rss_mb{{component="docling"}} {memory_rss}')
            
            metrics_lines.append(f"# HELP hades_component_supported_formats Number of supported formats")
            metrics_lines.append(f"# TYPE hades_component_supported_formats gauge")
            supported_count = infra_metrics.get("supported_format_count", 0)
            metrics_lines.append(f'hades_component_supported_formats{{component="docling"}} {supported_count}')
            
            # Performance metrics
            metrics_lines.append(f"# HELP hades_component_documents_total Total number of processed documents")
            metrics_lines.append(f"# TYPE hades_component_documents_total counter")
            metrics_lines.append(f'hades_component_documents_total{{component="docling"}} {perf_metrics.get("total_documents", 0)}')
            
            metrics_lines.append(f"# HELP hades_component_documents_successful_total Total number of successful documents")
            metrics_lines.append(f"# TYPE hades_component_documents_successful_total counter")
            metrics_lines.append(f'hades_component_documents_successful_total{{component="docling"}} {perf_metrics.get("successful_documents", 0)}')
            
            metrics_lines.append(f"# HELP hades_component_documents_failed_total Total number of failed documents")
            metrics_lines.append(f"# TYPE hades_component_documents_failed_total counter")
            metrics_lines.append(f'hades_component_documents_failed_total{{component="docling"}} {perf_metrics.get("failed_documents", 0)}')
            
            metrics_lines.append(f"# HELP hades_component_success_rate_percent Success rate percentage")
            metrics_lines.append(f"# TYPE hades_component_success_rate_percent gauge")
            metrics_lines.append(f'hades_component_success_rate_percent{{component="docling"}} {perf_metrics.get("success_rate_percent", 0)}')
            
            metrics_lines.append(f"# HELP hades_component_documents_per_second Documents processed per second")
            metrics_lines.append(f"# TYPE hades_component_documents_per_second gauge")
            metrics_lines.append(f'hades_component_documents_per_second{{component="docling"}} {perf_metrics.get("documents_per_second", 0)}')
            
            metrics_lines.append(f"# HELP hades_component_avg_processing_time_seconds Average processing time in seconds")
            metrics_lines.append(f"# TYPE hades_component_avg_processing_time_seconds gauge")
            metrics_lines.append(f'hades_component_avg_processing_time_seconds{{component="docling"}} {perf_metrics.get("average_processing_time", 0)}')
            
            # Format distribution metrics
            format_counts = perf_metrics.get("format_distribution", {})
            if format_counts:
                metrics_lines.append(f"# HELP hades_component_format_count Number of documents processed by format")
                metrics_lines.append(f"# TYPE hades_component_format_count counter")
                for format_name, count in format_counts.items():
                    metrics_lines.append(f'hades_component_format_count{{component="docling",format="{format_name}"}} {count}')
            
            return "\n".join(metrics_lines) + "\n"
            
        except Exception as e:
            self.logger.error(f"Failed to export Prometheus metrics: {e}")
            return f"# Error exporting metrics: {str(e)}\n"
    
    def process_documents(
        self, 
        file_paths: List[Union[str, Path]], 
        **kwargs: Any
    ) -> List[ProcessedDocument]:
        """Process multiple documents from file paths."""
        if not self._docling_available:
            return [self._create_error_document(fp, "Docling not available") for fp in file_paths]
        
        results = []
        for file_path in file_paths:
            result = self.process_document(file_path, **kwargs)
            results.append(result)
        
        return results
    
    def process_document(
        self, 
        file_path: Union[str, Path], 
        **kwargs: Any
    ) -> ProcessedDocument:
        """Process a single document using Docling for any supported format."""
        start_time = datetime.now(timezone.utc)
        
        # Update request statistics
        self._stats["total_documents"] += 1
        self._stats["last_processing_time"] = start_time.isoformat()
        
        if not self._docling_available:
            # Track error
            self._stats["failed_documents"] += 1
            error_doc = self._create_error_document(file_path, "Docling not available")
            self._track_error("Docling not available")
            return error_doc
        
        try:
            # Import Docling
            from docling.document_converter import DocumentConverter
            
            # Create converter
            converter = DocumentConverter()
            
            # Convert document (works for PDF, DOCX, PPTX, HTML, images, etc.)
            conversion_result = converter.convert(str(file_path))
            
            # Extract content - use markdown as primary content
            markdown_content = conversion_result.document.export_to_markdown()
            
            # Extract metadata
            doc_metadata = {
                'processed_by': 'docling',
                'conversion_status': str(conversion_result.status),
                'file_path': str(file_path),
                'file_size': Path(file_path).stat().st_size,
                'processed_at': datetime.now(timezone.utc).isoformat()
            }
            
            # Add document-specific metadata if available
            if hasattr(conversion_result.document, 'num_pages'):
                doc_metadata['num_pages'] = conversion_result.document.num_pages
            if hasattr(conversion_result.document, 'title'):
                doc_metadata['title'] = conversion_result.document.title
            if hasattr(conversion_result.document, 'author'):
                doc_metadata['author'] = conversion_result.document.author
            if hasattr(conversion_result.document, 'creation_date'):
                doc_metadata['creation_date'] = str(conversion_result.document.creation_date)
            
            # Determine content type and category based on file extension
            file_ext = Path(file_path).suffix.lower()
            content_type_map = {
                '.pdf': 'application/pdf',
                '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                '.doc': 'application/msword',
                '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
                '.ppt': 'application/vnd.ms-powerpoint',
                '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                '.xls': 'application/vnd.ms-excel',
                '.html': 'text/html',
                '.htm': 'text/html',
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg'
            }
            
            content_type = content_type_map.get(file_ext, 'application/octet-stream')
            
            # Calculate processing time and update statistics
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self._stats["successful_documents"] += 1
            self._stats["total_processing_time"] += processing_time
            
            # Track format statistics
            format_key = file_ext.lstrip('.')
            format_counts = self._stats.get("format_counts", {})
            if isinstance(format_counts, dict):
                format_counts[format_key] = format_counts.get(format_key, 0) + 1
                self._stats["format_counts"] = format_counts
            
            # Convert to contract format
            return ProcessedDocument(
                id=Path(file_path).stem,
                content=markdown_content,
                content_type=content_type,
                format=format_key,
                content_category=ContentCategory.DOCUMENT,
                entities=[],  # Docling doesn't extract entities by default
                sections=[],  # Could be extracted from markdown headers if needed
                metadata=doc_metadata,
                error=None,
                processing_time=processing_time
            )
            
        except Exception as e:
            # Calculate processing time and update error statistics
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self._stats["failed_documents"] += 1
            self._stats["total_processing_time"] += processing_time
            
            error_msg = f"Docling processing failed for {file_path}: {e}"
            self.logger.error(error_msg)
            self._track_error(error_msg)
            
            error_doc = self._create_error_document(file_path, str(e))
            error_doc.processing_time = processing_time
            return error_doc
    
    def process(self, input_data: DocumentProcessingInput) -> DocumentProcessingOutput:
        """Process documents according to the contract."""
        documents = []
        errors = []
        
        try:
            doc = self.process_document(
                input_data.file_path,
                **input_data.processing_options
            )
            documents.append(doc)
            
            if doc.error:
                errors.append(doc.error)
            
        except Exception as e:
            errors.append(str(e))
        
        metadata = ComponentMetadata(
            component_type=self.component_type,
            component_name=self.name,
            component_version=self.version,
            processed_at=datetime.now(timezone.utc),
            config=self._config,
            status=ProcessingStatus.SUCCESS if not errors else ProcessingStatus.ERROR
        )
        
        return DocumentProcessingOutput(
            documents=documents,
            metadata=metadata,
            errors=errors
        )
    
    def process_batch(
        self, 
        input_batch: List[DocumentProcessingInput]
    ) -> List[DocumentProcessingOutput]:
        """Process multiple documents in batch."""
        return [self.process(input_data) for input_data in input_batch]
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats that Docling can process."""
        if self._docling_available:
            return [
                # Document formats
                '.pdf',
                '.docx', '.doc',
                '.pptx', '.ppt', 
                '.xlsx', '.xls',
                '.html', '.htm',
                # Image formats (with OCR)
                '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff',
                # Text formats
                '.txt', '.md', '.markdown',
                # Other supported formats
                '.rtf', '.odt', '.ods', '.odp'
            ]
        return []
    
    def can_process(self, file_path: str) -> bool:
        """Check if this processor can handle the file."""
        if not self._docling_available:
            return False
        
        file_ext = Path(file_path).suffix.lower()
        return file_ext in self.get_supported_formats()
    
    def estimate_processing_time(self, input_data: DocumentProcessingInput) -> float:
        """Estimate processing time for given input."""
        if not self._docling_available:
            return 0.0
        
        try:
            file_size = Path(input_data.file_path).stat().st_size
            # Docling is slower for complex documents: 500KB per second
            return max(0.5, file_size / (500 * 1024))
        except Exception:
            return 2.0  # Default estimate for Docling
    
    def _check_docling_availability(self) -> bool:
        """Check if Docling is available."""
        try:
            import docling
            return True
        except ImportError:
            return False
    
    def _track_error(self, error_msg: str) -> None:
        """Track an error in statistics."""
        errors_list = self._stats.get("errors", [])
        if isinstance(errors_list, list):
            errors_list.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": error_msg
            })
            
            # Keep only last 50 errors to prevent memory growth
            if len(errors_list) > 50:
                errors_list = errors_list[-50:]
            
            self._stats["errors"] = errors_list
    
    def _get_recent_errors(self, count: int = 5) -> List[Any]:
        """Get recent errors with proper type checking."""
        errors_list = self._stats.get("errors", [])
        if isinstance(errors_list, list) and len(errors_list) >= count:
            return errors_list[-count:]
        elif isinstance(errors_list, list):
            return errors_list
        else:
            return []
    
    def _get_error_count(self) -> int:
        """Get error count with proper type checking."""
        errors_list = self._stats.get("errors", [])
        if isinstance(errors_list, list):
            return len(errors_list)
        else:
            return 0
    
    def _create_error_document(self, file_path: Union[str, Path], error: str) -> ProcessedDocument:
        """Create an error document."""
        return ProcessedDocument(
            id=f"error_{Path(file_path).name}",
            content="",
            content_type="text/plain",
            format="unknown",
            content_category=ContentCategory.UNKNOWN,
            error=error
        )