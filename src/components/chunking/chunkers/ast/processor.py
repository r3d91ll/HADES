"""
AST-Aware Code Chunking Component

This module provides AST-aware code chunking that uses symbol table information
to create intelligent code chunks based on semantic boundaries like functions,
classes, and logical code blocks.
"""

import ast
import logging
import psutil
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone

# Import component contracts and protocols
from src.types.components.contracts import (
    ComponentType,
    ComponentMetadata,
    ChunkingInput,
    ChunkingOutput,
    TextChunk,
    ProcessingStatus
)
from src.types.components.protocols import Chunker

# Import Python AST processor for symbol extraction
from src.components.docproc.python_ast.processor import PythonASTProcessor


class ASTAwareCodeChunker(Chunker):
    """
    AST-aware code chunking component implementing Chunker protocol.
    
    This component uses AST analysis to create intelligent code chunks
    based on semantic boundaries like functions, classes, and logical blocks.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize AST-aware code chunker.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self._config = config or {}
        
        # Component metadata
        self._metadata = ComponentMetadata(
            component_type=ComponentType.CHUNKING,
            component_name="ast_code",
            component_version="1.0.0",
            config=self._config
        )
        
        # Configuration options
        self._chunk_strategy = self._config.get('chunk_strategy', 'semantic')  # semantic, hybrid, size_based
        self._max_chunk_size = self._config.get('max_chunk_size', 2048)
        self._min_chunk_size = self._config.get('min_chunk_size', 100)
        self._include_imports = self._config.get('include_imports', True)
        self._include_docstrings = self._config.get('include_docstrings', True)
        self._context_lines = self._config.get('context_lines', 2)  # Lines of context around chunks
        
        # Initialize AST processor
        self._ast_processor = PythonASTProcessor({
            'include_docstrings': self._include_docstrings,
            'include_private_members': True,
            'extract_relationships': False  # Not needed for chunking
        })
        
        # Monitoring and metrics tracking
        self._stats = {
            "total_chunks_created": 0,
            "successful_chunks": 0,
            "failed_chunks": 0,
            "total_processing_time": 0.0,
            "total_characters_processed": 0,
            "semantic_chunks": 0,
            "function_chunks": 0,
            "class_chunks": 0,
            "fallback_chunks": 0,
            "last_processing_time": None,
            "initialization_count": 1,
            "errors": [],
            "chunk_strategy_counts": {}
        }
        self._startup_time = datetime.now(timezone.utc)
        
        self.logger.info(f"Initialized AST-aware code chunker with strategy: {self._chunk_strategy}")
    
    @property
    def name(self) -> str:
        """Component name for identification."""
        return "ast_code"
    
    @property
    def version(self) -> str:
        """Component version string."""
        return "1.0.0"
    
    @property
    def component_type(self) -> ComponentType:
        """Type of component."""
        return ComponentType.CHUNKING
    
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
        self._chunk_strategy = self._config.get('chunk_strategy', 'semantic')
        self._max_chunk_size = self._config.get('max_chunk_size', 2048)
        self._min_chunk_size = self._config.get('min_chunk_size', 100)
        self._include_imports = self._config.get('include_imports', True)
        self._include_docstrings = self._config.get('include_docstrings', True)
        self._context_lines = self._config.get('context_lines', 2)
        
        self.logger.info("Updated AST-aware code chunker configuration")
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        if not isinstance(config, dict):
            return False
        
        # Validate chunk strategy
        if 'chunk_strategy' in config:
            valid_strategies = ['semantic', 'hybrid', 'size_based']
            if config['chunk_strategy'] not in valid_strategies:
                return False
        
        # Validate size parameters
        size_params = ['max_chunk_size', 'min_chunk_size', 'context_lines']
        for param in size_params:
            if param in config:
                if not isinstance(config[param], int) or config[param] < 0:
                    return False
        
        # Validate boolean parameters
        bool_params = ['include_imports', 'include_docstrings']
        for param in bool_params:
            if param in config:
                if not isinstance(config[param], bool):
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
                "chunk_strategy": {
                    "type": "string",
                    "enum": ["semantic", "hybrid", "size_based"],
                    "default": "semantic",
                    "description": "Chunking strategy to use"
                },
                "max_chunk_size": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 2048,
                    "description": "Maximum chunk size in characters"
                },
                "min_chunk_size": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 100,
                    "description": "Minimum chunk size in characters"
                },
                "include_imports": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to include import statements in chunks"
                },
                "include_docstrings": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to include docstrings in chunks"
                },
                "context_lines": {
                    "type": "integer",
                    "minimum": 0,
                    "default": 2,
                    "description": "Number of context lines around chunks"
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
            # Test basic AST chunking functionality
            test_code = '''
def hello_world():
    """Simple test function."""
    print("Hello, world!")
    return True

class TestClass:
    """Test class for validation."""
    
    def __init__(self):
        self.value = 42
    
    def get_value(self):
        return self.value
'''
            test_input = ChunkingInput(
                text=test_code,
                document_id="health_check",
                chunk_size=500,
                chunk_overlap=50
            )
            
            chunks = self._perform_ast_chunking(test_input)
            return len(chunks) > 0
            
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
                "chunk_strategy": self._chunk_strategy,
                "max_chunk_size": self._max_chunk_size,
                "min_chunk_size": self._min_chunk_size,
                "include_imports": self._include_imports,
                "include_docstrings": self._include_docstrings,
                "context_lines": self._context_lines,
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
            chunks_per_second = self._stats["total_chunks_created"] / max(uptime, 1.0)
            avg_processing_time = (
                self._stats["total_processing_time"] / max(self._stats["total_chunks_created"], 1)
            )
            success_rate = (
                self._stats["successful_chunks"] / max(self._stats["total_chunks_created"], 1) * 100
            )
            avg_chars_per_chunk = (
                self._stats["total_characters_processed"] / max(self._stats["total_chunks_created"], 1)
            )
            
            return {
                "component_name": self.name,
                "total_chunks_created": self._stats["total_chunks_created"],
                "successful_chunks": self._stats["successful_chunks"],
                "failed_chunks": self._stats["failed_chunks"],
                "success_rate_percent": round(success_rate, 2),
                "chunks_per_second": round(chunks_per_second, 3),
                "average_processing_time": round(avg_processing_time, 3),
                "total_processing_time": round(self._stats["total_processing_time"], 3),
                "total_characters_processed": self._stats["total_characters_processed"],
                "avg_chars_per_chunk": round(avg_chars_per_chunk, 2),
                "semantic_chunks": self._stats["semantic_chunks"],
                "function_chunks": self._stats["function_chunks"],
                "class_chunks": self._stats["class_chunks"],
                "fallback_chunks": self._stats["fallback_chunks"],
                "last_processing_time": self._stats["last_processing_time"],
                "initialization_count": self._stats["initialization_count"],
                "strategy_distribution": self._stats["chunk_strategy_counts"],
                "recent_errors": self._stats["errors"][-5:],  # Last 5 errors
                "error_count": len(self._stats["errors"]),
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
            metrics_lines.append(f'hades_component_uptime_seconds{{component="ast_code_chunker"}} {infra_metrics.get("uptime_seconds", 0)}')
            
            metrics_lines.append(f"# HELP hades_component_memory_rss_mb Memory RSS usage in MB")
            metrics_lines.append(f"# TYPE hades_component_memory_rss_mb gauge")
            memory_rss = infra_metrics.get("memory_usage", {}).get("rss_mb", 0)
            metrics_lines.append(f'hades_component_memory_rss_mb{{component="ast_code_chunker"}} {memory_rss}')
            
            # Performance metrics
            metrics_lines.append(f"# HELP hades_component_chunks_total Total number of processed chunks")
            metrics_lines.append(f"# TYPE hades_component_chunks_total counter")
            metrics_lines.append(f'hades_component_chunks_total{{component="ast_code_chunker"}} {perf_metrics.get("total_chunks_created", 0)}')
            
            metrics_lines.append(f"# HELP hades_component_chunks_successful_total Total number of successful chunks")
            metrics_lines.append(f"# TYPE hades_component_chunks_successful_total counter")
            metrics_lines.append(f'hades_component_chunks_successful_total{{component="ast_code_chunker"}} {perf_metrics.get("successful_chunks", 0)}')
            
            metrics_lines.append(f"# HELP hades_component_chunks_failed_total Total number of failed chunks")
            metrics_lines.append(f"# TYPE hades_component_chunks_failed_total counter")
            metrics_lines.append(f'hades_component_chunks_failed_total{{component="ast_code_chunker"}} {perf_metrics.get("failed_chunks", 0)}')
            
            metrics_lines.append(f"# HELP hades_component_success_rate_percent Success rate percentage")
            metrics_lines.append(f"# TYPE hades_component_success_rate_percent gauge")
            metrics_lines.append(f'hades_component_success_rate_percent{{component="ast_code_chunker"}} {perf_metrics.get("success_rate_percent", 0)}')
            
            metrics_lines.append(f"# HELP hades_component_chunks_per_second Chunks processed per second")
            metrics_lines.append(f"# TYPE hades_component_chunks_per_second gauge")
            metrics_lines.append(f'hades_component_chunks_per_second{{component="ast_code_chunker"}} {perf_metrics.get("chunks_per_second", 0)}')
            
            # Chunk type distribution metrics
            chunk_types = ["semantic_chunks", "function_chunks", "class_chunks", "fallback_chunks"]
            for chunk_type in chunk_types:
                metrics_lines.append(f"# HELP hades_component_{chunk_type} Number of {chunk_type}")
                metrics_lines.append(f"# TYPE hades_component_{chunk_type} counter")
                metrics_lines.append(f'hades_component_{chunk_type}{{component="ast_code_chunker"}} {perf_metrics.get(chunk_type, 0)}')
            
            return "\\n".join(metrics_lines) + "\\n"
            
        except Exception as e:
            self.logger.error(f"Failed to export Prometheus metrics: {e}")
            return f"# Error exporting metrics: {str(e)}\\n"
    
    def chunk(self, input_data: ChunkingInput) -> ChunkingOutput:
        """
        Chunk code using AST-aware strategies.
        
        Args:
            input_data: Input data conforming to ChunkingInput contract
            
        Returns:
            Output data conforming to ChunkingOutput contract
        """
        errors: List[str] = []
        chunks: List[TextChunk] = []
        
        try:
            start_time = datetime.now(timezone.utc)
            
            # Update request statistics
            self._stats["last_processing_time"] = start_time.isoformat()
            
            # Perform AST-aware chunking
            chunks = self._perform_ast_chunking(input_data)
            
            # Calculate processing time
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Update statistics
            self._stats["total_chunks_created"] += len(chunks)
            self._stats["successful_chunks"] += len(chunks)
            self._stats["total_processing_time"] += processing_time
            self._stats["total_characters_processed"] += len(input_data.text)
            
            # Track strategy usage
            strategy_key = self._chunk_strategy
            self._stats["chunk_strategy_counts"][strategy_key] = self._stats["chunk_strategy_counts"].get(strategy_key, 0) + 1
            
            # Update metadata
            metadata = ComponentMetadata(
                component_type=self.component_type,
                component_name=self.name,
                component_version=self.version,
                processing_time=processing_time,
                processed_at=datetime.now(timezone.utc),
                config=self._config,
                status=ProcessingStatus.SUCCESS if not errors else ProcessingStatus.ERROR
            )
            
            return ChunkingOutput(
                chunks=chunks,
                metadata=metadata,
                processing_stats={
                    "processing_time": processing_time,
                    "chunks_created": len(chunks),
                    "characters_processed": len(input_data.text),
                    "chunking_strategy": self._chunk_strategy,
                    "chunk_size_used": self._max_chunk_size,
                    "semantic_chunks": self._count_chunks_by_type(chunks, "semantic"),
                    "function_chunks": self._count_chunks_by_type(chunks, "function"),
                    "class_chunks": self._count_chunks_by_type(chunks, "class"),
                    "fallback_chunks": self._count_chunks_by_type(chunks, "fallback"),
                    "throughput_chars_per_second": len(input_data.text) / max(processing_time, 0.001),
                    "throughput_chunks_per_second": len(chunks) / max(processing_time, 0.001)
                },
                errors=errors
            )
            
        except Exception as e:
            error_msg = f"AST code chunking failed: {str(e)}"
            errors.append(error_msg)
            self.logger.error(error_msg)
            
            # Track error statistics
            self._stats["failed_chunks"] += 1
            self._track_error(error_msg)
            
            metadata = ComponentMetadata(
                component_type=self.component_type,
                component_name=self.name,
                component_version=self.version,
                processed_at=datetime.now(timezone.utc),
                config=self._config,
                status=ProcessingStatus.ERROR
            )
            
            return ChunkingOutput(
                chunks=[],
                metadata=metadata,
                processing_stats={},
                errors=errors
            )
    
    def _perform_ast_chunking(self, input_data: ChunkingInput) -> List[TextChunk]:
        """Perform AST-aware chunking based on configured strategy."""
        text = input_data.text
        
        # Try to detect if this is Python code
        if not self._is_python_code(text):
            self.logger.info("Content doesn't appear to be Python code, falling back to line-based chunking")
            return self._fallback_line_chunking(text, input_data)
        
        try:
            # Parse AST
            tree = ast.parse(text)
            
            if self._chunk_strategy == 'semantic':
                return self._semantic_chunking(tree, text, input_data)
            elif self._chunk_strategy == 'hybrid':
                return self._hybrid_chunking(tree, text, input_data)
            else:  # size_based
                return self._size_based_chunking(tree, text, input_data)
                
        except SyntaxError as e:
            self.logger.warning(f"Syntax error in code, falling back to line-based chunking: {e}")
            return self._fallback_line_chunking(text, input_data)
        except Exception as e:
            self.logger.error(f"AST parsing failed, falling back to line-based chunking: {e}")
            return self._fallback_line_chunking(text, input_data)
    
    def _is_python_code(self, text: str) -> bool:
        """Check if text appears to be Python code."""
        # Simple heuristics to detect Python code
        python_keywords = ['def ', 'class ', 'import ', 'from ', 'if ', 'for ', 'while ', 'try:', 'except:', 'return ', 'print(']
        lines = text.split('\\n')[:50]  # Check first 50 lines
        
        keyword_count = 0
        for line in lines:
            stripped = line.strip()
            if any(keyword in stripped for keyword in python_keywords):
                keyword_count += 1
        
        # If we find Python keywords in at least 5% of lines, assume it's Python
        threshold = max(1, len([l for l in lines if l.strip()]) * 0.05)  # Only count non-empty lines
        self.logger.debug(f"Python detection: {keyword_count} keywords found, threshold: {threshold}")
        return keyword_count >= threshold
    
    def _semantic_chunking(self, tree: ast.AST, text: str, input_data: ChunkingInput) -> List[TextChunk]:
        """Chunk code based on semantic boundaries (functions, classes)."""
        chunks = []
        lines = text.split('\\n')
        chunk_id = 0
        
        # Extract imports as a separate chunk if requested
        if self._include_imports:
            import_chunk = self._extract_imports_chunk(tree, lines, input_data, chunk_id)
            if import_chunk:
                chunks.append(import_chunk)
                chunk_id += 1
                self._stats["semantic_chunks"] += 1
        
        # Process top-level elements
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                chunk = self._create_function_chunk(node, lines, input_data, chunk_id)
                chunks.append(chunk)
                chunk_id += 1
                self._stats["function_chunks"] += 1
                
            elif isinstance(node, ast.ClassDef):
                chunk = self._create_class_chunk(node, lines, input_data, chunk_id)
                chunks.append(chunk)
                chunk_id += 1
                self._stats["class_chunks"] += 1
                
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                # Already handled in imports chunk
                continue
                
            else:
                # Handle other statements (module-level code)
                chunk = self._create_statement_chunk(node, lines, input_data, chunk_id)
                if chunk:
                    chunks.append(chunk)
                    chunk_id += 1
                    self._stats["semantic_chunks"] += 1
        
        return chunks
    
    def _hybrid_chunking(self, tree: ast.AST, text: str, input_data: ChunkingInput) -> List[TextChunk]:
        """Hybrid chunking that combines semantic and size-based approaches."""
        semantic_chunks = self._semantic_chunking(tree, text, input_data)
        
        # Split large semantic chunks based on size
        refined_chunks = []
        chunk_id = 0
        
        for chunk in semantic_chunks:
            if len(chunk.text) <= self._max_chunk_size:
                # Chunk is acceptable size
                chunk.id = f"{input_data.document_id}_chunk_{chunk_id}"
                chunk.chunk_index = chunk_id
                refined_chunks.append(chunk)
                chunk_id += 1
            else:
                # Split large chunk
                sub_chunks = self._split_large_chunk(chunk, input_data, chunk_id)
                refined_chunks.extend(sub_chunks)
                chunk_id += len(sub_chunks)
        
        return refined_chunks
    
    def _size_based_chunking(self, tree: ast.AST, text: str, input_data: ChunkingInput) -> List[TextChunk]:
        """Size-based chunking with AST awareness for better boundaries."""
        lines = text.split('\\n')
        chunks = []
        chunk_id = 0
        
        current_lines = []
        current_size = 0
        
        # Create line-to-node mapping for better boundary detection
        line_to_node = self._create_line_to_node_mapping(tree)
        
        for i, line in enumerate(lines):
            line_number = i + 1
            
            # Check if adding this line would exceed max size
            if current_size + len(line) > self._max_chunk_size and current_lines:
                # Look for a good boundary nearby
                split_point = self._find_best_split_point(current_lines, line_to_node)
                
                # Create chunk
                chunk_text = '\\n'.join(current_lines[:split_point])
                chunk = TextChunk(
                    id=f"{input_data.document_id}_chunk_{chunk_id}",
                    text=chunk_text,
                    start_index=0,  # Would need proper calculation
                    end_index=len(chunk_text),
                    chunk_index=chunk_id,
                    metadata={
                        "chunking_method": "ast_size_based",
                        "chunk_type": "size_based",
                        "line_count": len(current_lines[:split_point])
                    }
                )
                chunks.append(chunk)
                chunk_id += 1
                self._stats["fallback_chunks"] += 1
                
                # Keep remaining lines for next chunk
                current_lines = current_lines[split_point:] + [line]
                current_size = sum(len(l) for l in current_lines)
            else:
                current_lines.append(line)
                current_size += len(line)
        
        # Add final chunk
        if current_lines:
            chunk_text = '\\n'.join(current_lines)
            chunk = TextChunk(
                id=f"{input_data.document_id}_chunk_{chunk_id}",
                text=chunk_text,
                start_index=0,
                end_index=len(chunk_text),
                chunk_index=chunk_id,
                metadata={
                    "chunking_method": "ast_size_based",
                    "chunk_type": "size_based",
                    "line_count": len(current_lines)
                }
            )
            chunks.append(chunk)
            self._stats["fallback_chunks"] += 1
        
        return chunks
    
    def _extract_imports_chunk(self, tree: ast.AST, lines: List[str], input_data: ChunkingInput, chunk_id: int) -> Optional[TextChunk]:
        """Extract all import statements as a single chunk."""
        import_lines = []
        
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                start_line = getattr(node, 'lineno', 1) - 1
                end_line = getattr(node, 'end_lineno', start_line + 1) - 1
                
                for line_idx in range(start_line, min(end_line + 1, len(lines))):
                    if line_idx < len(lines):
                        import_lines.append(lines[line_idx])
        
        if not import_lines:
            return None
        
        chunk_text = '\\n'.join(import_lines)
        return TextChunk(
            id=f"{input_data.document_id}_chunk_{chunk_id}",
            text=chunk_text,
            start_index=0,
            end_index=max(1, len(chunk_text)),
            chunk_index=chunk_id,
            metadata={
                "chunking_method": "ast_semantic",
                "chunk_type": "imports",
                "import_count": len([line for line in import_lines if line.strip().startswith(('import ', 'from '))])
            }
        )
    
    def _create_function_chunk(self, node: ast.FunctionDef, lines: List[str], input_data: ChunkingInput, chunk_id: int) -> TextChunk:
        """Create a chunk for a function definition."""
        # Get line information first
        start_line = getattr(node, 'lineno', 1) - 1
        end_line = getattr(node, 'end_lineno', None)
        
        # Try to use AST unparse for the function source
        try:
            chunk_text = ast.unparse(node)
        except:
            # Fallback to line-based extraction
            # If end_lineno is not available, estimate based on indentation
            if end_line is None:
                end_line = self._find_function_end(lines, start_line)
            else:
                end_line = end_line - 1  # Convert to 0-based indexing
            
            # Add context lines if configured
            context_start = max(0, start_line - self._context_lines)
            context_end = min(len(lines), end_line + 1 + self._context_lines)
            
            chunk_lines = lines[context_start:context_end]
            chunk_text = '\\n'.join(chunk_lines)
        
        # Ensure we have some content
        if not chunk_text.strip():
            chunk_text = f"def {node.name}(): pass  # AST extraction failed"
        
        # Set default end_line if still None
        if end_line is None:
            end_line = start_line + 10
        
        # Extract function metadata
        parameters = [arg.arg for arg in node.args.args]
        decorators = [ast.unparse(d) for d in node.decorator_list]
        docstring = ast.get_docstring(node)
        
        return TextChunk(
            id=f"{input_data.document_id}_chunk_{chunk_id}",
            text=chunk_text,
            start_index=0,
            end_index=max(1, len(chunk_text)),
            chunk_index=chunk_id,
            metadata={
                "chunking_method": "ast_semantic",
                "chunk_type": "function",
                "function_name": node.name,
                "parameters": parameters,
                "decorators": decorators,
                "has_docstring": docstring is not None,
                "is_async": isinstance(node, ast.AsyncFunctionDef),
                "line_range": [start_line + 1, end_line + 1]
            }
        )
    
    def _create_class_chunk(self, node: ast.ClassDef, lines: List[str], input_data: ChunkingInput, chunk_id: int) -> TextChunk:
        """Create a chunk for a class definition."""
        # Get line information first
        start_line = getattr(node, 'lineno', 1) - 1
        end_line = getattr(node, 'end_lineno', None)
        
        # Try to use AST unparse for the class source
        try:
            chunk_text = ast.unparse(node)
        except:
            # Fallback to line-based extraction
            # If end_lineno is not available, estimate based on indentation
            if end_line is None:
                end_line = self._find_class_end(lines, start_line)
            else:
                end_line = end_line - 1  # Convert to 0-based indexing
            
            # Add context lines if configured
            context_start = max(0, start_line - self._context_lines)
            context_end = min(len(lines), end_line + 1 + self._context_lines)
            
            chunk_lines = lines[context_start:context_end]
            chunk_text = '\\n'.join(chunk_lines)
        
        # Ensure we have some content
        if not chunk_text.strip():
            chunk_text = f"class {node.name}: pass  # AST extraction failed"
        
        # Set default end_line if still None
        if end_line is None:
            end_line = start_line + 20
        
        # Extract class metadata
        base_classes = [ast.unparse(base) for base in node.bases]
        decorators = [ast.unparse(d) for d in node.decorator_list]
        methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
        docstring = ast.get_docstring(node)
        
        return TextChunk(
            id=f"{input_data.document_id}_chunk_{chunk_id}",
            text=chunk_text,
            start_index=0,
            end_index=max(1, len(chunk_text)),
            chunk_index=chunk_id,
            metadata={
                "chunking_method": "ast_semantic",
                "chunk_type": "class",
                "class_name": node.name,
                "base_classes": base_classes,
                "decorators": decorators,
                "methods": methods,
                "has_docstring": docstring is not None,
                "line_range": [start_line + 1, end_line + 1]
            }
        )
    
    def _create_statement_chunk(self, node: ast.AST, lines: List[str], input_data: ChunkingInput, chunk_id: int) -> Optional[TextChunk]:
        """Create a chunk for other statements."""
        start_line = getattr(node, 'lineno', 1) - 1
        end_line = getattr(node, 'end_lineno', start_line + 1) - 1
        
        if start_line >= len(lines) or end_line < 0:
            return None
        
        chunk_lines = lines[start_line:end_line + 1]
        chunk_text = '\\n'.join(chunk_lines)
        
        # Skip empty or very small chunks
        if len(chunk_text.strip()) < self._min_chunk_size:
            return None
        
        return TextChunk(
            id=f"{input_data.document_id}_chunk_{chunk_id}",
            text=chunk_text,
            start_index=0,
            end_index=max(1, len(chunk_text)),
            chunk_index=chunk_id,
            metadata={
                "chunking_method": "ast_semantic",
                "chunk_type": "statement",
                "statement_type": type(node).__name__,
                "line_range": [start_line + 1, end_line + 1]
            }
        )
    
    def _fallback_line_chunking(self, text: str, input_data: ChunkingInput) -> List[TextChunk]:
        """Fallback to simple line-based chunking when AST parsing fails."""
        lines = text.split('\\n')
        chunks = []
        chunk_id = 0
        
        current_lines = []
        current_size = 0
        
        for line in lines:
            if current_size + len(line) > self._max_chunk_size and current_lines:
                # Create chunk
                chunk_text = '\\n'.join(current_lines)
                chunk = TextChunk(
                    id=f"{input_data.document_id}_chunk_{chunk_id}",
                    text=chunk_text,
                    start_index=0,
                    end_index=max(1, len(chunk_text)),  # Ensure end_index > start_index
                    chunk_index=chunk_id,
                    metadata={
                        "chunking_method": "fallback_line_based",
                        "chunk_type": "fallback",
                        "line_count": len(current_lines)
                    }
                )
                chunks.append(chunk)
                chunk_id += 1
                self._stats["fallback_chunks"] += 1
                
                current_lines = [line]
                current_size = len(line)
            else:
                current_lines.append(line)
                current_size += len(line)
        
        # Add final chunk
        if current_lines:
            chunk_text = '\\n'.join(current_lines)
            chunk = TextChunk(
                id=f"{input_data.document_id}_chunk_{chunk_id}",
                text=chunk_text,
                start_index=0,
                end_index=len(chunk_text),
                chunk_index=chunk_id,
                metadata={
                    "chunking_method": "fallback_line_based",
                    "chunk_type": "fallback",
                    "line_count": len(current_lines)
                }
            )
            chunks.append(chunk)
            self._stats["fallback_chunks"] += 1
        
        return chunks
    
    def _create_line_to_node_mapping(self, tree: ast.AST) -> Dict[int, ast.AST]:
        """Create mapping from line numbers to AST nodes."""
        line_to_node = {}
        
        for node in ast.walk(tree):
            if hasattr(node, 'lineno'):
                line_to_node[node.lineno] = node
        
        return line_to_node
    
    def _find_best_split_point(self, lines: List[str], line_to_node: Dict[int, ast.AST]) -> int:
        """Find the best point to split a chunk."""
        # Look for natural boundaries like function/class starts
        for i in range(len(lines) - 1, 0, -1):
            line_num = i + 1
            if line_num in line_to_node:
                node = line_to_node[line_num]
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    return i
        
        # Fallback to simple split
        return len(lines) // 2
    
    def _split_large_chunk(self, chunk: TextChunk, input_data: ChunkingInput, start_chunk_id: int) -> List[TextChunk]:
        """Split a large chunk into smaller ones."""
        lines = chunk.text.split('\\n')
        sub_chunks = []
        chunk_id = start_chunk_id
        
        current_lines = []
        current_size = 0
        
        for line in lines:
            if current_size + len(line) > self._max_chunk_size and current_lines:
                # Create sub-chunk
                chunk_text = '\\n'.join(current_lines)
                sub_chunk = TextChunk(
                    id=f"{input_data.document_id}_chunk_{chunk_id}",
                    text=chunk_text,
                    start_index=0,
                    end_index=len(chunk_text),
                    chunk_index=chunk_id,
                    metadata={
                        **chunk.metadata,
                        "is_split_chunk": True,
                        "original_chunk_type": chunk.metadata.get("chunk_type", "unknown")
                    }
                )
                sub_chunks.append(sub_chunk)
                chunk_id += 1
                
                current_lines = [line]
                current_size = len(line)
            else:
                current_lines.append(line)
                current_size += len(line)
        
        # Add final sub-chunk
        if current_lines:
            chunk_text = '\\n'.join(current_lines)
            sub_chunk = TextChunk(
                id=f"{input_data.document_id}_chunk_{chunk_id}",
                text=chunk_text,
                start_index=0,
                end_index=len(chunk_text),
                chunk_index=chunk_id,
                metadata={
                    **chunk.metadata,
                    "is_split_chunk": True,
                    "original_chunk_type": chunk.metadata.get("chunk_type", "unknown")
                }
            )
            sub_chunks.append(sub_chunk)
        
        return sub_chunks
    
    def _count_chunks_by_type(self, chunks: List[TextChunk], chunk_type: str) -> int:
        """Count chunks of a specific type."""
        return len([c for c in chunks if c.metadata.get("chunk_type") == chunk_type])
    
    def estimate_chunks(self, input_data: ChunkingInput) -> int:
        """
        Estimate number of chunks that will be generated.
        
        Args:
            input_data: Input data to estimate for
            
        Returns:
            Estimated number of chunks
        """
        try:
            # Simple estimation based on text length and max chunk size
            text_length = len(input_data.text)
            estimated_chunks = max(1, text_length // self._max_chunk_size)
            
            # Adjust for semantic chunking (functions/classes typically create more chunks)
            if self._chunk_strategy == 'semantic':
                estimated_chunks = int(estimated_chunks * 1.5)
            
            return estimated_chunks
            
        except Exception:
            return 1
    
    def supports_content_type(self, content_type: str) -> bool:
        """
        Check if chunker supports the given content type.
        
        Args:
            content_type: Content type to check
            
        Returns:
            True if content type is supported, False otherwise
        """
        code_types = ['code', 'python', 'text/x-python', 'application/x-python']
        return content_type.lower() in code_types
    
    def get_optimal_chunk_size(self, content_type: str) -> int:
        """
        Get the optimal chunk size for a given content type.
        
        Args:
            content_type: Content type
            
        Returns:
            Optimal chunk size in characters
        """
        # For code, we use semantic chunking so size is less critical
        return self._max_chunk_size
    
    def _find_function_end(self, lines: List[str], start_line: int) -> int:
        """Find the end line of a function based on indentation."""
        if start_line >= len(lines):
            return len(lines) - 1
        
        # Find the base indentation of the function
        function_line = lines[start_line].lstrip()
        if not function_line.startswith('def '):
            return start_line + 10  # Fallback
        
        base_indent = len(lines[start_line]) - len(function_line)
        
        # Look for the end of the function (next line with same or less indentation)
        for i in range(start_line + 1, len(lines)):
            line = lines[i]
            if line.strip():  # Non-empty line
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= base_indent:
                    return i - 1
        
        return len(lines) - 1
    
    def _find_class_end(self, lines: List[str], start_line: int) -> int:
        """Find the end line of a class based on indentation."""
        if start_line >= len(lines):
            return len(lines) - 1
        
        # Find the base indentation of the class
        class_line = lines[start_line].lstrip()
        if not class_line.startswith('class '):
            return start_line + 20  # Fallback
        
        base_indent = len(lines[start_line]) - len(class_line)
        
        # Look for the end of the class (next line with same or less indentation)
        for i in range(start_line + 1, len(lines)):
            line = lines[i]
            if line.strip():  # Non-empty line
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= base_indent:
                    return i - 1
        
        return len(lines) - 1
    
    def _track_error(self, error_msg: str) -> None:
        """Track an error in statistics."""
        self._stats["errors"].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": error_msg
        })
        
        # Keep only last 50 errors to prevent memory growth
        if len(self._stats["errors"]) > 50:
            self._stats["errors"] = self._stats["errors"][-50:]