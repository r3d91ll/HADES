"""
File Processor for Bootstrap Pipeline

Processes files for Sequential-ISNE modality-specific storage including content extraction,
chunking, and preparation for graph construction.
"""

import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator, Tuple
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .config import BootstrapConfig
from .chunking_manager import SemanticChunkingManager
from .embedding_manager import EmbeddingManager
from src.storage.incremental.sequential_isne_types import (
    CodeFile, DocumentationFile, ConfigFile,
    FileType, CodeFileType, DocumentationType, ConfigFileType,
    ProcessingStatus, classify_file_type, get_specific_file_type,
    Chunk, ChunkType, Embedding
)
from src.components.docproc.factory import create_docproc_component
from src.types.components.contracts import DocumentProcessingInput

logger = logging.getLogger(__name__)


class FileProcessor:
    """
    Processes files for Sequential-ISNE storage.
    
    Handles:
    - Content extraction and validation
    - Modality-specific metadata extraction
    - Text chunking for analysis
    - Batch processing for performance
    """
    
    def __init__(self, config: BootstrapConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Thread-safe counters
        self._lock = threading.Lock()
        self._processed_count = 0
        self._error_count = 0
        
        # Initialize document processing components
        self._initialize_docproc_components()
        
        # Initialize semantic chunking manager
        self.chunking_manager = SemanticChunkingManager(config)
        
        # Initialize embedding manager
        self.embedding_manager = EmbeddingManager(config)
        
        self.logger.info("Initialized FileProcessor with docproc, semantic chunking, and embeddings")
    
    def _initialize_docproc_components(self) -> None:
        """Initialize document processing components."""
        try:
            # Initialize core document processor for Python/code files
            self.core_docproc = create_docproc_component('core', {
                'extract_imports': True,
                'extract_functions': True,
                'extract_classes': True,
                'include_docstrings': True
            })
            self.logger.info("✅ Initialized Core DocProc for code files")
            
            # Initialize Docling processor for text/documentation files  
            self.docling_docproc = create_docproc_component('docling', {
                'convert_to_markdown': True,
                'extract_tables': True,
                'extract_images': True,
                'preserve_formatting': True
            })
            self.logger.info("✅ Initialized Docling DocProc for documentation files")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize docproc components: {e}")
            raise RuntimeError(f"DocProc initialization failed: {e}")
    
    def process_files(self, discovered_files: Dict[str, Dict[str, Any]]) -> Iterator[Tuple[str, Any, List[Chunk], List[Embedding]]]:
        """
        Process discovered files into modality-specific objects with embeddings.
        
        Args:
            discovered_files: Dictionary of file paths to file metadata
            
        Yields:
            Tuples of (file_path, file_object, chunks, embeddings)
        """
        self.logger.info(f"Processing {len(discovered_files)} files...")
        
        if self.config.enable_parallel_processing and len(discovered_files) > 1:
            yield from self._process_files_parallel(discovered_files)
        else:
            yield from self._process_files_sequential(discovered_files)
        
        self.logger.info(f"Completed processing: {self._processed_count} successful, {self._error_count} errors")
    
    def _process_files_sequential(self, discovered_files: Dict[str, Dict[str, Any]]) -> Iterator[Tuple[str, Any, List[Chunk], List[Embedding]]]:
        """Process files sequentially."""
        for file_path, file_info in discovered_files.items():
            try:
                result = self._process_single_file(file_path, file_info)
                if result:
                    file_obj, chunks, embeddings = result
                    yield file_path, file_obj, chunks, embeddings
                    
                    with self._lock:
                        self._processed_count += 1
                        
            except Exception as e:
                self.logger.error(f"Error processing file {file_path}: {e}")
                with self._lock:
                    self._error_count += 1
    
    def _process_files_parallel(self, discovered_files: Dict[str, Dict[str, Any]]) -> Iterator[Tuple[str, Any, List[Chunk], List[Embedding]]]:
        """Process files in parallel using thread pool."""
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all files for processing
            future_to_file = {
                executor.submit(self._process_single_file, file_path, file_info): file_path
                for file_path, file_info in discovered_files.items()
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        file_obj, chunks, embeddings = result
                        yield file_path, file_obj, chunks, embeddings
                        
                        with self._lock:
                            self._processed_count += 1
                            
                except Exception as e:
                    self.logger.error(f"Error processing file {file_path}: {e}")
                    with self._lock:
                        self._error_count += 1
    
    def _process_single_file(self, file_path: str, file_info: Dict[str, Any]) -> Optional[Tuple[Any, List[Chunk], List[Embedding]]]:
        """Process a single file through docproc → chunking → embeddings into modality-specific object, chunks, and embeddings."""
        try:
            # Step 1: Use DocProc to process the file (raw file -> structured JSON with metadata)
            processed_document = self._process_file_with_docproc(file_path)
            if processed_document is None:
                return None
            
            # Step 2: Create content hash from processed content
            processed_content = processed_document.content
            content_hash = hashlib.sha256(processed_content.encode('utf-8')).hexdigest()
            
            # Step 3: Determine file type
            file_type = classify_file_type(file_path)
            
            # Step 4: Create modality-specific object using processed content and extracted metadata
            if file_type == FileType.CODE:
                file_obj = self._create_code_file_from_processed(file_path, file_info, processed_document, content_hash)
            elif file_type == FileType.DOCUMENTATION:
                file_obj = self._create_documentation_file_from_processed(file_path, file_info, processed_document, content_hash)
            elif file_type == FileType.CONFIG:
                file_obj = self._create_config_file_from_processed(file_path, file_info, processed_document, content_hash)
            else:
                self.logger.debug(f"Skipping file with unknown type: {file_path}")
                return None
            
            # Step 5: Create semantic chunks using processed content (NOT raw content)
            chunks = self.chunking_manager.chunk_file_content(file_path, processed_content, file_type)
            
            # Step 6: Generate embeddings for chunks (NEW: dual storage with content preservation)
            embeddings = []
            if chunks:
                chunk_embedding_pairs = self.embedding_manager.generate_embeddings_for_chunks(
                    chunks, file_path, file_type
                )
                
                # Separate chunks and embeddings while preserving links
                updated_chunks = []
                embeddings = []
                for chunk, embedding in chunk_embedding_pairs:
                    updated_chunks.append(chunk)  # Chunk with embedding_id link
                    embeddings.append(embedding)  # Embedding with vector
                
                chunks = updated_chunks  # Use chunks with embedding links
                self.logger.debug(f"Generated {len(embeddings)} embeddings for {file_path}")
            
            return file_obj, chunks, embeddings
            
        except Exception as e:
            self.logger.error(f"Failed to process file {file_path}: {e}")
            return None
    
    def _process_file_with_docproc(self, file_path: str) -> Optional[Any]:
        """Process a file using appropriate docproc component based on file type."""
        try:
            file_type = classify_file_type(file_path)
            
            # Create docproc input
            docproc_input = DocumentProcessingInput(
                file_path=file_path,
                processing_options={
                    'extract_metadata': True,
                    'preserve_structure': True
                },
                metadata={'file_type': file_type.value}
            )
            
            # Select appropriate docproc component
            if file_type == FileType.CODE:
                # Use core docproc for code files (extracts AST info, preserves code structure)
                result = self.core_docproc.process(docproc_input)
                self.logger.debug(f"Processed code file {file_path} with Core DocProc")
            elif file_type == FileType.DOCUMENTATION:
                # Use Docling for documentation (converts to markdown, extracts structure)
                result = self.docling_docproc.process(docproc_input)
                self.logger.debug(f"Processed documentation file {file_path} with Docling DocProc")
            else:
                # Use core docproc as fallback for config and other files
                result = self.core_docproc.process(docproc_input)
                self.logger.debug(f"Processed config file {file_path} with Core DocProc (fallback)")
            
            # Return the first processed document
            if result.documents:
                return result.documents[0]
            else:
                self.logger.warning(f"No processed documents returned for {file_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"DocProc failed for {file_path}: {e}")
            return None
    
    def _read_file_content(self, file_path: Path) -> Optional[str]:
        """Safely read file content."""
        try:
            # Try UTF-8 first
            return file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            try:
                # Fallback to latin-1
                return file_path.read_text(encoding='latin-1')
            except Exception as e:
                self.logger.debug(f"Could not read file {file_path}: {e}")
                return None
        except Exception as e:
            self.logger.debug(f"Error reading file {file_path}: {e}")
            return None
    
    def _create_code_file(self, file_path: str, file_info: Dict[str, Any], content: str, content_hash: str) -> CodeFile:
        """Create CodeFile object."""
        path = Path(file_path)
        
        return CodeFile(
            file_path=file_path,
            file_name=path.name,
            directory=str(path.parent),
            extension=path.suffix.lower(),
            file_type=get_specific_file_type(file_path, FileType.CODE),
            content=content,
            content_hash=content_hash,
            size=len(content.encode('utf-8')),
            modified_time=datetime.fromtimestamp(file_info.get('modified_time', 0)),
            directory_depth=file_info.get('directory_depth', 0),
            lines_of_code=file_info.get('lines_of_code', 0),
            
            # Code-specific metadata
            ast_metadata=file_info.get('ast_metadata', {}),
            imports=file_info.get('imports', []),
            functions=file_info.get('functions', []),
            classes=file_info.get('classes', []),
            complexity_score=file_info.get('complexity_score', 0.0),
            
            # Processing status
            processing_status=ProcessingStatus.PENDING
        )
    
    def _create_documentation_file(self, file_path: str, file_info: Dict[str, Any], content: str, content_hash: str) -> DocumentationFile:
        """Create DocumentationFile object."""
        path = Path(file_path)
        
        return DocumentationFile(
            file_path=file_path,
            file_name=path.name,
            directory=str(path.parent),
            extension=path.suffix.lower(),
            file_type=get_specific_file_type(file_path, FileType.DOCUMENTATION),
            content=content,
            content_hash=content_hash,
            size=len(content.encode('utf-8')),
            modified_time=datetime.fromtimestamp(file_info.get('modified_time', 0)),
            directory_depth=file_info.get('directory_depth', 0),
            word_count=file_info.get('word_count', 0),
            
            # Documentation-specific metadata
            document_structure=file_info.get('document_structure', {}),
            headings=file_info.get('headings', []),
            links=file_info.get('links', []),
            code_references=file_info.get('code_references', []),
            readability_score=file_info.get('readability_score', 0.0),
            
            # Processing status
            processing_status=ProcessingStatus.PENDING
        )
    
    def _create_config_file(self, file_path: str, file_info: Dict[str, Any], content: str, content_hash: str) -> ConfigFile:
        """Create ConfigFile object."""
        path = Path(file_path)
        
        return ConfigFile(
            file_path=file_path,
            file_name=path.name,
            directory=str(path.parent),
            extension=path.suffix.lower(),
            file_type=get_specific_file_type(file_path, FileType.CONFIG),
            content=content,
            content_hash=content_hash,
            size=len(content.encode('utf-8')),
            modified_time=datetime.fromtimestamp(file_info.get('modified_time', 0)),
            directory_depth=file_info.get('directory_depth', 0),
            
            # Config-specific metadata
            parsed_config=file_info.get('parsed_config', {}),
            config_schema=file_info.get('config_schema', {}),
            validation_status=file_info.get('validation_status'),
            dependencies=file_info.get('dependencies', []),
            
            # Processing status
            processing_status=ProcessingStatus.PENDING
        )
    
    def _create_code_file_from_processed(self, file_path: str, file_info: Dict[str, Any], processed_doc: Any, content_hash: str) -> CodeFile:
        """Create CodeFile object from docproc-processed document."""
        path = Path(file_path)
        
        # Extract metadata from processed document
        metadata = processed_doc.metadata or {}
        
        return CodeFile(
            file_path=file_path,
            file_name=path.name,
            directory=str(path.parent),
            extension=path.suffix.lower(),
            file_type=get_specific_file_type(file_path, FileType.CODE),
            content=processed_doc.content,  # This is the processed content (structured JSON for code)
            content_hash=content_hash,
            size=len(processed_doc.content.encode('utf-8')),
            modified_time=datetime.fromtimestamp(file_info.get('modified_time', 0)),
            directory_depth=file_info.get('directory_depth', 0),
            lines_of_code=file_info.get('lines_of_code', 0),
            
            # Code-specific metadata extracted by docproc
            ast_metadata=metadata.get('ast_metadata', {}),
            imports=metadata.get('imports', []),
            functions=metadata.get('functions', []),
            classes=metadata.get('classes', []),
            complexity_score=metadata.get('complexity_score', 0.0),
            
            # Processing status
            processing_status=ProcessingStatus.PENDING
        )
    
    def _create_documentation_file_from_processed(self, file_path: str, file_info: Dict[str, Any], processed_doc: Any, content_hash: str) -> DocumentationFile:
        """Create DocumentationFile object from docproc-processed document."""
        path = Path(file_path)
        
        # Extract metadata from processed document
        metadata = processed_doc.metadata or {}
        
        return DocumentationFile(
            file_path=file_path,
            file_name=path.name,
            directory=str(path.parent),
            extension=path.suffix.lower(),
            file_type=get_specific_file_type(file_path, FileType.DOCUMENTATION),
            content=processed_doc.content,  # This is the processed content (markdown for docs)
            content_hash=content_hash,
            size=len(processed_doc.content.encode('utf-8')),
            modified_time=datetime.fromtimestamp(file_info.get('modified_time', 0)),
            directory_depth=file_info.get('directory_depth', 0),
            word_count=file_info.get('word_count', 0),
            
            # Documentation-specific metadata extracted by docproc
            document_structure=metadata.get('document_structure', {}),
            headings=metadata.get('headings', []),
            links=metadata.get('links', []),
            code_references=metadata.get('code_references', []),
            readability_score=metadata.get('readability_score', 0.0),
            
            # Processing status
            processing_status=ProcessingStatus.PENDING
        )
    
    def _create_config_file_from_processed(self, file_path: str, file_info: Dict[str, Any], processed_doc: Any, content_hash: str) -> ConfigFile:
        """Create ConfigFile object from docproc-processed document."""
        path = Path(file_path)
        
        # Extract metadata from processed document
        metadata = processed_doc.metadata or {}
        
        return ConfigFile(
            file_path=file_path,
            file_name=path.name,
            directory=str(path.parent),
            extension=path.suffix.lower(),
            file_type=get_specific_file_type(file_path, FileType.CONFIG),
            content=processed_doc.content,  # This is the processed content
            content_hash=content_hash,
            size=len(processed_doc.content.encode('utf-8')),
            modified_time=datetime.fromtimestamp(file_info.get('modified_time', 0)),
            directory_depth=file_info.get('directory_depth', 0),
            
            # Config-specific metadata extracted by docproc
            parsed_config=metadata.get('parsed_config', {}),
            config_schema=metadata.get('config_schema', {}),
            validation_status=metadata.get('validation_status'),
            dependencies=metadata.get('dependencies', []),
            
            # Processing status
            processing_status=ProcessingStatus.PENDING
        )
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics including chunking information."""
        with self._lock:
            stats = {
                'processed_count': self._processed_count,
                'error_count': self._error_count,
                'success_rate': self._processed_count / max(self._processed_count + self._error_count, 1)
            }
            
            # Add chunking statistics
            if hasattr(self, 'chunking_manager'):
                stats['chunking'] = self.chunking_manager.get_chunking_stats()
            
            return stats