"""
Hybrid Document Processing Stage for ISNE Bootstrap Pipeline

Routes different file types to appropriate processors:
- Python files → Core processor (for AST analysis)
- PDFs/complex docs → Docling processor
- Other text files → Core processor
"""

import logging
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.components.docproc.factory import create_docproc_component
from src.types.components.contracts import DocumentProcessingInput
from .base import BaseBootstrapStage

logger = logging.getLogger(__name__)


@dataclass
class HybridDocumentProcessingResult:
    """Result of hybrid document processing stage."""
    success: bool
    documents: List[Any]
    stats: Dict[str, Any]
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None


class HybridDocumentProcessingStage(BaseBootstrapStage):
    """Hybrid document processing stage that routes files to appropriate processors."""
    
    def __init__(self):
        """Initialize hybrid document processing stage."""
        super().__init__("hybrid_document_processing")
        
        # Initialize both processors
        self.core_processor = None
        self.docling_processor = None
        
        # File routing configuration
        self.core_extensions = {
            '.py', '.pyw',  # Python - MUST use core for AST
            '.js', '.ts',   # JavaScript/TypeScript
            '.java', '.cpp', '.c', '.h',  # Other code
            '.md', '.markdown',  # Markdown
            '.txt', '.text',  # Plain text
            '.json', '.yaml', '.yml',  # Data files (including conversations!)
            '.xml', '.html'  # Markup
        }
        
        self.docling_extensions = {
            '.pdf',  # PDFs - MUST use docling
            '.doc', '.docx',  # Word documents
            '.ppt', '.pptx',  # PowerPoint
            '.xls', '.xlsx',  # Excel
            '.odt', '.ods', '.odp'  # OpenDocument
        }
        
    def _initialize_processors(self, config: Any):
        """Initialize processors lazily on first use."""
        if self.core_processor is None:
            logger.info("Initializing core processor for code/text files...")
            self.core_processor = create_docproc_component("core", config.core_config if hasattr(config, 'core_config') else {})
            
        if self.docling_processor is None:
            logger.info("Initializing docling processor for PDFs/complex documents...")
            self.docling_processor = create_docproc_component("docling", config.docling_config if hasattr(config, 'docling_config') else {})
    
    def _route_file(self, file_path: Path) -> str:
        """
        Route file to appropriate processor based on extension.
        
        Returns:
            "core" or "docling"
        """
        ext = file_path.suffix.lower()
        
        if ext in self.core_extensions:
            return "core"
        elif ext in self.docling_extensions:
            return "docling"
        else:
            # Default to core for unknown text-like files
            # Could also check file content/magic bytes here
            return "core"
    
    def execute(self, input_files: List[Path], config: Any) -> HybridDocumentProcessingResult:
        """
        Execute hybrid document processing stage.
        
        Args:
            input_files: List of input file paths to process
            config: DocumentProcessingConfig object
            
        Returns:
            HybridDocumentProcessingResult with processed documents and stats
        """
        logger.info(f"Starting hybrid document processing stage with {len(input_files)} files")
        
        # Initialize processors
        self._initialize_processors(config)
        
        # Initialize tracking
        documents = []
        stats = {
            "files_processed": 0,
            "documents_generated": 0,
            "total_content_chars": 0,
            "files_by_processor": {"core": 0, "docling": 0},
            "processor_stats": {"core": {}, "docling": {}},
            "file_details": []
        }
        
        try:
            # Route and process files
            core_files = []
            docling_files = []
            
            for file_path in input_files:
                processor_type = self._route_file(file_path)
                if processor_type == "core":
                    core_files.append(file_path)
                else:
                    docling_files.append(file_path)
            
            logger.info(f"File routing: {len(core_files)} to core (Python/text), {len(docling_files)} to docling (PDFs)")
            
            # Process with core processor
            if core_files:
                logger.info(f"Processing {len(core_files)} files with core processor...")
                try:
                    core_docs = self.core_processor.process_documents(core_files)
                    
                    for i, doc in enumerate(core_docs):
                        file_path = core_files[i]
                        
                        # Skip empty documents
                        if hasattr(doc, 'content') and doc.content and doc.content.strip():
                            documents.append(doc)
                            stats["documents_generated"] += 1
                            stats["total_content_chars"] += len(doc.content)
                            stats["files_by_processor"]["core"] += 1
                            
                            # Log special file types specifically
                            if file_path.suffix.lower() in ['.py', '.pyw']:
                                logger.info(f"  ✓ {file_path.name}: Python file processed with AST support")
                            elif 'conversation' in file_path.name.lower() and file_path.suffix.lower() == '.json':
                                logger.info(f"  ✓ {file_path.name}: Conversation data processed (chat history)")
                            else:
                                logger.info(f"  ✓ {file_path.name}: {len(doc.content)} characters")
                                
                            stats["file_details"].append({
                                "filename": file_path.name,
                                "file_path": str(file_path),
                                "processor": "core",
                                "documents_created": 1,
                                "characters": len(doc.content),
                                "format": file_path.suffix.lower(),
                                "has_ast": file_path.suffix.lower() in ['.py', '.pyw'],
                                "is_conversation": 'conversation' in file_path.name.lower() and file_path.suffix.lower() == '.json'
                            })
                        else:
                            logger.warning(f"  ⚠️ {file_path.name}: Empty content, skipping")
                            
                    stats["files_processed"] += len(core_files)
                    
                except Exception as e:
                    logger.error(f"Core processor batch failed: {e}")
                    # Fall back to individual processing
                    for file_path in core_files:
                        try:
                            doc = self.core_processor.process_document(file_path)
                            if hasattr(doc, 'content') and doc.content and doc.content.strip():
                                documents.append(doc)
                                stats["documents_generated"] += 1
                                stats["total_content_chars"] += len(doc.content)
                                stats["files_by_processor"]["core"] += 1
                                logger.info(f"  ✓ {file_path.name}: {len(doc.content)} characters")
                        except Exception as e:
                            logger.error(f"  ❌ {file_path.name}: {e}")
                    
                    stats["files_processed"] += len(core_files)
            
            # Process with docling processor  
            if docling_files:
                logger.info(f"Processing {len(docling_files)} files with docling processor...")
                try:
                    docling_docs = self.docling_processor.process_documents(docling_files)
                    
                    for i, doc in enumerate(docling_docs):
                        file_path = docling_files[i]
                        
                        # Skip empty documents
                        if hasattr(doc, 'content') and doc.content and doc.content.strip():
                            documents.append(doc)
                            stats["documents_generated"] += 1
                            stats["total_content_chars"] += len(doc.content)
                            stats["files_by_processor"]["docling"] += 1
                            
                            logger.info(f"  ✓ {file_path.name}: {len(doc.content)} characters (PDF extracted)")
                            
                            stats["file_details"].append({
                                "filename": file_path.name,
                                "file_path": str(file_path),
                                "processor": "docling",
                                "documents_created": 1,
                                "characters": len(doc.content),
                                "format": file_path.suffix.lower()
                            })
                        else:
                            logger.warning(f"  ⚠️ {file_path.name}: Empty content, skipping")
                            
                    stats["files_processed"] += len(docling_files)
                    
                except Exception as e:
                    logger.error(f"Docling processor batch failed: {e}")
                    # Fall back to individual processing
                    for file_path in docling_files:
                        try:
                            doc = self.docling_processor.process_document(file_path)
                            if hasattr(doc, 'content') and doc.content and doc.content.strip():
                                documents.append(doc)
                                stats["documents_generated"] += 1
                                stats["total_content_chars"] += len(doc.content)
                                stats["files_by_processor"]["docling"] += 1
                                logger.info(f"  ✓ {file_path.name}: {len(doc.content)} characters")
                        except Exception as e:
                            logger.error(f"  ❌ {file_path.name}: {e}")
                    
                    stats["files_processed"] += len(docling_files)
            
            # Check results
            if not documents:
                error_msg = "No documents were successfully processed"
                logger.error(error_msg)
                return HybridDocumentProcessingResult(
                    success=False,
                    documents=[],
                    stats=stats,
                    error_message=error_msg
                )
            
            # Success summary
            logger.info(f"Hybrid processing completed:")
            logger.info(f"  Total: {len(documents)} documents from {stats['files_processed']} files")
            logger.info(f"  Core: {stats['files_by_processor']['core']} files (Python/text with AST)")
            logger.info(f"  Docling: {stats['files_by_processor']['docling']} files (PDFs)")
            logger.info(f"  Characters: {stats['total_content_chars']:,} total")
            
            # Log Python AST files specifically
            python_files = [f for f in stats['file_details'] if f.get('has_ast', False)]
            if python_files:
                logger.info(f"  🐍 Python files with AST: {len(python_files)}")
            
            return HybridDocumentProcessingResult(
                success=True,
                documents=documents,
                stats=stats
            )
            
        except Exception as e:
            error_msg = f"Hybrid document processing stage failed: {e}"
            error_traceback = traceback.format_exc()
            logger.error(error_msg)
            logger.debug(error_traceback)
            
            return HybridDocumentProcessingResult(
                success=False,
                documents=[],
                stats=stats,
                error_message=error_msg,
                error_traceback=error_traceback
            )
    
    def validate_inputs(self, input_files: List[Path], config: Any) -> List[str]:
        """Validate inputs for hybrid processing."""
        errors = []
        
        if not input_files:
            errors.append("No input files provided")
            return errors
        
        # Check if files exist
        missing_files = [f for f in input_files if not f.exists()]
        if missing_files:
            errors.append(f"Missing input files: {[str(f) for f in missing_files]}")
        
        # Check for at least one supported file
        all_extensions = self.core_extensions | self.docling_extensions
        supported_files = [f for f in input_files if f.exists() and f.suffix.lower() in all_extensions]
        
        if not supported_files:
            errors.append(f"No files with supported formats found")
        
        return errors
    
    def get_expected_outputs(self) -> List[str]:
        """Get list of expected output keys."""
        return ["documents", "stats"]
    
    def estimate_duration(self, input_size: int) -> float:
        """Estimate stage duration based on input size."""
        # PDFs take longer, estimate 2 seconds per file average
        return max(30, input_size * 2)
    
    def get_resource_requirements(self, input_size: int) -> Dict[str, Any]:
        """Get estimated resource requirements."""
        return {
            "memory_mb": max(500, input_size * 10),  # More memory for PDFs
            "cpu_cores": 2,  # Can use multiple cores
            "disk_mb": input_size * 5,
            "network_required": False
        }