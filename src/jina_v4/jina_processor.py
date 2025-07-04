"""
Jina v4 Unified Document Processor

This module implements the core document processing pipeline using Jina v4's multimodal
embeddings. It replaces the previous multi-component architecture with a single, unified
processor that handles all document operations.

Key Capabilities:
- Document parsing (multimodal: text, images, code)
- Embedding generation (single-vector and multi-vector modes)
- Late-chunking with semantic boundary detection
- Attention-based keyword extraction
- Hierarchical relationship preservation
- Docling integration for advanced document parsing
- Fallback parsers for common document formats

Recent Updates:
- Implemented Docling-based document parsing with OCR support
- Added comprehensive fallback parsing for PDF, DOCX, and text files
- Enhanced multimodal support with image extraction capabilities
- Integrated table extraction and metadata preservation

Module Organization:
- JinaV4Processor: Main processor class
  - Document parsing methods (_parse_document)
  - Embedding generation (_generate_embeddings)
  - Late chunking implementation (_perform_late_chunking)
  - Keyword extraction (_extract_keywords)
  - Relationship computation (_compute_relationships)

Implementation Status:
- ⚠️ _parse_document: Placeholder - needs Docling integration
- ⚠️ _extract_embeddings_local: Placeholder - needs vLLM direct access
- ⚠️ _extract_keywords: Placeholder - needs attention extraction
- ⚠️ Semantic operations: Placeholder - needs actual calculations
- ⚠️ Image processing: Placeholder - needs multimodal implementation

Related Resources:
- Research Paper: jina-embeddings-v4.pdf (co-located)
- Configuration: /config/jina_v4/config.yaml
- vLLM Integration: vllm_integration.py
- ISNE Adapter: isne_adapter.py
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import numpy as np
import torch
from datetime import datetime

from ..types.jina_v4.processor import DocumentInput, ProcessingResult
from ..types.common import DocumentType
import time
import asyncio

from .vllm_integration import VLLMEmbeddingExtractor
from .ast_analyzer import ASTAnalyzer

logger = logging.getLogger(__name__)


class JinaV4Processor:
    """
    Unified document processor using Jina v4.
    
    This single component replaces the entire document processing pipeline,
    handling everything from raw documents to ISNE-ready chunks with embeddings
    and keywords.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Jina v4 processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = config.get('device', 'cuda:0')
        self.model_name = config.get('model', 'jinaai/jina-embeddings-v4')
        self.max_input_tokens = config.get('max_input_tokens', 8192)
        self.output_mode = config.get('output_mode', 'multi-vector')
        self.batch_size = config.get('batch_size', 32)
        
        # Late-chunking config
        self.late_chunking_config = config.get('late_chunking', {})
        self.late_chunking_enabled = self.late_chunking_config.get('enabled', True)
        
        # Keyword config
        self.keyword_config = config.get('keywords', {})
        self.keywords_enabled = self.keyword_config.get('enabled', True)
        
        # LoRA adapter
        self.lora_adapter = config.get('lora_adapter', 'retrieval')
        
        # Initialize vLLM integration
        self._initialize_model()
        
        # Initialize AST analyzer for code files
        self.ast_analyzer = ASTAnalyzer()
        
        # Temporary storage for extracted images
        self._extracted_images: List[Dict[str, Any]] = []
        
        # Stats
        self.stats = {
            'documents_processed': 0,
            'total_chunks': 0,
            'total_tokens': 0,
            'processing_time': 0.0
        }
        
        logger.info(f"JinaV4Processor initialized on {self.device}")
    
    def _initialize_model(self) -> None:
        """Initialize the Jina v4 model via vLLM."""
        try:
            logger.info(f"Initializing Jina v4 with vLLM: {self.model_name}")
            
            # Initialize vLLM embedding extractor
            self.embedding_extractor = VLLMEmbeddingExtractor(self.config)
            
            # Log GPU memory if available
            if torch.cuda.is_available() and 'cuda' in self.device:
                device_idx = int(self.device.split(':')[1]) if ':' in self.device else 0
                allocated = torch.cuda.memory_allocated(device_idx) / 1024**3
                total = torch.cuda.get_device_properties(device_idx).total_memory / 1024**3
                logger.info(f"GPU {device_idx} memory: {allocated:.2f}GB / {total:.2f}GB")
                
                # With 96GB VRAM, we have plenty of room
                if total >= 90:
                    logger.info("Detected high-memory GPU, enabling aggressive batching")
                    self.batch_size = max(self.batch_size, 64)
        
        except Exception as e:
            logger.error(f"Failed to initialize Jina v4 model: {e}")
            raise
    
    async def process(
        self,
        file_path: Optional[str] = None,
        content: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a document through the entire pipeline.
        
        Args:
            input_data: Dictionary containing:
                - file_path: Path to the document
                - content: Optional pre-loaded content
                - options: Processing options
                
        Returns:
            Dictionary containing:
                - chunks: List of processed chunks with embeddings and keywords
                - document_metadata: Metadata about the processing
        """
        start_time = time.time()
        
        try:
            # Use provided parameters
            if options is None:
                options = {}
            
            logger.info(f"Processing document: {file_path}")
            
            # Step 1: Load and parse document (multimodal)
            document_data = self._parse_document(file_path, content, options)
            
            # Step 2: Generate embeddings (multi-vector mode)
            embeddings_data = await self._generate_embeddings(
                document_data['text'],
                document_data.get('images', []),
                options,
                ast_analysis=document_data.get('ast_analysis')
            )
            
            # Step 3: Late-chunking (if enabled)
            if self.late_chunking_enabled:
                chunks = self._perform_late_chunking(
                    document_data['text'],
                    embeddings_data,
                    options,
                    ast_analysis=document_data.get('ast_analysis')
                )
            else:
                # Single chunk if late-chunking disabled
                chunks = [{
                    'text': document_data['text'],
                    'embeddings': embeddings_data['embeddings'],
                    'metadata': {'start_token': 0, 'end_token': len(embeddings_data['tokens'])}
                }]
            
            # Step 4: Extract keywords for each chunk
            if self.keywords_enabled:
                chunks = self._extract_keywords(chunks, document_data['text'])
            
            # Step 5: Compute chunk relationships
            chunks = self._compute_relationships(chunks)
            
            # Update stats
            processing_time = time.time() - start_time
            self.stats['documents_processed'] += 1
            self.stats['total_chunks'] += len(chunks)
            self.stats['total_tokens'] += embeddings_data['total_tokens']
            self.stats['processing_time'] += processing_time
            
            # Prepare output
            result = {
                'chunks': chunks,
                'document_metadata': {
                    'file_path': str(file_path) if file_path else None,
                    'total_tokens': embeddings_data['total_tokens'],
                    'total_chunks': len(chunks),
                    'processing_time': processing_time,
                    'model_used': self.model_name,
                    'device': self.device,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            logger.info(f"Document processed: {len(chunks)} chunks in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise
    
    def _parse_document(
        self, 
        file_path: Optional[Union[str, Path]], 
        content: Optional[str],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Parse document content, handling multiple formats including multimodal.
        
        For Python files, uses AST analysis to extract semantic structure.
        
        Returns:
            Dictionary with 'text', 'ast_analysis', and optionally 'images'
        """
        result: Dict[str, Any] = {
            'text': '',
            'ast_analysis': None,
            'images': []
        }
        
        # Clear any previously extracted images
        self._extracted_images = []
        
        # Determine content and file type
        if content:
            text = content
            is_python = options.get('file_type') == 'python' or (
                file_path and str(file_path).endswith('.py')
            )
        elif file_path:
            file_path = Path(file_path)
            
            # Check if it's a Python file
            is_python = file_path.suffix == '.py'
            
            if is_python and file_path.exists():
                # Read Python file
                try:
                    text = file_path.read_text(encoding='utf-8')
                except Exception as e:
                    logger.error(f"Failed to read Python file {file_path}: {e}")
                    text = f"[Error reading {file_path}]"
            else:
                # Use Docling for multimodal document parsing
                text = self._parse_with_docling(file_path, is_python)
        else:
            raise ValueError("Either file_path or content must be provided")
        
        result['text'] = text
        
        # Perform AST analysis for Python files
        if is_python and self.config.get('features', {}).get('ast_analysis', {}).get('enabled', True):
            try:
                logger.info(f"Performing AST analysis for Python file")
                ast_analysis = self.ast_analyzer.analyze_code(
                    text, 
                    filename=str(file_path) if file_path else "unknown.py"
                )
                result['ast_analysis'] = ast_analysis
                
                # Log some stats
                logger.info(
                    f"AST analysis complete: "
                    f"{ast_analysis['stats']['classes']} classes, "
                    f"{ast_analysis['stats']['functions']} functions, "
                    f"{ast_analysis['stats']['imports']} imports"
                )
            except Exception as e:
                logger.warning(f"AST analysis failed: {e}")
                # Continue without AST analysis
        
        # Add any extracted images
        if self._extracted_images:
            result['images'] = self._extracted_images
            logger.info(f"Including {len(self._extracted_images)} extracted images")
            
        return result
    
    def _parse_with_docling(self, file_path: Path, is_python: bool = False) -> str:
        """
        Parse documents using Docling for multimodal support.
        
        Supports: PDF, DOCX, PPTX, XLSX, images, markdown, and more.
        """
        try:
            from docling.document_converter import DocumentConverter
            from docling.datamodel.base_models import InputFormat
            from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
            from docling.pipeline.simple_pipeline import SimplePipeline
            
            # Determine file format
            suffix = file_path.suffix.lower()
            
            # Initialize appropriate converter based on file type
            if suffix == '.pdf':
                # Use standard PDF pipeline for better quality
                converter = DocumentConverter(
                    pipeline_cls=StandardPdfPipeline,
                    pdf_backend='pypdfium2',  # or 'pdfplumber' for better table extraction
                    ocr_enabled=True  # Enable OCR for scanned PDFs
                )
            else:
                # Use simple pipeline for other formats
                converter = DocumentConverter(
                    pipeline_cls=SimplePipeline
                )
            
            # Convert document
            logger.info(f"Parsing {file_path} with Docling")
            result = converter.convert(str(file_path))
            
            # Extract text content
            if result.document:
                # Get main text content
                text_parts = []
                
                # Extract title if available
                if hasattr(result.document, 'title') and result.document.title:
                    text_parts.append(f"Title: {result.document.title}\n")
                
                # Extract main content
                if hasattr(result.document, 'main_text') and result.document.main_text:
                    text_parts.append(result.document.main_text)
                elif hasattr(result.document, 'text') and result.document.text:
                    text_parts.append(result.document.text)
                else:
                    # Fallback: iterate through document elements
                    for element in result.document:
                        if hasattr(element, 'text') and element.text:
                            text_parts.append(element.text)
                
                # Extract tables if present
                if hasattr(result.document, 'tables') and result.document.tables:
                    text_parts.append("\n\nTables:")
                    for i, table in enumerate(result.document.tables):
                        text_parts.append(f"\nTable {i+1}:")
                        # Convert table to text representation
                        if hasattr(table, 'to_text'):
                            text_parts.append(table.to_text())
                        else:
                            text_parts.append(str(table))
                
                # Extract metadata
                if hasattr(result.document, 'metadata') and result.document.metadata:
                    metadata_str = "\n\nMetadata:\n"
                    for key, value in result.document.metadata.items():
                        metadata_str += f"- {key}: {value}\n"
                    text_parts.append(metadata_str)
                
                text = "\n".join(text_parts)
                
                # Store images for multimodal processing
                if hasattr(result.document, 'images') and result.document.images:
                    # Images will be processed separately in multimodal pipeline
                    self._extracted_images = result.document.images
                    logger.info(f"Extracted {len(self._extracted_images)} images from document")
                
                logger.info(f"Successfully parsed {file_path}: {len(text)} characters extracted")
                return text
            else:
                logger.warning(f"Docling returned empty document for {file_path}")
                return f"[Empty document: {file_path}]"
                
        except ImportError as e:
            logger.error(f"Docling not installed: {e}")
            logger.info("Falling back to basic text extraction")
            return self._basic_text_extraction(file_path)
            
        except Exception as e:
            logger.error(f"Error parsing {file_path} with Docling: {e}")
            logger.info("Falling back to basic text extraction")
            return self._basic_text_extraction(file_path)
    
    def _basic_text_extraction(self, file_path: Path) -> str:
        """
        Basic text extraction fallback for common file types.
        """
        try:
            suffix = file_path.suffix.lower()
            
            if suffix in ['.txt', '.md', '.json', '.yaml', '.yml', '.xml', '.csv']:
                # Plain text files
                return file_path.read_text(encoding='utf-8')
                
            elif suffix == '.pdf':
                # Try PyPDF2 as fallback
                try:
                    import PyPDF2
                    text_parts = []
                    with open(file_path, 'rb') as pdf_file:
                        pdf_reader = PyPDF2.PdfReader(pdf_file)
                        for page_num in range(len(pdf_reader.pages)):
                            page = pdf_reader.pages[page_num]
                            text_parts.append(page.extract_text())
                    return "\n".join(text_parts)
                except Exception as e:
                    logger.error(f"PyPDF2 extraction failed: {e}")
                    return f"[PDF extraction failed: {file_path.name}]"
                    
            elif suffix in ['.docx', '.doc']:
                # Try python-docx as fallback
                try:
                    from docx import Document
                    doc = Document(file_path)
                    text_parts = []
                    for paragraph in doc.paragraphs:
                        text_parts.append(paragraph.text)
                    return "\n".join(text_parts)
                except Exception as e:
                    logger.error(f"python-docx extraction failed: {e}")
                    return f"[DOCX extraction failed: {file_path.name}]"
                    
            elif suffix in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                # Image files - return placeholder for now
                return f"[Image file: {file_path.name}]"
                
            else:
                # Unknown format
                return f"[Unsupported format: {file_path.suffix}]"
                
        except Exception as e:
            logger.error(f"Basic text extraction failed for {file_path}: {e}")
            return f"[Error extracting text from {file_path}]"
    
    async def _generate_embeddings(
        self,
        text: str,
        images: List[Any],
        options: Dict[str, Any],
        ast_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate embeddings using Jina v4 via vLLM.
        
        Returns:
            Dictionary with embeddings and metadata
        """
        try:
            # Handle task-specific instructions if enabled
            instruction = None
            if self.config.get('advanced', {}).get('instructions', {}).get('enabled', False):
                task_type = options.get('task_type', 'retrieval')
                if task_type == 'retrieval':
                    instruction = self.config['advanced']['instructions'].get(
                        'retrieval_instruction', 'Encode this passage for retrieval:'
                    )
                elif task_type == 'query':
                    instruction = self.config['advanced']['instructions'].get(
                        'query_instruction', 'Encode this query for retrieval:'
                    )
                
            # Select LoRA adapter if configured
            adapter = options.get('lora_adapter', self.lora_adapter)
            
            # Extract embeddings using vLLM integration
            result = await self.embedding_extractor.extract_embeddings(
                texts=text,
                adapter=adapter,
                instruction=instruction,
                batch_size=self.batch_size
            )
            
            # Process multimodal inputs if present
            if images and self.config.get('features', {}).get('multimodal', {}).get('enabled', True):
                # TODO: Implement image embedding extraction
                logger.warning("Image processing not yet implemented")
                
            # Format result to match expected structure
            embeddings_data = {
                'embeddings': result['embeddings'],
                'tokens': None,  # Not available from vLLM API
                'total_tokens': result['metadata']['token_counts'],
                'mode': result['mode']
            }
                
            return embeddings_data
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def _perform_late_chunking(
        self,
        text: str,
        embeddings_data: Dict[str, Any],
        options: Dict[str, Any],
        ast_analysis: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic late-chunking based on embeddings.
        
        For Python code, uses AST symbol boundaries for intelligent chunking.
        For other content, uses embedding similarity for semantic boundaries.
        """
        chunks = []
        
        # If we have AST analysis, use symbol-based chunking
        if ast_analysis and ast_analysis.get('symbols'):
            logger.info("Using AST-based chunking for Python code")
            
            # Extract symbol chunks using AST analyzer
            symbol_chunks = self.ast_analyzer.extract_symbol_chunks(
                text, 
                max_chunk_size=self.late_chunking_config.get('max_chunk_tokens', 512)
            )
            
            # Process each symbol chunk
            for idx, symbol_chunk in enumerate(symbol_chunks):
                # Get embeddings for this chunk
                # In practice, we'd need to map text positions to embedding indices
                # For now, we'll use placeholder embeddings
                
                chunk_data = {
                    'id': f'symbol_{idx:03d}',
                    'text': symbol_chunk['content'],
                    'embeddings': embeddings_data['embeddings'],  # Placeholder
                    'metadata': {
                        'chunk_type': 'code_symbol',
                        'symbol_name': symbol_chunk['symbol'],
                        'symbol_type': symbol_chunk['type'],
                        'line_range': symbol_chunk['line_range'],
                        'dependencies': symbol_chunk.get('dependencies', []),
                        'docstring': symbol_chunk.get('docstring'),
                        'parent_symbol': symbol_chunk.get('parent'),
                        'has_visual_content': False
                    }
                }
                
                # Add AST-specific keywords
                if 'keywords' in ast_analysis:
                    chunk_data['ast_keywords'] = {
                        'symbols': [symbol_chunk['symbol']],
                        'libraries': [lib for lib in ast_analysis['keywords'].get('libraries', [])
                                    if lib in symbol_chunk['content']],
                        'patterns': ast_analysis['keywords'].get('patterns', [])
                    }
                
                chunks.append(chunk_data)
            
            # Add a summary chunk with imports and overall structure
            if ast_analysis.get('imports'):
                import_text = "# File imports and structure\n"
                for name, module in ast_analysis['imports'].items():
                    import_text += f"import {module} as {name}\n" if name != module else f"import {module}\n"
                
                chunks.insert(0, {
                    'id': 'imports_summary',
                    'text': import_text,
                    'embeddings': embeddings_data['embeddings'],  # Placeholder
                    'metadata': {
                        'chunk_type': 'code_imports',
                        'imports': ast_analysis['imports'],
                        'structure': ast_analysis.get('structure', {}),
                        'stats': ast_analysis.get('stats', {})
                    }
                })
                
        else:
            # Standard semantic chunking for non-code content
            logger.info("Using standard semantic chunking")
            
            # Get chunking parameters
            min_tokens = self.late_chunking_config.get('min_chunk_tokens', 128)
            max_tokens = self.late_chunking_config.get('max_chunk_tokens', 512)
            similarity_threshold = self.late_chunking_config.get('similarity_threshold', 0.85)
            
            # In actual implementation, this would:
            # 1. Compute pairwise similarities between token embeddings
            # 2. Find semantic boundaries where similarity drops
            # 3. Create chunks at those boundaries
            # 4. Ensure chunks are within size limits
            
            # Placeholder: create fixed-size chunks
            tokens = embeddings_data.get('tokens', text.split())
            embeddings = embeddings_data['embeddings']
            
            chunk_size = max_tokens
            for i in range(0, len(tokens), chunk_size // 2):  # 50% overlap
                end = min(i + chunk_size, len(tokens))
                
                chunk_tokens = tokens[i:end]
                chunk_text = ' '.join(chunk_tokens) if isinstance(chunk_tokens[0], str) else str(chunk_tokens)
                
                # Get embeddings for this chunk
                if self.output_mode == 'multi-vector':
                    chunk_embeddings = embeddings[i:end] if isinstance(embeddings, list) else embeddings
                else:
                    chunk_embeddings = embeddings  # Single vector applies to all
                    
                chunks.append({
                    'id': f'chunk_{len(chunks):03d}',
                    'text': chunk_text,
                    'embeddings': chunk_embeddings,
                    'metadata': {
                        'chunk_type': 'text',
                        'start_token': i,
                        'end_token': end,
                        'semantic_density': np.random.rand(),  # Placeholder
                        'has_visual_content': False
                    }
                })
            
        return chunks
    
    def _extract_keywords(
        self,
        chunks: List[Dict[str, Any]],
        full_text: str
    ) -> List[Dict[str, Any]]:
        """
        Extract keywords for each chunk using Jina v4's understanding.
        
        This replaces the need for a separate keyword extraction model.
        """
        for chunk in chunks:
            # In actual implementation, Jina v4 would extract keywords
            # based on the chunk's content and full document context
            
            # Placeholder: extract simple keywords
            words = chunk['text'].lower().split()
            keywords = [w for w in words if len(w) > 5][:10]  # Simple heuristic
            
            chunk['keywords'] = keywords
            
        return chunks
    
    def _compute_relationships(
        self, 
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Compute semantic relationships between chunks.
        
        This helps ISNE build better graphs by providing pre-computed similarities.
        """
        for i, chunk in enumerate(chunks):
            relationships = []
            
            for j, other_chunk in enumerate(chunks):
                if i != j:
                    # Compute similarity between chunks
                    # In practice, this would use cosine similarity of embeddings
                    similarity = np.random.rand() * 0.5 + 0.5  # Placeholder: 0.5-1.0
                    
                    if similarity > 0.7:  # Threshold for significant relationship
                        relationships.append({
                            'target_chunk': other_chunk['id'],
                            'similarity': float(similarity),
                            'type': 'semantic'
                        })
            
            chunk['relationships'] = relationships
            
        return chunks
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            **self.stats,
            'avg_chunks_per_doc': (
                self.stats['total_chunks'] / max(1, self.stats['documents_processed'])
            ),
            'avg_processing_time': (
                self.stats['processing_time'] / max(1, self.stats['documents_processed'])
            ),
            'tokens_per_second': (
                self.stats['total_tokens'] / max(0.01, self.stats['processing_time'])
            )
        }
    
    async def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'embedding_extractor'):
            await self.embedding_extractor.close()
        logger.info("Resources cleaned up")
    
    def clear_cache(self):
        """Clear any internal caches."""
        # Would clear model caches, etc.
        logger.info("Cache cleared")
    
    def __repr__(self):
        return (
            f"JinaV4Processor(model={self.model_name}, device={self.device}, "
            f"output_mode={self.output_mode}, late_chunking={self.late_chunking_enabled})"
        )