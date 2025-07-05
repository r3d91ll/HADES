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
- Jina v4 multimodal document parsing (placeholder for implementation)
- Fallback parsers for common document formats

Recent Updates:
- Implemented placeholder for Jina v4 multimodal document parsing
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
- ⚠️ _parse_document: Placeholder - needs Jina v4 multimodal integration
- ⚠️ _extract_embeddings_local: Placeholder - needs vLLM direct access
- ⚠️ _extract_keywords: Placeholder - needs attention extraction
- ⚠️ Semantic operations: Placeholder - needs actual calculations
- ⚠️ Image processing: Placeholder - needs multimodal implementation

Related Resources:
- Research Paper: jina-embeddings-v4.pdf (co-located)
- Configuration: src/config/jina_v4/config.yaml
- vLLM Integration: vllm_integration.py
- ISNE Adapter: isne_adapter.py
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import numpy as np
from datetime import datetime

# Ensure version compatibility  
from src.utils.version_checker import ensure_torch_available
ensure_torch_available()

import torch

from ..types.jina_v4.processor import DocumentInput, ProcessingResult
from ..types.common import DocumentType
import time
import asyncio

from .vllm_integration import VLLMEmbeddingExtractor
from .ast_analyzer import ASTAnalyzer
from ..bridges.bridge_detector import BridgeDetector
from .parsers import XMLStructureParser, YAMLJSONParser, ImageParser, LaTeXParser, TOMLParser

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
        
        # Initialize bridge detector
        self.bridge_detector = BridgeDetector({
            'confidence_thresholds': {
                'citation_match': 0.9,
                'algorithm_name': 0.8,
                'symbol_match': 0.85,
                'fuzzy_match': 0.6
            }
        })
        
        # Temporary storage for extracted images
        self._extracted_images: List[Dict[str, Any]] = []
        
        # Temporary storage for detected bridges
        self._detected_bridges: List[Any] = []
        
        # Temporary storage for parsed metadata
        self._current_config_metadata: Dict[str, Any] = {}
        
        # Temporary storage for multimodal content
        self._current_image_data = None
        self._current_visual_features: Dict[str, Any] = {}
        self._has_multimodal_content = False
        
        # Temporary storage for LaTeX content
        self._current_latex_metadata: Dict[str, Any] = {}
        self._current_math_content: Dict[str, Any] = {}
        self._has_algorithmic_content = False
        
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
    
    async def process_directory(self, directory_path: str, **kwargs) -> Dict[str, Any]:
        """Process all files in a directory.
        
        Args:
            directory_path: Path to directory
            **kwargs: Additional options
            
        Returns:
            Processing results
        """
        # TODO: Implement directory processing
        logger.warning("process_directory not fully implemented")
        return {
            "status": "completed",
            "files_processed": 0,
            "errors": []
        }
    
    async def process_file(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Process a single file.
        
        Args:
            file_path: Path to file
            **kwargs: Additional options
            
        Returns:
            Processing results
        """
        # Use the main process method
        result = await self.process(file_path=file_path)
        return result
    
    async def _process_document(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process a document.
        
        Args:
            content: Document content
            metadata: Document metadata
            
        Returns:
            Processing results
        """
        # Process using the main process method
        result = await self.process(content=content)
        return {
            "embedding": result.get("embeddings", [None])[0] if result.get("embeddings") else None,
            "metadata": {**metadata, **result.get("metadata", {})}
        }
    
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
            # Include multimodal content if present
            images_for_embedding = document_data.get('images', [])
            if document_data.get('multimodal', {}).get('has_images'):
                # Add the parsed image data
                images_for_embedding.append(document_data['multimodal']['image_data'])
            
            embeddings_data = await self._generate_embeddings(
                document_data['text'],
                images_for_embedding,
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
            'images': [],
            'bridges': []
        }
        
        # Clear any previously extracted content
        self._extracted_images = []
        self._detected_bridges = []
        self._current_image_data = None
        self._current_visual_features = {}
        self._has_multimodal_content = False
        self._current_latex_metadata = {}
        self._current_math_content = {}
        self._has_algorithmic_content = False
        
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
                # Use Jina v4 for multimodal document parsing
                text = self._parse_with_jina_v4(file_path, is_python)
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
                
                # Detect bridges in Python code
                bridges = self.bridge_detector.detect_bridges_in_code(
                    file_path if file_path else Path("unknown.py"),
                    ast_analysis
                )
                self._detected_bridges.extend(bridges)
                
            except Exception as e:
                logger.warning(f"AST analysis failed: {e}")
                # Continue without AST analysis
        
        # Add any extracted images
        if self._extracted_images:
            result['images'] = self._extracted_images
            logger.info(f"Including {len(self._extracted_images)} extracted images")
        
        # Add detected bridges
        if self._detected_bridges:
            result['bridges'] = self._detected_bridges
            logger.info(f"Detected {len(self._detected_bridges)} theory-practice bridges")
        
        # Add multimodal content if present
        if self._has_multimodal_content:
            result['multimodal'] = {
                'has_images': bool(self._current_image_data),
                'image_data': self._current_image_data,
                'visual_features': self._current_visual_features
            }
            logger.info("Document contains multimodal content")
        
        # Add LaTeX content if present
        if self._current_latex_metadata or self._current_math_content:
            result['latex'] = {
                'metadata': self._current_latex_metadata,
                'math_content': self._current_math_content,
                'has_algorithms': self._has_algorithmic_content
            }
            logger.info(f"Document contains LaTeX content with {self._current_latex_metadata.get('equation_count', 0)} equations")
            
        return result
    
    def _parse_with_jina_v4(self, file_path: Path, is_python: bool = False) -> str:
        """
        Parse documents using Jina v4's multimodal capabilities with file-type awareness.
        
        Different file types receive specialized processing to maintain theory-practice bridges
        while leveraging Jina v4's unified embedding space.
        
        Args:
            file_path: Path to the document
            is_python: Whether this is a Python source file
            
        Returns:
            Extracted text content with structural metadata preserved
        """
        suffix = file_path.suffix.lower()
        
        # Route to appropriate parser based on file type
        if suffix == '.pdf':
            logger.info(f"Processing PDF with structure extraction: {file_path}")
            return self._parse_pdf_with_structure(file_path)
            
        elif suffix in ['.md', '.markdown']:
            logger.info(f"Processing Markdown with hierarchy parsing: {file_path}")
            return self._parse_markdown_with_links(file_path)
            
        elif suffix in ['.tex', '.latex']:
            logger.info(f"Processing LaTeX document: {file_path}")
            return self._parse_latex_document(file_path)
            
        elif suffix in ['.rst', '.adoc']:
            logger.info(f"Processing structured document: {file_path}")
            return self._parse_structured_document(file_path)
            
        elif suffix in ['.ipynb']:
            logger.info(f"Processing Jupyter notebook: {file_path}")
            return self._parse_jupyter_notebook(file_path)
            
        elif suffix in ['.yaml', '.yml', '.json', '.toml', '.xml']:
            logger.info(f"Processing configuration file: {file_path}")
            return self._parse_config_with_schema(file_path)
            
        else:
            # Fall back to basic extraction for other types
            logger.info(f"Using basic extraction for {suffix} file: {file_path}")
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
                    
            elif suffix in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg']:
                # Image files - use multimodal processing
                logger.info(f"Processing image with multimodal parser: {file_path}")
                return self._parse_image_multimodal(file_path)
                
            else:
                # Unknown format
                return f"[Unsupported format: {file_path.suffix}]"
                
        except Exception as e:
            logger.error(f"Basic text extraction failed for {file_path}: {e}")
            return f"[Error extracting text from {file_path}]"
    
    def _parse_image_multimodal(self, file_path: Path) -> str:
        """
        Parse image using multimodal capabilities of Jina v4.
        
        Returns:
            Structured text representation with embedded image data
        """
        try:
            parser = ImageParser(self.config)
            result = parser.parse(file_path)
            
            # Store image data for multimodal embedding
            self._current_image_data = result.get('image_data')
            
            # Store visual features
            self._current_visual_features = result.get('visual_features', {})
            
            # Store detected bridges
            if result.get('bridges'):
                bridges = result['bridges']
                # Enhance bridges with image context
                for bridge in bridges:
                    bridge['source_modality'] = 'image'
                    bridge['visual_context'] = result.get('image_type', 'unknown')
                self._detected_bridges.extend(bridges)
            
            # Mark that we have multimodal content
            self._has_multimodal_content = True
            
            # Return structured content
            content = result.get('content', f"[Image: {file_path.name}]")
            return str(content) if content is not None else f"[Image: {file_path.name}]"
            
        except Exception as e:
            logger.error(f"Error parsing image {file_path}: {e}")
            return f"[Image file: {file_path.name}]"
    
    def _parse_pdf_with_structure(self, file_path: Path) -> str:
        """
        Parse PDF with structure extraction for theory-practice bridges.
        
        Extracts:
        - Section headings and hierarchy
        - Figures and their captions
        - Tables and their titles
        - Citations and references
        - Algorithm blocks
        - Mathematical equations
        
        Returns:
            Structured text with metadata annotations
        """
        try:
            import fitz  # PyMuPDF
            import re
            
            doc = fitz.open(str(file_path))
            structured_content = []
            
            # Track document structure
            self._current_pdf_metadata: Dict[str, List[Any]] = {
                'sections': [],
                'figures': [],
                'tables': [],
                'citations': [],
                'algorithms': [],
                'equations': []
            }
            
            for page_num, page in enumerate(doc):
                # Extract text with positioning
                blocks = page.get_text("dict")
                
                # Process text blocks to identify structure
                for block in blocks.get("blocks", []):
                    if block.get("type") == 0:  # Text block
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                text = span.get("text", "").strip()
                                if not text:
                                    continue
                                
                                # Detect section headings (larger font, bold)
                                font_size = span.get("size", 0)
                                font_flags = span.get("flags", 0)
                                is_bold = font_flags & 2**4
                                
                                if font_size > 12 and is_bold:
                                    # Likely a heading
                                    self._current_pdf_metadata['sections'].append({
                                        'text': text,
                                        'page': page_num + 1,
                                        'level': self._estimate_heading_level(font_size)
                                    })
                                    structured_content.append(f"\n## {text}\n")
                                
                                # Detect citations (look for [1], (Author, Year), etc.)
                                citation_patterns = [
                                    r'\[(\d+)\]',  # [1], [2], etc.
                                    r'\(([A-Z][a-z]+(?:\s+et\s+al\.)?),?\s*(\d{4})\)',  # (Author, 2023)
                                ]
                                for pattern in citation_patterns:
                                    citations = re.findall(pattern, text)
                                    if citations:
                                        self._current_pdf_metadata['citations'].extend(citations)
                                
                                # Detect algorithm blocks
                                if re.match(r'^Algorithm\s+\d+', text, re.IGNORECASE):
                                    self._current_pdf_metadata['algorithms'].append({
                                        'title': text,
                                        'page': page_num + 1
                                    })
                                
                                # Add text to content
                                structured_content.append(text)
                    
                    elif block.get("type") == 1:  # Image block
                        # Track figures
                        self._current_pdf_metadata['figures'].append({
                            'page': page_num + 1,
                            'bbox': block.get("bbox"),
                            'extracted': False  # Will be extracted if needed
                        })
                
                # Extract tables (heuristic: look for grid patterns)
                tables = page.find_tables()
                if tables:
                    for table in tables:
                        self._current_pdf_metadata['tables'].append({
                            'page': page_num + 1,
                            'bbox': table.bbox,
                            'cells': len(table.cells)
                        })
            
            doc.close()
            
            # Detect bridges in PDF
            bridges = self.bridge_detector.detect_bridges_in_pdf(
                file_path,
                self._current_pdf_metadata
            )
            self._detected_bridges.extend(bridges)
            
            # Add metadata summary at the beginning
            metadata_summary = f"[PDF Structure Analysis]\n"
            metadata_summary += f"Sections: {len(self._current_pdf_metadata['sections'])}\n"
            metadata_summary += f"Figures: {len(self._current_pdf_metadata['figures'])}\n"
            metadata_summary += f"Tables: {len(self._current_pdf_metadata['tables'])}\n"
            metadata_summary += f"Citations: {len(set(str(c) for c in self._current_pdf_metadata['citations']))}\n"
            metadata_summary += f"Algorithms: {len(self._current_pdf_metadata['algorithms'])}\n"
            metadata_summary += f"Bridges: {len(bridges)}\n\n"
            
            return metadata_summary + '\n'.join(structured_content)
            
        except ImportError:
            logger.warning("PyMuPDF not available, trying alternative PDF parser")
            # Fall back to PyPDF2
            return self._basic_text_extraction(file_path)
        except Exception as e:
            logger.error(f"Error parsing PDF structure: {e}")
            return self._basic_text_extraction(file_path)
    
    def _estimate_heading_level(self, font_size: float) -> int:
        """Estimate heading level based on font size."""
        if font_size >= 18:
            return 1
        elif font_size >= 14:
            return 2
        elif font_size >= 12:
            return 3
        else:
            return 4
    
    def _parse_markdown_with_links(self, file_path: Path) -> str:
        """
        Parse Markdown with hierarchy and link analysis for theory-practice bridges.
        
        Extracts:
        - Heading hierarchy
        - Internal and external links
        - Code blocks with language
        - API references
        - Cross-references to other documents
        
        Returns:
            Structured text with link metadata preserved
        """
        try:
            import re
            
            content = file_path.read_text(encoding='utf-8')
            
            # Track markdown structure
            self._current_markdown_metadata: Dict[str, List[Any]] = {
                'headings': [],
                'links': [],
                'code_blocks': [],
                'api_refs': [],
                'cross_refs': []
            }
            
            lines = content.split('\n')
            current_heading_stack: List[str] = []
            
            for line_num, line in enumerate(lines):
                # Extract headings
                heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
                if heading_match:
                    level = len(heading_match.group(1))
                    text = heading_match.group(2)
                    
                    # Update heading stack
                    current_heading_stack = current_heading_stack[:level-1]
                    current_heading_stack.append(text)
                    
                    self._current_markdown_metadata['headings'].append({
                        'level': level,
                        'text': text,
                        'line': line_num + 1,
                        'path': '/'.join(current_heading_stack)
                    })
                
                # Extract links
                # Standard markdown links: [text](url)
                link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
                for match in re.finditer(link_pattern, line):
                    link_text = match.group(1)
                    link_url = match.group(2)
                    
                    link_info = {
                        'text': link_text,
                        'url': link_url,
                        'line': line_num + 1,
                        'context': current_heading_stack[-1] if current_heading_stack else 'root'
                    }
                    
                    # Classify link type
                    if link_url.startswith('#'):
                        link_info['type'] = 'internal_anchor'
                    elif link_url.startswith('http'):
                        link_info['type'] = 'external'
                    elif link_url.endswith('.md'):
                        link_info['type'] = 'cross_reference'
                        self._current_markdown_metadata['cross_refs'].append(link_info)
                    elif link_url.endswith(('.py', '.js', '.java')):
                        link_info['type'] = 'code_reference'
                    else:
                        link_info['type'] = 'file_reference'
                    
                    self._current_markdown_metadata['links'].append(link_info)
                
                # Extract API references (look for class/function mentions)
                api_pattern = r'`([A-Z][a-zA-Z0-9_]*(?:\.[a-zA-Z0-9_]+)*)`'
                for match in re.finditer(api_pattern, line):
                    api_ref = match.group(1)
                    if '.' in api_ref or api_ref[0].isupper():  # Likely a class or module
                        self._current_markdown_metadata['api_refs'].append({
                            'name': api_ref,
                            'line': line_num + 1,
                            'context': current_heading_stack[-1] if current_heading_stack else 'root'
                        })
            
            # Extract code blocks
            code_block_pattern = r'```(\w*)\n(.*?)\n```'
            for match in re.finditer(code_block_pattern, content, re.DOTALL):
                language = match.group(1) or 'plain'
                code = match.group(2)
                
                self._current_markdown_metadata['code_blocks'].append({
                    'language': language,
                    'content': code,
                    'length': len(code.split('\n')),
                    'start_pos': match.start()
                })
            
            # Detect bridges in Markdown
            bridges = self.bridge_detector.detect_bridges_in_markdown(
                file_path,
                self._current_markdown_metadata
            )
            self._detected_bridges.extend(bridges)
            
            # Add metadata summary
            metadata_summary = f"[Markdown Structure Analysis]\n"
            metadata_summary += f"Headings: {len(self._current_markdown_metadata['headings'])}\n"
            metadata_summary += f"Links: {len(self._current_markdown_metadata['links'])}\n"
            metadata_summary += f"Code blocks: {len(self._current_markdown_metadata['code_blocks'])}\n"
            metadata_summary += f"API references: {len(self._current_markdown_metadata['api_refs'])}\n"
            metadata_summary += f"Cross-references: {len(self._current_markdown_metadata['cross_refs'])}\n"
            metadata_summary += f"Bridges: {len(bridges)}\n\n"
            
            return metadata_summary + content
            
        except Exception as e:
            logger.error(f"Error parsing markdown structure: {e}")
            return self._basic_text_extraction(file_path)
    
    def _parse_latex_document(self, file_path: Path) -> str:
        """
        Parse LaTeX document with equation and citation extraction.
        
        Extracts:
        - Document structure (sections, subsections)
        - Mathematical content (equations, theorems, algorithms)
        - Citations and bibliography
        - Cross-references and labels
        - Theory-practice bridges
        
        Returns:
            Structured text with LaTeX metadata preserved
        """
        try:
            parser = LaTeXParser(self.config)
            result = parser.parse(file_path)
            
            # Store LaTeX-specific metadata
            self._current_latex_metadata = result.get('metadata', {})
            
            # Store mathematical content for specialized processing
            self._current_math_content = result.get('math_content', {})
            
            # Detect bridges in LaTeX
            if result.get('bridges'):
                bridges = result['bridges']
                # Enhance bridges with LaTeX context
                for bridge in bridges:
                    bridge['source_type'] = 'latex'
                    bridge['doc_type'] = result.get('doc_type', 'unknown')
                self._detected_bridges.extend(bridges)
            
            # If this is a research paper with algorithms, mark for special handling
            if result.get('doc_type') == 'research_paper' and result.get('math_content', {}).get('algorithms'):
                self._has_algorithmic_content = True
            
            # Return structured content
            content = result.get('content', '')
            return str(content) if content is not None else ''
            
        except Exception as e:
            logger.error(f"Error parsing LaTeX document {file_path}: {e}")
            return self._basic_text_extraction(file_path)
    
    def _parse_structured_document(self, file_path: Path) -> str:
        """Parse other structured documents (RST, AsciiDoc, LaTeX)."""
        # For now, use basic extraction
        # TODO: Add specific parsers for each format
        return self._basic_text_extraction(file_path)
    
    def _parse_jupyter_notebook(self, file_path: Path) -> str:
        """Parse Jupyter notebook preserving code-output relationships."""
        try:
            import json
            
            with open(file_path, 'r') as f:
                notebook = json.load(f)
            
            content_parts = []
            
            for cell in notebook.get('cells', []):
                cell_type = cell.get('cell_type')
                
                if cell_type == 'markdown':
                    content_parts.append(''.join(cell.get('source', [])))
                elif cell_type == 'code':
                    # Add code
                    content_parts.append('```python')
                    content_parts.append(''.join(cell.get('source', [])))
                    content_parts.append('```')
                    
                    # Add outputs if present
                    outputs = cell.get('outputs', [])
                    if outputs:
                        content_parts.append('\n[Output]')
                        for output in outputs:
                            if 'text' in output:
                                content_parts.append(''.join(output['text']))
                            elif 'data' in output and 'text/plain' in output['data']:
                                content_parts.append(''.join(output['data']['text/plain']))
            
            return '\n'.join(content_parts)
            
        except Exception as e:
            logger.error(f"Error parsing Jupyter notebook: {e}")
            return self._basic_text_extraction(file_path)
    
    def _parse_config_with_schema(self, file_path: Path) -> str:
        """Parse configuration files preserving schema information."""
        try:
            suffix = file_path.suffix.lower()
            parser: Union[XMLStructureParser, YAMLJSONParser, TOMLParser]
            
            if suffix == '.xml':
                logger.info(f"Processing XML configuration: {file_path}")
                parser = XMLStructureParser(self.config)
                result = parser.parse(file_path)
                
                # Store metadata and bridges
                self._current_config_metadata = result.get('metadata', {})
                if result.get('bridges'):
                    self._detected_bridges.extend(result['bridges'])
                
                content = result.get('content', '')
                return str(content) if content is not None else ''
            
            elif suffix in ['.yaml', '.yml', '.json']:
                logger.info(f"Processing {suffix} configuration: {file_path}")
                parser = YAMLJSONParser(self.config)
                result = parser.parse(file_path)
                
                # Store metadata and bridges
                self._current_config_metadata = result.get('metadata', {})
                if result.get('bridges'):
                    self._detected_bridges.extend(result['bridges'])
                
                content = result.get('content', '')
                return str(content) if content is not None else ''
            
            elif suffix == '.toml':
                logger.info(f"Processing TOML configuration: {file_path}")
                parser = TOMLParser(self.config)
                result = parser.parse(file_path)
                
                # Store metadata and bridges
                self._current_config_metadata = result.get('metadata', {})
                if result.get('bridges'):
                    self._detected_bridges.extend(result['bridges'])
                
                content = result.get('content', '')
                return str(content) if content is not None else ''
            
            else:
                return self._basic_text_extraction(file_path)
                
        except Exception as e:
            logger.error(f"Error parsing config file {file_path}: {e}")
            return self._basic_text_extraction(file_path)
    
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
            
            # Determine task and prompt_name from options
            task = options.get('task', 'retrieval')
            prompt_name = options.get('prompt_name', 'passage')
            
            # Extract embeddings using vLLM integration with Jina v4 capabilities
            result = await self.embedding_extractor.extract_embeddings(
                texts=text,
                adapter=adapter,
                instruction=instruction,
                batch_size=self.batch_size,
                task=task,
                prompt_name=prompt_name
            )
            
            # Process multimodal inputs if present
            if images and self.config.get('features', {}).get('multimodal', {}).get('enabled', True):
                # Process images through Jina v4
                logger.info(f"Processing {len(images)} images for multimodal embedding")
                
                # Extract image data for processing
                image_inputs = []
                for img_data in images:
                    if isinstance(img_data, dict):
                        if 'path' in img_data:
                            image_inputs.append(img_data['path'])
                        elif 'array' in img_data:
                            image_inputs.append(img_data['array'])
                        elif 'url' in img_data:
                            image_inputs.append(img_data['url'])
                    elif isinstance(img_data, str):
                        image_inputs.append(img_data)
                
                if image_inputs:
                    # Extract image embeddings using Jina v4
                    image_result = await self.embedding_extractor.extract_image_embeddings(
                        images=image_inputs,
                        task=task,
                        batch_size=self.batch_size // 4  # Smaller batch for images
                    )
                    
                    # Store image embeddings
                    result['image_embeddings'] = image_result['embeddings']
                    result['is_multimodal'] = True
                    result['metadata']['image_count'] = len(image_inputs)
                
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
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if hasattr(self, 'embedding_extractor'):
            await self.embedding_extractor.close()
        logger.info("Resources cleaned up")
    
    def clear_cache(self) -> None:
        """Clear any internal caches."""
        # Would clear model caches, etc.
        logger.info("Cache cleared")
    
    async def process_with_direct_transformers(
        self,
        texts: Union[str, List[str]],
        images: Optional[Union[str, List[str]]] = None,
        task: str = 'retrieval',
        prompt_name: Optional[str] = None,
        return_multivector: bool = False
    ) -> Dict[str, Any]:
        """
        Process texts/images directly using transformers without vLLM.
        
        This method provides direct access to Jina v4's native capabilities
        using the transformers library.
        
        Args:
            texts: Text(s) to encode
            images: Optional image(s) to encode
            task: Task type (retrieval, text-matching, code)
            prompt_name: Prompt name (query, passage)
            return_multivector: Whether to return multi-vector embeddings
            
        Returns:
            Dictionary with embeddings and metadata
        """
        try:
            # Lazy import to avoid dependency issues
            from transformers import AutoModel
            import torch
            
            # Initialize model if not already done
            if not hasattr(self, '_direct_model'):
                logger.info("Initializing direct transformers model")
                self._direct_model = AutoModel.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if 'cuda' in self.device else torch.float32
                )
                self._direct_model.to(self.device)
                self._direct_model.eval()
            
            # Ensure inputs are lists
            if isinstance(texts, str):
                texts = [texts]
            if images and isinstance(images, str):
                images = [images]
            
            embeddings = []
            
            # Process text embeddings
            if texts:
                with torch.no_grad():
                    text_embeddings = self._direct_model.encode_text(
                        texts=texts,
                        task=task,
                        prompt_name=prompt_name,
                        return_multivector=return_multivector,
                        batch_size=self.batch_size,
                        max_length=self.max_input_tokens
                    )
                    
                    # Convert to numpy
                    if isinstance(text_embeddings, torch.Tensor):
                        text_embeddings = text_embeddings.cpu().numpy()
                    
                    embeddings.extend(text_embeddings if isinstance(text_embeddings, list) else [text_embeddings])
            
            # Process image embeddings
            if images:
                with torch.no_grad():
                    image_embeddings = self._direct_model.encode_image(
                        images=images,
                        task=task,
                        return_multivector=return_multivector,
                        batch_size=self.batch_size // 4  # Smaller batch for images
                    )
                    
                    # Convert to numpy
                    if isinstance(image_embeddings, torch.Tensor):
                        image_embeddings = image_embeddings.cpu().numpy()
                    
                    embeddings.extend(image_embeddings if isinstance(image_embeddings, list) else [image_embeddings])
            
            return {
                'embeddings': embeddings,
                'task': task,
                'prompt_name': prompt_name,
                'multivector': return_multivector,
                'metadata': {
                    'model': self.model_name,
                    'device': self.device,
                    'num_texts': len(texts) if texts else 0,
                    'num_images': len(images) if images else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error in direct transformers processing: {e}")
            raise
    
    def __repr__(self) -> str:
        return (
            f"JinaV4Processor(model={self.model_name}, device={self.device}, "
            f"output_mode={self.output_mode}, late_chunking={self.late_chunking_enabled})"
        )