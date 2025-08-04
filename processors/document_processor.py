"""
High-level Document Processing Interface

Combines PDF extraction, chunking, and embedding generation into
a simple, unified interface for processing academic documents.
"""

from typing import List, Dict, Optional, Union
from pathlib import Path
import json
import logging
from dataclasses import dataclass

# from ..embeddings import LocalJinaGPU, LocalJinaConfig, BatchEmbeddingProcessor
# Temporarily disabled due to corrupted file
from .arxiv_loader import ArxivLoader

# Note: GPUConfig and ProgressTracker will be added when those modules are implemented


@dataclass
class ProcessingConfig:
    """Configuration for document processing"""
    chunking_strategy: str = "late"  # "late", "section", "fixed"
    chunk_size: int = 512
    use_gpu: bool = True
    batch_size: int = 64
    max_workers: int = 4
    

@dataclass
class ProcessingResult:
    """Result of document processing"""
    document_id: str
    chunks: List[str]
    embeddings: List[List[float]]
    metadata: Dict
    

class DocumentProcessor:
    """
    High-level interface for processing academic documents.
    
    Example:
        processor = DocumentProcessor()
        results = processor.process_documents(
            input_dir="/mnt/data/arxiv_data/pdf",
            num_documents=2000,
            output_dir="./results"
        )
        
        # Process with custom configuration
        config = ProcessingConfig(
            chunking_strategy="section",
            use_gpu=True,
            batch_size=128
        )
        processor = DocumentProcessor(config=config)
    """
    
    def __init__(
        self, 
        config: ProcessingConfig = None,
        jina_config: JinaConfig = None,
        gpu_config: Dict = None  # GPU configuration dict
    ):
        self.config = config or ProcessingConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        if not jina_config:
            raise ValueError("JinaConfig is required for document processing")
        self.jina_client = JinaClient(jina_config)
        self.chunker = LateChucker(use_gpu=self.config.use_gpu)
        self.batch_processor = BatchEmbeddingProcessor(
            jina_config=jina_config,
            use_gpu=self.config.use_gpu
        )
        
    def process_documents(
        self,
        input_dir: Union[str, Path],
        num_documents: Optional[int] = None,
        output_dir: Optional[Union[str, Path]] = None
    ) -> List[ProcessingResult]:
        """
        Process documents from input directory.
        
        Args:
            input_dir: Directory containing PDF files
            num_documents: Number of documents to process (None = all)
            output_dir: Directory to save results (optional)
            
        Returns:
            List of ProcessingResult objects
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir) if output_dir else Path("./results")
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load documents
        if "arxiv_data" in input_path.parts:
            # Use ArxivLoader for arXiv dataset
            loader = ArxivLoader(input_path.parent)
            papers = loader.load_papers(
                num_papers=num_documents or 2000,
                sampling_strategy="diverse"
            )
            pdf_paths = [p["pdf_path"] for p in papers]
        else:
            # Load PDFs from directory
            pdf_paths = list(input_path.glob("*.pdf"))
            if num_documents:
                pdf_paths = pdf_paths[:num_documents]
        
        self.logger.info(f"Processing {len(pdf_paths)} documents")
        
        # Process documents
        results = self.batch_processor.process_documents(
            pdf_paths=pdf_paths,
            output_dir=output_path,
            batch_size=self.config.batch_size
        )
        
        # Convert to ProcessingResult objects
        processing_results = []
        for pdf_path in pdf_paths:
            doc_id = Path(pdf_path).stem
            result_file = output_path / f"{doc_id}_embeddings.json"
            
            if result_file.exists():
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    
                processing_results.append(ProcessingResult(
                    document_id=doc_id,
                    chunks=[c["content"] for c in data["chunks"]],
                    embeddings=data["embeddings"],
                    metadata=data["metadata"]
                ))
        
        return processing_results
    
    def process_single_document(
        self,
        pdf_path: Union[str, Path]
    ) -> ProcessingResult:
        """Process a single PDF document"""
        pdf_path = Path(pdf_path)
        
        # Chunk document
        chunk_result = self.chunker.chunk_document(pdf_path)
        
        if not chunk_result["success"]:
            raise RuntimeError(f"Failed to chunk document: {chunk_result.get('error')}")
        
        # Generate embeddings
        chunks = chunk_result["chunks"]
        chunk_texts = [c["content"] for c in chunks]
        
        embeddings = self.jina_client.encode_batch(chunk_texts)
        
        return ProcessingResult(
            document_id=pdf_path.stem,
            chunks=chunk_texts,
            embeddings=embeddings,
            metadata=chunk_result["metadata"]
        )