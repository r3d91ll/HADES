"""
Model Engine Integration Test

This test validates the integration between the preprocessing pipeline, chunking, 
and the model_engine for embedding generation. It processes both code (Python) and
text (PDF) files through the full pipeline and verifies that:

1. Files are read and preprocessed correctly
2. Appropriate chunking is applied based on file type
3. The model_engine can be initialized with vLLM
4. Chunks can be prepared for embedding

This provides confidence that the model_engine refactor works in a real-world
context with actual content processing.
"""

import os
import sys
import json
import pytest
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import new document processing module
from src.docproc.core import process_document
from src.docproc.adapters import get_adapter_for_format

# Legacy import kept for backward compatibility during transition
from src.ingest.pre_processor.manager import PreprocessorManager

# Import model_engine components
from src.model_engine.adapters.vllm_adapter import VLLMAdapter
from src.model_engine.server_manager import ServerManager
from src.config.model_config import ModelConfig

# Import ISNE components
from src.isne.processors.chunking_processor import ChunkingProcessor
from src.isne.types.models import IngestDocument

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_files(tmp_path):
    """Create sample Python and use a real PDF file for testing."""
    # Create sample Python file
    py_path = tmp_path / "sample.py"
    py_path.write_text('''
import os
import sys
from typing import List, Dict, Any

class ModelProcessor:
    """A class for processing language models."""
    
    def __init__(self, model_name: str, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """Initialize the model processor."""
        print(f"Initializing model: {self.model_name}")
        self.is_initialized = True
        return self.is_initialized
        
    def process_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Process a batch of texts."""
        if not self.is_initialized:
            raise RuntimeError("Model not initialized")
            
        results = []
        for text in texts:
            # Simulate processing
            result = {
                "text": text,
                "score": len(text) / 100,
                "tokens": len(text.split())
            }
            results.append(result)
            
        return results
        
def main():
    """Main function to demonstrate the processor."""
    processor = ModelProcessor("gpt-3.5-turbo")
    processor.initialize()
    
    sample_texts = [
        "This is a sample text for processing.",
        "Language models are powerful tools for NLP tasks."
    ]
    
    results = processor.process_batch(sample_texts)
    for idx, result in enumerate(results):
        print(f"Result {idx+1}: {result}")
    
if __name__ == "__main__":
    main()
''')

    # Use a real PDF file for testing now that we have proper PDF support
    source_pdf = Path("/home/todd/ML-Lab/Olympus/HADES-PathRAG/docs/2502.14902v1.pdf")
    pdf_path = tmp_path / "2502.14902v1.pdf"
    
    # Copy the real PDF to the temporary directory
    import shutil
    shutil.copy(source_pdf, pdf_path)
    
    # For reference, this PDF is approximately the following (commented out):
    """
    # Original PDF was used instead of this text representation
    # Neural Language Models: A Technical Overview

    ## Abstract

    This document provides a technical overview of neural language models,
    their architecture, training procedures, and applications. We explore
    transformer-based models and their implications for natural language processing.

    ## 1. Introduction

    Language models have revolutionized natural language processing in recent years.
    These models can understand and generate human language with remarkable accuracy.
    The development of transformer architecture by Vaswani et al. has been particularly
    significant in advancing the state of the art.

    ### 1.1 Historical Context

    Early language models relied on n-gram statistics and simple neural networks.
    These approaches had limited context windows and struggled with long-range dependencies.
    The introduction of recurrent neural networks (RNNs) and later, long short-term memory
    (LSTM) networks improved performance but still had limitations.

    ## 2. Transformer Architecture

    The transformer architecture introduced the self-attention mechanism, which allows
    the model to weigh the importance of different words in a sequence when processing
    a specific word. This eliminated the sequential bottleneck of RNNs.

    ### 2.1 Self-Attention Mechanism

    The self-attention mechanism computes attention scores between all pairs of words
    in a sequence, allowing the model to focus on relevant parts of the input regardless
    of their distance. This is implemented through query, key, and value projections.

    ### 2.2 Multi-Head Attention

    Multi-head attention extends the self-attention mechanism by applying multiple
    attention operations in parallel. This allows the model to attend to information
    from different representation subspaces at different positions.

    ## 3. Training Methodology

    Training large language models requires significant computational resources and
    careful optimization. Techniques like batch normalization, gradient accumulation,
    and mixed-precision training have been crucial for scaling these models.

    ### 3.1 Pretraining Objectives

    Modern language models typically use self-supervised objectives for pretraining.
    These include masked language modeling (MLM), next sentence prediction (NSP),
    and causal language modeling (CLM).

    ### 3.2 Fine-tuning Approaches

    After pretraining, models can be fine-tuned on specific tasks with smaller amounts
    of labeled data. This transfer learning approach has been remarkably effective.

    ## 4. Scaling Laws

    Research has shown that performance scales predictably with model size, dataset size,
    and computational budget. These scaling laws have guided the development of increasingly
    large models with billions of parameters.

    ## 5. Future Directions

    Future research will focus on improving efficiency, interpretability, and safety.
    As these models continue to scale and become more capable, researchers must address
    the ethical implications and potential misuse.

    ## 6. Conclusion

    Neural language models represent a significant advance in artificial intelligence.
    As they continue to evolve and improve their understanding of human language,
    they will enable increasingly sophisticated applications and push the boundaries
    of artificial intelligence.
    """
    
    return py_path, pdf_path


class TestModelEngineIntegration:
    """Integration tests for the model_engine with the preprocessing pipeline."""
    
    @pytest.fixture
    def sample_files(self, tmp_path):
        """Create sample files for testing."""
        return create_sample_files(tmp_path)
    
    @pytest.fixture
    def output_dir(self, tmp_path):
        """Create output directory for chunks."""
        output_path = tmp_path / "output"
        output_path.mkdir(exist_ok=True)
        return output_path
    
    def test_preprocessing_to_chunking_pipeline(self, sample_files, output_dir):
        """
        Test the pipeline from preprocessing to chunking with both code and text.
        
        This test verifies that documents can be read, preprocessed, and chunked correctly.
        """
        py_path, pdf_path = sample_files
        
        # Step 1: Process Python file using new docproc module
        py_processed = process_document(py_path, options={"create_symbol_table": True})
        
        assert py_processed is not None, "Python preprocessing failed"
        assert "content" in py_processed, "Python content missing"
        assert "symbols" in py_processed, "Symbol extraction failed"
        assert "language" in py_processed, "Language detection failed"
        assert py_processed["language"] == "python", "File not recognized as Python"
        
        # Step 2: Create IngestDocument for chunking
        py_doc = IngestDocument(
            id=str(py_path),
            content=py_processed["content"],
            source=str(py_path),
            document_type="python",
            metadata={
                "path": str(py_path),
                "type": "python",
                "symbols": py_processed.get("symbols", [])
            }
        )
        
        # Step 3: Process PDF file using new docproc module
        pdf_processed = process_document(pdf_path)
        
        assert pdf_processed is not None, "PDF preprocessing failed"
        assert "content" in pdf_processed, "PDF content missing"
        assert "format" in pdf_processed, "Format detection failed"
        assert pdf_processed["format"] == "pdf", "File not recognized as PDF"
        
        # Step 4: Create IngestDocument for chunking
        pdf_doc = IngestDocument(
            id=str(pdf_path),
            content=pdf_processed["content"],
            source=str(pdf_path),
            document_type="pdf",
            metadata={
                "path": str(pdf_path),
                "type": "pdf"
            }
        )
        
        # Step 5: Initialize Chunking Processor
        chunking_processor = ChunkingProcessor(
            chunk_size=500,
            chunk_overlap=50,
            splitting_strategy="paragraph"
        )
        
        # Step 6: Process documents
        chunking_result = chunking_processor.process([py_doc, pdf_doc])
        
        assert len(chunking_result.documents) > 2, "Chunking should produce more documents than input"
        assert chunking_result.errors == [], "Chunking produced errors"
        
        # Step 7: Write chunks to output files for verification
        # Print document IDs for debugging
        logger.info(f"Python document ID: {py_doc.id}")
        logger.info(f"PDF document ID: {pdf_doc.id}")
        
        # Examine all chunks to debug filtering
        for idx, doc in enumerate(chunking_result.documents):
            logger.info(f"Chunk {idx}: type={doc.document_type}, parent_id={doc.metadata.get('parent_id', 'None')}")
        
        # Filter chunks by document_type instead of parent_id
        py_chunks = [doc for doc in chunking_result.documents if doc.document_type == "python"]
        pdf_chunks = [doc for doc in chunking_result.documents if doc.document_type == "pdf"]
        
        # Write Python chunks
        py_chunks_file = output_dir / "python_chunks.json"
        with open(py_chunks_file, "w") as f:
            json.dump([{
                "id": chunk.id,
                "content": chunk.content,
                "metadata": chunk.metadata
            } for chunk in py_chunks], f, indent=2)
        
        # Write PDF chunks
        pdf_chunks_file = output_dir / "pdf_chunks.json"
        with open(pdf_chunks_file, "w") as f:
            json.dump([{
                "id": chunk.id,
                "content": chunk.content,
                "metadata": chunk.metadata
            } for chunk in pdf_chunks], f, indent=2)
        
        logger.info(f"Wrote {len(py_chunks)} Python chunks to {py_chunks_file}")
        logger.info(f"Wrote {len(pdf_chunks)} PDF chunks to {pdf_chunks_file}")
        
        assert len(py_chunks) > 0, "No Python chunks produced"
        assert len(pdf_chunks) > 0, "No PDF chunks produced"
        
        return py_chunks, pdf_chunks
    
    @pytest.mark.asyncio
    async def test_model_engine_initialization(self):
        """
        Test that the model_engine can be initialized with the proper configuration.
        
        This verifies that the model_engine can load and initialize a model adapter
        without errors.
        """
        # Skip test if no GPU or not in CI pipeline to avoid long download times
        if not os.environ.get("CI") and not os.path.exists("/dev/nvidia0"):
            pytest.skip("Skipping model initialization test on non-GPU environment")
        
        # Initialize model adapter with a small model for testing
        adapter = VLLMAdapter(
            model_name="BAAI/bge-small-en-v1.5",  # Using a smaller model for testing
            server_url=None,  # Will use default
            batch_size=4,  # Small batch size for testing
            device="cuda" if os.path.exists("/dev/nvidia0") else "cpu",
            normalize_embeddings=True
        )
        
        # Check if adapter is properly configured
        assert adapter.model_name == "BAAI/bge-small-en-v1.5", "Model name not set correctly"
        assert adapter.batch_size == 4, "Batch size not set correctly"
        
        # Note: We're not checking is_available since that would require a running server
        logger.info(f"Model adapter initialized with model: {adapter.model_name}")
    
    def test_prepare_chunks_for_embedding(self, sample_files, output_dir):
        """
        Test preparing chunks for embedding with the model_engine.
        
        This test verifies that chunks can be properly prepared for the embedding
        model without actually running the model.
        """
        # Get chunks from the preprocessing pipeline
        py_chunks, pdf_chunks = self.test_preprocessing_to_chunking_pipeline(sample_files, output_dir)
        
        # Combine chunks for embedding
        all_chunks = py_chunks + pdf_chunks
        texts_to_embed = [chunk.content for chunk in all_chunks]
        
        # Prepare embeddings file (would normally be sent to the model)
        embeddings_file = output_dir / "chunks_for_embedding.json"
        with open(embeddings_file, "w") as f:
            json.dump({
                "chunk_count": len(texts_to_embed),
                "chunks": [
                    {
                        "id": chunk.id,
                        "content": chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content,
                        "type": chunk.metadata.get("type", "unknown"),
                        "token_count": len(chunk.content.split())
                    }
                    for chunk in all_chunks
                ]
            }, f, indent=2)
        
        logger.info(f"Prepared {len(texts_to_embed)} chunks for embedding, saved to {embeddings_file}")
        
        # This is as far as we can go without actually running the model
        # In a full integration test, we would:
        # 1. Initialize a vLLM server
        # 2. Send chunks for embedding
        # 3. Process the embeddings
        
        assert len(texts_to_embed) > 0, "No texts prepared for embedding"
        assert os.path.exists(embeddings_file), "Embeddings file not created"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
