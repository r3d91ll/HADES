"""
Tests for the GPU-orchestrated batch engine.

This module contains tests for the orchestrator and pipeline stages.
"""

import sys
import os
import asyncio
import pytest
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import patch, MagicMock, AsyncMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.engine.orchestrator import PipelineOrchestrator, run_pipeline
from src.engine.batch_types import PipelineBatch
from src.engine.stages.base import StageBase
from src.engine.stages.docproc_stage import DocProcStage
from src.engine.stages.chunk_stage import ChunkStage


class MockStage(StageBase):
    """Mock stage for testing."""
    
    def __init__(self, name: str, device: str = "cpu", sleep_time: float = 0.01):
        """Initialize the mock stage."""
        super().__init__(device)
        self.name = name
        self.sleep_time = sleep_time
        self.processed_batches = 0
    
    async def process_batch(self, batch: PipelineBatch) -> PipelineBatch:
        """Process a batch by sleeping and incrementing a counter."""
        # Simulate processing time
        await asyncio.sleep(self.sleep_time)
        
        # Update batch metadata
        batch.metadata[f"{self.name}_processed"] = True
        
        # Increment counter
        self.processed_batches += 1
        
        return batch


@pytest.fixture
def temp_docs():
    """Create temporary document files for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        for i in range(5):
            file_path = Path(temp_dir) / f"test_{i}.txt"
            with open(file_path, "w") as f:
                f.write(f"Test document {i}\n")
        
        yield temp_dir


@pytest.mark.asyncio
async def test_orchestrator_initialization():
    """Test that the orchestrator initializes correctly."""
    # Create orchestrator
    orchestrator = PipelineOrchestrator()
    
    # Check that configuration was loaded
    assert orchestrator.config is not None
    assert orchestrator.config.pipeline.batch_size == 128
    
    # Check that stage devices were set up
    assert "DocProc" in orchestrator.stage_devices
    assert "Chunk" in orchestrator.stage_devices


@pytest.mark.asyncio
async def test_create_batch_from_files():
    """Test creating a batch from files."""
    # Create orchestrator
    orchestrator = PipelineOrchestrator()
    
    # Create temporary files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        file_paths = []
        for i in range(3):
            file_path = Path(temp_dir) / f"test_{i}.txt"
            with open(file_path, "w") as f:
                f.write(f"Test document {i}\n")
            file_paths.append(file_path)
        
        # Create batch
        batch = await orchestrator.create_batch_from_files(file_paths)
        
        # Check batch
        assert batch.batch_id.startswith("batch_")
        assert len(batch.docs) == 3
        assert all("path" in doc for doc in batch.docs)
        assert all("id" in doc for doc in batch.docs)
        assert all("type" in doc for doc in batch.docs)
        assert all(doc["type"] == "txt" for doc in batch.docs)


@pytest.mark.asyncio
async def test_pipeline_with_mock_stages():
    """Test the pipeline with mock stages."""
    # Create orchestrator
    orchestrator = PipelineOrchestrator(batch_size=2, queue_depth=2)
    
    # Replace stage creation with mock stages
    def create_mock_stages():
        orchestrator.stages = {
            "DocProc": MockStage("DocProc", "cpu", 0.01),
            "Chunk": MockStage("Chunk", "cpu", 0.01)
        }
    
    orchestrator.create_stages = create_mock_stages
    
    # Start the pipeline
    await orchestrator.start()
    
    try:
        # Create test batch
        batch = PipelineBatch(
            batch_id="test_batch",
            docs=[
                {"id": "doc1", "path": "/test/doc1.txt", "type": "txt"},
                {"id": "doc2", "path": "/test/doc2.txt", "type": "txt"}
            ]
        )
        
        # Process the batch
        await orchestrator.queues["input"].put(batch)
        
        # Wait for processing to complete
        await orchestrator.queues["input"].join()
        for queue_name, queue in orchestrator.queues.items():
            if queue_name != "input":
                await queue.join()
        
        # Check that all stages processed the batch
        for stage_name, stage in orchestrator.stages.items():
            assert stage.processed_batches == 1, f"Stage {stage_name} did not process the batch"
    finally:
        # Stop the pipeline
        await orchestrator.stop()


@pytest.mark.asyncio
async def test_process_directory(temp_docs):
    """Test processing a directory of files."""
    # Create orchestrator
    orchestrator = PipelineOrchestrator(batch_size=2, queue_depth=2)
    
    # Replace stage creation with mock stages
    def create_mock_stages():
        orchestrator.stages = {
            "DocProc": MockStage("DocProc", "cpu", 0.01),
            "Chunk": MockStage("Chunk", "cpu", 0.01)
        }
    
    orchestrator.create_stages = create_mock_stages
    
    # Start the pipeline
    await orchestrator.start()
    
    try:
        # Process the directory
        await orchestrator.process_directory(temp_docs, batch_size=2)
        
        # Wait for processing to complete
        await orchestrator.queues["input"].join()
        for queue_name, queue in orchestrator.queues.items():
            if queue_name != "input":
                await queue.join()
        
        # Check metrics
        assert orchestrator.metrics["total_batches"] > 0
        assert orchestrator.metrics["total_documents"] == 5  # 5 test files
        
        # Check that all stages processed all batches
        expected_batches = 3  # 5 files with batch size 2 = 3 batches
        for stage_name, stage in orchestrator.stages.items():
            assert stage.processed_batches == expected_batches, \
                f"Stage {stage_name} processed {stage.processed_batches} batches, expected {expected_batches}"
    finally:
        # Stop the pipeline
        await orchestrator.stop()


@pytest.mark.asyncio
async def test_run_pipeline(temp_docs):
    """Test the run_pipeline function."""
    # Patch the PipelineOrchestrator class
    with patch("src.engine.orchestrator.PipelineOrchestrator") as mock_orchestrator_class:
        # Create a mock orchestrator instance
        mock_orchestrator = MagicMock()
        
        # Configure the mock to handle async methods
        mock_orchestrator.start = AsyncMock()
        mock_orchestrator.process_directory = AsyncMock()
        mock_orchestrator.stop = AsyncMock()
        mock_orchestrator.queues = {"input": AsyncMock()}
        mock_orchestrator.queues["input"].join = AsyncMock()
        
        # Set up metrics
        mock_orchestrator.get_metrics.return_value = {
            "total_batches": 3,
            "total_documents": 5,
            "total_chunks": 10,
            "errors": 0,
            "elapsed_seconds": 0.1
        }
        mock_orchestrator_class.return_value = mock_orchestrator
        
        # Run the pipeline
        metrics = await run_pipeline(
            input_dir=temp_docs,
            batch_size=2,
            queue_depth=2
        )
        
        # Check that the orchestrator was created and used correctly
        mock_orchestrator_class.assert_called_once()
        mock_orchestrator.start.assert_called_once()
        mock_orchestrator.process_directory.assert_called_once()
        mock_orchestrator.stop.assert_called_once()
        mock_orchestrator.get_metrics.assert_called_once()
        
        # Check metrics
        assert metrics["total_batches"] == 3
        assert metrics["total_documents"] == 5
        assert metrics["total_chunks"] == 10
        assert metrics["errors"] == 0


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
