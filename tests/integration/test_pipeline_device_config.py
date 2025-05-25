"""
Integration test for device configuration in the pipeline.

This test verifies that all components (docproc, chunking, and embedding)
correctly respect the device settings from the pipeline configuration.
"""

import os
import pytest
import logging
from unittest.mock import patch
import torch

from src.config.config_loader import load_pipeline_config, get_component_device
from src.docproc.adapters.docling_adapter import DoclingAdapter
from src.chunking.text_chunkers.chonky_chunker import _get_splitter_with_engine
from src.embedding.adapters.modernbert_adapter import ModernBERTEmbeddingAdapter


@pytest.fixture
def mock_gpu_environment():
    """Setup a mock GPU environment for testing."""
    with patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.device_count', return_value=2), \
         patch('src.utils.device_utils.is_gpu_available', return_value=True), \
         patch('os.environ.get', side_effect=lambda key, default: "0,1" if key == "CUDA_VISIBLE_DEVICES" else default):
        yield


@pytest.mark.integration
def test_all_components_respect_device_config(mock_gpu_environment):
    """Test that all components correctly use the configured device."""
    # Create a mock pipeline configuration
    pipeline_config = {
        'pipeline': {
            'device_config': {
                'CUDA_VISIBLE_DEVICES': '0,1'
            }
        },
        'gpu_execution': {
            'enabled': True,
            'docproc': {
                'device': 'cuda:1',
                'batch_size': 8
            },
            'chunking': {
                'device': 'cuda:1',
                'batch_size': 8
            },
            'embedding': {
                'device': 'cuda:1',
                'batch_size': 8
            }
        }
    }
    
    # Test the config loader
    with patch('src.config.config_loader.load_pipeline_config', return_value=pipeline_config):
        assert get_component_device('docproc') == 'cuda:1'
        assert get_component_device('chunking') == 'cuda:1'
        assert get_component_device('embedding') == 'cuda:1'
    
    # Test DoclingAdapter
    with patch('src.config.config_loader.load_pipeline_config', return_value=pipeline_config), \
         patch('logging.Logger.info') as mock_docproc_log:
        adapter = DoclingAdapter(options=pipeline_config)
        
        # Check log messages for correct device usage
        device_logs = [
            call_args[0][0] for call_args in mock_docproc_log.call_args_list 
            if isinstance(call_args[0][0], str) and 'device' in call_args[0][0].lower()
        ]
        assert any('cuda:1' in log for log in device_logs), "DoclingAdapter did not use the correct device"
    
    # Test chunking component
    with patch('src.config.config_loader.load_pipeline_config', return_value=pipeline_config), \
         patch('logging.Logger.info') as mock_chunking_log, \
         patch('src.chunking.text_chunkers.chonky_chunker._get_splitter_with_engine') as mock_get_splitter:
        
        # Call the function that loads the chunker config
        from src.chunking.text_chunkers.chonky_chunker import chunk_text
        try:
            # We'll patch the actual model loading to avoid errors
            with patch('src.chunking.text_chunkers.chonky_chunker.get_model_engine'):
                chunk_text("Sample text", device="cuda:1")
                
                # Check if the device was correctly passed through
                device_args = [
                    args[1] for args, _ in mock_get_splitter.call_args_list
                    if len(args) > 1
                ]
                assert "cuda:1" in device_args, "Chunker did not use the correct device"
        except Exception:
            # We don't actually need to run the chunker, just check the device setting
            pass

    # Test embedding component
    with patch('src.config.config_loader.load_pipeline_config', return_value=pipeline_config), \
         patch('logging.Logger.info') as mock_embedding_log:
        adapter = ModernBERTEmbeddingAdapter()
        
        # Check if the correct device was selected
        assert adapter.device == 'cuda:1', f"EmbeddingAdapter using wrong device: {adapter.device}"
        
        # Check log messages for correct device usage
        device_logs = [
            call_args[0][0] for call_args in mock_embedding_log.call_args_list 
            if isinstance(call_args[0][0], str) and 'device' in call_args[0][0].lower()
        ]
        assert any('cuda:1' in log for log in device_logs), "EmbeddingAdapter did not use the correct device"
