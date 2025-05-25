"""
Tests for device configuration handling across components.

These tests verify that the components correctly read and respect
the device settings from the pipeline configuration.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
import yaml
from pathlib import Path

# Import the config loader
from src.config.config_loader import (
    load_pipeline_config,
    get_component_device,
    get_device_config
)


def test_get_component_device():
    """Test that component devices are correctly retrieved from config."""
    # Mock pipeline configuration
    mock_config = {
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
    
    # Mock the load_pipeline_config function
    with patch('src.config.config_loader.load_pipeline_config', return_value=mock_config):
        # Test each component
        assert get_component_device('docproc') == 'cuda:1'
        assert get_component_device('chunking') == 'cuda:1'
        assert get_component_device('embedding') == 'cuda:1'
        
        # Test non-existent component
        assert get_component_device('nonexistent') is None


def test_get_device_config():
    """Test that the correct device mode is detected."""
    # Test GPU mode
    with patch('src.config.config_loader.load_pipeline_config', return_value={
        'gpu_execution': {'enabled': True}
    }):
        config = get_device_config()
        assert config['mode'] == 'gpu'
    
    # Test CPU mode
    with patch('src.config.config_loader.load_pipeline_config', return_value={
        'gpu_execution': {'enabled': False},
        'cpu_execution': {'enabled': True}
    }):
        config = get_device_config()
        assert config['mode'] == 'cpu'
    
    # Test when neither is explicitly enabled
    with patch('src.config.config_loader.load_pipeline_config', return_value={
        'gpu_execution': {'enabled': False},
        'cpu_execution': {'enabled': False}
    }):
        config = get_device_config()
        assert config['mode'] == 'auto'


# Integration test of docproc adapter with configuration
@pytest.mark.integration
def test_docling_adapter_uses_correct_device():
    """Test that DoclingAdapter correctly uses the configured device."""
    from src.docproc.adapters.docling_adapter import DoclingAdapter
    
    # Create a mock pipeline config
    pipeline_config = {
        'gpu_execution': {
            'enabled': True,
            'docproc': {
                'device': 'cuda:1'
            }
        }
    }
    
    # Initialize with config
    with patch('torch.cuda.is_available', return_value=True), \
         patch('src.utils.device_utils.is_gpu_available', return_value=True):
        adapter = DoclingAdapter(options={'gpu_execution': pipeline_config['gpu_execution']})
        
        # We can't directly test the device usage without actually loading models
        # But we can check if the adapter correctly parsed the configuration
        # by looking at the debug logs
        import logging
        with patch('logging.Logger.info') as mock_log:
            # Trigger the device setup again
            adapter._setup_device()
            
            # Check if the correct device was selected
            found_device_log = False
            for call_args in mock_log.call_args_list:
                if "Using configured device" in call_args[0][0] and "cuda:1" in call_args[0][0]:
                    found_device_log = True
                    break
            
            assert found_device_log, "DoclingAdapter did not correctly use the configured device"
