"""
Tests for the preprocessor configuration components.

This module tests both config.py and config_models.py functionality.
"""
import json
import os
import tempfile
import pytest
from unittest.mock import patch, mock_open, MagicMock

from src.ingest.pre_processor.config import load_config, save_config, get_default_config
from src.ingest.pre_processor.config_models import PreProcessorConfig


class TestPreProcessorConfigModels:
    """Tests for the PreProcessorConfig data model."""
    
    def test_initialization(self):
        """Test basic initialization of PreProcessorConfig."""
        config = PreProcessorConfig(
            input_dir="./input",
            output_dir="./output"
        )
        
        assert config.input_dir == "./input"
        assert config.output_dir == "./output"
        assert config.python == {"enabled": True}
        assert config.markdown == {"enabled": True}
        assert config.docling == {"enabled": True}
    
    def test_initialization_with_custom_processors(self):
        """Test initialization with custom processor configurations."""
        python_config = {"enabled": True, "chunk_size": 1000}
        markdown_config = {"enabled": False, "chunk_size": 1500}
        
        config = PreProcessorConfig(
            input_dir="./input",
            output_dir="./output",
            python=python_config,
            markdown=markdown_config
        )
        
        assert config.python == python_config
        assert config.markdown == markdown_config
        assert config.docling == {"enabled": True}  # Default
    
    def test_from_dict_method(self):
        """Test creating a config from a dictionary."""
        config_dict = {
            "input_dir": "./custom_input",
            "output_dir": "./custom_output",
            "python": {"enabled": False, "custom_option": True},
            "markdown": {"enabled": True, "chunk_size": 2000}
        }
        
        config = PreProcessorConfig.from_dict(config_dict)
        
        assert config.input_dir == "./custom_input"
        assert config.output_dir == "./custom_output"
        assert config.python == {"enabled": False, "custom_option": True}
        assert config.markdown == {"enabled": True, "chunk_size": 2000}
        assert config.docling == {"enabled": True}  # Default since not in input dict
    
    def test_from_dict_with_defaults(self):
        """Test creating a config from an empty dictionary."""
        config = PreProcessorConfig.from_dict({})
        
        assert config.input_dir == "."  # Default
        assert config.output_dir == "./output"  # Default
        assert config.python == {"enabled": True}
        assert config.markdown == {"enabled": True}
        assert config.docling == {"enabled": True}
    
    def test_to_dict_method(self):
        """Test converting a config to a dictionary."""
        config = PreProcessorConfig(
            input_dir="./test_input",
            output_dir="./test_output",
            python={"enabled": True, "option": "value"},
            markdown={"enabled": False}
        )
        
        config_dict = config.to_dict()
        
        assert config_dict["input_dir"] == "./test_input"
        assert config_dict["output_dir"] == "./test_output"
        assert config_dict["python"] == {"enabled": True, "option": "value"}
        assert config_dict["markdown"] == {"enabled": False}
        assert config_dict["docling"] == {"enabled": True}


class TestConfigFunctions:
    """Tests for the configuration functions in config.py."""
    
    @pytest.fixture
    def valid_config_json(self):
        """Sample valid configuration JSON."""
        return json.dumps({
            "input_dir": "./test_input",
            "output_dir": "./test_output",
            "python": {"enabled": True, "chunk_size": 800},
            "markdown": {"enabled": False}
        })
    
    @pytest.fixture
    def valid_config_object(self):
        """Sample valid configuration object."""
        return PreProcessorConfig(
            input_dir="./test_input",
            output_dir="./test_output",
            python={"enabled": True, "chunk_size": 800},
            markdown={"enabled": False}
        )
    
    def test_load_config_success(self, valid_config_json):
        """Test loading a valid configuration file."""
        # Mock file existence and open operations
        with patch("os.path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=valid_config_json)):
                config = load_config("fake_path.json")
                
                assert isinstance(config, PreProcessorConfig)
                assert config.input_dir == "./test_input"
                assert config.output_dir == "./test_output"
                assert config.python == {"enabled": True, "chunk_size": 800}
                assert config.markdown == {"enabled": False}
    
    def test_load_config_file_not_found(self):
        """Test loading a non-existent configuration file."""
        with patch("os.path.exists", return_value=False):
            with pytest.raises(FileNotFoundError):
                load_config("nonexistent_path.json")
    
    def test_load_config_invalid_json(self):
        """Test loading an invalid JSON configuration file."""
        with patch("os.path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data="Not a valid JSON")):
                with pytest.raises(json.JSONDecodeError):
                    load_config("invalid_path.json")
    
    def test_save_config_success(self, valid_config_object):
        """Test saving a configuration file successfully."""
        mock_file = mock_open()
        
        with patch("os.makedirs") as mock_makedirs:
            with patch("builtins.open", mock_file):
                result = save_config(valid_config_object, "test_path.json")
                
                assert result is True
                mock_makedirs.assert_called_once()
                mock_file.assert_called_once_with("test_path.json", "w")
                mock_file().write.assert_called()  # Check if write was called
    
    def test_save_config_io_error(self, valid_config_object):
        """Test saving a configuration file with IO error."""
        with patch("os.makedirs", side_effect=IOError("Test IO Error")):
            result = save_config(valid_config_object, "test_path.json")
            assert result is False
    
    def test_save_config_type_error(self):
        """Test saving a configuration with TypeError."""
        # Create a config with a non-serializable type to cause TypeError during json.dump
        invalid_config = MagicMock()
        invalid_config.to_dict.side_effect = TypeError("Test Type Error")
        
        with patch("os.makedirs"):
            result = save_config(invalid_config, "test_path.json")
            assert result is False
    
    def test_get_default_config(self):
        """Test getting the default configuration."""
        default_config = get_default_config()
        
        assert isinstance(default_config, PreProcessorConfig)
        assert default_config.input_dir == "."
        assert default_config.output_dir == "./output"
        assert default_config.python.get("enabled") is True
        assert default_config.python.get("chunk_size") == 1000
        assert default_config.python.get("overlap") == 200
        assert default_config.markdown.get("enabled") is True
        assert default_config.markdown.get("chunk_size") == 1500
        assert "enabled" in default_config.docling
