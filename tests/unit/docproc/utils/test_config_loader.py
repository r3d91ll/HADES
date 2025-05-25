"""
Unit tests for config_loader module.

This module tests the configuration loading functionality for document processing.
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.docproc.utils.config_loader import (
    get_config,
    get_file_type_map,
    get_extension_to_format_map,
    get_format_config,
    get_option
)


class TestConfigLoader:
    """Tests for the config_loader module."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create a mock config structure similar to what would be returned by load_config
        self.mock_config = {
            "input_dir": Path("/test/input"),
            "output_dir": Path("/test/output"),
            "exclude_patterns": ["__pycache__", ".git"],
            "recursive": True,
            "max_workers": 4,
            "file_type_map": {
                "python": [".py", ".pyi"],
                "markdown": [".md", ".markdown"],
                "text": [".txt"]
            },
            "preprocessor_config": {
                "python": {
                    "extract_docstrings": True,
                    "include_imports": True,
                    "max_line_length": 100
                },
                "markdown": {
                    "extract_headers": True,
                    "extract_code_blocks": True
                }
            },
            "options": {
                "verbose": True,
                "log_level": "info"
            }
        }

    @patch("src.docproc.utils.config_loader._cached_config", None)
    @patch("src.docproc.utils.config_loader.load_config")
    def test_get_config_loads_config(self, mock_load_config):
        """Test that get_config loads configuration if not cached."""
        mock_load_config.return_value = self.mock_config
        
        # Call get_config for the first time
        config = get_config()
        
        # Check that load_config was called once
        mock_load_config.assert_called_once()
        
        # Check returned config is correct
        assert config == self.mock_config
        
        # Call get_config again
        get_config()
        
        # Check that load_config was still called only once (using cache)
        assert mock_load_config.call_count == 1

    @patch("src.docproc.utils.config_loader._cached_config", None)
    @patch("src.docproc.utils.config_loader.load_config")
    def test_get_config_with_reload(self, mock_load_config):
        """Test that get_config reloads configuration when reload=True."""
        mock_load_config.return_value = self.mock_config
        
        # Call get_config first time
        get_config()
        
        # Call get_config with reload=True
        get_config(reload=True)
        
        # Check that load_config was called twice
        assert mock_load_config.call_count == 2

    @patch("src.docproc.utils.config_loader._cached_config", None)
    @patch("src.docproc.utils.config_loader.load_config")
    @patch("src.docproc.utils.config_loader.logger")
    def test_get_config_handles_errors(self, mock_logger, mock_load_config):
        """Test that get_config handles loading errors gracefully."""
        # Make load_config raise an exception
        mock_load_config.side_effect = Exception("Config error")
        
        # Call get_config
        config = get_config()
        
        # Check that error was logged
        mock_logger.error.assert_called_once()
        mock_logger.warning.assert_called_once()
        
        # Check that a default config was created
        assert config is not None
        assert "input_dir" in config
        assert "output_dir" in config
        assert "exclude_patterns" in config
        assert "recursive" in config
        assert "max_workers" in config
        assert "file_type_map" in config
        assert "preprocessor_config" in config
        assert "options" in config

    @patch("src.docproc.utils.config_loader.get_config")
    def test_get_file_type_map(self, mock_get_config):
        """Test get_file_type_map returns the file_type_map from config."""
        mock_get_config.return_value = self.mock_config
        
        file_type_map = get_file_type_map()
        
        assert file_type_map == self.mock_config["file_type_map"]
        assert "python" in file_type_map
        assert ".py" in file_type_map["python"]
        assert ".pyi" in file_type_map["python"]

    @patch("src.docproc.utils.config_loader.get_file_type_map")
    def test_get_extension_to_format_map(self, mock_get_file_type_map):
        """Test get_extension_to_format_map correctly inverts the file_type_map."""
        mock_get_file_type_map.return_value = {
            "python": [".py", ".pyi"],
            "markdown": [".md", ".markdown"],
            "text": [".txt"]
        }
        
        extension_map = get_extension_to_format_map()
        
        # Check structure
        assert ".py" in extension_map
        assert ".pyi" in extension_map
        assert ".md" in extension_map
        assert ".markdown" in extension_map
        assert ".txt" in extension_map
        
        # Check values
        assert extension_map[".py"] == "python"
        assert extension_map[".pyi"] == "python"
        assert extension_map[".md"] == "markdown"
        assert extension_map[".markdown"] == "markdown"
        assert extension_map[".txt"] == "text"

    @patch("src.docproc.utils.config_loader.get_config")
    def test_get_format_config_existing_format(self, mock_get_config):
        """Test get_format_config returns config for an existing format."""
        mock_get_config.return_value = self.mock_config
        
        # Get config for an existing format
        python_config = get_format_config("python")
        
        # Check it returned the correct config
        assert python_config == self.mock_config["preprocessor_config"]["python"]
        assert python_config["extract_docstrings"] is True
        assert python_config["include_imports"] is True
        assert python_config["max_line_length"] == 100

    @patch("src.docproc.utils.config_loader.get_config")
    def test_get_format_config_missing_format(self, mock_get_config):
        """Test get_format_config returns empty dict for a missing format."""
        mock_get_config.return_value = self.mock_config
        
        # Get config for a non-existent format
        missing_config = get_format_config("html")
        
        # Check it returned an empty dict
        assert missing_config == {}

    @patch("src.docproc.utils.config_loader.get_format_config")
    def test_get_option_existing_option(self, mock_get_format_config):
        """Test get_option returns value for an existing option."""
        mock_get_format_config.return_value = {
            "extract_docstrings": True,
            "include_imports": True,
            "max_line_length": 100
        }
        
        # Get an existing option
        option = get_option("python", "extract_docstrings")
        
        # Check it returned the correct value
        assert option is True
        
        # Check format was passed correctly
        mock_get_format_config.assert_called_once_with("python")

    @patch("src.docproc.utils.config_loader.get_format_config")
    def test_get_option_missing_option(self, mock_get_format_config):
        """Test get_option returns default for a missing option."""
        mock_get_format_config.return_value = {
            "extract_docstrings": True,
            "include_imports": True
        }
        
        # Get a non-existent option with default
        option = get_option("python", "missing_option", default="default_value")
        
        # Check it returned the default
        assert option == "default_value"

    @patch("src.docproc.utils.config_loader.get_format_config")
    def test_get_option_without_default(self, mock_get_format_config):
        """Test get_option returns None for missing option without default."""
        mock_get_format_config.return_value = {
            "extract_docstrings": True
        }
        
        # Get a non-existent option without default
        option = get_option("python", "missing_option")
        
        # Check it returned None
        assert option is None

    @patch("src.docproc.utils.config_loader.get_format_config")
    def test_get_option_with_falsy_value(self, mock_get_format_config):
        """Test get_option correctly returns falsy values."""
        mock_get_format_config.return_value = {
            "extract_docstrings": False,
            "max_depth": 0,
            "empty_string": ""
        }
        
        # Check falsy boolean
        assert get_option("python", "extract_docstrings") is False
        
        # Check zero
        assert get_option("python", "max_depth") == 0
        
        # Check empty string
        assert get_option("python", "empty_string") == ""
        
    @patch("src.docproc.utils.config_loader._cached_config")
    def test_cached_config_used(self, mock_cached_config):
        """Test that cached config is used when available."""
        # Set up the mock cached config
        mock_cached_config.return_value = self.mock_config
        
        # When get_config is called, it should use the cached value
        with patch("src.docproc.utils.config_loader.load_config") as mock_load_config:
            config = get_config()
            
            # Check that load_config was not called
            mock_load_config.assert_not_called()
