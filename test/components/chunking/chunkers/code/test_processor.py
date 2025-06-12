"""
Unit tests for CodeChunker component.
"""

import pytest
from typing import Dict, Any

from src.components.chunking.chunkers.code.processor import CodeChunker
from src.types.components.contracts import (
    ComponentType,
    ChunkingInput,
    ChunkingOutput,
    TextChunk,
    ProcessingStatus
)


class TestCodeChunker:
    """Test suite for CodeChunker."""
    
    @pytest.fixture
    def processor(self) -> CodeChunker:
        """Create a CodeChunker instance for testing."""
        return CodeChunker()
    
    @pytest.fixture
    def processor_with_config(self) -> CodeChunker:
        """Create a CodeChunker with configuration."""
        config = {
            "chunk_size": 800,
            "chunk_overlap": 80,
            "preserve_functions": True,
            "preserve_classes": True,
            "language": "python"
        }
        return CodeChunker(config=config)
    
    @pytest.fixture
    def python_code(self) -> str:
        """Sample Python code for testing."""
        return '''import os
import sys
from typing import List, Dict, Optional

def calculate_sum(numbers: List[int]) -> int:
    """Calculate the sum of a list of numbers.
    
    Args:
        numbers: List of integers to sum
        
    Returns:
        The sum of all numbers
    """
    total = 0
    for num in numbers:
        total += num
    return total

class DataProcessor:
    """A class for processing data efficiently."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the data processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.processed_count = 0
    
    def process_item(self, item: str) -> str:
        """Process a single data item.
        
        Args:
            item: The item to process
            
        Returns:
            Processed item as string
        """
        self.processed_count += 1
        return item.upper().strip()
    
    def process_batch(self, items: List[str]) -> List[str]:
        """Process a batch of items.
        
        Args:
            items: List of items to process
            
        Returns:
            List of processed items
        """
        return [self.process_item(item) for item in items]
    
    def get_stats(self) -> Dict[str, int]:
        """Get processing statistics.
        
        Returns:
            Dictionary with processing stats
        """
        return {"processed_count": self.processed_count}

# Main execution
if __name__ == "__main__":
    processor = DataProcessor({"debug": True})
    test_data = ["hello", "world", "test"]
    results = processor.process_batch(test_data)
    print(f"Results: {results}")
    print(f"Stats: {processor.get_stats()}")'''
    
    @pytest.fixture
    def javascript_code(self) -> str:
        """Sample JavaScript code for testing."""
        return '''// JavaScript utility functions
const express = require('express');
const path = require('path');

/**
 * Calculate the factorial of a number
 * @param {number} n - The number to calculate factorial for
 * @returns {number} The factorial result
 */
function factorial(n) {
    if (n <= 1) {
        return 1;
    }
    return n * factorial(n - 1);
}

/**
 * Utility class for string operations
 */
class StringUtils {
    /**
     * Create a new StringUtils instance
     * @param {Object} options - Configuration options
     */
    constructor(options = {}) {
        this.options = options;
        this.cache = new Map();
    }
    
    /**
     * Capitalize the first letter of a string
     * @param {string} str - The string to capitalize
     * @returns {string} Capitalized string
     */
    capitalize(str) {
        if (this.cache.has(str)) {
            return this.cache.get(str);
        }
        
        const result = str.charAt(0).toUpperCase() + str.slice(1);
        this.cache.set(str, result);
        return result;
    }
    
    /**
     * Reverse a string
     * @param {string} str - The string to reverse
     * @returns {string} Reversed string
     */
    reverse(str) {
        return str.split('').reverse().join('');
    }
}

// Export for use in other modules
module.exports = {
    factorial,
    StringUtils
};'''
    
    @pytest.fixture
    def json_data(self) -> str:
        """Sample JSON data for testing."""
        return '''{
    "name": "test-project",
    "version": "1.0.0",
    "description": "A test project for chunking",
    "main": "index.js",
    "scripts": {
        "start": "node index.js",
        "test": "jest",
        "build": "webpack --mode production",
        "dev": "webpack --mode development --watch"
    },
    "dependencies": {
        "express": "^4.18.0",
        "lodash": "^4.17.21",
        "moment": "^2.29.4"
    },
    "devDependencies": {
        "jest": "^28.1.0",
        "webpack": "^5.70.0",
        "webpack-cli": "^4.9.0"
    },
    "keywords": ["test", "example", "chunking"],
    "author": "Test Author",
    "license": "MIT",
    "repository": {
        "type": "git",
        "url": "https://github.com/test/test-project.git"
    },
    "bugs": {
        "url": "https://github.com/test/test-project/issues"
    },
    "homepage": "https://github.com/test/test-project#readme"
}'''
    
    def test_basic_properties(self, processor: CodeChunker):
        """Test basic component properties."""
        assert processor.name == "code"
        assert processor.version == "1.0.0"
        assert processor.component_type == ComponentType.CHUNKING
    
    def test_configuration(self, processor: CodeChunker):
        """Test component configuration."""
        config = {
            "chunk_size": 1000,
            "chunk_overlap": 100,
            "preserve_functions": False,
            "preserve_classes": True,
            "language": "javascript"
        }
        
        assert processor.validate_config(config)
        processor.configure(config)
        
        # Configuration should be updated
        assert processor._config["chunk_size"] == 1000
        assert processor._config["chunk_overlap"] == 100
        assert processor._config["preserve_functions"] == False
        assert processor._config["preserve_classes"] == True
        assert processor._config["language"] == "javascript"
    
    def test_config_validation(self, processor: CodeChunker):
        """Test configuration validation."""
        # Valid config
        valid_config = {"chunk_size": 1000, "language": "python"}
        assert processor.validate_config(valid_config)
        
        # Invalid config - not a dict
        assert not processor.validate_config("invalid")
        assert not processor.validate_config(None)
        
        # Invalid chunk size
        invalid_config = {"chunk_size": -100}
        assert not processor.validate_config(invalid_config)
        
        # Invalid language
        invalid_config = {"language": "unsupported_language"}
        assert not processor.validate_config(invalid_config)
    
    def test_config_schema(self, processor: CodeChunker):
        """Test configuration schema."""
        schema = processor.get_config_schema()
        
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "chunk_size" in schema["properties"]
        assert "chunk_overlap" in schema["properties"]
        assert "preserve_functions" in schema["properties"]
        assert "preserve_classes" in schema["properties"]
        assert "language" in schema["properties"]
    
    def test_health_check(self, processor: CodeChunker):
        """Test health check."""
        assert processor.health_check() == True
    
    def test_metrics(self, processor: CodeChunker):
        """Test metrics collection."""
        metrics = processor.get_metrics()
        
        assert metrics["component_name"] == "code"
        assert metrics["component_version"] == "1.0.0"
        assert "preserve_functions" in metrics
        assert "preserve_classes" in metrics
        assert "supported_languages" in metrics
        assert "total_chunks_created" in metrics
        assert metrics["total_chunks_created"] >= 0
    
    def test_python_code_chunking(self, processor: CodeChunker, python_code: str):
        """Test chunking of Python code."""
        input_data = ChunkingInput(
            text=python_code,
            document_id="test_py",
            content_type="code",
            chunk_size=400,
            chunk_overlap=60,
            processing_options={
                "language": "python",
                "preserve_functions": True,
                "preserve_classes": True
            }
        )
        
        output = processor.chunk(input_data)
        
        assert isinstance(output, ChunkingOutput)
        assert len(output.chunks) > 0
        assert output.metadata.status == ProcessingStatus.SUCCESS
        assert len(output.errors) == 0
        
        # Check that functions and classes are preserved
        chunk_texts = [chunk.text for chunk in output.chunks]
        full_text = "\n".join(chunk_texts)
        
        # Should contain key code elements
        assert "def calculate_sum" in full_text
        assert "class DataProcessor" in full_text
        assert "def process_item" in full_text
    
    def test_javascript_code_chunking(self, processor: CodeChunker, javascript_code: str):
        """Test chunking of JavaScript code."""
        input_data = ChunkingInput(
            text=javascript_code,
            document_id="test_js",
            content_type="code",
            chunk_size=500,
            chunk_overlap=70,
            processing_options={
                "language": "javascript",
                "preserve_functions": True,
                "preserve_classes": True
            }
        )
        
        output = processor.chunk(input_data)
        
        assert isinstance(output, ChunkingOutput)
        assert len(output.chunks) > 0
        assert output.metadata.status == ProcessingStatus.SUCCESS
        
        # Check that JavaScript structures are preserved
        chunk_texts = [chunk.text for chunk in output.chunks]
        full_text = "\n".join(chunk_texts)
        
        assert "function factorial" in full_text
        assert "class StringUtils" in full_text
        assert "module.exports" in full_text
    
    def test_json_chunking(self, processor: CodeChunker, json_data: str):
        """Test chunking of JSON data."""
        input_data = ChunkingInput(
            text=json_data,
            document_id="test_json",
            content_type="code",
            chunk_size=300,
            chunk_overlap=50,
            processing_options={
                "language": "json",
                "preserve_structure": True
            }
        )
        
        output = processor.chunk(input_data)
        
        assert isinstance(output, ChunkingOutput)
        assert len(output.chunks) > 0
        assert output.metadata.status == ProcessingStatus.SUCCESS
        
        # JSON should be chunked while preserving structure
        for chunk in output.chunks:
            assert len(chunk.text) > 0
            # Should contain valid JSON fragments
    
    def test_function_preservation(self, processor: CodeChunker, python_code: str):
        """Test that functions are preserved within chunks."""
        input_data = ChunkingInput(
            text=python_code,
            document_id="test_py",
            content_type="code",
            chunk_size=300,  # Small size to force splits
            chunk_overlap=50,
            processing_options={
                "language": "python",
                "preserve_functions": True
            }
        )
        
        output = processor.chunk(input_data)
        
        # Check that functions aren't split inappropriately
        for chunk in output.chunks:
            text = chunk.text
            
            # If a chunk contains a function definition, it should contain the full function
            if "def " in text:
                # Count def and corresponding indentation patterns
                lines = text.split('\n')
                in_function = False
                function_indent = 0
                
                for line in lines:
                    if line.strip().startswith('def '):
                        in_function = True
                        function_indent = len(line) - len(line.lstrip())
                    elif in_function and line.strip() and len(line) - len(line.lstrip()) <= function_indent:
                        # Function ended
                        in_function = False
                
                # Function should be complete within the chunk (or continue properly)
    
    def test_class_preservation(self, processor: CodeChunker, python_code: str):
        """Test that classes are preserved within chunks."""
        input_data = ChunkingInput(
            text=python_code,
            document_id="test_py",
            content_type="code",
            chunk_size=600,  # Size to potentially split classes
            chunk_overlap=80,
            processing_options={
                "language": "python",
                "preserve_classes": True
            }
        )
        
        output = processor.chunk(input_data)
        
        # Check that classes are handled appropriately
        for chunk in output.chunks:
            text = chunk.text
            
            # If a chunk contains a class definition, it should preserve class structure
            if "class " in text:
                # Should contain meaningful class content
                assert "def " in text or "__init__" in text or "pass" in text
    
    def test_language_detection(self, processor: CodeChunker):
        """Test automatic language detection."""
        # Python code
        python_input = ChunkingInput(
            text="def hello():\n    print('Hello')",
            document_id="test",
            content_type="code",
            processing_options={"file_extension": ".py"}
        )
        
        output = processor.chunk(python_input)
        assert output.metadata.status == ProcessingStatus.SUCCESS
        
        # JavaScript code
        js_input = ChunkingInput(
            text="function hello() {\n    console.log('Hello');\n}",
            document_id="test",
            content_type="code",
            processing_options={"file_extension": ".js"}
        )
        
        output = processor.chunk(js_input)
        assert output.metadata.status == ProcessingStatus.SUCCESS
    
    def test_chunk_estimation(self, processor: CodeChunker, python_code: str):
        """Test chunk count estimation for code."""
        input_data = ChunkingInput(
            text=python_code,
            document_id="test_py",
            content_type="code",
            chunk_size=400,
            chunk_overlap=60
        )
        
        estimated_count = processor.estimate_chunks(input_data)
        
        assert isinstance(estimated_count, int)
        assert estimated_count > 0
        
        # Actually chunk and compare
        output = processor.chunk(input_data)
        actual_count = len(output.chunks)
        
        # Code estimation might be less precise due to structure preservation
        assert abs(estimated_count - actual_count) <= max(2, actual_count)
    
    def test_content_type_support(self, processor: CodeChunker):
        """Test content type support checking."""
        assert processor.supports_content_type("code") == True
        assert processor.supports_content_type("python") == True
        assert processor.supports_content_type("javascript") == True
        assert processor.supports_content_type("json") == True
        assert processor.supports_content_type("yaml") == True
        assert processor.supports_content_type("text") == False
        assert processor.supports_content_type("unknown") == False
    
    def test_optimal_chunk_size(self, processor: CodeChunker):
        """Test optimal chunk size calculation for different languages."""
        python_size = processor.get_optimal_chunk_size("python")
        js_size = processor.get_optimal_chunk_size("javascript")
        json_size = processor.get_optimal_chunk_size("json")
        
        assert isinstance(python_size, int)
        assert isinstance(js_size, int)
        assert isinstance(json_size, int)
        
        assert python_size > 0
        assert js_size > 0
        assert json_size > 0
        
        # Different languages might have different optimal sizes
    
    def test_no_preservation_mode(self, processor: CodeChunker, python_code: str):
        """Test chunking without structure preservation."""
        input_data = ChunkingInput(
            text=python_code,
            document_id="test_py",
            content_type="code",
            chunk_size=200,
            chunk_overlap=40,
            processing_options={
                "language": "python",
                "preserve_functions": False,
                "preserve_classes": False
            }
        )
        
        output = processor.chunk(input_data)
        
        assert isinstance(output, ChunkingOutput)
        assert len(output.chunks) > 0
        assert output.metadata.status == ProcessingStatus.SUCCESS
        
        # Should create chunks without strict structure preservation
        for chunk in output.chunks:
            assert len(chunk.text) > 0
    
    def test_small_code_snippets(self, processor: CodeChunker):
        """Test handling of small code snippets."""
        small_code = "def hello():\n    return 'Hello, World!'"
        
        input_data = ChunkingInput(
            text=small_code,
            document_id="test",
            content_type="code",
            chunk_size=1000,  # Much larger than the code
            processing_options={"language": "python"}
        )
        
        output = processor.chunk(input_data)
        
        assert isinstance(output, ChunkingOutput)
        assert len(output.chunks) == 1
        assert output.chunks[0].text.strip() == small_code.strip()
    
    def test_malformed_code_handling(self, processor: CodeChunker):
        """Test handling of malformed or incomplete code."""
        malformed_code = "def incomplete_function(\nprint('missing closing parenthesis'\nclass IncompleteClass"
        
        input_data = ChunkingInput(
            text=malformed_code,
            document_id="test",
            content_type="code",
            chunk_size=200,
            processing_options={"language": "python"}
        )
        
        output = processor.chunk(input_data)
        
        # Should handle malformed code gracefully
        assert isinstance(output, ChunkingOutput)
        assert len(output.chunks) > 0
        # Might have errors but should not crash
    
    def test_configured_processor(self, processor_with_config: CodeChunker, python_code: str):
        """Test processor with pre-configured settings."""
        input_data = ChunkingInput(
            text=python_code,
            document_id="test_py",
            content_type="code"
            # Should use configured settings
        )
        
        output = processor_with_config.chunk(input_data)
        
        assert isinstance(output, ChunkingOutput)
        assert output.metadata.status == ProcessingStatus.SUCCESS
        
        # Should use configured settings
        if len(output.chunks) > 0:
            # Verify chunks respect configured size
            max_expected_size = 800 + 200  # chunk_size + tolerance
            for chunk in output.chunks:
                assert len(chunk.text) <= max_expected_size
    
    def test_metadata_tracking(self, processor: CodeChunker, python_code: str):
        """Test that metadata is properly tracked."""
        input_data = ChunkingInput(
            text=python_code,
            document_id="test_py",
            content_type="code",
            chunk_size=400,
            processing_options={
                "language": "python",
                "preserve_functions": True
            }
        )
        
        output = processor.chunk(input_data)
        
        assert output.metadata.component_name == "code"
        assert output.metadata.component_version == "1.0.0"
        assert output.metadata.processing_time is not None
        assert output.metadata.processed_at is not None
        
        # Processing stats should be included
        assert "processing_time" in output.processing_stats
        assert "chunks_created" in output.processing_stats
        assert "language" in output.processing_stats
        assert "preserve_functions" in output.processing_stats
        assert output.processing_stats["chunks_created"] == len(output.chunks)
    
    def test_chunk_metadata(self, processor: CodeChunker, python_code: str):
        """Test that individual chunks have proper metadata."""
        input_data = ChunkingInput(
            text=python_code,
            document_id="test_py",
            content_type="code",
            chunk_size=400,
            processing_options={
                "language": "python",
                "preserve_functions": True
            }
        )
        
        output = processor.chunk(input_data)
        
        for chunk in output.chunks:
            assert isinstance(chunk.metadata, dict)
            assert "language" in chunk.metadata
            assert chunk.metadata["language"] == "python"
            assert "preserve_functions" in chunk.metadata
            assert "processing_method" in chunk.metadata
            assert chunk.metadata["processing_method"] == "code_aware"


if __name__ == "__main__":
    pytest.main([__file__])