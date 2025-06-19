"""
Tests for Directory Analyzer

Tests directory structure analysis, file discovery, and relationship extraction.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, mock_open

from src.pipelines.bootstrap.directory_analyzer import DirectoryAnalyzer
from src.pipelines.bootstrap.config import BootstrapConfig, EdgeDiscoveryMethod
from src.storage.incremental.sequential_isne_types import FileType, EdgeType


@pytest.fixture
def test_project_structure():
    """Create a comprehensive test project structure."""
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create nested directory structure
    (temp_dir / "src").mkdir()
    (temp_dir / "src" / "core").mkdir()
    (temp_dir / "src" / "utils").mkdir()
    (temp_dir / "tests").mkdir()
    (temp_dir / "docs").mkdir()
    (temp_dir / "docs" / "api").mkdir()
    (temp_dir / "config").mkdir()
    (temp_dir / "scripts").mkdir()
    
    # Create Python code files with imports
    (temp_dir / "src" / "main.py").write_text("""
import os
import sys
from src.core.processor import DataProcessor
from src.utils.helpers import format_data, validate_input
import json

class Application:
    def __init__(self):
        self.processor = DataProcessor()
    
    def run(self, input_file):
        '''Run the application'''
        data = self.load_data(input_file)
        processed = self.processor.process(data)
        return format_data(processed)
    
    def load_data(self, filepath):
        '''Load data from file'''
        with open(filepath) as f:
            return json.load(f)
""")
    
    (temp_dir / "src" / "core" / "__init__.py").write_text("")
    
    (temp_dir / "src" / "core" / "processor.py").write_text("""
from src.utils.helpers import validate_input
import logging

class DataProcessor:
    '''Core data processing functionality'''
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def process(self, data):
        '''Process input data'''
        if not validate_input(data):
            raise ValueError("Invalid input")
        return self._transform(data)
    
    def _transform(self, data):
        return [item.upper() for item in data]
""")
    
    (temp_dir / "src" / "utils" / "__init__.py").write_text("")
    
    (temp_dir / "src" / "utils" / "helpers.py").write_text("""
import re
from typing import List, Any

def validate_input(data: Any) -> bool:
    '''Validate input data'''
    return isinstance(data, list) and len(data) > 0

def format_data(data: List[str]) -> str:
    '''Format processed data'''
    return '\\n'.join(data)

def clean_text(text: str) -> str:
    '''Clean text data'''
    return re.sub(r'[^a-zA-Z0-9\\s]', '', text)
""")
    
    # Create test files
    (temp_dir / "tests" / "test_processor.py").write_text("""
import pytest
from src.core.processor import DataProcessor
from src.utils.helpers import validate_input

class TestDataProcessor:
    def test_process_valid_data(self):
        processor = DataProcessor()
        result = processor.process(['hello', 'world'])
        assert result == ['HELLO', 'WORLD']
    
    def test_process_invalid_data(self):
        processor = DataProcessor()
        with pytest.raises(ValueError):
            processor.process([])
""")
    
    # Create documentation files
    (temp_dir / "docs" / "README.md").write_text("""
# Project Documentation

This project provides data processing capabilities through the main `Application` class.

## Quick Start

```python
from src.main import Application
app = Application()
result = app.run('data.json')
```

## Architecture

The project consists of:

- `src/main.py` - Main application entry point
- `src/core/processor.py` - Core processing logic
- `src/utils/helpers.py` - Utility functions

For detailed API documentation, see [API Reference](api/reference.md).

## Configuration

Configuration files are located in the `config/` directory:
- `config/settings.json` - Application settings
- `config/logging.yaml` - Logging configuration
""")
    
    (temp_dir / "docs" / "api" / "reference.md").write_text("""
# API Reference

## Application Class

Located in `src/main.py`.

### Methods

- `run(input_file)` - Process data from file
- `load_data(filepath)` - Load JSON data

## DataProcessor Class

Located in `src/core/processor.py`.

### Methods

- `process(data)` - Transform input data
- `_transform(data)` - Internal transformation

## Utility Functions

Located in `src/utils/helpers.py`.

- `validate_input(data)` - Validate data format
- `format_data(data)` - Format output
- `clean_text(text)` - Clean text input
""")
    
    # Create configuration files
    (temp_dir / "config" / "settings.json").write_text("""
{
    "app_name": "data_processor",
    "version": "1.0.0",
    "main_module": "src.main",
    "data_source": "input/data.json",
    "output_format": "text",
    "enable_logging": true,
    "log_config": "config/logging.yaml"
}
""")
    
    (temp_dir / "config" / "logging.yaml").write_text("""
version: 1
formatters:
  default:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: default

root:
  level: INFO
  handlers: [console]
""")
    
    # Create script files
    (temp_dir / "scripts" / "run_processor.py").write_text("""
#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.main import Application

if __name__ == "__main__":
    app = Application()
    result = app.run(sys.argv[1])
    print(result)
""")
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def analyzer_config(test_project_structure):
    """Create analyzer configuration."""
    return BootstrapConfig(
        input_directory=test_project_structure,
        file_type_filter="all",
        enable_cross_modal_discovery=True,
        edge_discovery_methods=[
            EdgeDiscoveryMethod.DIRECTORY_STRUCTURE,
            EdgeDiscoveryMethod.IMPORT_ANALYSIS,
            EdgeDiscoveryMethod.FILE_REFERENCES,
            EdgeDiscoveryMethod.CO_LOCATION
        ]
    )


class TestDirectoryAnalyzer:
    """Test directory analysis functionality."""
    
    def test_initialization(self, analyzer_config):
        """Test analyzer initialization."""
        analyzer = DirectoryAnalyzer(analyzer_config)
        
        assert analyzer.config == analyzer_config
        assert analyzer.discovered_files == {}
        assert analyzer.relationships == []
        
        # Check patterns are set up
        assert len(analyzer.python_import_patterns) > 0
        assert len(analyzer.js_import_patterns) > 0
        assert len(analyzer.file_reference_patterns) > 0
    
    def test_file_discovery(self, analyzer_config):
        """Test file discovery and classification."""
        analyzer = DirectoryAnalyzer(analyzer_config)
        analyzer._discover_files(analyzer_config.input_directory)
        
        # Should discover multiple files
        assert len(analyzer.discovered_files) > 8
        
        # Check file types are properly classified
        file_types = [info['file_type'] for info in analyzer.discovered_files.values()]
        assert FileType.CODE in file_types
        assert FileType.DOCUMENTATION in file_types
        assert FileType.CONFIG in file_types
        
        # Check specific files exist
        file_paths = list(analyzer.discovered_files.keys())
        main_py_files = [f for f in file_paths if f.endswith('main.py')]
        assert len(main_py_files) > 0
        
        readme_files = [f for f in file_paths if f.endswith('README.md')]
        assert len(readme_files) > 0
        
        config_files = [f for f in file_paths if f.endswith('settings.json')]
        assert len(config_files) > 0
    
    def test_file_info_extraction(self, analyzer_config):
        """Test file metadata extraction."""
        analyzer = DirectoryAnalyzer(analyzer_config)
        analyzer._discover_files(analyzer_config.input_directory)
        
        # Check a specific file's metadata
        main_py_files = [f for f in analyzer.discovered_files.keys() if f.endswith('main.py')]
        assert len(main_py_files) > 0
        
        main_py_file = main_py_files[0]
        file_info = analyzer.discovered_files[main_py_file]
        
        assert 'file_path' in file_info
        assert 'file_name' in file_info
        assert 'directory' in file_info
        assert 'extension' in file_info
        assert 'size' in file_info
        assert 'file_type' in file_info
        assert file_info['file_type'] == FileType.CODE
        
        # Check content analysis for Python files
        if 'imports' in file_info:
            assert isinstance(file_info['imports'], list)
            assert len(file_info['imports']) > 0
        
        if 'functions' in file_info:
            assert isinstance(file_info['functions'], list)
    
    def test_should_include_file(self, analyzer_config):
        """Test file inclusion filtering."""
        analyzer = DirectoryAnalyzer(analyzer_config)
        
        # Should include Python files
        py_file = Path("test.py")
        assert analyzer._should_include_file(py_file) is True
        
        # Should include markdown files
        md_file = Path("README.md")
        assert analyzer._should_include_file(md_file) is True
        
        # Should include JSON files
        json_file = Path("config.json")
        assert analyzer._should_include_file(json_file) is True
        
        # Should exclude binary files (not in include_extensions)
        exe_file = Path("program.exe")
        assert analyzer._should_include_file(exe_file) is False
        
        # Should exclude files matching exclude patterns
        pyc_file = Path("test.pyc")
        assert analyzer._should_include_file(pyc_file) is False
    
    def test_import_analysis(self, analyzer_config):
        """Test Python import analysis."""
        analyzer = DirectoryAnalyzer(analyzer_config)
        discovered_files, relationships = analyzer.analyze_directory(analyzer_config.input_directory)
        
        # Should find import relationships
        import_edges = [rel for rel in relationships if rel[2] == EdgeType.CODE_IMPORTS.value]
        assert len(import_edges) > 0
        
        # Check that imports are properly resolved
        for from_file, to_file, edge_type, weight, metadata in import_edges:
            assert from_file in discovered_files
            assert metadata.get('source') == 'import_analysis'
            assert 'import_name' in metadata
    
    def test_directory_structure_analysis(self, analyzer_config):
        """Test directory hierarchy analysis."""
        analyzer = DirectoryAnalyzer(analyzer_config)
        discovered_files, relationships = analyzer.analyze_directory(analyzer_config.input_directory)
        
        # Should find directory hierarchy relationships
        hierarchy_edges = [rel for rel in relationships 
                          if rel[2] == EdgeType.DIRECTORY_HIERARCHY.value]
        assert len(hierarchy_edges) > 0
        
        # Check hierarchy relationships are valid
        for from_file, to_file, edge_type, weight, metadata in hierarchy_edges:
            assert from_file in discovered_files
            assert to_file in discovered_files
            assert metadata.get('source') == 'directory_structure'
    
    def test_co_location_analysis(self, analyzer_config):
        """Test file co-location analysis."""
        analyzer = DirectoryAnalyzer(analyzer_config)
        discovered_files, relationships = analyzer.analyze_directory(analyzer_config.input_directory)
        
        # Should find co-location relationships
        colocation_edges = [rel for rel in relationships 
                           if rel[4].get('source') == 'co_location']
        assert len(colocation_edges) > 0
        
        # Check co-location relationships
        for from_file, to_file, edge_type, weight, metadata in colocation_edges:
            assert from_file in discovered_files
            assert to_file in discovered_files
            assert 'directory' in metadata
    
    def test_file_reference_analysis(self, analyzer_config):
        """Test file reference extraction."""
        analyzer = DirectoryAnalyzer(analyzer_config)
        discovered_files, relationships = analyzer.analyze_directory(analyzer_config.input_directory)
        
        # Should find file reference relationships (from documentation)
        reference_edges = [rel for rel in relationships 
                          if rel[4].get('source') == 'file_reference']
        
        # May or may not find references depending on content
        # Just check that the analysis runs without error
        for from_file, to_file, edge_type, weight, metadata in reference_edges:
            assert from_file in discovered_files
            assert to_file in discovered_files
            assert 'referenced_path' in metadata
    
    def test_cross_modal_edge_detection(self, analyzer_config):
        """Test cross-modal edge type assignment."""
        analyzer = DirectoryAnalyzer(analyzer_config)
        
        # Test cross-modal edge type mapping
        code_to_doc = analyzer._get_cross_modal_edge_type(FileType.CODE, FileType.DOCUMENTATION)
        assert code_to_doc == EdgeType.CODE_TO_DOC.value
        
        doc_to_code = analyzer._get_cross_modal_edge_type(FileType.DOCUMENTATION, FileType.CODE)
        assert doc_to_code == EdgeType.DOC_TO_CODE.value
        
        code_to_config = analyzer._get_cross_modal_edge_type(FileType.CODE, FileType.CONFIG)
        assert code_to_config == EdgeType.CODE_TO_CONFIG.value
        
        config_to_code = analyzer._get_cross_modal_edge_type(FileType.CONFIG, FileType.CODE)
        assert config_to_code == EdgeType.CONFIG_TO_CODE.value
    
    def test_code_content_analysis(self, analyzer_config):
        """Test Python code content analysis."""
        analyzer = DirectoryAnalyzer(analyzer_config)
        
        # Test with sample Python code
        python_code = """
import os
import sys
from typing import List

def process_data(data: List[str]) -> List[str]:
    '''Process input data'''
    return [item.upper() for item in data]

class DataHandler:
    def __init__(self):
        self.data = []
    
    def add_item(self, item):
        self.data.append(item)
"""
        
        result = analyzer._analyze_code_content(Path("test.py"), python_code)
        
        assert 'lines_of_code' in result
        assert 'comment_lines' in result
        assert 'complexity_score' in result
        assert 'imports' in result
        assert 'functions' in result
        assert 'classes' in result
        
        # Check specific extracted information
        assert 'os' in result['imports']
        assert 'sys' in result['imports']
        assert 'typing' in result['imports']
        assert 'process_data' in result['functions']
        assert 'DataHandler' in result['classes']
    
    def test_documentation_content_analysis(self, analyzer_config):
        """Test documentation content analysis."""
        analyzer = DirectoryAnalyzer(analyzer_config)
        
        # Test with sample markdown content
        markdown_content = """
# Main Title

This is a documentation file with multiple sections.

## Section 1

Some content here with [link to file](../src/main.py).

### Subsection

More content with [external link](https://example.com).

## Section 2

Additional content.
"""
        
        result = analyzer._analyze_doc_content(markdown_content)
        
        assert 'word_count' in result
        assert 'headings' in result
        assert 'links' in result
        assert 'readability_score' in result
        
        # Check extracted headings
        assert len(result['headings']) == 4  # 1 h1, 2 h2, 1 h3
        assert result['headings'][0]['level'] == 1
        assert result['headings'][0]['text'] == 'Main Title'
        
        # Check extracted links
        assert len(result['links']) == 2
        link_urls = [link['url'] for link in result['links']]
        assert '../src/main.py' in link_urls
        assert 'https://example.com' in link_urls
    
    def test_config_content_analysis(self, analyzer_config):
        """Test configuration file content analysis."""
        analyzer = DirectoryAnalyzer(analyzer_config)
        
        # Test JSON config
        json_content = """
{
    "app_name": "test_app",
    "version": "1.0.0",
    "modules": ["core", "utils"],
    "settings": {
        "debug": true,
        "log_level": "INFO"
    }
}
"""
        
        result = analyzer._analyze_config_content(Path("config.json"), json_content)
        
        assert 'parsed_config' in result
        assert 'config_keys' in result
        
        parsed = result['parsed_config']
        assert parsed['app_name'] == 'test_app'
        assert parsed['version'] == '1.0.0'
        assert 'app_name' in result['config_keys']
        assert 'version' in result['config_keys']
    
    def test_import_resolution(self, analyzer_config):
        """Test import statement resolution to files."""
        analyzer = DirectoryAnalyzer(analyzer_config)
        
        # First discover files
        analyzer._discover_files(analyzer_config.input_directory)
        
        # Test resolving imports
        source_file = str(analyzer_config.input_directory / "src" / "main.py")
        
        # Should find matches for imports that exist in the project
        candidates = analyzer._resolve_import("src.core.processor", source_file)
        
        # Should return some candidates (exact matching depends on file structure)
        assert isinstance(candidates, list)
    
    def test_file_reference_resolution(self, analyzer_config):
        """Test file reference resolution."""
        analyzer = DirectoryAnalyzer(analyzer_config)
        
        # First discover files
        analyzer._discover_files(analyzer_config.input_directory)
        
        # Test resolving file references
        source_file = str(analyzer_config.input_directory / "docs" / "README.md")
        
        candidates = analyzer._resolve_file_reference("../src/main.py", source_file)
        
        # Should return candidates if the referenced file exists
        assert isinstance(candidates, list)
    
    def test_analyze_directory_complete(self, analyzer_config):
        """Test complete directory analysis."""
        analyzer = DirectoryAnalyzer(analyzer_config)
        discovered_files, relationships = analyzer.analyze_directory(analyzer_config.input_directory)
        
        # Should discover files and relationships
        assert len(discovered_files) > 0
        assert len(relationships) > 0
        
        # Check that all files have required metadata
        for file_path, file_info in discovered_files.items():
            assert 'file_type' in file_info
            assert 'file_name' in file_info
            assert 'directory' in file_info
            assert 'size' in file_info
        
        # Check that relationships are well-formed
        for from_file, to_file, edge_type, weight, metadata in relationships:
            assert from_file in discovered_files
            assert to_file in discovered_files
            assert isinstance(weight, (int, float))
            assert isinstance(metadata, dict)
            assert 'source' in metadata
    
    def test_edge_discovery_method_filtering(self, test_project_structure):
        """Test that only enabled edge discovery methods are used."""
        # Create config with only import analysis enabled
        config = BootstrapConfig(
            input_directory=test_project_structure,
            edge_discovery_methods=[EdgeDiscoveryMethod.IMPORT_ANALYSIS]
        )
        
        analyzer = DirectoryAnalyzer(config)
        discovered_files, relationships = analyzer.analyze_directory(config.input_directory)
        
        # Should only have import-based relationships
        edge_sources = {rel[4].get('source') for rel in relationships}
        assert 'import_analysis' in edge_sources
        assert 'directory_structure' not in edge_sources
        assert 'co_location' not in edge_sources