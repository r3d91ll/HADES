"""
Pre-processor configuration loader for HADES-PathRAG (YAML-based).

Loads, merges, and validates pre-processor configuration from YAML.
"""
import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union, List, cast
from src.types.common import (
    PreProcessorConfig,
    MetadataExtractionConfig,
    EntityExtractionConfig,
    ChunkingPreparationConfig
)

# Defaults (should match those in preprocessor_config.yaml)
DEFAULT_FILE_TYPE_MAP: Dict[str, List[str]] = {
    'python': ['.py'],
    'javascript': ['.js', '.jsx', '.ts', '.tsx'],
    'java': ['.java'],
    'cpp': ['.cpp', '.hpp', '.cc', '.h'],
    'markdown': ['.md', '.markdown'],
    'pdf': ['.pdf'],
    'json': ['.json'],
    'yaml': ['.yaml', '.yml'],
    'csv': ['.csv'],
    'text': ['.txt'],
    'html': ['.html', '.htm'],
    'xml': ['.xml'],
    'toml': ['.toml'],
}

# Default content categories
DEFAULT_CONTENT_CATEGORIES: Dict[str, List[str]] = {
    'code': ['python', 'javascript', 'java', 'cpp', 'json', 'yaml', 'xml', 'toml'],
    'text': ['markdown', 'pdf', 'csv', 'text', 'html', 'docx', 'xlsx', 'pptx'],
}

DEFAULT_PREPROCESSOR_CONFIG: Dict[str, Dict[str, Any]] = {
    'python': {
        'create_symbol_table': True,
        'extract_docstrings': True,
        'analyze_imports': True,
    },
    'markdown': {
        'extract_mermaid': True,
        'extract_code_blocks': True,
        'extract_links': True,
    }
}

DEFAULTS: Dict[str, Any] = {
    'version': 1,
    'exclude_patterns': ['__pycache__', '.git'],
    'recursive': True,
    'max_workers': 4,
    'file_type_map': DEFAULT_FILE_TYPE_MAP,
    'preprocessor_config': DEFAULT_PREPROCESSOR_CONFIG,
    'options': {},
}

CONFIG_PATH = Path(__file__).parent / 'preprocessor_config.yaml'

# Default settings for new configuration sections
DEFAULT_METADATA_EXTRACTION = {
    'extract_title': True,
    'extract_authors': True,
    'extract_date': True,
    'use_filename_as_title': True,
    'detect_language': True,
}

DEFAULT_ENTITY_EXTRACTION = {
    'extract_named_entities': True,
    'extract_technical_terms': True,
    'min_confidence': 0.7,
}

DEFAULT_CHUNKING_PREPARATION = {
    'add_section_markers': True,
    'preserve_metadata': True,
    'mark_chunk_boundaries': True,
}

# Update defaults dictionary
DEFAULTS.update({
    'metadata_extraction': DEFAULT_METADATA_EXTRACTION,
    'entity_extraction': DEFAULT_ENTITY_EXTRACTION,
    'chunking_preparation': DEFAULT_CHUNKING_PREPARATION,
})


def load_config(config_path: Optional[Union[str, Path]] = None) -> PreProcessorConfig:
    """
    Load configuration from a YAML file, merging with defaults.
    """
    path = Path(config_path) if config_path else CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r') as f:
        user_config = yaml.safe_load(f) or {}

    # Create fresh dictionaries with proper types
    exclude_patterns: List[str] = list(DEFAULTS['exclude_patterns'])
    recursive: bool = bool(DEFAULTS['recursive'])
    max_workers: int = int(DEFAULTS['max_workers'])
    options: Dict[str, Any] = dict(DEFAULTS['options'])
    
    # Update from user config
    if 'exclude_patterns' in user_config and isinstance(user_config['exclude_patterns'], list):
        exclude_patterns = list(user_config['exclude_patterns'])
    if 'recursive' in user_config:
        recursive = bool(user_config['recursive'])
    if 'max_workers' in user_config:
        max_workers = int(user_config['max_workers'])
    if 'options' in user_config and isinstance(user_config['options'], dict):
        options = dict(user_config['options'])
    
    # Ensure the file_type_map exists with sensible defaults
    if 'file_type_map' not in user_config:
        file_type_map = DEFAULT_FILE_TYPE_MAP
    else:
        # Merge with defaults, preferring loaded config
        merged_map = DEFAULT_FILE_TYPE_MAP.copy()
        merged_map.update(user_config['file_type_map'])
        file_type_map = merged_map

    # Ensure the content_categories exist with sensible defaults
    if 'content_categories' not in user_config:
        content_categories = DEFAULT_CONTENT_CATEGORIES
    else:
        # Merge with defaults, preferring loaded config
        merged_categories = DEFAULT_CONTENT_CATEGORIES.copy()
        merged_categories.update(user_config['content_categories'])
        content_categories = merged_categories

    # Handle nested dictionaries
    for key, value in user_config.get('file_type_map', {}).items():
        if isinstance(value, list):
            file_type_map[key] = list(value)
    
    preprocessor_config: Dict[str, Dict[str, Any]] = dict(DEFAULT_PREPROCESSOR_CONFIG)
    if 'preprocessor_config' in user_config and isinstance(user_config['preprocessor_config'], dict):
        # Merge preprocessor config
        for key, value in user_config['preprocessor_config'].items():
            if isinstance(value, dict):
                if key in preprocessor_config:
                    # Update existing dictionary
                    preprocessor_config[key].update(value)
                else:
                    # Add new dictionary
                    preprocessor_config[key] = dict(value)
    
    # Handle the new configuration sections
    metadata_extraction_dict = dict(DEFAULT_METADATA_EXTRACTION)
    if 'metadata_extraction' in user_config and isinstance(user_config['metadata_extraction'], dict):
        metadata_extraction_dict.update(user_config['metadata_extraction'])
    # Convert to proper TypedDict
    metadata_extraction = cast(MetadataExtractionConfig, metadata_extraction_dict)
    
    entity_extraction_dict = dict(DEFAULT_ENTITY_EXTRACTION)
    if 'entity_extraction' in user_config and isinstance(user_config['entity_extraction'], dict):
        entity_extraction_dict.update(user_config['entity_extraction'])
    # Convert to proper TypedDict
    entity_extraction = cast(EntityExtractionConfig, entity_extraction_dict)
    
    chunking_preparation_dict = dict(DEFAULT_CHUNKING_PREPARATION)
    if 'chunking_preparation' in user_config and isinstance(user_config['chunking_preparation'], dict):
        chunking_preparation_dict.update(user_config['chunking_preparation'])
    # Convert to proper TypedDict
    chunking_preparation = cast(ChunkingPreparationConfig, chunking_preparation_dict)

    # Build PreProcessorConfig TypedDict
    result = {
        'version': user_config.get('version', DEFAULTS['version']),
        'recursive': recursive,
        'max_workers': max_workers,
        'exclude_patterns': exclude_patterns,
        'file_type_map': file_type_map,
        'content_categories': content_categories,
        'preprocessor_config': preprocessor_config,
        'metadata_extraction': metadata_extraction,
        'entity_extraction': entity_extraction,
        'chunking_preparation': chunking_preparation,
        'options': options,
    }
    
    return result
