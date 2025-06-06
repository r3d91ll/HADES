"""
Configuration type definitions for document processing.

This module centralizes the configuration types used throughout the document
processing pipeline, including format adapters, entity extraction, and metadata
handling. These types provide a consistent interface for configuring the
document processing behavior.
"""

from typing import Dict, Any, List, Optional, Union, TypedDict, Set, Literal


class FormatDetectionConfig(TypedDict, total=False):
    """Configuration for document format detection."""
    
    use_content_analysis: bool
    use_file_extension: bool
    min_confidence: float
    default_format: str
    extension_map: Dict[str, str]
    content_analyzers: List[str]


class TextCleaningConfig(TypedDict, total=False):
    """Configuration for text cleaning operations."""
    
    remove_control_chars: bool
    normalize_whitespace: bool
    fix_encoding_errors: bool
    remove_duplicate_linebreaks: bool
    max_consecutive_linebreaks: int
    replace_tabs_with_spaces: bool
    tab_size: int
    strip_html_tags: bool
    preserve_line_breaks: bool
    preserve_paragraph_breaks: bool


class AdapterConfig(TypedDict, total=False):
    """Base configuration for document adapters."""
    
    extract_metadata: bool
    extract_entities: bool
    validation_level: Literal["strict", "warn", "none"]
    max_content_length: int
    include_raw_content: bool
    text_cleaning: TextCleaningConfig


# Format-specific configuration types

class MarkdownConfig(AdapterConfig, total=False):
    """Configuration for Markdown document processing."""
    
    extract_frontmatter: bool
    extract_headers: bool
    extract_links: bool
    extract_images: bool
    extract_code_blocks: bool
    flatten_headers: bool
    preserve_linebreaks: bool
    extensions: List[str]


class HtmlConfig(AdapterConfig, total=False):
    """Configuration for HTML document processing."""
    
    extract_metadata_tags: bool
    extract_schema_org: bool
    extract_microdata: bool
    extract_rdfa: bool
    extract_json_ld: bool
    extract_open_graph: bool
    extract_twitter_cards: bool
    extract_links: bool
    extract_images: bool
    clean_html: bool
    strip_scripts: bool
    strip_styles: bool
    strip_comments: bool
    follow_redirects: bool
    max_redirect_depth: int
    timeout: int


class PdfConfig(AdapterConfig, total=False):
    """Configuration for PDF document processing."""
    
    extract_images: bool
    extract_tables: bool
    extract_forms: bool
    extract_annotations: bool
    extract_embedded_files: bool
    extract_structure: bool
    extract_fonts: bool
    extract_text_layout: bool
    password: Optional[str]
    ocr_enabled: bool
    ocr_language: str
    ocr_dpi: int
    ocr_mode: str
    extraction_mode: Literal["text", "layout", "raw", "structured"]
    page_range: Optional[str]


class PythonConfig(AdapterConfig, total=False):
    """Configuration for Python code processing."""
    
    extract_docstrings: bool
    extract_imports: bool
    extract_classes: bool
    extract_functions: bool
    extract_methods: bool
    extract_variables: bool
    extract_decorators: bool
    extract_type_hints: bool
    resolve_imports: bool
    analyze_complexity: bool
    max_complexity: int
    include_nested_defs: bool
    parse_mode: Literal["ast", "tree-sitter", "hybrid"]


class JsonConfig(AdapterConfig, total=False):
    """Configuration for JSON document processing."""
    
    pretty_print: bool
    flatten_arrays: bool
    flatten_objects: bool
    max_depth: int
    include_paths: List[str]
    exclude_paths: List[str]
    schema_validation: bool
    schema_path: Optional[str]
    extract_schema: bool


class YamlConfig(AdapterConfig, total=False):
    """Configuration for YAML document processing."""
    
    pretty_print: bool
    flatten_arrays: bool
    flatten_objects: bool
    max_depth: int
    include_paths: List[str]
    exclude_paths: List[str]
    schema_validation: bool
    schema_path: Optional[str]
    allow_duplicate_keys: bool
    allow_anchors: bool


class DocumentProcessorConfig(TypedDict, total=False):
    """Master configuration for document processing."""
    
    format_detection: FormatDetectionConfig
    text_cleaning: TextCleaningConfig
    formats: Dict[str, AdapterConfig]
    validation_level: Literal["strict", "warn", "none"]
    default_format: str
    cache_enabled: bool
    cache_dir: Optional[str]
    max_document_size_mb: float
    parallel_processing: bool
    max_workers: int
    timeout: int
