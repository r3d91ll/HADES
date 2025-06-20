"""
Text Chunking Types

Type definitions for text-based chunking components.
"""

from typing import Dict, Any, List, Optional, Literal
from pydantic import Field
from datetime import datetime, timezone

from ...common import BaseHADESModel


class TextChunkingConfig(BaseHADESModel):
    """Configuration for text chunking component."""
    
    # Text-specific chunking
    chunking_method: Literal["sentence", "paragraph", "semantic", "sliding_window"] = Field(
        default="sentence",
        description="Text chunking method"
    )
    
    # Language processing
    language: str = Field(default="en", description="Text language")
    use_nltk: bool = Field(default=True, description="Use NLTK for sentence detection")
    use_spacy: bool = Field(default=False, description="Use spaCy for processing")
    
    # Sentence-based chunking
    sentences_per_chunk: int = Field(default=3, ge=1, description="Target sentences per chunk")
    sentence_overlap: int = Field(default=1, ge=0, description="Sentence overlap between chunks")
    
    # Paragraph-based chunking
    paragraphs_per_chunk: int = Field(default=1, ge=1, description="Paragraphs per chunk")
    paragraph_separator: str = Field(default="\n\n", description="Paragraph separator")
    
    # Semantic chunking
    similarity_threshold: float = Field(default=0.7, ge=0, le=1, description="Semantic similarity threshold")
    
    # Text cleaning
    remove_extra_whitespace: bool = Field(default=True, description="Remove extra whitespace")
    normalize_line_breaks: bool = Field(default=True, description="Normalize line breaks")
    remove_empty_lines: bool = Field(default=True, description="Remove empty lines")


class TextChunkingMetrics(BaseHADESModel):
    """Metrics for text chunking operations."""
    
    # Text-specific metrics
    total_sentences_processed: int = Field(default=0, ge=0, description="Total sentences processed")
    total_paragraphs_processed: int = Field(default=0, ge=0, description="Total paragraphs processed")
    avg_sentences_per_chunk: float = Field(default=0.0, ge=0, description="Average sentences per chunk")
    avg_words_per_chunk: float = Field(default=0.0, ge=0, description="Average words per chunk")
    
    # Language processing metrics
    sentence_detection_time: float = Field(default=0.0, ge=0, description="Time spent on sentence detection")
    language_processing_time: float = Field(default=0.0, ge=0, description="Time spent on language processing")
    
    # Quality metrics
    text_coherence_score: float = Field(default=0.0, ge=0, le=1, description="Text coherence score")
    readability_score: float = Field(default=0.0, ge=0, description="Text readability score")


# Export all types
__all__ = [
    "TextChunkingConfig",
    "TextChunkingMetrics"
]