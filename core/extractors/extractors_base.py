#!/usr/bin/env python3
"""
Base Extractor Interface

Defines the contract for all document extraction implementations.
Following the Conveyance Framework: extractors transform raw documents
into structured information, preserving the WHAT while enhancing accessibility.

Theory Connection:
Extractors are critical for the CONVEYANCE dimension - they transform
unstructured documents into actionable, structured data. High-quality
extraction directly increases the C value by making information more
readily usable.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of document extraction."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    equations: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    images: List[Dict[str, Any]] = field(default_factory=list)
    code_blocks: List[Dict[str, Any]] = field(default_factory=list)
    references: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    processing_time: float = 0.0


@dataclass
class ExtractorConfig:
    """Configuration for extractors."""
    use_gpu: bool = True
    batch_size: int = 1
    timeout_seconds: int = 300
    extract_equations: bool = True
    extract_tables: bool = True
    extract_images: bool = True
    extract_code: bool = True
    extract_references: bool = True
    max_pages: Optional[int] = None
    ocr_enabled: bool = False


class ExtractorBase(ABC):
    """
    Abstract base class for all extractors.

    Defines the interface that all extraction implementations must follow
    to ensure consistency across different document types and approaches.
    """

    def __init__(self, config: Optional[ExtractorConfig] = None):
        """
        Initialize extractor with configuration.

        Args:
            config: Extraction configuration
        """
        self.config = config or ExtractorConfig()

    @abstractmethod
    def extract(self,
               file_path: Union[str, Path],
               **kwargs) -> ExtractionResult:
        """
        Extract content from a document.

        Args:
            file_path: Path to the document
            **kwargs: Additional extraction options

        Returns:
            ExtractionResult with extracted content
        """
        pass

    @abstractmethod
    def extract_batch(self,
                     file_paths: List[Union[str, Path]],
                     **kwargs) -> List[ExtractionResult]:
        """
        Extract content from multiple documents.

        Args:
            file_paths: List of document paths
            **kwargs: Additional extraction options

        Returns:
            List of ExtractionResult objects
        """
        pass

    def validate_file(self, file_path: Union[str, Path]) -> bool:
        """
        Validate that a file can be processed.

        Args:
            file_path: Path to validate

        Returns:
            True if file can be processed
        """
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File does not exist: {path}")
            return False
        if not path.is_file():
            logger.error(f"Path is not a file: {path}")
            return False
        if path.stat().st_size == 0:
            logger.error(f"File is empty: {path}")
            return False
        return True

    @property
    @abstractmethod
    def supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        pass

    @property
    def supports_gpu(self) -> bool:
        """Whether this extractor can use GPU acceleration."""
        return False

    @property
    def supports_batch(self) -> bool:
        """Whether this extractor supports batch processing."""
        return True

    @property
    def supports_ocr(self) -> bool:
        """Whether this extractor supports OCR."""
        return False

    def get_extractor_info(self) -> Dict[str, Any]:
        """
        Get information about the extractor.

        Returns:
            Dictionary with extractor metadata
        """
        return {
            "class": self.__class__.__name__,
            "supported_formats": self.supported_formats,
            "supports_gpu": self.supports_gpu,
            "supports_batch": self.supports_batch,
            "supports_ocr": self.supports_ocr,
            "config": {
                "use_gpu": self.config.use_gpu,
                "batch_size": self.config.batch_size,
                "timeout_seconds": self.config.timeout_seconds
            }
        }
