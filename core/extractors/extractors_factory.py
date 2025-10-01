#!/usr/bin/env python3
"""
Extractor Factory

Factory pattern for creating extractor instances based on file type and configuration.
Supports automatic format detection and fallback strategies.

Theory Connection:
The factory enables adaptive extraction strategies based on document type,
maximizing the CONVEYANCE dimension by selecting the optimal extractor
for each document format.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .extractors_base import ExtractorBase, ExtractorConfig

logger = logging.getLogger(__name__)


class ExtractorFactory:
    """
    Factory for creating extractor instances.

    Manages the instantiation of different extractor types based on
    file format and configuration, with support for fallbacks.
    """

    # Registry of available extractors
    _extractors: Dict[str, type] = {}

    # Format to extractor mapping
    _format_mapping: Dict[str, str] = {
        '.pdf': 'docling',
        '.tex': 'latex',
        '.py': 'code',
        '.js': 'code',
        '.ts': 'code',
        '.java': 'code',
        '.cpp': 'code',
        '.c': 'code',
        '.rs': 'code',
        '.go': 'code',
    }

    @classmethod
    def register(cls, name: str, extractor_class: type):
        """
        Register an extractor class.

        Args:
            name: Name to register under
            extractor_class: Extractor class to register
        """
        cls._extractors[name] = extractor_class
        logger.info(f"Registered extractor: {name}")

    @classmethod
    def create(cls,
              file_path: Optional[Union[str, Path]] = None,
              extractor_type: Optional[str] = None,
              config: Optional[ExtractorConfig] = None,
              **kwargs) -> ExtractorBase:
        """
        Create an extractor instance.

        Args:
            file_path: Optional file path for auto-detection
            extractor_type: Explicit extractor type
            config: Extraction configuration
            **kwargs: Additional arguments for the extractor

        Returns:
            Extractor instance

        Raises:
            ValueError: If no suitable extractor found
        """
        # Create config if not provided
        if config is None:
            config = ExtractorConfig(**kwargs)

        # Determine extractor type
        if extractor_type is None and file_path is not None:
            extractor_type = cls._determine_extractor_type(file_path)
        elif extractor_type is None:
            extractor_type = 'docling'  # Default

        if extractor_type not in cls._extractors:
            # Try to import and register on-demand
            cls._auto_register(extractor_type)

        if extractor_type not in cls._extractors:
            available = list(cls._extractors.keys())
            raise ValueError(
                f"No extractor registered for type '{extractor_type}'. "
                f"Available: {available}"
            )

        extractor_class = cls._extractors[extractor_type]
        logger.info(f"Creating {extractor_type} extractor")

        return extractor_class(config)

    @classmethod
    def _determine_extractor_type(cls, file_path: Union[str, Path]) -> str:
        """
        Determine extractor type from file path.

        Args:
            file_path: File path

        Returns:
            Extractor type string
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        # Check format mapping
        if suffix in cls._format_mapping:
            return cls._format_mapping[suffix]

        # Check for LaTeX files
        if suffix in ['.tex', '.bib', '.cls', '.sty']:
            return 'latex'

        # Check for code files
        code_extensions = ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h',
                          '.rs', '.go', '.rb', '.php', '.swift', '.kt']
        if suffix in code_extensions:
            return 'code'

        # Default to Docling for PDFs and unknown formats
        return 'docling'

    @classmethod
    def _auto_register(cls, extractor_type: str):
        """
        Attempt to auto-register an extractor type.

        Args:
            extractor_type: Type of extractor to register
        """
        try:
            if extractor_type == 'docling':
                from .extractors_docling import DoclingExtractor
                cls.register('docling', DoclingExtractor)
            elif extractor_type == 'latex':
                from .extractors_latex import LaTeXExtractor
                cls.register('latex', LaTeXExtractor)
            elif extractor_type == 'code':
                from .extractors_code import CodeExtractor
                cls.register('code', CodeExtractor)
            elif extractor_type == 'treesitter':
                from .extractors_treesitter import TreeSitterExtractor
                cls.register('treesitter', TreeSitterExtractor)
            elif extractor_type == 'robust':
                from .extractors_robust import RobustExtractor
                cls.register('robust', RobustExtractor)
            else:
                logger.warning(f"Unknown extractor type: {extractor_type}")
        except ImportError as e:
            logger.error(f"Failed to import {extractor_type} extractor: {e}")

    @classmethod
    def create_for_file(cls,
                       file_path: Union[str, Path],
                       config: Optional[ExtractorConfig] = None,
                       **kwargs) -> ExtractorBase:
        """
        Create the best extractor for a given file.

        Args:
            file_path: Path to the file
            config: Extraction configuration
            **kwargs: Additional arguments

        Returns:
            Extractor instance
        """
        return cls.create(file_path=file_path, config=config, **kwargs)

    @classmethod
    def list_available(cls) -> Dict[str, Any]:
        """
        List available extractors.

        Returns:
            Dictionary of available extractors with their info
        """
        available = {}
        for name, extractor_class in cls._extractors.items():
            try:
                available[name] = {
                    "class": extractor_class.__name__,
                    "module": extractor_class.__module__
                }
            except Exception as e:
                logger.warning(f"Failed to get info for {name}: {e}")
                available[name] = {"error": str(e)}

        return available

    @classmethod
    def get_format_mapping(cls) -> Dict[str, str]:
        """
        Get the format to extractor mapping.

        Returns:
            Dictionary mapping file extensions to extractor types
        """
        return cls._format_mapping.copy()
