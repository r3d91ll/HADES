"""
Extractors Module

Provides document extraction capabilities for various file formats.
All extractors follow the Conveyance Framework, maximizing the CONVEYANCE
dimension by transforming unstructured documents into actionable data.

This module replaces core.framework.extractors with a cleaner structure.
"""

from .extractors_base import ExtractionResult, ExtractorBase, ExtractorConfig
from .extractors_factory import ExtractorFactory

# Auto-register available extractors
try:
    from .extractors_docling import DoclingExtractor
    ExtractorFactory.register("docling", DoclingExtractor)
except ImportError:
    pass

try:
    from .extractors_latex import LaTeXExtractor
    ExtractorFactory.register("latex", LaTeXExtractor)
except ImportError:
    pass

try:
    from .extractors_code import CodeExtractor
    ExtractorFactory.register("code", CodeExtractor)
except ImportError:
    pass

try:
    from .extractors_treesitter import TreeSitterExtractor
    ExtractorFactory.register("treesitter", TreeSitterExtractor)
except ImportError:
    pass

try:
    from .extractors_robust import RobustExtractor
    ExtractorFactory.register("robust", RobustExtractor)
except ImportError:
    pass

# Make all classes available at module level
try:
    from .extractors_docling import DoclingExtractor
except ImportError:
    DoclingExtractor = None  # type: ignore[misc]

try:
    from .extractors_latex import LaTeXExtractor
except ImportError:
    LaTeXExtractor = None  # type: ignore[misc]

try:
    from .extractors_code import CodeExtractor
except ImportError:
    CodeExtractor = None  # type: ignore[misc]

try:
    from .extractors_treesitter import TreeSitterExtractor
except ImportError:
    TreeSitterExtractor = None  # type: ignore[misc]

try:
    from .extractors_robust import RobustExtractor
except ImportError:
    RobustExtractor = None  # type: ignore[misc]

# Backward compatibility exports
__all__ = [
    'ExtractorBase',
    'ExtractorConfig',
    'ExtractionResult',
    'ExtractorFactory',
    'DoclingExtractor',
    'LaTeXExtractor',
    'CodeExtractor',
    'TreeSitterExtractor',
    'RobustExtractor',
]

# Convenience function for backward compatibility
def create_extractor_for_file(file_path, **kwargs):
    """
    Create an extractor for a given file (backward compatibility).

    Args:
        file_path: Path to the file
        **kwargs: Additional configuration

    Returns:
        Extractor instance
    """
    return ExtractorFactory.create_for_file(file_path, **kwargs)
