"""vLLM-based model engine for HADES-PathRAG.

DEPRECATED: This module is deprecated. Import from src.model_engine.engines.vllm instead.

This file is kept for backward compatibility only and will be removed in a future version.
"""
from __future__ import annotations

import logging
import warnings

# Import from the new consolidated implementation
from src.model_engine.engines.vllm.engine import VLLMModelEngine

# Show deprecation warning
warnings.warn(
    "The module src.model_engine.engines.vllm.vllm_engine is deprecated. "
    "Import VLLMModelEngine from src.model_engine.engines.vllm instead.",
    DeprecationWarning,
    stacklevel=2
)

# Set up logging for backward compatibility
logger = logging.getLogger(__name__)

# Nothing else needed - the VLLMModelEngine class is imported from engine.py
