"""
vLLM Model Engine Component

This component provides vLLM-specific model serving capabilities
with optimized batching and GPU acceleration.
"""

from .processor import VLLMModelEngine

__all__ = ["VLLMModelEngine"]