"""
Model adapter implementations for HADES-PathRAG.

This package contains adapters for different model backends, providing a
consistent interface for embedding, completion, and chat functionality.
"""

from src.model_engine.adapters.vllm_adapter import VLLMAdapter, start_vllm_server

__all__ = ["VLLMAdapter", "start_vllm_server"]
