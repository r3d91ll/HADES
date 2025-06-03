"""vLLM-based model engine for HADES-PathRAG.

This engine uses vLLM for managing models in a separate process. It provides efficient
GPU memory management and high-performance inference for large language models.
"""
# Import the VLLMModelEngine directly from the engine module to avoid duplication
from src.model_engine.engines.vllm.engine import VLLMModelEngine

# Re-export any other necessary types or functions
from src.config.vllm_config import VLLMConfig, ModelMode
from src.model_engine.vllm_session import get_vllm_manager, VLLMProcessManager

# Set version
__version__ = "1.0.0"

# This is just an import module now - all implementation is in engine.py
