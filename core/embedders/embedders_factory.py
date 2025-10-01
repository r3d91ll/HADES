#!/usr/bin/env python3
"""
Embedder Factory - HADES Standardized on Jina v4

Factory pattern for creating embedder instances based on configuration.
HADES Lab has standardized on Jina v4 embeddings for consistency and performance.

Architecture Decision:
- Production: JinaV4Embedder (transformers-based, 32k context, late chunking)
- Deprecated: SentenceTransformersEmbedder (moved to Acheron/)

Extension Guide for Future Models:
1. Create new class inheriting from EmbedderBase
2. Implement required methods (embed_texts, embed_with_late_chunking, etc.)
3. Register in factory's _auto_register method
4. Update _determine_embedder_type for model name detection

Theory Connection:
The factory pattern enables flexible switching between embedding models
while maintaining consistent interfaces, supporting the WHO dimension
(different models as agents) without changing the WHAT processing logic.
"""

import logging
from typing import Any, Dict, Optional

from .embedders_base import EmbedderBase, EmbeddingConfig

logger = logging.getLogger(__name__)


class EmbedderFactory:
    """
    Factory for creating embedder instances.

    Manages the instantiation of different embedder types based on
    configuration, with support for fallbacks and auto-detection.
    """

    # Registry of available embedders
    _embedders: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, embedder_class: type):
        """
        Register an embedder class.

        Args:
            name: Name to register under
            embedder_class: Embedder class to register
        """
        cls._embedders[name] = embedder_class
        logger.info(f"Registered embedder: {name}")

    @classmethod
    def create(cls,
              model_name: str = "jinaai/jina-embeddings-v4",
              config: Optional[EmbeddingConfig] = None,
              **kwargs) -> EmbedderBase:
        """
        Create an embedder instance.

        Args:
            model_name: Name or path of the model
            config: Embedding configuration
            **kwargs: Additional arguments for the embedder

        Returns:
            Embedder instance

        Raises:
            ValueError: If no suitable embedder found
        """
        # Create config if not provided
        if config is None:
            config = EmbeddingConfig(model_name=model_name, **kwargs)

        # Determine embedder type based on model name
        embedder_type = cls._determine_embedder_type(model_name)

        if embedder_type not in cls._embedders:
            # Try to import and register on-demand
            cls._auto_register(embedder_type)

        if embedder_type not in cls._embedders:
            available = list(cls._embedders.keys())
            raise ValueError(
                f"No embedder registered for type '{embedder_type}'. "
                f"Available: {available}"
            )

        embedder_class = cls._embedders[embedder_type]
        logger.info(f"Creating {embedder_type} embedder for model: {model_name}")

        return embedder_class(config)

    @classmethod
    def _determine_embedder_type(cls, model_name: str) -> str:
        """
        Determine embedder type from model name.

        Args:
            model_name: Model name or path

        Returns:
            Embedder type string
        """
        # HADES standardized on the SentenceTransformers-backed Jina embedder.
        return "jina"

    @classmethod
    def _auto_register(cls, embedder_type: str):
        """
        Attempt to auto-register an embedder type.

        Args:
            embedder_type: Type of embedder to register
        """
        try:
            if embedder_type == "jina":
                from .embedders_jina import JinaV4Embedder
                cls.register("jina", JinaV4Embedder)
            else:
                logger.warning(f"Unknown embedder type: {embedder_type}")
        except ImportError as e:
            logger.error(f"Failed to import {embedder_type} embedder: {e}")

    @classmethod
    def list_available(cls) -> Dict[str, Any]:
        """
        List available embedders.

        Returns:
            Dictionary of available embedders with their info
        """
        available = {}
        for name, embedder_class in cls._embedders.items():
            try:
                # Try to get class-level info without instantiation
                available[name] = {
                    "class": embedder_class.__name__,
                    "module": embedder_class.__module__
                }
            except Exception as e:
                logger.warning(f"Failed to get info for {name}: {e}")
                available[name] = {"error": str(e)}

        return available
