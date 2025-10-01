#!/usr/bin/env python3
"""SentenceTransformers-backed Jina v4 embedder.

The upstream Jina embeddings model is now exposed via sentence-transformers:
https://huggingface.co/jinaai/jina-embeddings-v4

We wrap it with the same interface expected by the rest of HADES. Late chunking
reuses the sentence-transformers heuristic implementation while prompt selection
(`query` vs `passage`) matches Jina's recommended usage.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from .embedders_base import EmbeddingConfig
from .embedders_sentence import (
    SentenceTransformersEmbedder,
    STChunkWithEmbedding,
)

ChunkWithEmbedding = STChunkWithEmbedding

logger = logging.getLogger(__name__)


class JinaV4Embedder(SentenceTransformersEmbedder):
    """SentenceTransformers wrapper for jinaai/jina-embeddings-v4."""

    DEFAULT_MODEL = "jinaai/jina-embeddings-v4"

    def __init__(self, config: Optional[EmbeddingConfig] = None) -> None:
        if config is None:
            config = EmbeddingConfig(model_name=self.DEFAULT_MODEL)
        else:
            if not config.model_name:
                config.model_name = self.DEFAULT_MODEL
            if config.model_name != self.DEFAULT_MODEL:
                logger.info("Overriding model name to %s", self.DEFAULT_MODEL)
                config.model_name = self.DEFAULT_MODEL

        # Always trust remote code for Jina's custom modules
        config.trust_remote_code = True

        super().__init__(config)

        # Get embedding dimension from model, with fallback to 2048 for Jina v4
        try:
            dim_method = getattr(self.model, "get_sentence_embedding_dimension", None)
            if dim_method and callable(dim_method):
                dim_value = dim_method()
                self._dim = int(dim_value) if dim_value is not None else 2048
            else:
                self._dim = 2048
        except (TypeError, AttributeError):
            logger.warning("Could not determine embedding dimension, using default 2048")
            self._dim = 2048
        # Jina v4 tolerates long contexts; allow wider chunks by default
        self.chunk_size_tokens = self.config.chunk_size_tokens or 1024
        self.chunk_overlap_tokens = self.config.chunk_overlap_tokens or 256

    def _load_model(self, model_name: str, device: str) -> SentenceTransformer:
        """Override parent to set default_task for Jina v4."""
        kwargs: Dict[str, Any] = {
            "device": device,
            "trust_remote_code": getattr(self.config, "trust_remote_code", False),
            "local_files_only": getattr(self.config, "local_files_only", False),
            "model_kwargs": {"default_task": "retrieval"},
        }
        cache_dir = getattr(self.config, "cache_dir", None)
        if cache_dir:
            kwargs["cache_folder"] = cache_dir
        return SentenceTransformer(model_name, **kwargs)

    def embed_texts(
        self,
        texts: List[str],
        task: str = "retrieval",
        batch_size: Optional[int] = None,
        prompt_name: Optional[str] = None,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        jina_prompt = prompt_name or ("passage" if "passage" in task else "query")
        embeddings = self.model.encode(
            texts,
            prompt_name=jina_prompt,
            batch_size=batch_size or self.config.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32, copy=False)

    def embed_single(
        self,
        text: str,
        task: str = "retrieval",
        prompt_name: Optional[str] = None,
    ) -> np.ndarray:
        emb = self.embed_texts([text], task=task, batch_size=1, prompt_name=prompt_name)
        return emb[0] if emb.size else np.zeros(self._dim, dtype=np.float32)

    def embed_code(self, code_snippets: List[str], batch_size: int = 4) -> np.ndarray:
        return self.embed_texts(
            code_snippets,
            task="code",
            prompt_name="passage",
            batch_size=batch_size,
        )

    def embed_images(self, images):  # type: ignore[override]
        raise NotImplementedError("Image embeddings are not supported via sentence-transformers")

    def embed_multimodal(self, pairs):  # type: ignore[override]
        raise NotImplementedError("Multimodal embeddings are not supported via sentence-transformers")

    @property
    def embedding_dimension(self) -> int:
        return self._dim

    @property
    def max_sequence_length(self) -> int:
        # Jina embeddings support very long contexts; ST wrapper will tokenize/chunk as needed.
        return 32768

    @property
    def supports_images(self) -> bool:
        return False

    @property
    def supports_multimodal(self) -> bool:
        return False
