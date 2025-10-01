#!/usr/bin/env python3
"""
Sentence-Transformers Embedder (fallback)

Lightweight embedder using sentence-transformers for environments where
the Jina v4 model (flash-attn / custom modules) is unavailable.

Implements a simple late-chunking approximation: split by approximate token
counts with overlap and embed each chunk.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - optional import
    torch = None  # type: ignore

from sentence_transformers import SentenceTransformer

from .embedders_base import EmbedderBase, EmbeddingConfig


@dataclass
class STChunkWithEmbedding:
    text: str
    embedding: np.ndarray
    start_char: int
    end_char: int
    start_token: int
    end_token: int
    chunk_index: int
    total_chunks: int
    context_window_used: int


class SentenceTransformersEmbedder(EmbedderBase):
    """Minimal sentence-transformers embedder with chunking support."""

    def __init__(self, config: Optional[EmbeddingConfig] = None) -> None:
        super().__init__(config)
        model_name = self.config.model_name or "sentence-transformers/all-mpnet-base-v2"
        device = self.config.device or (
            "cuda" if (torch and torch.cuda.is_available()) else "cpu"
        )
        self.model = self._load_model(model_name, device)
        self._dim = self._infer_embedding_dim(default=768)
        # Defaults for chunking if provided
        self.chunk_size_tokens = self.config.chunk_size_tokens or 512
        self.chunk_overlap_tokens = self.config.chunk_overlap_tokens or 128

    def _load_model(self, model_name: str, device: str) -> SentenceTransformer:
        """Instantiate the underlying sentence-transformers model."""

        kwargs: Dict[str, Any] = {
            "device": device,
            "trust_remote_code": getattr(self.config, "trust_remote_code", False),
            "local_files_only": getattr(self.config, "local_files_only", False),
        }
        cache_dir = getattr(self.config, "cache_dir", None)
        if cache_dir:
            kwargs["cache_folder"] = cache_dir
        return SentenceTransformer(model_name, **kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _infer_embedding_dim(self, default: int) -> int:
        """Determine the output embedding dimension with robust fallbacks."""

        value: Optional[int] = None

        getter = getattr(self.model, "get_sentence_embedding_dimension", None)
        if callable(getter):
            try:
                value = getter()
            except Exception as exc:  # pragma: no cover - defensive guard
                logging.getLogger(__name__).debug(
                    "SentenceTransformer dimension getter failed: %s", exc
                )

        if not value:
            # Walk modules from the tail to find a pooling layer that exposes the size
            for module in reversed(list(self.model)):  # type: ignore[arg-type]
                module_getter = getattr(module, "get_sentence_embedding_dimension", None)
                if callable(module_getter):
                    try:
                        value = module_getter()
                    except Exception:  # pragma: no cover - defensive guard
                        continue
                    if value:
                        break

        if not value:
            logging.getLogger(__name__).warning(
                "Falling back to default embedding dimension %s for %s",
                default,
                getattr(self.config, "model_name", "unknown"),
            )
            value = default

        return int(value)

    def embed_texts(self,
                    texts: List[str],
                    task: str = "retrieval",
                    batch_size: Optional[int] = None,
                    prompt_name: Optional[str] = None) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        emb = self.model.encode(
            texts,
            batch_size=batch_size or self.config.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return emb.astype(np.float32, copy=False)

    def embed_single(self,
                     text: str,
                     task: str = "retrieval",
                     prompt_name: Optional[str] = None) -> np.ndarray:
        arr = self.embed_texts([text], task=task, batch_size=1, prompt_name=prompt_name)
        return arr[0] if arr.size else np.zeros(self._dim, dtype=np.float32)

    @property
    def embedding_dimension(self) -> int:
        return self._dim

    @property
    def max_sequence_length(self) -> int:
        # Typical ST models accept 512 tokens
        return 512

    @property
    def supports_late_chunking(self) -> bool:
        return True

    def embed_with_late_chunking(self, text: str, task: str = "retrieval") -> List[STChunkWithEmbedding]:
        if not text:
            return []
        tokens = text.split()
        if not tokens:
            return []
        size = max(8, int(self.chunk_size_tokens))
        overlap = max(0, int(self.chunk_overlap_tokens))
        step = max(1, size - overlap)

        # Build chunks (token-based approximation)
        chunks: List[Dict[str, Any]] = []
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i : i + size]
            chunk_text = " ".join(chunk_tokens)
            if not chunk_text:
                continue
            chunks.append(
                {
                    "text": chunk_text,
                    "start_token": i,
                    "end_token": min(i + size, len(tokens)),
                }
            )

        # Embed chunk texts
        embeddings = self.embed_texts([c["text"] for c in chunks], task=task, batch_size=self.config.batch_size)

        out: List[STChunkWithEmbedding] = []
        for idx, (c, emb) in enumerate(zip(chunks, embeddings)):
            # Char offsets (approximate; not used downstream for exact spans)
            start_char = len(" ".join(tokens[: c["start_token"]]))
            end_char = start_char + len(c["text"])
            out.append(
                STChunkWithEmbedding(
                    text=c["text"],
                    embedding=emb.astype(np.float32, copy=False),
                    start_char=start_char,
                    end_char=end_char,
                    start_token=c["start_token"],
                    end_token=c["end_token"],
                    chunk_index=idx,
                    total_chunks=len(chunks),
                    context_window_used=len(tokens),
                )
            )
        return out
