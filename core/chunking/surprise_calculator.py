"""
Bayesian Surprise for Semantic Boundary Detection
==================================================

Based on "Bytes Are All You Need: Transformers Operating Directly On File Bytes"
(arXiv:2407.09450v2)

Uses token-level surprise (-log P(token | context)) to detect:
1. Semantic boundaries in documentation (for chunking)
2. Code quality signals (mixed concerns detection)

Surprise peaks indicate topic shifts or context changes, providing natural
boundaries for semantic chunking that align with human information structure.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class SurpriseCalculator:
    """
    Calculate surprise scores for semantic boundary detection.

    Surprise is defined as -log P(token | context), representing how
    unexpected a token is given the preceding context. High surprise
    indicates a topic shift or semantic boundary.
    """

    def __init__(self, model=None, tokenizer=None, device: str = "cpu"):
        """
        Initialize SurpriseCalculator.

        Args:
            model: Causal language model for computing token probabilities
            tokenizer: Tokenizer for the model
            device: Device to run model on ("cpu" or "cuda")
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        if model is not None:
            self.model.eval()
            self.model.to(device)

    @classmethod
    def from_pretrained(
        cls, model_name: str = "gpt2", device: Optional[str] = None
    ) -> SurpriseCalculator:
        """
        Load a pretrained model for surprise calculation.

        Args:
            model_name: HuggingFace model name (default: "gpt2")
            device: Device to use ("cpu" or "cuda", auto-detected if None)

        Returns:
            SurpriseCalculator instance
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers library required for SurpriseCalculator. "
                "Install with: pip install transformers"
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading surprise model: {model_name} on {device}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # Ensure padding token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return cls(model, tokenizer, device)

    def calculate_surprise_scores(
        self, text: str, window_size: int = 50, stride: int = 25
    ) -> np.ndarray:
        """
        Calculate surprise at each token position.

        Args:
            text: Input text
            window_size: Context window size for prediction
            stride: Stride between windows (for efficiency)

        Returns:
            Array of surprise scores (one per token)
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be initialized")

        # Tokenize input
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) == 0:
            return np.array([])

        surprises = []

        # Calculate surprise for each position
        for i in range(0, len(tokens), stride):
            # Get context window
            start_idx = max(0, i - window_size)
            context = tokens[start_idx:i]

            if len(context) == 0:
                # No context for first tokens, assign default surprise
                surprises.extend([0.5] * min(stride, len(tokens) - i))
                continue

            # Target tokens (next stride tokens)
            end_idx = min(i + stride, len(tokens))
            targets = tokens[i:end_idx]

            if len(targets) == 0:
                break

            # Prepare input
            input_ids = torch.tensor([context], device=self.device)

            # Get model predictions
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :]  # Last position predictions
                probs = torch.softmax(logits, dim=-1)

            # Calculate surprise for each target token
            for target in targets:
                surprise = -torch.log(probs[target] + 1e-10).item()
                surprises.append(surprise)

        return np.array(surprises[: len(tokens)])

    def detect_boundaries(
        self,
        text: str,
        threshold_percentile: float = 85,
        min_chunk_size: int = 100,
        window_size: int = 50,
    ) -> List[int]:
        """
        Detect semantic boundaries using surprise peaks.

        High surprise scores indicate topic shifts or semantic boundaries,
        providing natural chunking points for documentation.

        Args:
            text: Input text
            threshold_percentile: Percentile for surprise threshold (default: 85)
            min_chunk_size: Minimum chunk size in characters (default: 100)
            window_size: Context window for surprise calculation

        Returns:
            List of character offsets for chunk boundaries
        """
        if not text:
            return [0]

        # Calculate surprises
        surprises = self.calculate_surprise_scores(text, window_size=window_size)

        if len(surprises) == 0:
            return [0, len(text)]

        # Determine threshold from percentile
        threshold = np.percentile(surprises, threshold_percentile)

        # Find peaks above threshold
        boundaries = [0]
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        for i, surprise in enumerate(surprises):
            if surprise > threshold:
                # Convert token index to character offset
                # Decode tokens up to this point to get character offset
                char_offset = len(self.tokenizer.decode(tokens[: i + 1]))

                # Enforce minimum chunk size
                if char_offset - boundaries[-1] >= min_chunk_size:
                    boundaries.append(char_offset)

        # Add final boundary
        if boundaries[-1] != len(text):
            boundaries.append(len(text))

        return boundaries

    def calculate_code_quality_metrics(
        self, code: str, function_spans: List[Tuple[int, int]]
    ) -> dict:
        """
        Calculate surprise-based code quality metrics.

        High surprise within a function indicates mixed concerns or
        unexpected context shifts, suggesting poor code structure.

        Args:
            code: Source code text
            function_spans: List of (start_line, end_line) for each function

        Returns:
            Dict mapping function index to quality metrics:
            {
                "func_0": {
                    "mean_surprise": float,
                    "variance": float,
                    "max_surprise": float,
                    "quality_signal": "well_structured" | "mixed_concerns" | "moderate"
                },
                ...
            }
        """
        if not function_spans:
            return {}

        # Convert line spans to character spans
        lines = code.split("\n")
        char_spans = []

        for start_line, end_line in function_spans:
            start_char = sum(len(line) + 1 for line in lines[: start_line - 1])
            end_char = sum(len(line) + 1 for line in lines[:end_line])
            char_spans.append((start_char, end_char))

        # Calculate surprises for full code
        surprises = self.calculate_surprise_scores(code)

        if len(surprises) == 0:
            return {}

        # Map characters back to tokens for indexing
        tokens = self.tokenizer.encode(code, add_special_tokens=False)

        metrics = {}

        for i, (start_char, end_char) in enumerate(char_spans):
            # Find token indices for this character range
            # (Approximate: decode tokens until we reach char offsets)
            start_token = 0
            end_token = len(tokens)

            for j in range(len(tokens)):
                decoded_len = len(self.tokenizer.decode(tokens[: j + 1]))
                if decoded_len >= start_char and start_token == 0:
                    start_token = j
                if decoded_len >= end_char:
                    end_token = j
                    break

            # Extract surprises for this function
            func_surprises = surprises[start_token:end_token]

            if len(func_surprises) == 0:
                continue

            mean_surprise = float(np.mean(func_surprises))
            variance = float(np.var(func_surprises))
            max_surprise = float(np.max(func_surprises))

            # Quality heuristics based on surprise levels
            if mean_surprise < 0.5:
                quality_signal = "well_structured"
            elif mean_surprise > 1.5:
                quality_signal = "mixed_concerns"
            else:
                quality_signal = "moderate"

            metrics[f"func_{i}"] = {
                "mean_surprise": mean_surprise,
                "variance": variance,
                "max_surprise": max_surprise,
                "quality_signal": quality_signal,
            }

        return metrics

    def chunk_text_with_surprise(
        self, text: str, target_chunk_size: int = 1000, threshold_percentile: float = 85
    ) -> List[dict]:
        """
        Chunk text at surprise-detected boundaries.

        Args:
            text: Input text to chunk
            target_chunk_size: Target chunk size in characters (soft limit)
            threshold_percentile: Percentile threshold for boundaries

        Returns:
            List of chunk dicts with keys: text, start_char, end_char, surprise_score
        """
        boundaries = self.detect_boundaries(
            text,
            threshold_percentile=threshold_percentile,
            min_chunk_size=target_chunk_size // 2,
        )

        chunks = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]

            chunks.append(
                {
                    "text": text[start:end],
                    "start_char": start,
                    "end_char": end,
                    "surprise_score": self._calculate_boundary_surprise_score(
                        text, start
                    ),
                }
            )

        return chunks

    def _calculate_boundary_surprise_score(self, text: str, boundary: int) -> float:
        """Calculate average surprise around a boundary point."""
        # Extract window around boundary
        window_size = 100
        start = max(0, boundary - window_size)
        end = min(len(text), boundary + window_size)
        window = text[start:end]

        if not window:
            return 0.0

        surprises = self.calculate_surprise_scores(window, window_size=25)
        return float(np.mean(surprises)) if len(surprises) > 0 else 0.0
