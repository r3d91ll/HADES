#!/usr/bin/env python3
"""
Generate GraphSAGE Training Data
=================================

Creates query-node pairs for supervised GraphSAGE training.

Strategy:
1. Synthetic queries from file names and docstrings
2. Positive examples from PathRAG successful retrievals
3. Negative sampling from random/low-scoring nodes

Output: training_data.pt with query embeddings and labeled node pairs

Usage:
    poetry run python scripts/generate_training_data.py --output models/training_data.pt
    poetry run python scripts/generate_training_data.py --dry-run --samples 10
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.database.arango.memory_client import (
    ArangoMemoryClient,
    resolve_memory_config,
)
from core.embedders.embedders_jina import JinaV4Embedder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingDataGenerator:
    """Generate query-node pairs for GraphSAGE training."""

    def __init__(self, client: ArangoMemoryClient, embedder: JinaV4Embedder):
        self.client = client
        self.embedder = embedder

    def generate_from_files(
        self,
        num_samples: int = 100,
    ) -> List[Tuple[str, np.ndarray, str, int]]:
        """
        Generate synthetic queries from file metadata.

        For each file:
        - Query 1: "What does {filename} do?"
        - Query 2: "How does {module_name} work?"
        - Query 3: Extract first sentence from docstring

        Returns:
            List of (query_text, query_embedding, node_id, label)
            label: 1 = positive, 0 = negative
        """
        logger.info("Generating training data from file metadata...")

        # Get all documents with embeddings
        aql = """
        FOR doc IN repo_docs
          FILTER doc.embedding != null
          LIMIT @limit
          RETURN {
            _id: doc._id,
            path: doc.path,
            text: doc.text
          }
        """

        docs = list(self.client.execute_query(aql, {"limit": num_samples}))
        logger.info(f"Found {len(docs)} documents")

        training_pairs = []

        for doc in docs:
            doc_id = doc["_id"]
            path = doc.get("path", "")
            text = doc.get("text", "")

            # Extract module name from path
            filename = Path(path).stem if path else "unknown"
            module_name = filename.replace("_", " ")

            queries = []

            # Query 1: Filename-based
            queries.append(f"What does {filename} do?")
            queries.append(f"How does {module_name} work?")

            # Query 2: Extract docstring summary
            docstring_summary = self._extract_docstring_summary(text)
            if docstring_summary:
                queries.append(docstring_summary)

            # Query 3: Function/class name queries (if code file)
            if path.endswith(".py"):
                functions = self._extract_function_names(text)
                for func in functions[:2]:  # Top 2 functions
                    queries.append(f"How does {func} function work?")

            # Generate embeddings for queries
            for query in queries:
                try:
                    query_emb = self.embedder.embed_single(
                        query,
                        task="retrieval",
                        prompt_name="query"
                    )

                    # Positive pair: (query, matching_doc, label=1)
                    training_pairs.append((query, query_emb, doc_id, 1))

                except Exception as exc:
                    logger.warning(f"Failed to embed query '{query}': {exc}")

        logger.info(f"Generated {len(training_pairs)} positive training pairs")
        return training_pairs

    def generate_negative_samples(
        self,
        positive_pairs: List[Tuple[str, np.ndarray, str, int]],
        neg_ratio: float = 2.0,
    ) -> List[Tuple[str, np.ndarray, str, int]]:
        """
        Generate negative samples by pairing queries with random non-matching nodes.

        Args:
            positive_pairs: Positive (query, embedding, node_id, 1) pairs
            neg_ratio: Number of negative samples per positive (default: 2.0)

        Returns:
            List of negative (query, embedding, node_id, 0) pairs
        """
        logger.info(f"Generating negative samples (ratio={neg_ratio})...")

        # Get all document IDs
        aql = """
        FOR doc IN repo_docs
          FILTER doc.embedding != null
          RETURN doc._id
        """
        all_doc_ids = list(self.client.execute_query(aql, {}))

        negative_pairs = []
        num_negatives = int(len(positive_pairs) * neg_ratio)

        for i in range(num_negatives):
            # Sample random positive pair
            query, query_emb, pos_doc_id, _ = positive_pairs[i % len(positive_pairs)]

            # Sample random negative document (different from positive)
            neg_doc_id = pos_doc_id
            while neg_doc_id == pos_doc_id:
                neg_doc_id = np.random.choice(all_doc_ids)

            # Negative pair: (query, non_matching_doc, label=0)
            negative_pairs.append((query, query_emb, neg_doc_id, 0))

        logger.info(f"Generated {len(negative_pairs)} negative training pairs")
        return negative_pairs

    def _extract_docstring_summary(self, text: str) -> str:
        """Extract first sentence from module docstring."""
        # Match triple-quoted docstrings
        match = re.search(r'"""(.*?)"""', text, re.DOTALL)
        if not match:
            match = re.search(r"'''(.*?)'''", text, re.DOTALL)

        if match:
            docstring = match.group(1).strip()
            # Get first sentence
            sentences = re.split(r'[.!?]\s+', docstring)
            if sentences:
                return sentences[0].strip()

        return ""

    def _extract_function_names(self, text: str) -> List[str]:
        """Extract function names from Python code."""
        # Match function definitions
        matches = re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', text)
        return matches


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate GraphSAGE training data")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("models/training_data.pt"),
        help="Output path for training data"
    )
    p.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of documents to sample (default: 100, use -1 for all)"
    )
    p.add_argument(
        "--neg-ratio",
        type=float,
        default=2.0,
        help="Negative samples per positive (default: 2.0)"
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview data generation without saving"
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    logger.info("Initializing ArangoDB client...")
    cfg = resolve_memory_config()
    client = ArangoMemoryClient(cfg)

    logger.info("Loading Jina v4 embedder...")
    embedder = JinaV4Embedder()

    logger.info("Generating training data...")
    generator = TrainingDataGenerator(client, embedder)

    # Generate positive samples
    positive_pairs = generator.generate_from_files(num_samples=args.samples)

    if not positive_pairs:
        logger.error("No training data generated!")
        return 1

    # Generate negative samples
    negative_pairs = generator.generate_negative_samples(
        positive_pairs,
        neg_ratio=args.neg_ratio
    )

    # Combine and shuffle
    all_pairs = positive_pairs + negative_pairs
    np.random.shuffle(all_pairs)

    # Print statistics
    logger.info("=" * 80)
    logger.info("Training Data Statistics")
    logger.info("=" * 80)
    logger.info(f"Positive pairs: {len(positive_pairs)}")
    logger.info(f"Negative pairs: {len(negative_pairs)}")
    logger.info(f"Total pairs: {len(all_pairs)}")
    logger.info(f"Positive ratio: {len(positive_pairs) / len(all_pairs):.2%}")
    logger.info("=" * 80)

    # Show sample queries
    logger.info("\nSample Queries:")
    for i, (query, _, node_id, label) in enumerate(all_pairs[:5], 1):
        label_str = "POS" if label == 1 else "NEG"
        logger.info(f"  [{label_str}] {query} → {node_id}")

    if args.dry_run:
        logger.info("\nDry run complete. Exiting without saving.")
        return 0

    # Save training data
    args.output.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nSaving training data to {args.output}...")

    # Convert to tensors
    queries_text = [pair[0] for pair in all_pairs]
    query_embeddings = torch.tensor(np.array([pair[1] for pair in all_pairs]), dtype=torch.float32)
    node_ids = [pair[2] for pair in all_pairs]
    labels = torch.tensor([pair[3] for pair in all_pairs], dtype=torch.long)

    torch.save({
        "queries": queries_text,
        "query_embeddings": query_embeddings,
        "node_ids": node_ids,
        "labels": labels,
        "metadata": {
            "num_samples": args.samples,
            "neg_ratio": args.neg_ratio,
            "total_pairs": len(all_pairs),
            "positive_pairs": len(positive_pairs),
            "negative_pairs": len(negative_pairs),
        }
    }, args.output)

    logger.info("✓ Training data generation complete!")
    logger.info(f"  Output: {args.output}")
    logger.info(f"  Ready for GraphSAGE training")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
