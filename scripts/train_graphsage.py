#!/usr/bin/env python3
"""
Train GraphSAGE Model
=====================

Trains multi-relational GraphSAGE on code graph for retrieval.

Prerequisites:
1. poetry run python scripts/export_graph_for_training.py
2. poetry run python scripts/generate_training_data.py

Usage:
    poetry run python scripts/train_graphsage.py
    poetry run python scripts/train_graphsage.py --epochs 50 --batch-size 64
    poetry run python scripts/train_graphsage.py --resume models/checkpoints/latest.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.gnn.trainer import GraphSAGETrainer, TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GraphSAGE model")

    # Data paths
    p.add_argument(
        "--graph-data",
        type=Path,
        default=Path("models/graph_data.pt"),
        help="Path to exported graph data"
    )
    p.add_argument(
        "--training-data",
        type=Path,
        default=Path("models/training_data.pt"),
        help="Path to training data (query-node pairs)"
    )

    # Training config
    p.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size")
    p.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Checkpointing
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("models/checkpoints"),
        help="Checkpoint directory"
    )
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from checkpoint"
    )

    return p.parse_args()


def load_data(graph_path: Path, training_path: Path):
    """Load graph and training data."""
    logger.info(f"Loading graph data from {graph_path}...")
    graph_checkpoint = torch.load(graph_path, weights_only=False)

    data = graph_checkpoint["data"]
    train_mask = graph_checkpoint["train_mask"]
    val_mask = graph_checkpoint["val_mask"]
    node_id_map = graph_checkpoint["node_id_map"]

    logger.info(f"Graph: {data.x.shape[0]} nodes, {sum(e.shape[1] for e in data.edge_index_dict.values())} edges")
    logger.info(f"Edge types: {list(data.edge_index_dict.keys())}")

    logger.info(f"Loading training data from {training_path}...")
    training_checkpoint = torch.load(training_path, weights_only=False)

    queries = training_checkpoint["queries"]
    query_embeddings = training_checkpoint["query_embeddings"]
    node_ids = training_checkpoint["node_ids"]
    labels = training_checkpoint["labels"]

    logger.info(f"Training pairs: {len(queries)}")
    logger.info(f"Positive: {labels.sum().item()}, Negative: {(labels == 0).sum().item()}")

    # Map node_ids to indices
    id_to_idx = {aid: int(idx) for idx, aid in node_id_map.items()}

    node_indices = []
    valid_mask = []
    for node_id in node_ids:
        if node_id in id_to_idx:
            node_indices.append(id_to_idx[node_id])
            valid_mask.append(True)
        else:
            node_indices.append(0)  # Placeholder
            valid_mask.append(False)
            logger.warning(f"Node {node_id} not found in graph")

    valid_mask = np.array(valid_mask)

    # Filter out invalid mappings
    query_embeddings = query_embeddings[valid_mask]
    node_indices = [node_indices[i] for i, v in enumerate(valid_mask) if v]
    labels = labels[valid_mask]

    logger.info(f"Valid training pairs after filtering: {len(node_indices)}")

    # Create train/val split for training data
    num_samples = len(node_indices)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    val_size = int(num_samples * 0.2)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_data_mask = np.zeros(num_samples, dtype=bool)
    val_data_mask = np.zeros(num_samples, dtype=bool)
    train_data_mask[train_indices] = True
    val_data_mask[val_indices] = True

    return {
        "data": data,
        "query_embeddings": query_embeddings,
        "node_indices": node_indices,
        "labels": labels,
        "train_mask": train_data_mask,
        "val_mask": val_data_mask,
        "edge_types": list(data.edge_index_dict.keys()),
    }


def main() -> int:
    args = parse_args()

    # Check prerequisites
    if not args.graph_data.exists():
        logger.error(f"Graph data not found: {args.graph_data}")
        logger.error("Run: poetry run python scripts/export_graph_for_training.py")
        return 1

    if not args.training_data.exists():
        logger.error(f"Training data not found: {args.training_data}")
        logger.error("Run: poetry run python scripts/generate_training_data.py")
        return 1

    # Load data
    try:
        data_dict = load_data(args.graph_data, args.training_data)
    except Exception as exc:
        logger.error(f"Failed to load data: {exc}")
        return 1

    # Create training config
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )

    # Initialize or resume trainer
    if args.resume and args.resume.exists():
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer = GraphSAGETrainer.load_checkpoint(args.resume, config)
    else:
        logger.info("Initializing new trainer...")
        trainer = GraphSAGETrainer(config, edge_types=data_dict["edge_types"])

    # Train
    try:
        trainer.fit(
            data=data_dict["data"],
            query_embeddings=data_dict["query_embeddings"],
            node_indices=data_dict["node_indices"],
            labels=data_dict["labels"],
            train_mask=data_dict["train_mask"],
            val_mask=data_dict["val_mask"],
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.save_checkpoint(is_best=False)
        logger.info("Checkpoint saved")
    except Exception as exc:
        logger.error(f"Training failed: {exc}")
        import traceback
        traceback.print_exc()
        return 1

    # Save final model
    final_path = args.checkpoint_dir / "final.pt"
    trainer.save_checkpoint(is_best=False)
    logger.info(f"Final model saved: {final_path}")

    # Print summary
    logger.info("=" * 80)
    logger.info("Training Summary")
    logger.info("=" * 80)
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
    logger.info(f"Total epochs: {trainer.current_epoch + 1}")
    logger.info(f"Checkpoints: {args.checkpoint_dir}")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
