#!/usr/bin/env python3
"""
GraphSAGE Trainer with Ranking Loss
====================================

Training pipeline for multi-relational GraphSAGE with contrastive/triplet loss.

Loss Function:
- Contrastive loss for query-node retrieval task
- Positive pairs: (query, relevant_node) → cosine_sim close to 1
- Negative pairs: (query, irrelevant_node) → cosine_sim close to 0

Optimization:
- Adam optimizer with learning rate scheduling
- Early stopping on validation loss
- Model checkpointing (best + latest)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .graphsage_model import MultiRelationalGraphSAGE

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Model architecture
    in_channels: int = 2048
    hidden_channels: int = 1024
    out_channels: int = 512
    num_layers: int = 3
    dropout: float = 0.3

    # Training
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5

    # Loss function
    margin: float = 0.5  # Contrastive loss margin
    temperature: float = 0.07  # Temperature for softmax (InfoNCE style)

    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4

    # Checkpointing
    checkpoint_dir: Path = Path("models/checkpoints")
    save_best_only: bool = True

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for retrieval.

    Positive pairs should have high cosine similarity.
    Negative pairs should have low cosine similarity.
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        query_embeddings: Tensor,
        node_embeddings: Tensor,
        labels: Tensor,
    ) -> Tensor:
        """
        Compute contrastive loss.

        Args:
            query_embeddings: [batch_size, embed_dim]
            node_embeddings: [batch_size, embed_dim]
            labels: [batch_size] (1=positive, 0=negative)

        Returns:
            Scalar loss
        """
        # Normalize embeddings
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        node_embeddings = F.normalize(node_embeddings, p=2, dim=1)

        # Cosine similarity
        cosine_sim = torch.sum(query_embeddings * node_embeddings, dim=1)

        # Contrastive loss
        # Positive: minimize (1 - sim)
        # Negative: minimize max(0, sim - margin)
        pos_loss = labels * (1 - cosine_sim)
        neg_loss = (1 - labels) * torch.clamp(cosine_sim - self.margin, min=0)

        loss = torch.mean(pos_loss + neg_loss)
        return loss


class GraphSAGETrainer:
    """
    Trainer for multi-relational GraphSAGE.
    """

    def __init__(
        self,
        config: TrainingConfig,
        edge_types: Optional[List[str]] = None,
    ):
        self.config = config
        self.device = torch.device(config.device)

        # Initialize model
        self.model = MultiRelationalGraphSAGE(
            in_channels=config.in_channels,
            hidden_channels=config.hidden_channels,
            out_channels=config.out_channels,
            num_layers=config.num_layers,
            dropout=config.dropout,
            edge_types=edge_types,
        ).to(self.device)

        # Query projection: map query embeddings (2048-dim) to match model output (512-dim)
        self.query_projection = nn.Linear(config.in_channels, config.out_channels).to(self.device)

        # Loss function
        self.criterion = ContrastiveLoss(margin=config.margin)

        # Optimizer (include both model and query projection)
        self.optimizer = Adam(
            list(self.model.parameters()) + list(self.query_projection.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []

        # Create checkpoint directory
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(
        self,
        data,
        query_embeddings: Tensor,
        node_indices: List[int],
        labels: Tensor,
        train_mask: np.ndarray,
    ) -> float:
        """
        Train for one epoch.

        Args:
            data: PyG Data object with graph structure
            query_embeddings: [num_queries, 2048] query embeddings
            node_indices: List of node indices for each query
            labels: [num_queries] binary labels (1=positive, 0=negative)
            train_mask: Boolean mask for training samples

        Returns:
            Average training loss
        """
        self.model.train()

        # Filter training samples
        train_idx = np.where(train_mask)[0]
        num_train = len(train_idx)

        # Shuffle training data
        perm = np.random.permutation(num_train)
        train_idx = train_idx[perm]

        total_loss = 0.0
        num_batches = 0

        for start_idx in range(0, num_train, self.config.batch_size):
            end_idx = min(start_idx + self.config.batch_size, num_train)
            batch_idx = train_idx[start_idx:end_idx]

            # Get batch data
            batch_queries = query_embeddings[batch_idx].to(self.device)
            batch_nodes = [node_indices[i] for i in batch_idx]
            batch_labels = labels[batch_idx].to(self.device)

            # Project queries to match node embedding dimension (2048 → 512)
            batch_queries_proj = self.query_projection(batch_queries)

            # Forward pass: compute node embeddings
            node_embs = self.model(data.x.to(self.device), {
                k: v.to(self.device) for k, v in data.edge_index_dict.items()
            })

            # Get embeddings for batch nodes
            batch_node_embs = node_embs[batch_nodes]

            # Compute loss
            loss = self.criterion(batch_queries_proj, batch_node_embs, batch_labels.float())

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    @torch.no_grad()
    def validate(
        self,
        data,
        query_embeddings: Tensor,
        node_indices: List[int],
        labels: Tensor,
        val_mask: np.ndarray,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Validate model.

        Returns:
            (val_loss, metrics_dict)
        """
        self.model.eval()

        val_idx = np.where(val_mask)[0]

        if len(val_idx) == 0:
            return 0.0, {}

        # Get validation data
        val_queries = query_embeddings[val_idx].to(self.device)
        val_nodes = [node_indices[i] for i in val_idx]
        val_labels = labels[val_idx].to(self.device)

        # Project queries (2048 → 512)
        val_queries_proj = self.query_projection(val_queries)

        # Forward pass
        node_embs = self.model(data.x.to(self.device), {
            k: v.to(self.device) for k, v in data.edge_index_dict.items()
        })

        val_node_embs = node_embs[val_nodes]

        # Compute loss
        loss = self.criterion(val_queries_proj, val_node_embs, val_labels.float())

        # Compute metrics
        metrics = self._compute_metrics(val_queries_proj, val_node_embs, val_labels)

        return loss.item(), metrics

    def _compute_metrics(
        self,
        query_embs: Tensor,
        node_embs: Tensor,
        labels: Tensor,
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        # Normalize embeddings
        query_embs = F.normalize(query_embs, p=2, dim=1)
        node_embs = F.normalize(node_embs, p=2, dim=1)

        # Cosine similarities
        cosine_sim = torch.sum(query_embs * node_embs, dim=1)

        # Accuracy (threshold at 0.5)
        predictions = (cosine_sim > 0.5).long()
        accuracy = (predictions == labels).float().mean().item()

        # Separate positive and negative
        pos_mask = labels == 1
        neg_mask = labels == 0

        pos_sim = cosine_sim[pos_mask].mean().item() if pos_mask.any() else 0.0
        neg_sim = cosine_sim[neg_mask].mean().item() if neg_mask.any() else 0.0

        return {
            "accuracy": accuracy,
            "pos_sim_mean": pos_sim,
            "neg_sim_mean": neg_sim,
        }

    def fit(
        self,
        data,
        query_embeddings: Tensor,
        node_indices: List[int],
        labels: Tensor,
        train_mask: np.ndarray,
        val_mask: np.ndarray,
    ):
        """
        Train GraphSAGE model.

        Args:
            data: PyG Data object
            query_embeddings: Query embeddings
            node_indices: Node index for each query
            labels: Binary labels
            train_mask: Training sample mask
            val_mask: Validation sample mask
        """
        logger.info("Starting GraphSAGE training...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model: {self.model}")
        logger.info(f"Train samples: {train_mask.sum()}, Val samples: {val_mask.sum()}")

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch(
                data, query_embeddings, node_indices, labels, train_mask
            )
            self.train_losses.append(train_loss)

            # Validate
            val_loss, metrics = self.validate(
                data, query_embeddings, node_indices, labels, val_mask
            )
            self.val_losses.append(val_loss)

            # Update learning rate
            self.scheduler.step(val_loss)

            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{self.config.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Acc: {metrics.get('accuracy', 0):.3f} | "
                f"Pos Sim: {metrics.get('pos_sim_mean', 0):.3f} | "
                f"Neg Sim: {metrics.get('neg_sim_mean', 0):.3f}"
            )

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if is_best or not self.config.save_best_only:
                self.save_checkpoint(is_best=is_best)

            # Early stopping
            if self.patience_counter >= self.config.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        logger.info("Training complete!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "query_projection_state_dict": self.query_projection.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "config": self.config.__dict__,
            "in_channels": self.config.in_channels,
            "hidden_channels": self.config.hidden_channels,
            "out_channels": self.config.out_channels,
            "num_layers": self.config.num_layers,
            "dropout": self.config.dropout,
            "edge_types": self.model.edge_types,
        }

        # Save latest
        latest_path = self.config.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

        # Save best
        if is_best:
            best_path = self.config.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint: {best_path}")

    @classmethod
    def load_checkpoint(cls, checkpoint_path: Path, config: Optional[TrainingConfig] = None):
        """Load trainer from checkpoint."""
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        if config is None:
            config = TrainingConfig(**checkpoint["config"])

        trainer = cls(config, edge_types=checkpoint["edge_types"])
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        if "query_projection_state_dict" in checkpoint:
            trainer.query_projection.load_state_dict(checkpoint["query_projection_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        trainer.current_epoch = checkpoint["epoch"]
        trainer.best_val_loss = checkpoint["best_val_loss"]
        trainer.train_losses = checkpoint["train_losses"]
        trainer.val_losses = checkpoint["val_losses"]

        return trainer
