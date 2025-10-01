#!/usr/bin/env python3
"""
Export Graph for GraphSAGE Training
====================================

Exports code graph from ArangoDB to PyTorch Geometric format.

Outputs:
- graph_data.pt: PyG Data object with nodes, edges, features
- node_id_map.json: Mapping from node index → ArangoDB _id
- train_val_split.pt: Train/validation masks for inductive split

Usage:
    poetry run python scripts/export_graph_for_training.py --output models/graph_data.pt
    poetry run python scripts/export_graph_for_training.py --dry-run  # Preview only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.database.arango.memory_client import (
    ArangoMemoryClient,
    resolve_memory_config,
)
from core.gnn.graph_builder import GraphBuilder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export graph for GraphSAGE training")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("models/graph_data.pt"),
        help="Output path for graph data (default: models/graph_data.pt)"
    )
    p.add_argument(
        "--collections",
        nargs="+",
        default=["repo_docs"],
        help="Node collections to include (default: repo_docs)"
    )
    p.add_argument(
        "--edge-types",
        nargs="+",
        default=None,
        help="Edge types to include (default: all)"
    )
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation set ratio (default: 0.2)"
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split (default: 42)"
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview graph structure without saving"
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    logger.info("Initializing ArangoDB client...")
    cfg = resolve_memory_config()
    client = ArangoMemoryClient(cfg)

    logger.info("Building graph from ArangoDB...")
    builder = GraphBuilder(client)

    try:
        data, node_id_map = builder.build_graph(
            include_collections=args.collections,
            edge_types=args.edge_types,
        )
    except Exception as exc:
        logger.error(f"Failed to build graph: {exc}")
        return 1

    # Print graph statistics
    logger.info("=" * 80)
    logger.info("Graph Statistics")
    logger.info("=" * 80)
    logger.info(f"Nodes: {data.x.shape[0]}")
    logger.info(f"Node features: {data.x.shape[1]}-dim (Jina v4)")
    logger.info(f"Edge types: {len(data.edge_index_dict)}")
    for edge_type, edge_index in data.edge_index_dict.items():
        logger.info(f"  - {edge_type}: {edge_index.shape[1]} edges")
    logger.info("=" * 80)

    if args.dry_run:
        logger.info("Dry run complete. Exiting without saving.")
        return 0

    # Create train/val split
    logger.info(f"Creating train/val split (val_ratio={args.val_ratio})...")
    train_mask, val_mask = builder.create_train_val_split(
        num_nodes=data.x.shape[0],
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Save graph data
    logger.info(f"Saving graph data to {args.output}...")
    torch.save({
        "data": data,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "node_id_map": node_id_map,
        "metadata": {
            "collections": args.collections,
            "edge_types": list(data.edge_index_dict.keys()),
            "num_nodes": data.x.shape[0],
            "feature_dim": data.x.shape[1],
            "val_ratio": args.val_ratio,
            "seed": args.seed,
        }
    }, args.output)

    # Save node ID mapping separately (for easy reference)
    node_map_path = args.output.parent / "node_id_map.json"
    logger.info(f"Saving node ID map to {node_map_path}...")
    with open(node_map_path, "w") as f:
        json.dump(node_id_map, f, indent=2)

    logger.info("✓ Graph export complete!")
    logger.info(f"  Graph data: {args.output}")
    logger.info(f"  Node map: {node_map_path}")
    logger.info(f"  Train: {train_mask.sum()} nodes, Val: {val_mask.sum()} nodes")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
