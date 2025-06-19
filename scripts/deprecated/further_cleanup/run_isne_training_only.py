#!/usr/bin/env python3
"""
Run ISNE Training Only
This script runs only the ISNE training stage using the already-constructed graph.
"""

import sys
import torch
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run ISNE training using existing graph data."""
    output_dir = Path("/home/todd/ML-Lab/Olympus/HADES/output/isne_comprehensive_training")
    
    # First, let's run just the ISNE training stage using the monitored pipeline
    # which already has the graph data in memory
    import subprocess
    
    logger.info("Running ISNE training stage directly...")
    
    # Create a minimal script to run just stage 5
    training_script = output_dir / "run_stage_5_only.py"
    with open(training_script, 'w') as f:
        f.write("""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.monitored_complete_isne_bootstrap_pipeline import MonitoredISNEBootstrapPipeline

# Initialize pipeline
pipeline = MonitoredISNEBootstrapPipeline(
    Path("/home/todd/ML-Lab/Olympus/HADES/output/isne_comprehensive_training/input_data"),
    Path("/home/todd/ML-Lab/Olympus/HADES/output/isne_comprehensive_training"),
    enable_alerts=True
)

# Load previous results
import json
with open("/home/todd/ML-Lab/Olympus/HADES/output/isne_comprehensive_training/20250614_074323_monitored_bootstrap_results.json") as f:
    pipeline.results = json.load(f)

# Run only stage 5
success = pipeline.stage_5_isne_training()
print(f"ISNE training {'succeeded' if success else 'failed'}")
""")
    
    # Run the training stage
    cmd = ["poetry", "run", "python3", str(training_script)]
    result = subprocess.run(cmd, cwd=str(output_dir.parent.parent))
    
    return result.returncode
    
    # Import ISNE trainer (this will fail if torch_geometric is not installed)
    try:
        from src.isne.training.trainer import ISNETrainer
        from src.isne.models.isne_model import ISNEModel
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Please install torch-geometric: poetry add torch-geometric")
        return 1
    
    # Setup ISNE configuration
    config = {
        "embedding_dim": 384,  # all-MiniLM-L6-v2 dimension
        "isne_dim": 128,
        "num_samples": 10,
        "learning_rate": 0.01,
        "num_epochs": 50,
        "batch_size": 256,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "checkpoint_interval": 10
    }
    
    logger.info(f"ISNE Configuration: {config}")
    logger.info(f"Using device: {config['device']}")
    
    # Initialize model
    model = ISNEModel(
        input_dim=config["embedding_dim"],
        hidden_dim=256,
        output_dim=config["isne_dim"],
        num_layers=3,
        dropout=0.1
    ).to(config["device"])
    
    # Initialize trainer
    trainer = ISNETrainer(
        model=model,
        learning_rate=config["learning_rate"],
        device=config["device"]
    )
    
    # Prepare training data
    node_features = graph_data["node_features"]
    edge_index = graph_data["edge_index"]
    node_metadata = graph_data.get("node_metadata", [])
    
    logger.info(f"Graph statistics:")
    logger.info(f"  Nodes: {node_features.size(0)}")
    logger.info(f"  Edges: {edge_index.size(1)}")
    logger.info(f"  Feature dimension: {node_features.size(1)}")
    
    # Train model
    logger.info("Starting ISNE training...")
    model_path = trainer.train(
        node_features=node_features,
        edge_index=edge_index,
        num_epochs=config["num_epochs"],
        batch_size=config["batch_size"],
        num_neighbors=config["num_samples"],
        checkpoint_dir=str(output_dir / "checkpoints"),
        checkpoint_interval=config["checkpoint_interval"]
    )
    
    # Save final model
    final_model_path = output_dir / "isne_model_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'training_completed': datetime.now().isoformat(),
        'node_count': node_features.size(0),
        'edge_count': edge_index.size(1),
        'model_path': str(model_path)
    }, final_model_path)
    
    logger.info(f"✓ ISNE model saved to: {final_model_path}")
    logger.info("✓ ISNE training completed successfully!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())