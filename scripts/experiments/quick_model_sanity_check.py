#!/usr/bin/env python3
"""
Quick ISNE Model Sanity Check

Performs basic validation of the trained ISNE model without requiring
the original graph data file. This is useful for immediate post-training
validation.

Usage:
    python scripts/quick_model_sanity_check.py --model-path <path_to_model.pth>
"""

import argparse
import sys
import torch
import torch.nn.functional as F
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.isne.models.isne_model import ISNEModel


def run_sanity_check(model_path: str):
    """Run basic sanity checks on the trained ISNE model."""
    
    print(f"🔍 Running ISNE model sanity check...")
    print(f"📁 Model: {model_path}")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_config = checkpoint['model_config']
    
    print(f"\n📊 Model Configuration:")
    print(f"   Nodes: {model_config['num_nodes']:,}")
    print(f"   Embedding dimension: {model_config['embedding_dim']}")
    print(f"   Learning rate: {model_config['learning_rate']}")
    
    # Initialize model
    model = ISNEModel(
        num_nodes=model_config['num_nodes'],
        embedding_dim=model_config['embedding_dim'],
        learning_rate=model_config['learning_rate'],
        device=device
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test 1: Check parameter initialization
    theta_params = model.theta.data
    print(f"\n🧪 Parameter Analysis:")
    print(f"   Theta shape: {theta_params.shape}")
    print(f"   Theta mean: {theta_params.mean().item():.6f}")
    print(f"   Theta std: {theta_params.std().item():.6f}")
    print(f"   Theta min: {theta_params.min().item():.6f}")
    print(f"   Theta max: {theta_params.max().item():.6f}")
    
    # Test 2: Forward pass with synthetic data
    print(f"\n🔧 Forward Pass Test:")
    num_test_nodes = min(100, model_config['num_nodes'])
    test_node_ids = torch.randperm(model_config['num_nodes'])[:num_test_nodes].to(device)
    
    # Create synthetic neighbor lists (random neighbors)
    test_neighbor_lists = []
    for node_id in test_node_ids:
        # Random 5-15 neighbors for each test node
        num_neighbors = torch.randint(5, 16, (1,)).item()
        neighbors = torch.randperm(model_config['num_nodes'])[:num_neighbors].tolist()
        test_neighbor_lists.append(neighbors)
    
    # Generate embeddings
    with torch.no_grad():
        embeddings = model.get_node_embeddings(test_node_ids, test_neighbor_lists)
    
    print(f"   Generated embeddings shape: {embeddings.shape}")
    print(f"   Embedding mean: {embeddings.mean().item():.6f}")
    print(f"   Embedding std: {embeddings.std().item():.6f}")
    print(f"   Embedding norm mean: {torch.norm(embeddings, dim=1).mean().item():.6f}")
    
    # Test 3: Embedding diversity
    if embeddings.shape[0] > 1:
        pairwise_dist = torch.cdist(embeddings, embeddings)
        mask = ~torch.eye(embeddings.shape[0], dtype=torch.bool, device=device)
        diversity = pairwise_dist[mask].mean()
        print(f"   Mean pairwise distance: {diversity.item():.6f}")
    
    # Test 4: Gradient check (should be None in eval mode)
    print(f"\n🎯 Model State:")
    print(f"   Training mode: {model.training}")
    print(f"   Requires grad: {model.theta.requires_grad}")
    
    # Test 5: Loss computation check
    if 'training_history' in checkpoint:
        losses = checkpoint['training_history']['losses']
        print(f"\n📈 Training History:")
        print(f"   Epochs trained: {len(losses)}")
        print(f"   Initial loss: {losses[0]:.6f}")
        print(f"   Final loss: {losses[-1]:.6f}")
        print(f"   Loss improvement: {losses[0] - losses[-1]:.6f}")
        
        # Check for convergence
        if len(losses) >= 5:
            recent_improvement = losses[-5] - losses[-1]
            print(f"   Recent improvement (last 5 epochs): {recent_improvement:.6f}")
            if abs(recent_improvement) < 0.1:
                print(f"   ⚠️  Model may have converged (small recent improvement)")
    
    # Quality assessment
    print(f"\n✅ Sanity Check Results:")
    
    quality_flags = []
    
    # Check parameter ranges
    if theta_params.std().item() < 0.001:
        quality_flags.append("❌ Parameters have very low variance - possible initialization issue")
    elif theta_params.std().item() > 10.0:
        quality_flags.append("⚠️  Parameters have very high variance - possible training instability")
    else:
        quality_flags.append("✅ Parameter variance looks normal")
    
    # Check embedding ranges
    if embeddings.std().item() < 0.001:
        quality_flags.append("❌ Embeddings have very low variance - possible collapse")
    elif embeddings.std().item() > 10.0:
        quality_flags.append("⚠️  Embeddings have very high variance - possible instability")
    else:
        quality_flags.append("✅ Embedding variance looks normal")
    
    # Check embedding norms
    embedding_norms = torch.norm(embeddings, dim=1)
    if embedding_norms.mean().item() < 0.1:
        quality_flags.append("❌ Embedding norms are very small - possible vanishing")
    elif embedding_norms.mean().item() > 100.0:
        quality_flags.append("⚠️  Embedding norms are very large - possible exploding")
    else:
        quality_flags.append("✅ Embedding norms look reasonable")
    
    # Print results
    for flag in quality_flags:
        print(f"   {flag}")
    
    # Overall assessment
    error_count = sum(1 for flag in quality_flags if flag.startswith("❌"))
    warning_count = sum(1 for flag in quality_flags if flag.startswith("⚠️"))
    
    print(f"\n🎯 Overall Assessment:")
    if error_count == 0 and warning_count == 0:
        print(f"   🎉 Model passes all sanity checks - ready for evaluation!")
        return True
    elif error_count == 0:
        print(f"   ⚠️  Model has {warning_count} warnings but should work for evaluation")
        return True
    else:
        print(f"   ❌ Model has {error_count} errors - may need retraining")
        return False


def main():
    parser = argparse.ArgumentParser(description='Quick ISNE model sanity check')
    parser.add_argument('--model-path', required=True, help='Path to the model file')
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)
    
    try:
        success = run_sanity_check(str(model_path))
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Error during sanity check: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()