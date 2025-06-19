#!/usr/bin/env python3
"""
Quick ISNE Model Quality Check

Provides immediate feedback on trained ISNE model quality without full evaluation.
Useful for quick sanity checks during/after training.

Usage:
    python scripts/quick_isne_check.py --model-path output/olympus_production/isne_model_final.pth
"""

import argparse
import json
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.isne.models.isne_model import ISNEModel, create_neighbor_lists_from_edge_index


def quick_quality_check(model_path: str, graph_data_path: str) -> dict:
    """Perform quick quality checks on ISNE model."""
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['model_config']
    
    model = ISNEModel(
        num_nodes=model_config['num_nodes'],
        embedding_dim=model_config['embedding_dim'],
        learning_rate=model_config['learning_rate'],
        device=device
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load graph data
    with open(graph_data_path, 'r') as f:
        graph_data = json.load(f)
    
    nodes = graph_data['nodes']
    edges = graph_data['edges']
    
    # Get original embeddings
    original_embeddings = torch.tensor([node['embedding'] for node in nodes], dtype=torch.float32, device=device)
    
    # Create adjacency lists
    edge_index = torch.tensor([[edge['source'], edge['target']] for edge in edges], dtype=torch.long).t().contiguous()
    adjacency_lists = create_neighbor_lists_from_edge_index(edge_index, len(nodes))
    
    # Generate ISNE embeddings for sample
    sample_size = min(1000, len(nodes))
    sample_indices = torch.randperm(len(nodes))[:sample_size]
    
    with torch.no_grad():
        sample_original = original_embeddings[sample_indices]
        sample_neighbor_lists = [adjacency_lists[i] for i in sample_indices]
        sample_isne = model.get_node_embeddings(sample_indices, sample_neighbor_lists)
    
    # Quick metrics
    results = {}
    
    # 1. Basic statistics
    results['model_info'] = {
        'num_nodes': len(nodes),
        'num_edges': len(edges),
        'embedding_dim': model_config['embedding_dim'],
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'sample_size': sample_size
    }
    
    # 2. Embedding similarity
    cosine_sim = F.cosine_similarity(sample_original, sample_isne, dim=1)
    results['similarity'] = {
        'cosine_mean': float(cosine_sim.mean()),
        'cosine_std': float(cosine_sim.std()),
        'cosine_min': float(cosine_sim.min()),
        'cosine_max': float(cosine_sim.max())
    }
    
    # 3. Embedding diversity
    pairwise_dist = torch.cdist(sample_isne, sample_isne)
    mask = ~torch.eye(sample_size, dtype=torch.bool, device=device)
    diversity = pairwise_dist[mask].mean()
    
    results['diversity'] = {
        'mean_pairwise_distance': float(diversity),
        'embedding_norm_mean': float(torch.norm(sample_isne, dim=1).mean()),
        'embedding_norm_std': float(torch.norm(sample_isne, dim=1).std())
    }
    
    # 4. Quality indicators
    quality_flags = []
    
    if cosine_sim.mean() < 0.1:
        quality_flags.append("LOW_SIMILARITY: ISNE embeddings very different from original")
    elif cosine_sim.mean() > 0.9:
        quality_flags.append("HIGH_SIMILARITY: ISNE embeddings too similar to original (potential underfitting)")
    
    if diversity < 0.1:
        quality_flags.append("LOW_DIVERSITY: Embeddings may be collapsing")
    
    if torch.norm(sample_isne, dim=1).mean() < 0.01:
        quality_flags.append("LOW_MAGNITUDE: Embedding magnitudes very small")
    elif torch.norm(sample_isne, dim=1).mean() > 100:
        quality_flags.append("HIGH_MAGNITUDE: Embedding magnitudes very large")
    
    # Check for NaN or infinite values
    if torch.isnan(sample_isne).any():
        quality_flags.append("NAN_VALUES: Found NaN values in embeddings")
    if torch.isinf(sample_isne).any():
        quality_flags.append("INF_VALUES: Found infinite values in embeddings")
    
    results['quality_flags'] = quality_flags
    
    # 5. Training info from checkpoint
    if 'training_stats' in checkpoint:
        training_stats = checkpoint['training_stats']
        results['training'] = {
            'epochs_trained': training_stats.get('epochs_trained', 'unknown'),
            'final_loss': training_stats.get('final_loss', 'unknown'),
            'training_time': training_stats.get('total_training_time_seconds', 'unknown')
        }
    
    return results


def print_quality_report(results: dict):
    """Print a human-readable quality report."""
    
    print("🔍 ISNE Model Quick Quality Check")
    print("=" * 50)
    
    # Model info
    info = results['model_info']
    print(f"📊 Model Information:")
    print(f"   Nodes: {info['num_nodes']:,}")
    print(f"   Edges: {info['num_edges']:,}")
    print(f"   Embedding Dimension: {info['embedding_dim']}")
    print(f"   Model Parameters: {info['model_parameters']:,}")
    print(f"   Sample Size: {info['sample_size']:,}")
    print()
    
    # Training info
    if 'training' in results:
        training = results['training']
        print(f"🎯 Training Information:")
        print(f"   Epochs: {training['epochs_trained']}")
        print(f"   Final Loss: {training['final_loss']}")
        if isinstance(training['training_time'], (int, float)):
            print(f"   Training Time: {training['training_time']:.1f}s")
        print()
    
    # Similarity metrics
    sim = results['similarity']
    print(f"🔗 Embedding Similarity (Original vs ISNE):")
    print(f"   Mean Cosine Similarity: {sim['cosine_mean']:.4f}")
    print(f"   Std Dev: {sim['cosine_std']:.4f}")
    print(f"   Range: [{sim['cosine_min']:.4f}, {sim['cosine_max']:.4f}]")
    print()
    
    # Diversity metrics
    div = results['diversity']
    print(f"🌟 Embedding Diversity:")
    print(f"   Mean Pairwise Distance: {div['mean_pairwise_distance']:.4f}")
    print(f"   Mean Embedding Norm: {div['embedding_norm_mean']:.4f}")
    print(f"   Norm Std Dev: {div['embedding_norm_std']:.4f}")
    print()
    
    # Quality assessment
    quality_flags = results['quality_flags']
    if not quality_flags:
        print("✅ Quality Assessment: No major issues detected")
    else:
        print("⚠️  Quality Flags:")
        for flag in quality_flags:
            print(f"   - {flag}")
    print()
    
    # Overall assessment
    sim_score = sim['cosine_mean']
    diversity_score = div['mean_pairwise_distance']
    
    print("📈 Overall Assessment:")
    
    if quality_flags:
        print("   🔴 ISSUES DETECTED - Review quality flags above")
    elif 0.3 <= sim_score <= 0.7 and diversity_score > 0.5:
        print("   🟢 GOOD - Model appears to be working well")
    elif sim_score < 0.3:
        print("   🟡 CONCERNING - Low similarity to original embeddings")
    elif sim_score > 0.7:
        print("   🟡 CONCERNING - Very high similarity (possible underfitting)")
    elif diversity_score < 0.5:
        print("   🟡 CONCERNING - Low embedding diversity")
    else:
        print("   🟡 MODERATE - Some metrics outside ideal range")
    
    print("\n💡 Next Steps:")
    if not quality_flags and 0.3 <= sim_score <= 0.7:
        print("   • Run full evaluation: ./scripts/run_isne_evaluation.sh")
        print("   • Test integration with PathRAG")
    else:
        print("   • Investigate quality issues before proceeding")
        print("   • Consider adjusting training parameters")
        print("   • Run full evaluation for detailed analysis")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Quick ISNE model quality check')
    parser.add_argument('--model-path', required=True, help='Path to trained ISNE model')
    parser.add_argument('--graph-data', help='Path to graph data (auto-detected if not provided)')
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)
    
    # Auto-detect graph data path
    if args.graph_data:
        graph_data_path = Path(args.graph_data)
    else:
        graph_data_path = model_path.parent / 'graph_data.json'
    
    if not graph_data_path.exists():
        print(f"❌ Graph data file not found: {graph_data_path}")
        print("   Please specify --graph-data path or ensure graph_data.json exists in model directory")
        sys.exit(1)
    
    try:
        results = quick_quality_check(str(model_path), str(graph_data_path))
        print_quality_report(results)
        
        # Exit with appropriate code
        if results['quality_flags']:
            sys.exit(1)  # Issues detected
        else:
            sys.exit(0)  # All good
            
    except Exception as e:
        print(f"❌ Quality check failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()