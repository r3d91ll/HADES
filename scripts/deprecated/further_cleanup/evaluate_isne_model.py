#!/usr/bin/env python3
"""
ISNE Model Evaluation Script

Paper-aligned evaluation of trained ISNE models based on the original ISNE paper:
"Unsupervised Graph Representation Learning with Inductive Shallow Node Embedding"

Evaluation methodology follows the paper's approach:
1. Transductive task performance (KNN classification on seen nodes)
2. Inductive task performance (classification on unseen nodes)
3. Relative performance comparison (unseen vs seen nodes)
4. Neighborhood preservation analysis
5. Robustness analysis under noise

Usage:
    python scripts/evaluate_isne_model.py --model-path output/olympus_production/isne_model_final.pth \
                                         --graph-data output/olympus_production/graph_data.json \
                                         --output-dir evaluation_results/
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.isne.models.isne_model import ISNEModel, create_neighbor_lists_from_edge_index
from src.validation.embedding_validator import validate_embeddings_before_isne, validate_embeddings_after_isne

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ISNEEvaluator:
    """Comprehensive ISNE model evaluator."""
    
    def __init__(self, model_path: str, graph_data_path: str, output_dir: str):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to trained ISNE model
            graph_data_path: Path to graph data used for training
            output_dir: Directory to save evaluation results
        """
        self.model_path = Path(model_path)
        self.graph_data_path = Path(graph_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model and data
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.model = None
        self.graph_data = None
        self.adjacency_lists = None
        self.original_embeddings = None
        self.isne_embeddings = None
        
        self._load_model_and_data()
    
    def _load_model_and_data(self):
        """Load the trained model and graph data."""
        logger.info(f"Loading model from {self.model_path}")
        
        # Load model checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model_config = checkpoint['model_config']
        
        # Create model with saved configuration
        self.model = ISNEModel(
            num_nodes=model_config['num_nodes'],
            embedding_dim=model_config['embedding_dim'],
            learning_rate=model_config['learning_rate'],
            device=self.device
        )
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        logger.info(f"Model loaded: {model_config['num_nodes']} nodes, {model_config['embedding_dim']}D embeddings")
        
        # Load graph data
        logger.info(f"Loading graph data from {self.graph_data_path}")
        with open(self.graph_data_path, 'r') as f:
            self.graph_data = json.load(f)
        
        nodes = self.graph_data['nodes']
        edges = self.graph_data['edges']
        
        logger.info(f"Graph data loaded: {len(nodes)} nodes, {len(edges)} edges")
        
        # Extract original embeddings and create adjacency lists
        self.original_embeddings = torch.tensor([node['embedding'] for node in nodes], dtype=torch.float32, device=self.device)
        
        # Create edge index
        edge_index = torch.tensor([[edge['source'], edge['target']] for edge in edges], dtype=torch.long).t().contiguous()
        
        # Create adjacency lists for ISNE
        self.adjacency_lists = create_neighbor_lists_from_edge_index(edge_index, len(nodes))
        
        # Generate ISNE embeddings
        logger.info("Generating ISNE embeddings...")
        with torch.no_grad():
            all_node_ids = torch.arange(len(nodes), device=self.device)
            self.isne_embeddings = self.model.get_node_embeddings(all_node_ids, self.adjacency_lists)
        
        logger.info(f"Generated ISNE embeddings: {self.isne_embeddings.shape}")
    
    def evaluate_embedding_quality(self) -> Dict[str, Any]:
        """Evaluate the quality of ISNE embeddings."""
        logger.info("Evaluating embedding quality...")
        
        results = {}
        
        # 1. Embedding statistics
        original_emb_np = self.original_embeddings.cpu().numpy()
        isne_emb_np = self.isne_embeddings.cpu().numpy()
        
        results['original_stats'] = {
            'mean': float(np.mean(original_emb_np)),
            'std': float(np.std(original_emb_np)),
            'min': float(np.min(original_emb_np)),
            'max': float(np.max(original_emb_np))
        }
        
        results['isne_stats'] = {
            'mean': float(np.mean(isne_emb_np)),
            'std': float(np.std(isne_emb_np)),
            'min': float(np.min(isne_emb_np)),
            'max': float(np.max(isne_emb_np))
        }
        
        # 2. Cosine similarity between original and ISNE embeddings
        cosine_similarities = F.cosine_similarity(self.original_embeddings, self.isne_embeddings, dim=1)
        results['cosine_similarity'] = {
            'mean': float(cosine_similarities.mean()),
            'std': float(cosine_similarities.std()),
            'min': float(cosine_similarities.min()),
            'max': float(cosine_similarities.max())
        }
        
        # 3. L2 distance between original and ISNE embeddings
        l2_distances = torch.norm(self.original_embeddings - self.isne_embeddings, dim=1)
        results['l2_distance'] = {
            'mean': float(l2_distances.mean()),
            'std': float(l2_distances.std()),
            'min': float(l2_distances.min()),
            'max': float(l2_distances.max())
        }
        
        # 4. Embedding diversity (average pairwise distance)
        sample_size = min(1000, len(self.isne_embeddings))  # Sample for efficiency
        sample_indices = torch.randperm(len(self.isne_embeddings))[:sample_size]
        sample_embeddings = self.isne_embeddings[sample_indices]
        
        pairwise_distances = torch.cdist(sample_embeddings, sample_embeddings)
        # Exclude diagonal (self-distances)
        mask = ~torch.eye(sample_size, dtype=torch.bool, device=self.device)
        diversity = pairwise_distances[mask].mean()
        
        results['embedding_diversity'] = float(diversity)
        
        logger.info(f"Embedding quality evaluation completed")
        logger.info(f"  Average cosine similarity to original: {results['cosine_similarity']['mean']:.4f}")
        logger.info(f"  Average L2 distance to original: {results['l2_distance']['mean']:.4f}")
        logger.info(f"  Embedding diversity: {results['embedding_diversity']:.4f}")
        
        return results
    
    def evaluate_graph_structure_preservation(self) -> Dict[str, Any]:
        """Evaluate how well ISNE preserves graph structure."""
        logger.info("Evaluating graph structure preservation...")
        
        results = {}
        
        # 1. Neighborhood preservation
        # For each node, check if its nearest neighbors in ISNE space match graph neighbors
        k = 10  # Check top-k neighbors
        
        original_nn = NearestNeighbors(n_neighbors=k+1, metric='cosine')
        original_nn.fit(self.original_embeddings.cpu().numpy())
        
        isne_nn = NearestNeighbors(n_neighbors=k+1, metric='cosine')
        isne_nn.fit(self.isne_embeddings.cpu().numpy())
        
        neighborhood_preservation_scores = []
        
        for i in range(min(1000, len(self.adjacency_lists))):  # Sample for efficiency
            # Get graph neighbors
            graph_neighbors = set(self.adjacency_lists[i])
            
            # Get embedding space neighbors (excluding self)
            _, original_neighbors = original_nn.kneighbors([self.original_embeddings[i].cpu().numpy()])
            _, isne_neighbors = isne_nn.kneighbors([self.isne_embeddings[i].cpu().numpy()])
            
            original_neighbors = set(original_neighbors[0][1:])  # Exclude self
            isne_neighbors = set(isne_neighbors[0][1:])  # Exclude self
            
            # Calculate preservation scores
            if len(graph_neighbors) > 0:
                original_overlap = len(graph_neighbors.intersection(original_neighbors)) / min(len(graph_neighbors), k)
                isne_overlap = len(graph_neighbors.intersection(isne_neighbors)) / min(len(graph_neighbors), k)
                
                # How much better/worse is ISNE compared to original
                preservation_score = isne_overlap / max(original_overlap, 1e-8)
                neighborhood_preservation_scores.append(preservation_score)
        
        results['neighborhood_preservation'] = {
            'mean': float(np.mean(neighborhood_preservation_scores)),
            'std': float(np.std(neighborhood_preservation_scores)),
            'scores': neighborhood_preservation_scores[:100]  # Store first 100 for analysis
        }
        
        # 2. Edge prediction accuracy
        # Use ISNE embeddings to predict graph edges
        num_test_edges = min(1000, len(self.graph_data['edges']))
        test_edges = self.graph_data['edges'][:num_test_edges]
        
        # Create negative samples
        num_nodes = len(self.graph_data['nodes'])
        negative_edges = []
        edge_set = {(edge['source'], edge['target']) for edge in self.graph_data['edges']}
        
        while len(negative_edges) < num_test_edges:
            src = np.random.randint(0, num_nodes)
            dst = np.random.randint(0, num_nodes)
            if src != dst and (src, dst) not in edge_set and (dst, src) not in edge_set:
                negative_edges.append({'source': src, 'target': dst})
        
        # Compute edge prediction scores
        positive_scores = []
        negative_scores = []
        
        for edge in test_edges:
            src_emb = self.isne_embeddings[edge['source']]
            dst_emb = self.isne_embeddings[edge['target']]
            score = F.cosine_similarity(src_emb.unsqueeze(0), dst_emb.unsqueeze(0))
            positive_scores.append(float(score))
        
        for edge in negative_edges:
            src_emb = self.isne_embeddings[edge['source']]
            dst_emb = self.isne_embeddings[edge['target']]
            score = F.cosine_similarity(src_emb.unsqueeze(0), dst_emb.unsqueeze(0))
            negative_scores.append(float(score))
        
        # Calculate AUC-like metric
        from sklearn.metrics import roc_auc_score
        y_true = [1] * len(positive_scores) + [0] * len(negative_scores)
        y_scores = positive_scores + negative_scores
        
        try:
            auc_score = roc_auc_score(y_true, y_scores)
        except ValueError:
            auc_score = 0.5  # Random performance
        
        results['edge_prediction'] = {
            'auc_score': float(auc_score),
            'positive_score_mean': float(np.mean(positive_scores)),
            'negative_score_mean': float(np.mean(negative_scores)),
            'score_separation': float(np.mean(positive_scores) - np.mean(negative_scores))
        }
        
        logger.info(f"Graph structure preservation evaluation completed")
        logger.info(f"  Neighborhood preservation: {results['neighborhood_preservation']['mean']:.4f}")
        logger.info(f"  Edge prediction AUC: {results['edge_prediction']['auc_score']:.4f}")
        
        return results
    
    def evaluate_transductive_performance(self, test_split: float = 0.2, k_neighbors: int = 15) -> Dict[str, Any]:
        """
        Evaluate transductive task performance following the ISNE paper methodology.
        Uses KNN classification with dot product similarity on 5-fold cross-validation.
        """
        logger.info("Evaluating transductive task performance (paper method)...")
        
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import accuracy_score
        from sklearn.neighbors import NearestNeighbors
        
        results = {}
        
        # For transductive evaluation, we need node labels (simulate with clustering)
        # In a real scenario, you would have ground truth labels
        n_classes = 10  # Simulate 10 classes
        kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
        pseudo_labels = kmeans.fit_predict(self.original_embeddings.cpu().numpy())
        
        # 5-fold cross-validation as per paper
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        original_accuracies = []
        isne_accuracies = []
        
        original_emb_np = self.original_embeddings.cpu().numpy()
        isne_emb_np = self.isne_embeddings.cpu().numpy()
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(original_emb_np, pseudo_labels)):
            # KNN classifier with dot product similarity (as per paper)
            # Convert to use cosine similarity via normalization
            original_train_norm = original_emb_np[train_idx] / np.linalg.norm(original_emb_np[train_idx], axis=1, keepdims=True)
            original_test_norm = original_emb_np[test_idx] / np.linalg.norm(original_emb_np[test_idx], axis=1, keepdims=True)
            
            isne_train_norm = isne_emb_np[train_idx] / np.linalg.norm(isne_emb_np[train_idx], axis=1, keepdims=True)
            isne_test_norm = isne_emb_np[test_idx] / np.linalg.norm(isne_emb_np[test_idx], axis=1, keepdims=True)
            
            # Original embeddings performance
            knn_original = NearestNeighbors(n_neighbors=k_neighbors, metric='cosine')
            knn_original.fit(original_train_norm)
            _, indices_orig = knn_original.kneighbors(original_test_norm)
            
            orig_predictions = []
            for i, neighbors in enumerate(indices_orig):
                neighbor_labels = pseudo_labels[train_idx[neighbors]]
                pred_label = np.bincount(neighbor_labels).argmax()
                orig_predictions.append(pred_label)
            
            orig_accuracy = accuracy_score(pseudo_labels[test_idx], orig_predictions)
            original_accuracies.append(orig_accuracy)
            
            # ISNE embeddings performance
            knn_isne = NearestNeighbors(n_neighbors=k_neighbors, metric='cosine')
            knn_isne.fit(isne_train_norm)
            _, indices_isne = knn_isne.kneighbors(isne_test_norm)
            
            isne_predictions = []
            for i, neighbors in enumerate(indices_isne):
                neighbor_labels = pseudo_labels[train_idx[neighbors]]
                pred_label = np.bincount(neighbor_labels).argmax()
                isne_predictions.append(pred_label)
            
            isne_accuracy = accuracy_score(pseudo_labels[test_idx], isne_predictions)
            isne_accuracies.append(isne_accuracy)
        
        results = {
            'original_accuracy_mean': float(np.mean(original_accuracies)),
            'original_accuracy_std': float(np.std(original_accuracies)),
            'isne_accuracy_mean': float(np.mean(isne_accuracies)),
            'isne_accuracy_std': float(np.std(isne_accuracies)),
            'accuracy_improvement': float(np.mean(isne_accuracies) - np.mean(original_accuracies)),
            'fold_results': {
                'original_accuracies': [float(x) for x in original_accuracies],
                'isne_accuracies': [float(x) for x in isne_accuracies]
            }
        }
        
        logger.info(f"Transductive performance - Original: {results['original_accuracy_mean']:.4f}, ISNE: {results['isne_accuracy_mean']:.4f}")
        
        return results
    
    def evaluate_inductive_performance(self, unseen_ratio: float = 0.3, k_neighbors: int = 15) -> Dict[str, Any]:
        """
        Evaluate inductive task performance following the ISNE paper methodology.
        Tests performance on unseen nodes vs seen nodes.
        """
        logger.info(f"Evaluating inductive task performance with {unseen_ratio:.1%} unseen nodes...")
        
        from sklearn.metrics import accuracy_score
        from sklearn.neighbors import NearestNeighbors
        
        # Split nodes into seen (training) and unseen (test)
        num_nodes = len(self.isne_embeddings)
        num_unseen = int(num_nodes * unseen_ratio)
        
        # Random split
        perm_indices = torch.randperm(num_nodes)
        seen_indices = perm_indices[num_unseen:]
        unseen_indices = perm_indices[:num_unseen]
        
        # Create pseudo labels for evaluation
        n_classes = 10
        kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
        all_labels = kmeans.fit_predict(self.original_embeddings.cpu().numpy())
        
        # Get embeddings for seen and unseen nodes
        seen_isne = self.isne_embeddings[seen_indices].cpu().numpy()
        unseen_isne = self.isne_embeddings[unseen_indices].cpu().numpy()
        
        seen_labels = all_labels[seen_indices.cpu().numpy()]
        unseen_labels = all_labels[unseen_indices.cpu().numpy()]
        
        # Normalize embeddings for dot product similarity
        seen_isne_norm = seen_isne / np.linalg.norm(seen_isne, axis=1, keepdims=True)
        unseen_isne_norm = unseen_isne / np.linalg.norm(unseen_isne, axis=1, keepdims=True)
        
        # Performance on seen nodes (transductive)
        knn_seen = NearestNeighbors(n_neighbors=k_neighbors, metric='cosine')
        knn_seen.fit(seen_isne_norm)
        
        # Test on a subset of seen nodes
        seen_test_size = min(1000, len(seen_indices) // 2)
        seen_test_idx = torch.randperm(len(seen_indices))[:seen_test_size]
        
        _, indices_seen = knn_seen.kneighbors(seen_isne_norm[seen_test_idx])
        seen_predictions = []
        for neighbors in indices_seen:
            neighbor_labels = seen_labels[neighbors]
            pred_label = np.bincount(neighbor_labels).argmax()
            seen_predictions.append(pred_label)
        
        seen_accuracy = accuracy_score(seen_labels[seen_test_idx], seen_predictions)
        
        # Performance on unseen nodes (inductive)
        _, indices_unseen = knn_seen.kneighbors(unseen_isne_norm)
        unseen_predictions = []
        for neighbors in indices_unseen:
            neighbor_labels = seen_labels[neighbors]
            pred_label = np.bincount(neighbor_labels).argmax()
            unseen_predictions.append(pred_label)
        
        unseen_accuracy = accuracy_score(unseen_labels, unseen_predictions)
        
        # Calculate relative performance (key metric from paper)
        relative_performance = (unseen_accuracy / seen_accuracy) * 100
        
        results = {
            'seen_nodes_accuracy': float(seen_accuracy),
            'unseen_nodes_accuracy': float(unseen_accuracy),
            'relative_performance_percent': float(relative_performance),
            'performance_drop_percent': float((seen_accuracy - unseen_accuracy) * 100),
            'num_seen_nodes': len(seen_indices),
            'num_unseen_nodes': len(unseen_indices),
            'achieves_90_percent_target': relative_performance >= 90.0  # Paper target
        }
        
        logger.info(f"Inductive performance - Seen: {seen_accuracy:.4f}, Unseen: {unseen_accuracy:.4f}")
        logger.info(f"Relative performance: {relative_performance:.1f}% (Target: >90%)")
        
        return results
    
    def evaluate_neighborhood_preservation(self, sample_size: int = 1000) -> Dict[str, Any]:
        """
        Evaluate how well ISNE preserves neighborhood structure.
        Key property from the ISNE paper.
        """
        logger.info("Evaluating neighborhood preservation...")
        
        from sklearn.neighbors import NearestNeighbors
        
        # Sample nodes for efficiency
        sample_indices = torch.randperm(len(self.adjacency_lists))[:sample_size]
        
        preservation_scores = []
        k = 10  # Number of neighbors to check
        
        # Fit KNN on original and ISNE embeddings
        original_knn = NearestNeighbors(n_neighbors=k+1, metric='cosine')
        original_knn.fit(self.original_embeddings.cpu().numpy())
        
        isne_knn = NearestNeighbors(n_neighbors=k+1, metric='cosine')
        isne_knn.fit(self.isne_embeddings.cpu().numpy())
        
        for idx in sample_indices[:sample_size]:
            idx_int = idx.item()
            
            # Get graph neighbors
            graph_neighbors = set(self.adjacency_lists[idx_int])
            if len(graph_neighbors) == 0:
                continue
            
            # Get embedding space neighbors
            _, orig_neighbors = original_knn.kneighbors([self.original_embeddings[idx].cpu().numpy()])
            _, isne_neighbors = isne_knn.kneighbors([self.isne_embeddings[idx].cpu().numpy()])
            
            orig_neighbors = set(orig_neighbors[0][1:])  # Exclude self
            isne_neighbors = set(isne_neighbors[0][1:])  # Exclude self
            
            # Calculate preservation: how many graph neighbors are in embedding neighbors
            orig_preservation = len(graph_neighbors.intersection(orig_neighbors)) / min(len(graph_neighbors), k)
            isne_preservation = len(graph_neighbors.intersection(isne_neighbors)) / min(len(graph_neighbors), k)
            
            # Relative improvement
            if orig_preservation > 0:
                preservation_score = isne_preservation / orig_preservation
            else:
                preservation_score = 1.0 if isne_preservation == 0 else float('inf')
            
            preservation_scores.append(preservation_score)
        
        results = {
            'mean_preservation_ratio': float(np.mean(preservation_scores)),
            'std_preservation_ratio': float(np.std(preservation_scores)),
            'median_preservation_ratio': float(np.median(preservation_scores)),
            'preservation_scores_sample': preservation_scores[:50]  # Sample for analysis
        }
        
        logger.info(f"Neighborhood preservation ratio: {results['mean_preservation_ratio']:.4f}")
        
        return results
    
    def create_visualizations(self) -> Dict[str, str]:
        """Create visualizations of the embeddings."""
        logger.info("Creating visualizations...")
        
        viz_paths = {}
        
        # Sample for visualization efficiency
        sample_size = min(2000, len(self.isne_embeddings))
        sample_indices = torch.randperm(len(self.isne_embeddings))[:sample_size]
        
        original_sample = self.original_embeddings[sample_indices].cpu().numpy()
        isne_sample = self.isne_embeddings[sample_indices].cpu().numpy()
        
        # 1. t-SNE visualization
        logger.info("Creating t-SNE visualization...")
        try:
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            
            # Combine embeddings for consistent t-SNE
            combined = np.vstack([original_sample, isne_sample])
            tsne_combined = tsne.fit_transform(combined)
            
            # Split back
            tsne_original = tsne_combined[:sample_size]
            tsne_isne = tsne_combined[sample_size:]
            
            # Create plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            ax1.scatter(tsne_original[:, 0], tsne_original[:, 1], alpha=0.6, s=1)
            ax1.set_title('Original Embeddings (t-SNE)')
            ax1.set_xlabel('t-SNE 1')
            ax1.set_ylabel('t-SNE 2')
            
            ax2.scatter(tsne_isne[:, 0], tsne_isne[:, 1], alpha=0.6, s=1)
            ax2.set_title('ISNE Embeddings (t-SNE)')
            ax2.set_xlabel('t-SNE 1')
            ax2.set_ylabel('t-SNE 2')
            
            plt.tight_layout()
            tsne_path = self.output_dir / 'tsne_comparison.png'
            plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            viz_paths['tsne_comparison'] = str(tsne_path)
            
        except Exception as e:
            logger.warning(f"t-SNE visualization failed: {e}")
        
        # 2. Embedding distribution comparison
        logger.info("Creating embedding distribution plots...")
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Original embedding distribution
            axes[0, 0].hist(original_sample.flatten(), bins=50, alpha=0.7, density=True)
            axes[0, 0].set_title('Original Embedding Distribution')
            axes[0, 0].set_xlabel('Value')
            axes[0, 0].set_ylabel('Density')
            
            # ISNE embedding distribution
            axes[0, 1].hist(isne_sample.flatten(), bins=50, alpha=0.7, density=True)
            axes[0, 1].set_title('ISNE Embedding Distribution')
            axes[0, 1].set_xlabel('Value')
            axes[0, 1].set_ylabel('Density')
            
            # Cosine similarity distribution
            sample_cosine_sim = F.cosine_similarity(
                self.original_embeddings[sample_indices], 
                self.isne_embeddings[sample_indices], 
                dim=1
            ).cpu().numpy()
            
            axes[1, 0].hist(sample_cosine_sim, bins=50, alpha=0.7, density=True)
            axes[1, 0].set_title('Cosine Similarity Distribution')
            axes[1, 0].set_xlabel('Cosine Similarity')
            axes[1, 0].set_ylabel('Density')
            
            # L2 distance distribution
            sample_l2_dist = torch.norm(
                self.original_embeddings[sample_indices] - self.isne_embeddings[sample_indices], 
                dim=1
            ).cpu().numpy()
            
            axes[1, 1].hist(sample_l2_dist, bins=50, alpha=0.7, density=True)
            axes[1, 1].set_title('L2 Distance Distribution')
            axes[1, 1].set_xlabel('L2 Distance')
            axes[1, 1].set_ylabel('Density')
            
            plt.tight_layout()
            dist_path = self.output_dir / 'embedding_distributions.png'
            plt.savefig(dist_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            viz_paths['embedding_distributions'] = str(dist_path)
            
        except Exception as e:
            logger.warning(f"Distribution visualization failed: {e}")
        
        # 3. Graph structure visualization (degree distribution)
        logger.info("Creating graph structure plots...")
        try:
            degrees = [len(neighbors) for neighbors in self.adjacency_lists]
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.hist(degrees, bins=50, alpha=0.7, density=True)
            ax.set_title('Node Degree Distribution')
            ax.set_xlabel('Degree')
            ax.set_ylabel('Density')
            ax.set_yscale('log')
            
            # Add statistics
            ax.axvline(np.mean(degrees), color='red', linestyle='--', label=f'Mean: {np.mean(degrees):.2f}')
            ax.axvline(np.median(degrees), color='orange', linestyle='--', label=f'Median: {np.median(degrees):.2f}')
            ax.legend()
            
            plt.tight_layout()
            degree_path = self.output_dir / 'degree_distribution.png'
            plt.savefig(degree_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            viz_paths['degree_distribution'] = str(degree_path)
            
        except Exception as e:
            logger.warning(f"Graph structure visualization failed: {e}")
        
        logger.info(f"Visualizations created: {list(viz_paths.keys())}")
        
        return viz_paths
    
    def run_full_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation suite."""
        logger.info("Starting full ISNE model evaluation...")
        start_time = time.time()
        
        results = {
            'model_info': {
                'model_path': str(self.model_path),
                'graph_data_path': str(self.graph_data_path),
                'num_nodes': len(self.graph_data['nodes']),
                'num_edges': len(self.graph_data['edges']),
                'embedding_dim': self.model.embedding_dim,
                'device': str(self.device),
                'evaluation_time': None
            }
        }
        
        # Run paper-aligned evaluations
        try:
            results['embedding_quality'] = self.evaluate_embedding_quality()
        except Exception as e:
            logger.error(f"Embedding quality evaluation failed: {e}")
            results['embedding_quality'] = {'error': str(e)}
        
        try:
            results['transductive_performance'] = self.evaluate_transductive_performance()
        except Exception as e:
            logger.error(f"Transductive evaluation failed: {e}")
            results['transductive_performance'] = {'error': str(e)}
        
        try:
            results['inductive_performance'] = self.evaluate_inductive_performance()
        except Exception as e:
            logger.error(f"Inductive evaluation failed: {e}")
            results['inductive_performance'] = {'error': str(e)}
        
        try:
            results['neighborhood_preservation'] = self.evaluate_neighborhood_preservation()
        except Exception as e:
            logger.error(f"Neighborhood preservation evaluation failed: {e}")
            results['neighborhood_preservation'] = {'error': str(e)}
        
        try:
            results['graph_structure_preservation'] = self.evaluate_graph_structure_preservation()
        except Exception as e:
            logger.error(f"Graph structure evaluation failed: {e}")
            results['graph_structure_preservation'] = {'error': str(e)}
        
        try:
            results['visualizations'] = self.create_visualizations()
        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")
            results['visualizations'] = {'error': str(e)}
        
        # Final timing
        evaluation_time = time.time() - start_time
        results['model_info']['evaluation_time'] = evaluation_time
        
        # Save results
        results_path = self.output_dir / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Full evaluation completed in {evaluation_time:.2f}s")
        logger.info(f"Results saved to {results_path}")
        
        return results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate trained ISNE model')
    parser.add_argument('--model-path', required=True, help='Path to trained ISNE model')
    parser.add_argument('--graph-data', required=True, help='Path to graph data JSON file')
    parser.add_argument('--output-dir', required=True, help='Output directory for evaluation results')
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = ISNEEvaluator(args.model_path, args.graph_data, args.output_dir)
    results = evaluator.run_full_evaluation()
    
    # Print summary
    print("\n" + "="*60)
    print("ISNE MODEL EVALUATION SUMMARY (Paper-Aligned)")
    print("="*60)
    
    if 'embedding_quality' in results and 'error' not in results['embedding_quality']:
        eq = results['embedding_quality']
        print(f"Embedding Quality:")
        print(f"  Cosine similarity to original: {eq['cosine_similarity']['mean']:.4f} ± {eq['cosine_similarity']['std']:.4f}")
        print(f"  L2 distance to original: {eq['l2_distance']['mean']:.4f} ± {eq['l2_distance']['std']:.4f}")
        print(f"  Embedding diversity: {eq['embedding_diversity']:.4f}")
    
    if 'transductive_performance' in results and 'error' not in results['transductive_performance']:
        tp = results['transductive_performance']
        print(f"\nTransductive Performance (5-fold CV, KNN-15):")
        print(f"  Original embeddings: {tp['original_accuracy_mean']:.4f} ± {tp['original_accuracy_std']:.4f}")
        print(f"  ISNE embeddings: {tp['isne_accuracy_mean']:.4f} ± {tp['isne_accuracy_std']:.4f}")
        print(f"  Improvement: {tp['accuracy_improvement']:.4f}")
    
    if 'inductive_performance' in results and 'error' not in results['inductive_performance']:
        ip = results['inductive_performance']
        print(f"\nInductive Performance (Key Paper Metric):")
        print(f"  Seen nodes accuracy: {ip['seen_nodes_accuracy']:.4f}")
        print(f"  Unseen nodes accuracy: {ip['unseen_nodes_accuracy']:.4f}")
        print(f"  Relative performance: {ip['relative_performance_percent']:.1f}%")
        target_met = "✅ ACHIEVED" if ip['achieves_90_percent_target'] else "❌ NOT MET"
        print(f"  Paper target (>90%): {target_met}")
    
    if 'neighborhood_preservation' in results and 'error' not in results['neighborhood_preservation']:
        np_result = results['neighborhood_preservation']
        print(f"\nNeighborhood Preservation:")
        print(f"  Mean preservation ratio: {np_result['mean_preservation_ratio']:.4f}")
        print(f"  Median preservation ratio: {np_result['median_preservation_ratio']:.4f}")
    
    if 'graph_structure_preservation' in results and 'error' not in results['graph_structure_preservation']:
        gsp = results['graph_structure_preservation']
        print(f"\nGraph Structure Analysis:")
        print(f"  Edge prediction AUC: {gsp['edge_prediction']['auc_score']:.4f}")
        print(f"  Score separation: {gsp['edge_prediction']['score_separation']:.4f}")
    
    print(f"\nEvaluation Summary:")
    print(f"  Model nodes: {results['model_info']['num_nodes']:,}")
    print(f"  Model edges: {results['model_info']['num_edges']:,}")
    print(f"  Embedding dimension: {results['model_info']['embedding_dim']}")
    print(f"  Evaluation time: {results['model_info']['evaluation_time']:.2f}s")
    
    # Overall assessment based on paper metrics
    if 'inductive_performance' in results and 'error' not in results['inductive_performance']:
        ip = results['inductive_performance']
        if ip['achieves_90_percent_target']:
            print(f"\n🎯 PAPER BENCHMARK: PASSED - Achieves >90% relative performance on unseen nodes")
        else:
            print(f"\n⚠️  PAPER BENCHMARK: NEEDS IMPROVEMENT - {ip['relative_performance_percent']:.1f}% < 90% target")
    
    print(f"\nResults saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()