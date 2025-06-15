"""
ISNE Model Evaluation Stage

Comprehensive evaluation stage for ISNE models following the research paper metrics.
Runs automatically after training completion to assess model quality.
"""

import json
import time
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

from ..config import StageConfig
from ...models.isne_model import ISNEModel

logger = logging.getLogger(__name__)


@dataclass
class ModelEvaluationConfig(StageConfig):
    """Configuration for model evaluation stage."""
    run_evaluation: bool = True  # Whether to run evaluation
    inductive_test_ratio: float = 0.3  # Ratio of nodes to use as "unseen" for inductive test
    k_neighbors: int = 15  # K for KNN classification
    cross_validation_folds: int = 5  # Number of CV folds
    sample_size_for_visualization: int = 1000  # Sample size for t-SNE plots
    save_visualizations: bool = True  # Whether to generate and save plots
    target_relative_performance: float = 0.9  # Target: unseen nodes should achieve >90% performance
    

@dataclass
class ModelEvaluationResult:
    """Result from model evaluation stage."""
    success: bool
    evaluation_metrics: Dict[str, Any]
    stats: Dict[str, Any]
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'evaluation_metrics': self.evaluation_metrics,
            'stats': self.stats,
            'error_message': self.error_message
        }


class ModelEvaluationStage:
    """ISNE model evaluation stage following research paper methodology."""
    
    def __init__(self):
        self.name = "model_evaluation"
    
    def validate_inputs(self, model_path: str, graph_data_path: str, config: ModelEvaluationConfig) -> List[str]:
        """Validate evaluation inputs."""
        errors = []
        
        if not Path(model_path).exists():
            errors.append(f"Model file not found: {model_path}")
        
        if not Path(graph_data_path).exists():
            errors.append(f"Graph data file not found: {graph_data_path}")
        
        if config.inductive_test_ratio <= 0 or config.inductive_test_ratio >= 1:
            errors.append(f"Invalid inductive test ratio: {config.inductive_test_ratio} (must be 0 < ratio < 1)")
        
        if config.k_neighbors <= 0:
            errors.append(f"Invalid k_neighbors: {config.k_neighbors} (must be > 0)")
        
        return errors
    
    def run(self, model_path: str, graph_data_path: str, output_dir: Path, 
            config: ModelEvaluationConfig) -> ModelEvaluationResult:
        """
        Run comprehensive ISNE model evaluation.
        
        Args:
            model_path: Path to trained model file
            graph_data_path: Path to graph_data.json
            output_dir: Output directory for evaluation results
            config: Evaluation configuration
            
        Returns:
            ModelEvaluationResult with evaluation metrics
        """
        
        if not config.run_evaluation:
            logger.info("Model evaluation disabled in config - skipping")
            return ModelEvaluationResult(
                success=True,
                evaluation_metrics={'status': 'skipped'},
                stats={'evaluation_skipped': True}
            )
        
        start_time = time.time()
        logger.info(f"Starting ISNE model evaluation...")
        logger.info(f"Model: {model_path}")
        logger.info(f"Graph data: {graph_data_path}")
        
        try:
            # Validate inputs
            validation_errors = self.validate_inputs(model_path, graph_data_path, config)
            if validation_errors:
                error_msg = f"Validation failed: {', '.join(validation_errors)}"
                logger.error(error_msg)
                return ModelEvaluationResult(
                    success=False,
                    evaluation_metrics={},
                    stats={},
                    error_message=error_msg
                )
            
            # Create evaluation output directory
            eval_output_dir = output_dir / "evaluation_results"
            eval_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Load model and data
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {device}")
            
            # Load model
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
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
            
            logger.info(f"Loaded model: {len(nodes)} nodes, {len(edges)} edges")
            
            # Run evaluation components
            evaluation_metrics = {}
            
            # 1. Basic model statistics
            evaluation_metrics['model_info'] = self._evaluate_model_info(model, nodes, edges, config)
            
            # 2. Embedding quality analysis
            evaluation_metrics['embedding_quality'] = self._evaluate_embedding_quality(
                model, nodes, edges, device, config
            )
            
            # 3. Inductive vs transductive performance (key ISNE paper metric)
            evaluation_metrics['inductive_performance'] = self._evaluate_inductive_performance(
                model, nodes, edges, device, config
            )
            
            # 4. Graph-aware evaluation
            evaluation_metrics['graph_awareness'] = self._evaluate_graph_awareness(
                model, nodes, edges, device, config
            )
            
            # Save evaluation results
            results_file = eval_output_dir / "evaluation_results.json"
            with open(results_file, 'w') as f:
                json.dump(evaluation_metrics, f, indent=2)
            
            # Generate visualizations if requested
            if config.save_visualizations:
                try:
                    self._generate_visualizations(
                        model, nodes, edges, device, eval_output_dir, config
                    )
                except Exception as e:
                    logger.warning(f"Visualization generation failed: {e}")
            
            # Calculate overall performance assessment
            overall_stats = self._calculate_overall_assessment(evaluation_metrics, config)
            
            duration = time.time() - start_time
            logger.info(f"Model evaluation completed in {duration:.2f}s")
            
            # Log key findings
            self._log_key_findings(evaluation_metrics, overall_stats)
            
            return ModelEvaluationResult(
                success=True,
                evaluation_metrics=evaluation_metrics,
                stats={
                    **overall_stats,
                    'evaluation_duration_seconds': duration,
                    'evaluation_output_dir': str(eval_output_dir),
                    'results_file': str(results_file)
                }
            )
            
        except Exception as e:
            error_msg = f"Model evaluation stage failed: {e}"
            error_traceback = traceback.format_exc()
            logger.error(error_msg)
            logger.debug(error_traceback)
            
            return ModelEvaluationResult(
                success=False,
                evaluation_metrics={},
                stats={},
                error_message=error_msg,
                error_traceback=error_traceback
            )
    
    def _evaluate_model_info(self, model: ISNEModel, nodes: List[Dict], edges: List[Dict], 
                           config: ModelEvaluationConfig) -> Dict[str, Any]:
        """Basic model information and statistics."""
        return {
            'num_nodes': len(nodes),
            'num_edges': len(edges),
            'embedding_dimension': model.theta.shape[1],
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'graph_density': (2 * len(edges)) / (len(nodes) * (len(nodes) - 1)) if len(nodes) > 1 else 0,
            'average_degree': (2 * len(edges)) / len(nodes) if len(nodes) > 0 else 0
        }
    
    def _evaluate_embedding_quality(self, model: ISNEModel, nodes: List[Dict], edges: List[Dict],
                                   device: torch.device, config: ModelEvaluationConfig) -> Dict[str, Any]:
        """Evaluate embedding quality metrics."""
        
        # Get original embeddings and generate ISNE embeddings for sample
        original_embeddings = torch.tensor([node['embedding'] for node in nodes], dtype=torch.float32, device=device)
        
        # Create adjacency lists from edges
        adjacency_lists = self._create_adjacency_lists(edges, len(nodes))
        
        # Sample for efficiency
        sample_size = min(config.sample_size_for_visualization, len(nodes))
        sample_indices = torch.randperm(len(nodes))[:sample_size]
        
        with torch.no_grad():
            sample_original = original_embeddings[sample_indices]
            sample_neighbor_lists = [adjacency_lists[i] for i in sample_indices]
            sample_isne = model.get_node_embeddings(sample_indices, sample_neighbor_lists)
        
        # Similarity analysis
        cosine_sim = F.cosine_similarity(sample_original, sample_isne, dim=1)
        
        # Diversity analysis
        pairwise_dist = torch.cdist(sample_isne, sample_isne)
        mask = ~torch.eye(sample_size, dtype=torch.bool, device=device)
        diversity = pairwise_dist[mask].mean()
        
        return {
            'sample_size': sample_size,
            'cosine_similarity': {
                'mean': float(cosine_sim.mean()),
                'std': float(cosine_sim.std()),
                'min': float(cosine_sim.min()),
                'max': float(cosine_sim.max())
            },
            'embedding_diversity': {
                'mean_pairwise_distance': float(diversity),
                'embedding_norm_mean': float(torch.norm(sample_isne, dim=1).mean()),
                'embedding_norm_std': float(torch.norm(sample_isne, dim=1).std())
            }
        }
    
    def _evaluate_inductive_performance(self, model: ISNEModel, nodes: List[Dict], edges: List[Dict],
                                       device: torch.device, config: ModelEvaluationConfig) -> Dict[str, Any]:
        """
        Evaluate inductive vs transductive performance - key ISNE paper metric.
        
        Tests whether unseen nodes achieve >90% relative performance compared to seen nodes.
        """
        
        logger.info("Running inductive vs transductive performance evaluation...")
        
        # Get all embeddings
        original_embeddings = torch.tensor([node['embedding'] for node in nodes], dtype=torch.float32, device=device)
        adjacency_lists = self._create_adjacency_lists(edges, len(nodes))
        
        # Split nodes into seen/unseen
        num_nodes = len(nodes)
        unseen_size = int(num_nodes * config.inductive_test_ratio)
        seen_size = num_nodes - unseen_size
        
        # Random split
        indices = torch.randperm(num_nodes)
        seen_indices = indices[:seen_size]
        unseen_indices = indices[seen_size:]
        
        logger.info(f"Split: {seen_size} seen nodes, {unseen_size} unseen nodes")
        
        with torch.no_grad():
            # Generate ISNE embeddings for both sets
            seen_isne = model.get_node_embeddings(
                seen_indices, 
                [adjacency_lists[i] for i in seen_indices]
            )
            unseen_isne = model.get_node_embeddings(
                unseen_indices, 
                [adjacency_lists[i] for i in unseen_indices]
            )
        
        # Create synthetic classification task based on embedding clusters
        # Use original embeddings to create "ground truth" clusters
        from sklearn.cluster import KMeans
        
        # Cluster original embeddings to create classification labels
        n_clusters = min(10, num_nodes // 50)  # Reasonable number of clusters
        if n_clusters >= 2:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            all_original = original_embeddings.cpu().numpy()
            cluster_labels = kmeans.fit_predict(all_original)
            
            seen_labels = cluster_labels[seen_indices.cpu().numpy()]
            unseen_labels = cluster_labels[unseen_indices.cpu().numpy()]
            
            # Train KNN on seen nodes, test on unseen nodes
            knn = KNeighborsClassifier(n_neighbors=min(config.k_neighbors, len(seen_indices)))
            
            # Transductive performance (seen nodes)
            knn.fit(seen_isne.cpu().numpy(), seen_labels)
            seen_pred = knn.predict(seen_isne.cpu().numpy())
            seen_accuracy = accuracy_score(seen_labels, seen_pred)
            
            # Inductive performance (unseen nodes)
            unseen_pred = knn.predict(unseen_isne.cpu().numpy())
            unseen_accuracy = accuracy_score(unseen_labels, unseen_pred)
            
            # Key metric: relative performance
            relative_performance = (unseen_accuracy / seen_accuracy) * 100 if seen_accuracy > 0 else 0
            achieves_target = relative_performance >= (config.target_relative_performance * 100)
            
            logger.info(f"Inductive evaluation results:")
            logger.info(f"  Seen accuracy: {seen_accuracy:.4f}")
            logger.info(f"  Unseen accuracy: {unseen_accuracy:.4f}")
            logger.info(f"  Relative performance: {relative_performance:.2f}%")
            logger.info(f"  Achieves >90% target: {achieves_target}")
            
            return {
                'seen_accuracy': float(seen_accuracy),
                'unseen_accuracy': float(unseen_accuracy),
                'relative_performance_percent': float(relative_performance),
                'target_threshold_percent': config.target_relative_performance * 100,
                'achieves_90_percent_target': achieves_target,
                'n_clusters': n_clusters,
                'seen_nodes': seen_size,
                'unseen_nodes': unseen_size,
                'evaluation_method': 'kmeans_clustering_knn_classification'
            }
        else:
            logger.warning("Too few nodes for clustering-based evaluation")
            return {
                'status': 'insufficient_data',
                'reason': 'too_few_nodes_for_clustering',
                'n_clusters': n_clusters,
                'min_required': 2
            }
    
    def _evaluate_graph_awareness(self, model: ISNEModel, nodes: List[Dict], edges: List[Dict],
                                 device: torch.device, config: ModelEvaluationConfig) -> Dict[str, Any]:
        """Evaluate how well ISNE embeddings capture graph structure."""
        
        # Create adjacency lists
        adjacency_lists = self._create_adjacency_lists(edges, len(nodes))
        
        # Sample nodes for efficiency
        sample_size = min(1000, len(nodes))
        sample_indices = torch.randperm(len(nodes))[:sample_size]
        
        # Get original embeddings and ISNE embeddings for sample
        original_embeddings = torch.tensor([node['embedding'] for node in nodes], dtype=torch.float32, device=device)
        
        with torch.no_grad():
            sample_original = original_embeddings[sample_indices]
            sample_neighbor_lists = [adjacency_lists[i] for i in sample_indices]
            sample_isne = model.get_node_embeddings(sample_indices, sample_neighbor_lists)
        
        # Calculate graph-aware metrics
        # 1. Neighborhood preservation: do similar nodes remain similar?
        original_sim_matrix = F.cosine_similarity(sample_original.unsqueeze(1), sample_original.unsqueeze(0), dim=2)
        isne_sim_matrix = F.cosine_similarity(sample_isne.unsqueeze(1), sample_isne.unsqueeze(0), dim=2)
        
        # Correlation between similarity matrices
        orig_flat = original_sim_matrix[torch.triu(torch.ones_like(original_sim_matrix), diagonal=1) == 1]
        isne_flat = isne_sim_matrix[torch.triu(torch.ones_like(isne_sim_matrix), diagonal=1) == 1]
        similarity_correlation = F.cosine_similarity(orig_flat.unsqueeze(0), isne_flat.unsqueeze(0)).item()
        
        # 2. Local structure preservation
        # For each node, check if its top-k neighbors remain similar
        k = min(10, sample_size - 1)
        neighborhood_preservation_scores = []
        
        for i in range(sample_size):
            # Top-k neighbors in original space
            orig_sims = original_sim_matrix[i]
            orig_sims[i] = -1  # Exclude self
            orig_top_k = torch.topk(orig_sims, k).indices
            
            # Top-k neighbors in ISNE space
            isne_sims = isne_sim_matrix[i]
            isne_sims[i] = -1  # Exclude self
            isne_top_k = torch.topk(isne_sims, k).indices
            
            # Calculate overlap
            overlap = len(set(orig_top_k.cpu().numpy()) & set(isne_top_k.cpu().numpy()))
            preservation_score = overlap / k
            neighborhood_preservation_scores.append(preservation_score)
        
        avg_neighborhood_preservation = np.mean(neighborhood_preservation_scores)
        
        return {
            'similarity_matrix_correlation': float(similarity_correlation),
            'neighborhood_preservation': {
                'average_score': float(avg_neighborhood_preservation),
                'k_neighbors': k,
                'score_distribution': {
                    'mean': float(np.mean(neighborhood_preservation_scores)),
                    'std': float(np.std(neighborhood_preservation_scores)),
                    'min': float(np.min(neighborhood_preservation_scores)),
                    'max': float(np.max(neighborhood_preservation_scores))
                }
            },
            'sample_size': sample_size
        }
    
    def _create_adjacency_lists(self, edges: List[Dict], num_nodes: int) -> List[List[int]]:
        """Create adjacency lists from edge list."""
        adjacency_lists = [[] for _ in range(num_nodes)]
        
        for edge in edges:
            source = edge['source']
            target = edge['target']
            if 0 <= source < num_nodes and 0 <= target < num_nodes:
                adjacency_lists[source].append(target)
                if source != target:  # Avoid duplicate for undirected edges
                    adjacency_lists[target].append(source)
        
        return adjacency_lists
    
    def _generate_visualizations(self, model: ISNEModel, nodes: List[Dict], edges: List[Dict],
                               device: torch.device, output_dir: Path, config: ModelEvaluationConfig):
        """Generate evaluation visualizations."""
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.manifold import TSNE
            
            # Sample for visualization
            sample_size = min(config.sample_size_for_visualization, len(nodes))
            sample_indices = torch.randperm(len(nodes))[:sample_size]
            
            original_embeddings = torch.tensor([node['embedding'] for node in nodes], dtype=torch.float32, device=device)
            adjacency_lists = self._create_adjacency_lists(edges, len(nodes))
            
            with torch.no_grad():
                sample_original = original_embeddings[sample_indices]
                sample_neighbor_lists = [adjacency_lists[i] for i in sample_indices]
                sample_isne = model.get_node_embeddings(sample_indices, sample_neighbor_lists)
            
            # t-SNE comparison
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, sample_size-1))
            
            orig_2d = tsne.fit_transform(sample_original.cpu().numpy())
            isne_2d = tsne.fit_transform(sample_isne.cpu().numpy())
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            ax1.scatter(orig_2d[:, 0], orig_2d[:, 1], alpha=0.6, s=20)
            ax1.set_title('Original Embeddings (t-SNE)')
            ax1.set_xlabel('t-SNE 1')
            ax1.set_ylabel('t-SNE 2')
            
            ax2.scatter(isne_2d[:, 0], isne_2d[:, 1], alpha=0.6, s=20)
            ax2.set_title('ISNE Embeddings (t-SNE)')
            ax2.set_xlabel('t-SNE 1')
            ax2.set_ylabel('t-SNE 2')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'tsne_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Embedding distribution comparison
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Norm distributions
            orig_norms = torch.norm(sample_original, dim=1).cpu().numpy()
            isne_norms = torch.norm(sample_isne, dim=1).cpu().numpy()
            
            axes[0, 0].hist(orig_norms, bins=50, alpha=0.7, label='Original', density=True)
            axes[0, 0].hist(isne_norms, bins=50, alpha=0.7, label='ISNE', density=True)
            axes[0, 0].set_title('Embedding Norm Distribution')
            axes[0, 0].set_xlabel('L2 Norm')
            axes[0, 0].set_ylabel('Density')
            axes[0, 0].legend()
            
            # Cosine similarity distribution
            cosine_sims = F.cosine_similarity(sample_original, sample_isne, dim=1).cpu().numpy()
            axes[0, 1].hist(cosine_sims, bins=50, alpha=0.7)
            axes[0, 1].set_title('Original vs ISNE Cosine Similarity')
            axes[0, 1].set_xlabel('Cosine Similarity')
            axes[0, 1].set_ylabel('Count')
            
            # Dimension-wise variance
            orig_var = torch.var(sample_original, dim=0).cpu().numpy()
            isne_var = torch.var(sample_isne, dim=0).cpu().numpy()
            
            dim_indices = np.arange(len(orig_var))
            axes[1, 0].scatter(dim_indices, orig_var, alpha=0.6, label='Original', s=10)
            axes[1, 0].scatter(dim_indices, isne_var, alpha=0.6, label='ISNE', s=10)
            axes[1, 0].set_title('Per-Dimension Variance')
            axes[1, 0].set_xlabel('Dimension')
            axes[1, 0].set_ylabel('Variance')
            axes[1, 0].legend()
            
            # Pairwise distance distribution
            orig_dists = torch.cdist(sample_original, sample_original)
            isne_dists = torch.cdist(sample_isne, sample_isne)
            
            mask = torch.triu(torch.ones_like(orig_dists), diagonal=1) == 1
            orig_dists_flat = orig_dists[mask].cpu().numpy()
            isne_dists_flat = isne_dists[mask].cpu().numpy()
            
            axes[1, 1].hist(orig_dists_flat, bins=50, alpha=0.7, label='Original', density=True)
            axes[1, 1].hist(isne_dists_flat, bins=50, alpha=0.7, label='ISNE', density=True)
            axes[1, 1].set_title('Pairwise Distance Distribution')
            axes[1, 1].set_xlabel('L2 Distance')
            axes[1, 1].set_ylabel('Density')
            axes[1, 1].legend()
            
            plt.tight_layout()
            plt.savefig(output_dir / 'embedding_distributions.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Visualizations saved to {output_dir}")
            
        except ImportError as e:
            logger.warning(f"Visualization dependencies not available: {e}")
        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
    
    def _calculate_overall_assessment(self, evaluation_metrics: Dict[str, Any], 
                                    config: ModelEvaluationConfig) -> Dict[str, Any]:
        """Calculate overall model assessment."""
        
        quality_flags = []
        
        # Check inductive performance
        if 'inductive_performance' in evaluation_metrics:
            inductive = evaluation_metrics['inductive_performance']
            if 'achieves_90_percent_target' in inductive:
                if inductive['achieves_90_percent_target']:
                    quality_flags.append("✅ Achieves >90% inductive performance target")
                else:
                    relative_perf = inductive.get('relative_performance_percent', 0)
                    quality_flags.append(f"⚠️ Inductive performance below target: {relative_perf:.1f}% < 90%")
        
        # Check embedding quality
        if 'embedding_quality' in evaluation_metrics:
            quality = evaluation_metrics['embedding_quality']
            cosine_mean = quality.get('cosine_similarity', {}).get('mean', 0)
            if cosine_mean > 0.8:
                quality_flags.append("✅ High similarity preservation with original embeddings")
            elif cosine_mean > 0.6:
                quality_flags.append("⚠️ Moderate similarity preservation with original embeddings")
            else:
                quality_flags.append("❌ Low similarity preservation with original embeddings")
        
        # Check graph awareness
        if 'graph_awareness' in evaluation_metrics:
            graph_aware = evaluation_metrics['graph_awareness']
            sim_corr = graph_aware.get('similarity_matrix_correlation', 0)
            neighborhood_pres = graph_aware.get('neighborhood_preservation', {}).get('average_score', 0)
            
            if sim_corr > 0.7 and neighborhood_pres > 0.6:
                quality_flags.append("✅ Good graph structure preservation")
            elif sim_corr > 0.5 or neighborhood_pres > 0.4:
                quality_flags.append("⚠️ Moderate graph structure preservation")
            else:
                quality_flags.append("❌ Poor graph structure preservation")
        
        # Overall assessment
        error_count = sum(1 for flag in quality_flags if flag.startswith("❌"))
        warning_count = sum(1 for flag in quality_flags if flag.startswith("⚠️"))
        success_count = sum(1 for flag in quality_flags if flag.startswith("✅"))
        
        if error_count == 0 and warning_count == 0:
            overall_status = "excellent"
        elif error_count == 0:
            overall_status = "good_with_warnings"
        else:
            overall_status = "needs_improvement"
        
        return {
            'overall_status': overall_status,
            'quality_flags': quality_flags,
            'success_count': success_count,
            'warning_count': warning_count,
            'error_count': error_count
        }
    
    def _log_key_findings(self, evaluation_metrics: Dict[str, Any], overall_stats: Dict[str, Any]):
        """Log key evaluation findings."""
        
        logger.info(f"\n{'='*60}")
        logger.info(f"ISNE MODEL EVALUATION RESULTS")
        logger.info(f"{'='*60}")
        
        # Model info
        if 'model_info' in evaluation_metrics:
            info = evaluation_metrics['model_info']
            logger.info(f"Model: {info['num_nodes']:,} nodes, {info['num_edges']:,} edges")
            logger.info(f"Parameters: {info['model_parameters']:,}")
            logger.info(f"Graph density: {info['graph_density']:.6f}")
        
        # Key metric: inductive performance
        if 'inductive_performance' in evaluation_metrics:
            inductive = evaluation_metrics['inductive_performance']
            if 'relative_performance_percent' in inductive:
                logger.info(f"\n🎯 Key ISNE Paper Metric:")
                logger.info(f"  Inductive performance: {inductive['relative_performance_percent']:.2f}%")
                logger.info(f"  Target (>90%): {'✅ ACHIEVED' if inductive.get('achieves_90_percent_target') else '❌ NOT ACHIEVED'}")
        
        # Overall assessment
        logger.info(f"\n📊 Overall Assessment: {overall_stats['overall_status'].upper()}")
        for flag in overall_stats['quality_flags']:
            logger.info(f"   {flag}")
        
        logger.info(f"{'='*60}\n")