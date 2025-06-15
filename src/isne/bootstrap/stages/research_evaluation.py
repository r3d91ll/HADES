"""
Research-Specific ISNE Evaluation

Enhanced evaluation metrics specifically designed for academic research RAG systems.
Focuses on research-relevant performance indicators.
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

from .model_evaluation import ModelEvaluationStage, ModelEvaluationResult, ModelEvaluationConfig

logger = logging.getLogger(__name__)


@dataclass
class ResearchEvaluationConfig(ModelEvaluationConfig):
    """Configuration for research-specific evaluation."""
    # Research-specific parameters
    evaluate_cross_domain_connections: bool = True
    evaluate_methodology_transfer: bool = True  
    evaluate_concept_bridging: bool = True
    academic_paper_ratio_threshold: float = 0.3  # Min ratio of academic papers
    
    # Academic paper detection patterns
    academic_indicators: List[str] = None
    
    def __post_init__(self):
        if self.academic_indicators is None:
            self.academic_indicators = [
                "abstract", "introduction", "methodology", "results", "conclusion",
                "references", "bibliography", "doi:", "arxiv:", "et al", "figure",
                "table", "equation", "theorem", "hypothesis"
            ]


class ResearchEvaluationStage(ModelEvaluationStage):
    """Research-specific ISNE model evaluation stage."""
    
    def __init__(self):
        super().__init__()
        self.name = "research_evaluation"
    
    def run(self, model_path: str, graph_data_path: str, output_dir: Path, 
            config: ResearchEvaluationConfig) -> ModelEvaluationResult:
        """
        Run research-specific ISNE model evaluation.
        
        Extends base evaluation with research-focused metrics.
        """
        
        # Run base evaluation first
        base_result = super().run(model_path, graph_data_path, output_dir, config)
        
        if not base_result.success:
            return base_result
        
        # Add research-specific evaluations
        try:
            research_metrics = self._evaluate_research_capabilities(
                model_path, graph_data_path, output_dir, config
            )
            
            # Merge with base evaluation
            combined_metrics = {
                **base_result.evaluation_metrics,
                'research_specific': research_metrics
            }
            
            # Update overall stats
            research_stats = self._calculate_research_assessment(research_metrics, config)
            combined_stats = {
                **base_result.stats,
                **research_stats
            }
            
            return ModelEvaluationResult(
                success=True,
                evaluation_metrics=combined_metrics,
                stats=combined_stats
            )
            
        except Exception as e:
            logger.error(f"Research evaluation failed: {e}")
            # Return base result if research evaluation fails
            return base_result
    
    def _evaluate_research_capabilities(self, model_path: str, graph_data_path: str, 
                                      output_dir: Path, config: ResearchEvaluationConfig) -> Dict[str, Any]:
        """Evaluate research-specific capabilities."""
        
        logger.info("Running research-specific evaluation...")
        
        research_metrics = {}
        
        # Load graph data
        with open(graph_data_path, 'r') as f:
            graph_data = json.load(f)
        
        nodes = graph_data['nodes']
        edges = graph_data['edges']
        
        # 1. Academic content analysis
        research_metrics['content_analysis'] = self._analyze_academic_content(nodes, config)
        
        # 2. Cross-domain connectivity
        if config.evaluate_cross_domain_connections:
            research_metrics['cross_domain_connectivity'] = self._evaluate_cross_domain_connections(
                nodes, edges, config
            )
        
        # 3. Methodology transfer capability
        if config.evaluate_methodology_transfer:
            research_metrics['methodology_transfer'] = self._evaluate_methodology_transfer(
                model_path, nodes, edges, config
            )
        
        # 4. Concept bridging analysis
        if config.evaluate_concept_bridging:
            research_metrics['concept_bridging'] = self._evaluate_concept_bridging(
                nodes, edges, config
            )
        
        return research_metrics
    
    def _analyze_academic_content(self, nodes: List[Dict], config: ResearchEvaluationConfig) -> Dict[str, Any]:
        """Analyze the academic content in the graph."""
        
        academic_indicators = config.academic_indicators
        total_nodes = len(nodes)
        academic_nodes = 0
        technical_nodes = 0
        
        academic_scores = []
        
        for node in nodes[:1000]:  # Sample for efficiency
            content = node.get('content', '') + ' ' + str(node.get('metadata', {}))
            content_lower = content.lower()
            
            # Count academic indicators
            score = sum(1 for indicator in academic_indicators if indicator in content_lower)
            academic_scores.append(score)
            
            if score >= 3:  # Threshold for academic content
                academic_nodes += 1
            else:
                technical_nodes += 1
        
        academic_ratio = academic_nodes / len(nodes[:1000]) if nodes else 0
        
        return {
            'total_sampled_nodes': min(1000, total_nodes),
            'academic_nodes': academic_nodes,
            'technical_nodes': technical_nodes,
            'academic_content_ratio': academic_ratio,
            'meets_academic_threshold': academic_ratio >= config.academic_paper_ratio_threshold,
            'avg_academic_score': np.mean(academic_scores) if academic_scores else 0,
            'academic_score_distribution': {
                'min': float(np.min(academic_scores)) if academic_scores else 0,
                'max': float(np.max(academic_scores)) if academic_scores else 0,
                'std': float(np.std(academic_scores)) if academic_scores else 0
            }
        }
    
    def _evaluate_cross_domain_connections(self, nodes: List[Dict], edges: List[Dict], 
                                         config: ResearchEvaluationConfig) -> Dict[str, Any]:
        """Evaluate connections between different research domains."""
        
        # Simple domain classification based on content
        domains = {
            'machine_learning': ['machine learning', 'neural network', 'deep learning', 'algorithm', 'model'],
            'actor_network_theory': ['actor network', 'ant', 'assemblage', 'network theory', 'latour'],
            'data_science': ['data science', 'analytics', 'visualization', 'statistics', 'data mining'],
            'computer_science': ['computer science', 'programming', 'software', 'system', 'computation']
        }
        
        # Classify nodes by domain
        node_domains = {}
        domain_counts = {domain: 0 for domain in domains}
        
        for i, node in enumerate(nodes[:1000]):  # Sample for efficiency
            content = (node.get('content', '') + ' ' + str(node.get('metadata', {}))).lower()
            
            node_domain = 'general'
            max_score = 0
            
            for domain, keywords in domains.items():
                score = sum(1 for keyword in keywords if keyword in content)
                if score > max_score:
                    max_score = score
                    node_domain = domain
            
            node_domains[i] = node_domain
            domain_counts[node_domain] += 1
        
        # Analyze cross-domain edges
        cross_domain_edges = 0
        same_domain_edges = 0
        
        for edge in edges[:10000]:  # Sample edges for efficiency
            source = edge.get('source', 0)
            target = edge.get('target', 0)
            
            if source in node_domains and target in node_domains:
                if node_domains[source] != node_domains[target]:
                    cross_domain_edges += 1
                else:
                    same_domain_edges += 1
        
        total_analyzed_edges = cross_domain_edges + same_domain_edges
        cross_domain_ratio = cross_domain_edges / total_analyzed_edges if total_analyzed_edges > 0 else 0
        
        return {
            'domain_distribution': domain_counts,
            'cross_domain_edges': cross_domain_edges,
            'same_domain_edges': same_domain_edges,
            'cross_domain_ratio': cross_domain_ratio,
            'enables_interdisciplinary_discovery': cross_domain_ratio > 0.2,  # 20% threshold
            'total_analyzed_edges': total_analyzed_edges
        }
    
    def _evaluate_methodology_transfer(self, model_path: str, nodes: List[Dict], edges: List[Dict],
                                     config: ResearchEvaluationConfig) -> Dict[str, Any]:
        """Evaluate the model's ability to transfer methodologies across papers."""
        
        # Load model for embedding analysis
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        from ...models.isne_model import ISNEModel
        
        model = ISNEModel(
            num_nodes=checkpoint['model_config']['num_nodes'],
            embedding_dim=checkpoint['model_config']['embedding_dim'],
            learning_rate=checkpoint['model_config']['learning_rate'],
            device=device
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Find methodology-related content
        methodology_nodes = []
        for i, node in enumerate(nodes[:1000]):
            content = (node.get('content', '') + ' ' + str(node.get('metadata', {}))).lower()
            
            methodology_keywords = ['method', 'approach', 'technique', 'algorithm', 'procedure', 'framework']
            if any(keyword in content for keyword in methodology_keywords):
                methodology_nodes.append(i)
        
        if len(methodology_nodes) < 10:
            return {
                'status': 'insufficient_methodology_content',
                'methodology_nodes_found': len(methodology_nodes)
            }
        
        # Create adjacency lists for methodology nodes
        adjacency_lists = self._create_adjacency_lists(edges, len(nodes))
        
        # Get embeddings for methodology nodes
        methodology_indices = torch.tensor(methodology_nodes[:100], device=device)  # Limit for efficiency
        methodology_neighbor_lists = [adjacency_lists[i] for i in methodology_indices]
        
        with torch.no_grad():
            methodology_embeddings = model.get_node_embeddings(methodology_indices, methodology_neighbor_lists)
        
        # Analyze methodology clustering
        if len(methodology_embeddings) > 1:
            # Compute pairwise similarities
            methodology_similarities = F.cosine_similarity(
                methodology_embeddings.unsqueeze(1), 
                methodology_embeddings.unsqueeze(0), 
                dim=2
            )
            
            # Remove diagonal (self-similarities)
            mask = ~torch.eye(len(methodology_embeddings), dtype=torch.bool, device=device)
            similarities = methodology_similarities[mask]
            
            methodology_coherence = float(similarities.mean())
            methodology_diversity = float(similarities.std())
        else:
            methodology_coherence = 0.0
            methodology_diversity = 0.0
        
        return {
            'methodology_nodes_analyzed': len(methodology_indices),
            'methodology_coherence': methodology_coherence,
            'methodology_diversity': methodology_diversity,
            'supports_methodology_transfer': methodology_coherence > 0.7 and methodology_diversity > 0.1,
            'transfer_quality': 'high' if methodology_coherence > 0.8 else 'medium' if methodology_coherence > 0.6 else 'low'
        }
    
    def _evaluate_concept_bridging(self, nodes: List[Dict], edges: List[Dict],
                                 config: ResearchEvaluationConfig) -> Dict[str, Any]:
        """Evaluate the model's ability to bridge different concepts."""
        
        # Define concept categories
        concept_categories = {
            'theoretical': ['theory', 'framework', 'concept', 'model', 'paradigm'],
            'empirical': ['data', 'experiment', 'study', 'analysis', 'evidence'],
            'methodological': ['method', 'approach', 'technique', 'procedure', 'algorithm'],
            'technological': ['technology', 'system', 'implementation', 'tool', 'platform']
        }
        
        # Classify nodes by concept category
        node_categories = {}
        category_counts = {cat: 0 for cat in concept_categories}
        
        for i, node in enumerate(nodes[:1000]):
            content = (node.get('content', '') + ' ' + str(node.get('metadata', {}))).lower()
            
            node_category = 'general'
            max_score = 0
            
            for category, keywords in concept_categories.items():
                score = sum(1 for keyword in keywords if keyword in content)
                if score > max_score:
                    max_score = score
                    node_category = category
            
            node_categories[i] = node_category
            category_counts[node_category] += 1
        
        # Analyze concept bridging through edges
        bridging_patterns = {}
        for cat1 in concept_categories:
            for cat2 in concept_categories:
                if cat1 != cat2:
                    bridging_patterns[f"{cat1}_to_{cat2}"] = 0
        
        for edge in edges[:10000]:  # Sample for efficiency
            source = edge.get('source', 0)
            target = edge.get('target', 0)
            
            if source in node_categories and target in node_categories:
                source_cat = node_categories[source]
                target_cat = node_categories[target]
                
                if source_cat != target_cat and source_cat != 'general' and target_cat != 'general':
                    pattern = f"{source_cat}_to_{target_cat}"
                    if pattern in bridging_patterns:
                        bridging_patterns[pattern] += 1
        
        total_bridges = sum(bridging_patterns.values())
        
        return {
            'concept_category_distribution': category_counts,
            'bridging_patterns': bridging_patterns,
            'total_concept_bridges': total_bridges,
            'enables_concept_bridging': total_bridges > 100,  # Threshold for meaningful bridging
            'strongest_bridge': max(bridging_patterns.items(), key=lambda x: x[1]) if bridging_patterns else None
        }
    
    def _calculate_research_assessment(self, research_metrics: Dict[str, Any], 
                                     config: ResearchEvaluationConfig) -> Dict[str, Any]:
        """Calculate overall research capability assessment."""
        
        assessment_flags = []
        
        # Content analysis assessment
        if 'content_analysis' in research_metrics:
            content = research_metrics['content_analysis']
            if content.get('meets_academic_threshold', False):
                assessment_flags.append("✅ Sufficient academic content for research RAG")
            else:
                ratio = content.get('academic_content_ratio', 0)
                assessment_flags.append(f"⚠️ Low academic content ratio: {ratio:.2f}")
        
        # Cross-domain connectivity assessment
        if 'cross_domain_connectivity' in research_metrics:
            cross_domain = research_metrics['cross_domain_connectivity']
            if cross_domain.get('enables_interdisciplinary_discovery', False):
                assessment_flags.append("✅ Enables interdisciplinary research discovery")
            else:
                ratio = cross_domain.get('cross_domain_ratio', 0)
                assessment_flags.append(f"⚠️ Limited cross-domain connectivity: {ratio:.2f}")
        
        # Methodology transfer assessment
        if 'methodology_transfer' in research_metrics:
            method = research_metrics['methodology_transfer']
            if method.get('supports_methodology_transfer', False):
                assessment_flags.append("✅ Supports methodology transfer across papers")
            else:
                coherence = method.get('methodology_coherence', 0)
                assessment_flags.append(f"⚠️ Limited methodology transfer capability: {coherence:.2f}")
        
        # Concept bridging assessment
        if 'concept_bridging' in research_metrics:
            bridging = research_metrics['concept_bridging']
            if bridging.get('enables_concept_bridging', False):
                assessment_flags.append("✅ Enables concept bridging across research areas")
            else:
                bridges = bridging.get('total_concept_bridges', 0)
                assessment_flags.append(f"⚠️ Limited concept bridging: {bridges} bridges")
        
        # Overall research capability
        success_count = sum(1 for flag in assessment_flags if flag.startswith("✅"))
        warning_count = sum(1 for flag in assessment_flags if flag.startswith("⚠️"))
        
        if success_count >= 3:
            research_status = "excellent_research_rag"
        elif success_count >= 2:
            research_status = "good_research_rag"
        elif success_count >= 1:
            research_status = "basic_research_rag"
        else:
            research_status = "limited_research_capability"
        
        return {
            'research_quality_flags': assessment_flags,
            'research_capability_status': research_status,
            'research_success_count': success_count,
            'research_warning_count': warning_count,
            'ready_for_research_applications': success_count >= 2
        }