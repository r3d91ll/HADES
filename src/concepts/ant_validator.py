"""
Actor-Network Theory (ANT) Validator

This module validates that knowledge graphs follow ANT principles,
ensuring proper heterogeneous network formation.

Key ANT Principles:
- Symmetry between human and non-human actors
- Heterogeneous networks (mixing different types of nodes)
- Translation networks (how knowledge moves between domains)
- No privileged perspectives (all nodes have equal potential agency)
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class ANTValidator:
    """
    Validates knowledge graphs against Actor-Network Theory principles.
    
    Ensures the graph represents a proper heterogeneous network where
    different types of knowledge actors interact meaningfully.
    """
    
    def __init__(self) -> None:
        """Initialize ANT validator with validation criteria."""
        # Minimum thresholds for a healthy ANT network
        self.min_heterogeneity_score = 0.3
        self.min_node_types = 3
        self.min_cross_type_edges = 0.25
        self.max_type_dominance = 0.5
        
        # Expected node types in a knowledge network
        self.expected_node_types = {
            'theory',      # Research papers, academic content
            'practice',    # Code implementations
            'documentation', # Guides, tutorials
            'data',        # Datasets, examples
            'query',       # User queries (temporary nodes)
            'concept',     # Abstract concepts
            'tool',        # Software tools, libraries
            'human',       # Authors, contributors
        }
        
        logger.info("Initialized ANT Validator")
    
    def validate_network(self, graph_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a knowledge network against ANT principles.
        
        Args:
            graph_stats: Statistics about the graph structure
            
        Returns:
            Validation results with scores and recommendations
        """
        validation_results: Dict[str, Any] = {
            'is_valid': True,
            'scores': {},
            'issues': [],
            'recommendations': [],
            'ant_metrics': {}
        }
        
        # 1. Check heterogeneity
        heterogeneity_score = self._calculate_heterogeneity(graph_stats)
        validation_results['scores']['heterogeneity'] = heterogeneity_score
        validation_results['ant_metrics']['heterogeneity_score'] = heterogeneity_score
        
        if heterogeneity_score < self.min_heterogeneity_score:
            validation_results['is_valid'] = False
            validation_results['issues'].append(
                f"Low heterogeneity score: {heterogeneity_score:.2f} < {self.min_heterogeneity_score}"
            )
            validation_results['recommendations'].append(
                "Add more diverse node types (theory, practice, documentation)"
            )
        
        # 2. Check node type diversity
        node_type_diversity = self._check_node_type_diversity(graph_stats)
        validation_results['scores']['node_diversity'] = node_type_diversity['score']
        validation_results['ant_metrics']['node_types'] = node_type_diversity['types']
        
        if node_type_diversity['count'] < self.min_node_types:
            validation_results['is_valid'] = False
            validation_results['issues'].append(
                f"Insufficient node types: {node_type_diversity['count']} < {self.min_node_types}"
            )
            validation_results['recommendations'].append(
                f"Missing node types: {node_type_diversity['missing']}"
            )
        
        # 3. Check cross-type relationships
        cross_type_ratio = self._calculate_cross_type_ratio(graph_stats)
        validation_results['scores']['cross_type_connections'] = cross_type_ratio
        validation_results['ant_metrics']['cross_type_ratio'] = cross_type_ratio
        
        if cross_type_ratio < self.min_cross_type_edges:
            validation_results['is_valid'] = False
            validation_results['issues'].append(
                f"Low cross-type connections: {cross_type_ratio:.2f} < {self.min_cross_type_edges}"
            )
            validation_results['recommendations'].append(
                "Create more connections between different node types"
            )
        
        # 4. Check for type dominance
        dominance_score = self._check_type_dominance(graph_stats)
        validation_results['scores']['balance'] = 1.0 - dominance_score
        validation_results['ant_metrics']['type_dominance'] = dominance_score
        
        if dominance_score > self.max_type_dominance:
            validation_results['issues'].append(
                f"Type dominance too high: {dominance_score:.2f} > {self.max_type_dominance}"
            )
            validation_results['recommendations'].append(
                "Balance node types to avoid single-type dominance"
            )
        
        # 5. Check translation networks
        translation_score = self._check_translation_networks(graph_stats)
        validation_results['scores']['translation_networks'] = translation_score
        validation_results['ant_metrics']['translation_score'] = translation_score
        
        # 6. Calculate overall ANT compliance score
        overall_score = self._calculate_overall_score(validation_results['scores'])
        validation_results['ant_metrics']['overall_score'] = overall_score
        validation_results['is_ant_compliant'] = overall_score > 0.7
        
        return validation_results
    
    def _calculate_heterogeneity(self, graph_stats: Dict[str, Any]) -> float:
        """
        Calculate network heterogeneity score.
        
        High score means good mix of different node types.
        """
        node_types = graph_stats.get('node_types', {})
        if not node_types:
            return 0.0
        
        total_nodes = sum(node_types.values())
        if total_nodes == 0:
            return 0.0
        
        # Calculate entropy of node type distribution
        entropy = 0.0
        for count in node_types.values():
            if count > 0:
                p = count / total_nodes
                entropy -= p * np.log(p)
        
        # Normalize by maximum possible entropy
        max_entropy = np.log(len(node_types)) if len(node_types) > 1 else 1.0
        heterogeneity = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return float(heterogeneity)
    
    def _check_node_type_diversity(self, graph_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Check diversity of node types in the network."""
        node_types = set(graph_stats.get('node_types', {}).keys())
        expected_types = self.expected_node_types
        
        missing_types = expected_types - node_types
        unexpected_types = node_types - expected_types
        
        # Score based on coverage of expected types
        coverage = len(node_types & expected_types) / len(expected_types)
        
        return {
            'count': len(node_types),
            'types': list(node_types),
            'missing': list(missing_types),
            'unexpected': list(unexpected_types),
            'score': coverage
        }
    
    def _calculate_cross_type_ratio(self, graph_stats: Dict[str, Any]) -> float:
        """
        Calculate ratio of edges that connect different node types.
        
        High ratio means good interconnection between different types.
        """
        edge_types = graph_stats.get('edge_type_distribution', {})
        total_edges = sum(edge_types.values())
        
        if total_edges == 0:
            return 0.0
        
        # Count cross-type edges
        cross_type_edges = 0
        for edge_type, count in edge_types.items():
            # Assuming edge type format like "theory_to_practice"
            if '_to_' in edge_type:
                parts = edge_type.split('_to_')
                if len(parts) == 2 and parts[0] != parts[1]:
                    cross_type_edges += count
        
        return float(cross_type_edges / total_edges)
    
    def _check_type_dominance(self, graph_stats: Dict[str, Any]) -> float:
        """
        Check if any single node type dominates the network.
        
        Returns dominance score (0 = balanced, 1 = single type dominates).
        """
        node_types = graph_stats.get('node_types', {})
        total_nodes = sum(node_types.values())
        
        if total_nodes == 0:
            return 1.0
        
        # Find maximum proportion
        max_proportion = max(count / total_nodes for count in node_types.values())
        
        return float(max_proportion)
    
    def _check_translation_networks(self, graph_stats: Dict[str, Any]) -> float:
        """
        Check for translation networks (knowledge moving between domains).
        
        Looks for paths that connect different knowledge domains.
        """
        # Check for theory-practice bridges
        bridge_count = graph_stats.get('theory_practice_bridges', 0)
        total_edges = graph_stats.get('total_edges', 1)
        
        # Check for multi-hop paths between different types
        cross_domain_paths = graph_stats.get('cross_domain_paths', 0)
        
        # Combined score
        bridge_ratio = bridge_count / total_edges if total_edges > 0 else 0
        path_score = min(cross_domain_paths / 100, 1.0)  # Normalize to 0-1
        
        return float((bridge_ratio + path_score) / 2)
    
    def _calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """Calculate overall ANT compliance score."""
        if not scores:
            return 0.0
        
        # Weighted average of different scores
        weights = {
            'heterogeneity': 0.3,
            'node_diversity': 0.2,
            'cross_type_connections': 0.25,
            'balance': 0.15,
            'translation_networks': 0.1
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in scores:
                total_score += scores[metric] * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def generate_ant_report(self, validation_results: Dict[str, Any]) -> str:
        """Generate human-readable ANT compliance report."""
        report = []
        report.append("=== Actor-Network Theory Compliance Report ===")
        report.append("")
        
        # Overall status
        if validation_results['is_ant_compliant']:
            report.append("✅ Network is ANT-compliant")
        else:
            report.append("❌ Network needs improvement for ANT compliance")
        
        report.append(f"Overall Score: {validation_results['ant_metrics']['overall_score']:.2f}/1.0")
        report.append("")
        
        # Detailed metrics
        report.append("Detailed Metrics:")
        metrics = validation_results['ant_metrics']
        report.append(f"- Heterogeneity Score: {metrics.get('heterogeneity_score', 0):.2f}")
        report.append(f"- Node Types: {len(metrics.get('node_types', []))}")
        report.append(f"- Cross-Type Ratio: {metrics.get('cross_type_ratio', 0):.2f}")
        report.append(f"- Type Dominance: {metrics.get('type_dominance', 0):.2f}")
        report.append(f"- Translation Networks: {metrics.get('translation_score', 0):.2f}")
        report.append("")
        
        # Issues
        if validation_results['issues']:
            report.append("Issues Found:")
            for issue in validation_results['issues']:
                report.append(f"- {issue}")
            report.append("")
        
        # Recommendations
        if validation_results['recommendations']:
            report.append("Recommendations:")
            for rec in validation_results['recommendations']:
                report.append(f"- {rec}")
        
        return "\n".join(report)
    
    def suggest_improvements(self, 
                           validation_results: Dict[str, Any],
                           current_graph: Any) -> List[Dict[str, Any]]:
        """
        Suggest specific improvements to make the network more ANT-compliant.
        
        Returns list of suggested actions.
        """
        suggestions = []
        
        # Low heterogeneity - suggest adding different node types
        if validation_results['scores'].get('heterogeneity', 1.0) < self.min_heterogeneity_score:
            suggestions.append({
                'action': 'add_diverse_content',
                'description': 'Add more diverse content types',
                'specific_actions': [
                    'Add research papers if mostly code',
                    'Add implementation examples if mostly theory',
                    'Add documentation to bridge theory and practice'
                ]
            })
        
        # Missing cross-type connections
        if validation_results['scores'].get('cross_type_connections', 1.0) < self.min_cross_type_edges:
            suggestions.append({
                'action': 'create_bridges',
                'description': 'Create connections between different node types',
                'specific_actions': [
                    'Link code implementations to their theoretical papers',
                    'Connect documentation to both theory and practice',
                    'Add query nodes that span multiple domains'
                ]
            })
        
        # Type dominance
        if validation_results['ant_metrics'].get('type_dominance', 0) > self.max_type_dominance:
            suggestions.append({
                'action': 'balance_network',
                'description': 'Balance the network by adding underrepresented types',
                'specific_actions': [
                    'Identify the dominant type and add complementary content',
                    'Create more relationships from minority types',
                    'Consider removing redundant nodes of dominant type'
                ]
            })
        
        return suggestions