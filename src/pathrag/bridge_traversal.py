"""
Bridge-Aware Query Enhancement for PathRAG

This module enhances PathRAG queries by leveraging theory-practice bridges
to find relevant content across different domains (code, documentation, research).
"""

import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from pathlib import Path
import networkx as nx

from src.types.hades.relationships_v2 import TheoryPracticeBridge
from src.types.pathrag.strategy import PathNode, RetrievalPath

logger = logging.getLogger(__name__)


class BridgeTraversal:
    """Traverses theory-practice bridges during PathRAG retrieval."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize bridge traversal with configuration."""
        self.config = config
        self.bridge_weights = config.get('bridge_weights', {
            'implements': 0.9,
            'algorithm_of': 0.85,
            'documented_in': 0.8,
            'based_on': 0.7,
            'references': 0.6,
            'used_by': 0.5,
            'example_in': 0.4
        })
        
        # Track visited bridges to avoid cycles
        self.visited_bridges: Set[str] = set()
    
    def enhance_retrieval_with_bridges(
        self,
        initial_nodes: List[str],
        bridges: List[TheoryPracticeBridge],
        knowledge_graph: nx.Graph,
        query_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enhance retrieval by following theory-practice bridges.
        
        Args:
            initial_nodes: Initial nodes from standard retrieval
            bridges: Available theory-practice bridges
            knowledge_graph: The knowledge graph
            query_context: Context about the query (e.g., is it asking for theory or practice?)
            
        Returns:
            Enhanced retrieval results with bridge-connected nodes
        """
        enhanced_results = {
            'primary_nodes': initial_nodes,
            'bridge_nodes': [],
            'bridge_paths': [],
            'context_type': self._determine_query_context(query_context)
        }
        
        # Determine query intent
        context_type = enhanced_results['context_type']
        
        # For each initial node, find relevant bridges
        for node_id in initial_nodes[:10]:  # Limit to top nodes
            node_path = self._get_node_path(node_id, knowledge_graph)
            
            if not node_path:
                continue
            
            # Find bridges involving this node
            relevant_bridges = self._find_relevant_bridges(
                node_path, bridges, context_type
            )
            
            # Traverse bridges to find connected nodes
            for bridge in relevant_bridges:
                bridge_key = self._get_bridge_key(bridge)
                if bridge_key in self.visited_bridges:
                    continue
                    
                self.visited_bridges.add(bridge_key)
                
                # Get the other end of the bridge
                target_path = self._get_bridge_target(bridge, node_path)
                if target_path:
                    target_node = self._find_node_by_path(target_path, knowledge_graph)
                    
                    if target_node and target_node not in enhanced_results['bridge_nodes']:
                        enhanced_results['bridge_nodes'].append(target_node)
                        
                        # Create bridge path information
                        bridge_path = {
                            'source': node_id,
                            'target': target_node,
                            'bridge_type': bridge.relationship,
                            'confidence': bridge.confidence,
                            'weight': self.bridge_weights.get(bridge.relationship, 0.5),
                            'context': self._get_bridge_context(bridge)
                        }
                        enhanced_results['bridge_paths'].append(bridge_path)
        
        return enhanced_results
    
    def calculate_bridge_aware_scores(
        self,
        node_scores: Dict[str, float],
        bridge_paths: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Adjust node scores based on theory-practice bridges.
        
        Args:
            node_scores: Original node scores
            bridge_paths: Bridge paths discovered
            
        Returns:
            Updated scores incorporating bridge relationships
        """
        updated_scores = node_scores.copy()
        
        # Boost scores for nodes connected by strong bridges
        for bridge_path in bridge_paths:
            source = bridge_path['source']
            target = bridge_path['target']
            bridge_weight = bridge_path['weight']
            
            # Propagate score through bridge
            if source in node_scores:
                boost = node_scores[source] * bridge_weight * 0.5
                updated_scores[target] = updated_scores.get(target, 0.0) + boost
            
            if target in node_scores:
                boost = node_scores[target] * bridge_weight * 0.5
                updated_scores[source] = updated_scores.get(source, 0.0) + boost
        
        return updated_scores
    
    def _determine_query_context(self, query_context: Dict[str, Any]) -> str:
        """Determine if query is seeking theory, practice, or both."""
        query_text = query_context.get('query', '').lower()
        
        # Theory-seeking patterns
        theory_patterns = [
            'how does', 'explain', 'theory', 'concept', 'algorithm',
            'paper', 'research', 'why', 'principle'
        ]
        
        # Practice-seeking patterns
        practice_patterns = [
            'implement', 'code', 'example', 'use', 'apply',
            'function', 'class', 'api', 'syntax'
        ]
        
        theory_score = sum(1 for pattern in theory_patterns if pattern in query_text)
        practice_score = sum(1 for pattern in practice_patterns if pattern in query_text)
        
        if theory_score > practice_score:
            return 'theory'
        elif practice_score > theory_score:
            return 'practice'
        else:
            return 'both'
    
    def _get_node_path(self, node_id: str, graph: nx.Graph) -> Optional[str]:
        """Get filesystem path for a node."""
        node_data = graph.nodes.get(node_id, {})
        return node_data.get('filesystem_path')
    
    def _find_relevant_bridges(
        self,
        node_path: str,
        bridges: List[TheoryPracticeBridge],
        context_type: str
    ) -> List[TheoryPracticeBridge]:
        """Find bridges relevant to a node and query context."""
        relevant = []
        
        for bridge in bridges:
            # Check if bridge involves this node
            if (bridge.source.path == node_path or 
                bridge.target.path == node_path):
                
                # Filter by context type
                if context_type == 'theory' and bridge.source.type == 'code':
                    # Looking for theory, current node is code
                    if bridge.target.type in ['research_paper', 'documentation']:
                        relevant.append(bridge)
                        
                elif context_type == 'practice' and bridge.source.type in ['research_paper', 'documentation']:
                    # Looking for practice, current node is theory
                    if bridge.target.type == 'code':
                        relevant.append(bridge)
                        
                elif context_type == 'both':
                    # Include all bridges
                    relevant.append(bridge)
        
        # Sort by confidence
        relevant.sort(key=lambda b: b.confidence, reverse=True)
        return relevant[:5]  # Limit to top 5 bridges per node
    
    def _get_bridge_key(self, bridge: TheoryPracticeBridge) -> str:
        """Get unique key for a bridge."""
        return f"{bridge.source.path}_{bridge.target.path}_{bridge.relationship}"
    
    def _get_bridge_target(
        self,
        bridge: TheoryPracticeBridge,
        source_path: str
    ) -> Optional[str]:
        """Get the target path from a bridge given the source."""
        if bridge.source.path == source_path:
            return bridge.target.path
        elif bridge.target.path == source_path and bridge.bidirectional:
            return bridge.source.path
        return None
    
    def _find_node_by_path(self, path: str, graph: nx.Graph) -> Optional[str]:
        """Find node ID by filesystem path."""
        for node_id, data in graph.nodes(data=True):
            if data.get('filesystem_path') == path:
                return node_id
        return None
    
    def _get_bridge_context(self, bridge: TheoryPracticeBridge) -> Dict[str, Any]:
        """Extract contextual information from a bridge."""
        context = {
            'source_type': bridge.source.type,
            'target_type': bridge.target.type,
            'relationship': bridge.relationship
        }
        
        # Add specific context based on types
        if bridge.source.section:
            context['source_section'] = bridge.source.section
        if bridge.target.section:
            context['target_section'] = bridge.target.section
        if bridge.source.symbol:
            context['source_symbol'] = bridge.source.symbol
        if bridge.target.symbol:
            context['target_symbol'] = bridge.target.symbol
        if bridge.notes:
            context['notes'] = bridge.notes
            
        return context
    
    def generate_bridge_narrative(
        self,
        bridge_paths: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate human-readable narratives for bridge connections."""
        narratives = []
        
        for bridge_path in bridge_paths:
            source = bridge_path['source']
            target = bridge_path['target']
            bridge_type = bridge_path['bridge_type']
            context = bridge_path['context']
            
            # Generate narrative based on bridge type
            if bridge_type == 'implements':
                narrative = f"{source} implements the algorithm described in {target}"
                if context.get('source_symbol'):
                    narrative += f" (specifically {context['source_symbol']})"
                    
            elif bridge_type == 'documented_in':
                narrative = f"{source} is documented in {target}"
                if context.get('target_section'):
                    narrative += f" (see section: {context['target_section']})"
                    
            elif bridge_type == 'based_on':
                narrative = f"{source} is based on concepts from {target}"
                
            elif bridge_type == 'used_by':
                narrative = f"{source} is used by {target}"
                if context.get('target_symbol'):
                    narrative += f" (in {context['target_symbol']})"
                    
            else:
                narrative = f"{source} {bridge_type} {target}"
            
            narratives.append(narrative)
        
        return narratives
    
    def create_unified_context(
        self,
        primary_nodes: List[Tuple[str, Dict[str, Any]]],
        bridge_nodes: List[Tuple[str, Dict[str, Any]]],
        bridge_narratives: List[str]
    ) -> str:
        """
        Create unified context combining primary and bridge-connected content.
        
        Args:
            primary_nodes: List of (node_id, node_data) tuples
            bridge_nodes: List of (node_id, node_data) tuples from bridges
            bridge_narratives: Human-readable bridge descriptions
            
        Returns:
            Unified context string
        """
        context_parts = []
        
        # Add primary content
        context_parts.append("=== Primary Results ===")
        for node_id, node_data in primary_nodes[:5]:
            content = node_data.get('content', '')
            if content:
                context_parts.append(f"[{node_id}]: {content[:200]}...")
        
        # Add bridge connections
        if bridge_narratives:
            context_parts.append("\n=== Theory-Practice Connections ===")
            for narrative in bridge_narratives[:3]:
                context_parts.append(f"- {narrative}")
        
        # Add bridge content
        if bridge_nodes:
            context_parts.append("\n=== Related Content via Bridges ===")
            for node_id, node_data in bridge_nodes[:3]:
                content = node_data.get('content', '')
                node_type = node_data.get('type', 'unknown')
                if content:
                    context_parts.append(f"[{node_type}:{node_id}]: {content[:200]}...")
        
        return "\n".join(context_parts)