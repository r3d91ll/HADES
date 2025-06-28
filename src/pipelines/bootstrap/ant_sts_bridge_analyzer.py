#!/usr/bin/env python3
"""
ANT/STS Bridge Analyzer for Theory-Practice Semantic Networks

This script analyzes the quality and distribution of theory-practice bridges
created by the supra-weight bootstrap pipeline, validating ANT/STS principles.
"""

import sys
from pathlib import Path
import json
import logging
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.pipelines.bootstrap.supra_weight import SupraWeightBootstrapPipeline
from src.pipelines.bootstrap.supra_weight.storage import ArangoSupraStorage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ANTSTSBridgeAnalyzer:
    """
    Analyzes theory-practice bridges from ANT/STS perspective.
    """
    
    def __init__(self, storage: ArangoSupraStorage):
        """Initialize analyzer with storage connection."""
        self.storage = storage
        self.bridge_stats = defaultdict(int)
        self.domain_connections = defaultdict(lambda: defaultdict(int))
        
    def analyze_bridges(self) -> Dict[str, Any]:
        """
        Analyze all bridges in the graph database.
        
        Returns:
            Comprehensive analysis of theory-practice bridges
        """
        logger.info("Starting ANT/STS bridge analysis...")
        
        # Query all edges with bridge metadata
        query = """
        FOR edge IN @@edges
            FILTER edge.relationships != null
            FOR rel IN edge.relationships
                FILTER rel.metadata.bridge_type != null
                COLLECT bridge_type = rel.metadata.bridge_type,
                        bridge_strength = rel.metadata.bridge_strength,
                        theory_domain = rel.metadata.theory_domain
                WITH COUNT INTO count
                RETURN {
                    bridge_type: bridge_type,
                    bridge_strength: bridge_strength,
                    theory_domain: theory_domain,
                    count: count
                }
        """
        
        cursor = self.storage.db.aql.execute(
            query,
            bind_vars={'@edges': self.storage.edge_collection.name}
        )
        
        # Process results
        bridges_by_type = defaultdict(int)
        bridges_by_strength = defaultdict(int)
        theory_domains = defaultdict(int)
        
        for result in cursor:
            bridge_type = result.get('bridge_type')
            bridge_strength = result.get('bridge_strength')
            theory_domain = result.get('theory_domain')
            count = result.get('count', 0)
            
            if bridge_type:
                bridges_by_type[bridge_type] += count
                
            if bridge_strength:
                bridges_by_strength[bridge_strength] += count
                
            if theory_domain and theory_domain != 'unknown':
                theory_domains[theory_domain] += count
                
        # Analyze actor-network formation
        actor_network_stats = self._analyze_actor_networks()
        
        # Analyze semantic bridge quality
        bridge_quality = self._analyze_bridge_quality()
        
        # Compile results
        analysis = {
            'total_bridges': sum(bridges_by_type.values()),
            'bridge_types': dict(bridges_by_type),
            'bridge_strengths': dict(bridges_by_strength),
            'theory_domains': dict(theory_domains),
            'actor_network_formation': actor_network_stats,
            'bridge_quality': bridge_quality,
            'ant_sts_validation': self._validate_ant_principles(
                bridges_by_type, actor_network_stats
            )
        }
        
        return analysis
        
    def _analyze_actor_networks(self) -> Dict[str, Any]:
        """Analyze actor-network formation patterns."""
        query = """
        LET nodes_with_types = (
            FOR node IN @@nodes
                COLLECT node_type = node.node_type WITH COUNT INTO count
                RETURN {type: node_type, count: count}
        )
        
        LET heterogeneous_connections = (
            FOR edge IN @@edges
                LET from_node = DOCUMENT(edge._from)
                LET to_node = DOCUMENT(edge._to)
                FILTER from_node.node_type != to_node.node_type
                COLLECT from_type = from_node.node_type,
                        to_type = to_node.node_type
                WITH COUNT INTO count
                RETURN {
                    from: from_type,
                    to: to_type,
                    count: count
                }
        )
        
        RETURN {
            node_types: nodes_with_types,
            heterogeneous_connections: heterogeneous_connections
        }
        """
        
        cursor = self.storage.db.aql.execute(
            query,
            bind_vars={
                '@nodes': self.storage.node_collection.name,
                '@edges': self.storage.edge_collection.name
            }
        )
        
        result = next(cursor, {})
        
        # Calculate heterogeneity score
        heterogeneous_count = sum(
            conn['count'] 
            for conn in result.get('heterogeneous_connections', [])
        )
        total_edges = self.storage.edge_collection.count()
        
        heterogeneity_score = (
            heterogeneous_count / total_edges if total_edges > 0 else 0
        )
        
        return {
            'node_types': result.get('node_types', []),
            'heterogeneous_connections': result.get('heterogeneous_connections', []),
            'heterogeneity_score': heterogeneity_score,
            'is_heterogeneous_network': heterogeneity_score > 0.3
        }
        
    def _analyze_bridge_quality(self) -> Dict[str, Any]:
        """Analyze the quality of semantic bridges."""
        query = """
        FOR edge IN @@edges
            FILTER edge.weight > 0
            LET has_bridge = (
                FOR rel IN edge.relationships
                    FILTER rel.metadata.bridge_type != null
                    RETURN 1
            )[0] != null
            COLLECT has_bridge_type = has_bridge
            AGGREGATE
                avg_weight = AVG(edge.weight),
                min_weight = MIN(edge.weight),
                max_weight = MAX(edge.weight),
                count = COUNT()
            RETURN {
                is_bridge: has_bridge_type,
                avg_weight: avg_weight,
                min_weight: min_weight,
                max_weight: max_weight,
                count: count
            }
        """
        
        cursor = self.storage.db.aql.execute(
            query,
            bind_vars={'@edges': self.storage.edge_collection.name}
        )
        
        results = list(cursor)
        
        bridge_edges = next((r for r in results if r['is_bridge']), {})
        non_bridge_edges = next((r for r in results if not r['is_bridge']), {})
        
        # Calculate quality metrics
        quality_score = 0.0
        if bridge_edges.get('avg_weight', 0) > 0:
            # Bridges should have higher average weight
            weight_ratio = (
                bridge_edges.get('avg_weight', 0) / 
                non_bridge_edges.get('avg_weight', 1)
            )
            quality_score = min(weight_ratio, 2.0) / 2.0  # Normalize to [0, 1]
            
        return {
            'bridge_edges': bridge_edges,
            'non_bridge_edges': non_bridge_edges,
            'quality_score': quality_score,
            'bridges_stronger': bridge_edges.get('avg_weight', 0) > 
                              non_bridge_edges.get('avg_weight', 0)
        }
        
    def _validate_ant_principles(self, 
                                bridge_types: Dict[str, int],
                                actor_network_stats: Dict) -> Dict[str, bool]:
        """Validate core ANT/STS principles."""
        total_bridges = sum(bridge_types.values())
        
        validations = {
            # 1. Heterogeneous networks
            'heterogeneous_networks': actor_network_stats.get('is_heterogeneous_network', False),
            
            # 2. Translation processes (theory → practice)
            'translation_processes': bridge_types.get('theory_to_implementation', 0) > 10,
            
            # 3. Intermediaries (documentation bridges)
            'intermediaries_present': bridge_types.get('documentation_bridge', 0) > 5,
            
            # 4. Black-boxing (complex → simple)
            'black_boxing': bridge_types.get('black_box_inscription', 0) > 0,
            
            # 5. Actor-network formation
            'network_formation': bridge_types.get('actor_network_formation', 0) > 0,
            
            # 6. Knowledge inscription
            'knowledge_inscription': bridge_types.get('knowledge_inscription', 0) > 0,
            
            # 7. Sociotechnical assembly
            'sociotechnical_assembly': bridge_types.get('sociotechnical_assembly', 0) > 0,
            
            # 8. Sufficient bridge density
            'sufficient_bridges': total_bridges >= 100
        }
        
        # Overall validation
        validations['ant_principles_validated'] = sum(validations.values()) >= 6
        
        return validations
        
    def generate_report(self, analysis: Dict[str, Any], output_file: Path):
        """Generate detailed analysis report."""
        report = []
        report.append("# ANT/STS Theory-Practice Bridge Analysis Report\n")
        
        # Summary
        report.append("## Summary\n")
        report.append(f"- Total Theory-Practice Bridges: {analysis['total_bridges']}")
        report.append(f"- Actor-Network Heterogeneity Score: {analysis['actor_network_formation']['heterogeneity_score']:.3f}")
        report.append(f"- Bridge Quality Score: {analysis['bridge_quality']['quality_score']:.3f}")
        report.append(f"- ANT Principles Validated: {analysis['ant_sts_validation']['ant_principles_validated']}\n")
        
        # Bridge Types
        report.append("## Bridge Type Distribution\n")
        for bridge_type, count in sorted(analysis['bridge_types'].items(), 
                                       key=lambda x: x[1], reverse=True):
            report.append(f"- {bridge_type}: {count}")
        report.append("")
        
        # Bridge Strengths
        report.append("## Bridge Strength Categories\n")
        for strength, count in analysis['bridge_strengths'].items():
            report.append(f"- {strength}: {count}")
        report.append("")
        
        # Theory Domains
        report.append("## Connected Theory Domains\n")
        for domain, count in sorted(analysis['theory_domains'].items(),
                                  key=lambda x: x[1], reverse=True):
            report.append(f"- {domain}: {count}")
        report.append("")
        
        # Actor-Network Analysis
        report.append("## Actor-Network Formation\n")
        report.append("### Node Type Distribution")
        for node_type in analysis['actor_network_formation']['node_types']:
            report.append(f"- {node_type['type']}: {node_type['count']}")
        report.append("")
        
        report.append("### Heterogeneous Connections")
        for conn in analysis['actor_network_formation']['heterogeneous_connections'][:10]:
            report.append(f"- {conn['from']} → {conn['to']}: {conn['count']}")
        report.append("")
        
        # ANT/STS Validation
        report.append("## ANT/STS Principles Validation\n")
        for principle, validated in analysis['ant_sts_validation'].items():
            if principle != 'ant_principles_validated':
                status = "✓" if validated else "✗"
                report.append(f"- {status} {principle.replace('_', ' ').title()}")
        report.append("")
        
        # Bridge Quality
        report.append("## Bridge Quality Analysis\n")
        bridge_stats = analysis['bridge_quality']['bridge_edges']
        non_bridge_stats = analysis['bridge_quality']['non_bridge_edges']
        
        report.append(f"### Bridge Edges ({bridge_stats.get('count', 0)} total)")
        report.append(f"- Average Weight: {bridge_stats.get('avg_weight', 0):.3f}")
        report.append(f"- Weight Range: [{bridge_stats.get('min_weight', 0):.3f}, {bridge_stats.get('max_weight', 0):.3f}]")
        report.append("")
        
        report.append(f"### Non-Bridge Edges ({non_bridge_stats.get('count', 0)} total)")
        report.append(f"- Average Weight: {non_bridge_stats.get('avg_weight', 0):.3f}")
        report.append(f"- Weight Range: [{non_bridge_stats.get('min_weight', 0):.3f}, {non_bridge_stats.get('max_weight', 0):.3f}]")
        report.append("")
        
        # Recommendations
        report.append("## Recommendations\n")
        
        if analysis['total_bridges'] < 100:
            report.append("- ⚠️ Insufficient theory-practice bridges. Consider adding more research papers or documentation.")
            
        if not analysis['ant_sts_validation']['heterogeneous_networks']:
            report.append("- ⚠️ Low network heterogeneity. Ensure diverse node types (code, docs, papers).")
            
        if analysis['bridge_quality']['quality_score'] < 0.5:
            report.append("- ⚠️ Low bridge quality. Review semantic similarity thresholds.")
            
        if analysis['ant_sts_validation']['ant_principles_validated']:
            report.append("- ✅ ANT/STS principles successfully validated!")
        else:
            report.append("- ❌ ANT/STS principles not fully validated. Review data diversity.")
            
        # Write report
        with open(output_file, 'w') as f:
            f.write('\n'.join(report))
            
        logger.info(f"Report saved to {output_file}")


def main():
    """Run ANT/STS bridge analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze ANT/STS theory-practice bridges")
    parser.add_argument("--database", default="isne_ant_sts", help="Database name")
    parser.add_argument("--output", default="ant_sts_analysis.md", help="Output report file")
    
    args = parser.parse_args()
    
    # Connect to database
    storage = ArangoSupraStorage(database_name=args.database)
    storage.initialize_collections()
    
    # Run analysis
    analyzer = ANTSTSBridgeAnalyzer(storage)
    analysis = analyzer.analyze_bridges()
    
    # Generate report
    output_path = Path(args.output)
    analyzer.generate_report(analysis, output_path)
    
    # Save raw analysis
    json_output = output_path.with_suffix('.json')
    with open(json_output, 'w') as f:
        json.dump(analysis, f, indent=2)
    logger.info(f"Raw analysis saved to {json_output}")
    
    # Print summary
    print(f"\nANT/STS Bridge Analysis Complete!")
    print(f"Total Bridges: {analysis['total_bridges']}")
    print(f"ANT Validated: {analysis['ant_sts_validation']['ant_principles_validated']}")
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()