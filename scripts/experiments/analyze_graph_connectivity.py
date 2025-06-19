#!/usr/bin/env python3
"""
Analyze Graph Connectivity

Compare graph connectivity metrics between different runs to validate
our hypothesis about graph density affecting inductive performance.
"""

import json
import sys
from pathlib import Path


def analyze_connectivity(graph_data_path: str):
    """Analyze connectivity metrics from graph_data.json."""
    
    print(f"📊 Analyzing graph connectivity: {graph_data_path}")
    
    try:
        # For large files, just read metadata instead of full graph
        if Path(graph_data_path).stat().st_size > 100_000_000:  # >100MB
            print("⚠️  Large graph file detected - reading metadata only")
            
            # Try to read just the metadata section
            with open(graph_data_path, 'r') as f:
                # Read first few lines to get metadata
                content = f.read(10000)  # First 10KB
                
            if '"metadata"' in content:
                # Extract metadata section
                import re
                metadata_match = re.search(r'"metadata":\s*({[^}]+})', content)
                if metadata_match:
                    metadata_str = metadata_match.group(1)
                    metadata = json.loads(metadata_str)
                    
                    print(f"📈 Graph Statistics (from metadata):")
                    print(f"   Nodes: {metadata.get('num_nodes', 'unknown'):,}")
                    print(f"   Edges: {metadata.get('num_edges', 'unknown'):,}")
                    
                    nodes = metadata.get('num_nodes', 0)
                    edges = metadata.get('num_edges', 0)
                    
                    if nodes > 0 and edges > 0:
                        avg_degree = (2 * edges) / nodes
                        density = (2 * edges) / (nodes * (nodes - 1)) if nodes > 1 else 0
                        
                        print(f"   Average degree: {avg_degree:.2f}")
                        print(f"   Graph density: {density:.8f}")
                        print(f"   Embedding dim: {metadata.get('embedding_dimension', 'unknown')}")
                        
                        return {
                            'num_nodes': nodes,
                            'num_edges': edges,
                            'avg_degree': avg_degree,
                            'density': density,
                            'source': 'metadata'
                        }
            
            print("❌ Could not extract metadata from large file")
            return None
        
        else:
            # Small file - read normally
            with open(graph_data_path, 'r') as f:
                data = json.load(f)
            
            nodes = data.get('nodes', [])
            edges = data.get('edges', [])
            
            print(f"📈 Graph Statistics:")
            print(f"   Nodes: {len(nodes):,}")
            print(f"   Edges: {len(edges):,}")
            
            if len(nodes) > 0 and len(edges) > 0:
                avg_degree = (2 * len(edges)) / len(nodes)
                density = (2 * len(edges)) / (len(nodes) * (len(nodes) - 1)) if len(nodes) > 1 else 0
                
                print(f"   Average degree: {avg_degree:.2f}")
                print(f"   Graph density: {density:.8f}")
                
                # Analyze edge weight distribution
                edge_weights = [e.get('weight', 1.0) for e in edges if isinstance(e, dict)]
                if edge_weights:
                    import numpy as np
                    print(f"   Edge weights - mean: {np.mean(edge_weights):.4f}, std: {np.std(edge_weights):.4f}")
                
                return {
                    'num_nodes': len(nodes),
                    'num_edges': len(edges),
                    'avg_degree': avg_degree,
                    'density': density,
                    'edge_weights': edge_weights[:100] if edge_weights else [],
                    'source': 'full_data'
                }
            
    except Exception as e:
        print(f"❌ Error analyzing graph: {e}")
        return None


def compare_connectivity():
    """Compare connectivity between old and new runs."""
    
    print("🔍 Graph Connectivity Comparison")
    print("="*50)
    
    # Paths to graph files
    old_graph = "output/olympus_production/graph_data.json"
    new_graph = "output/connectivity_test/graph_data.json"
    
    results = {}
    
    # Analyze old graph
    print("\n📊 Original Run (high threshold, low edges):")
    if Path(old_graph).exists():
        results['original'] = analyze_connectivity(old_graph)
    else:
        print("❌ Original graph data not found")
        # Use known stats from bootstrap summary
        results['original'] = {
            'num_nodes': 79987,
            'num_edges': 630851,
            'avg_degree': 15.8,
            'density': 0.0002,
            'source': 'summary'
        }
        print("📋 Using stats from bootstrap summary:")
        print(f"   Nodes: {results['original']['num_nodes']:,}")
        print(f"   Edges: {results['original']['num_edges']:,}")
        print(f"   Average degree: {results['original']['avg_degree']:.2f}")
        print(f"   Graph density: {results['original']['density']:.8f}")
    
    # Analyze new graph
    print(f"\n📊 New Run (lower threshold, more edges):")
    if Path(new_graph).exists():
        results['new'] = analyze_connectivity(new_graph)
    else:
        print("❌ New graph data not found - run connectivity test first")
        return
    
    # Compare results
    if results.get('original') and results.get('new'):
        print(f"\n🔄 Comparison:")
        
        orig = results['original']
        new = results['new']
        
        # Node comparison
        node_change = new['num_nodes'] - orig['num_nodes']
        print(f"   Nodes: {orig['num_nodes']:,} → {new['num_nodes']:,} ({node_change:+,})")
        
        # Edge comparison
        edge_change = new['num_edges'] - orig['num_edges']
        edge_pct = (edge_change / orig['num_edges']) * 100 if orig['num_edges'] > 0 else 0
        print(f"   Edges: {orig['num_edges']:,} → {new['num_edges']:,} ({edge_change:+,}, {edge_pct:+.1f}%)")
        
        # Degree comparison
        degree_change = new['avg_degree'] - orig['avg_degree']
        degree_pct = (degree_change / orig['avg_degree']) * 100 if orig['avg_degree'] > 0 else 0
        print(f"   Avg degree: {orig['avg_degree']:.2f} → {new['avg_degree']:.2f} ({degree_change:+.2f}, {degree_pct:+.1f}%)")
        
        # Density comparison
        density_change = new['density'] - orig['density']
        density_pct = (density_change / orig['density']) * 100 if orig['density'] > 0 else 0
        print(f"   Density: {orig['density']:.8f} → {new['density']:.8f} ({density_change:+.8f}, {density_pct:+.1f}%)")
        
        # Assessment
        print(f"\n📈 Connectivity Assessment:")
        if new['avg_degree'] > orig['avg_degree'] * 1.2:
            print("   ✅ Significant improvement in connectivity")
        elif new['avg_degree'] > orig['avg_degree']:
            print("   📈 Moderate improvement in connectivity")
        else:
            print("   ⚠️  No improvement in connectivity")
        
        if new['density'] > orig['density'] * 1.5:
            print("   ✅ Much denser graph structure")
        elif new['density'] > orig['density']:
            print("   📈 Denser graph structure")
        else:
            print("   ⚠️  Graph density not improved")


if __name__ == "__main__":
    compare_connectivity()