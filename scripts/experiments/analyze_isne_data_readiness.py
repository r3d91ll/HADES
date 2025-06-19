#!/usr/bin/env python3
"""
Advanced analysis of ArangoDB data readiness for ISNE training.

This script examines the data structures in more detail to determine exactly
what is available and what needs to be created for successful ISNE training.
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from database.arango_client import ArangoClient


def analyze_embedding_data(client: ArangoClient, db_name: str) -> Dict[str, Any]:
    """Analyze the embedding collection in detail."""
    print(f"\n--- EMBEDDING DATA ANALYSIS for {db_name} ---")
    
    try:
        # Query embedding collection for detailed analysis
        query = """
        FOR e IN embeddings 
        LIMIT 5
        RETURN {
            embedding_dim: e.embedding_dim,
            model_name: e.model_name,
            source_type: e.source_type,
            source_collection: e.source_collection,
            vector_length: LENGTH(e.vector),
            vector_sample: SLICE(e.vector, 0, 10),
            metadata_keys: ATTRIBUTES(e.metadata)
        }
        """
        results = list(client.execute_aql(query))
        
        if results:
            print(f"✓ Found {len(results)} embedding samples")
            sample = results[0]
            print(f"  - Model: {sample['model_name']}")
            print(f"  - Embedding dimension: {sample['embedding_dim']}")
            print(f"  - Vector length: {sample['vector_length']}")
            print(f"  - Source types: {set(r['source_type'] for r in results)}")
            print(f"  - Source collections: {set(r['source_collection'] for r in results)}")
            
            # Check vector quality
            vector_sample = sample['vector_sample']
            non_zero_count = sum(1 for v in vector_sample if v != 0)
            print(f"  - Vector sample (first 10): non-zero elements: {non_zero_count}/10")
            
            return {
                "has_embeddings": True,
                "embedding_dim": sample['embedding_dim'],
                "model_name": sample['model_name'],
                "vector_quality": "good" if non_zero_count > 2 else "sparse",
                "total_embeddings": client.get_collection_count("embeddings")
            }
        else:
            return {"has_embeddings": False}
            
    except Exception as e:
        print(f"❌ Error analyzing embeddings: {e}")
        return {"has_embeddings": False, "error": str(e)}


def analyze_node_structure(client: ArangoClient, db_name: str) -> Dict[str, Any]:
    """Analyze the node structure across collections."""
    print(f"\n--- NODE STRUCTURE ANALYSIS for {db_name} ---")
    
    node_collections = ["code_files", "documentation_files", "config_files", "chunks"]
    node_analysis = {}
    
    for collection in node_collections:
        try:
            count = client.get_collection_count(collection)
            if count > 0:
                # Get sample documents to understand structure
                query = f"""
                FOR doc IN {collection}
                LIMIT 3
                RETURN {{
                    _key: doc._key,
                    _id: doc._id,
                    has_content: HAS(doc, "content"),
                    has_embedding_id: HAS(doc, "embedding_id"),
                    content_length: HAS(doc, "content") ? LENGTH(doc.content) : 0,
                    metadata_keys: ATTRIBUTES(doc.metadata)
                }}
                """
                results = list(client.execute_aql(query))
                
                node_analysis[collection] = {
                    "count": count,
                    "has_content": all(r["has_content"] for r in results),
                    "has_embedding_refs": any(r["has_embedding_id"] for r in results),
                    "avg_content_length": np.mean([r["content_length"] for r in results]),
                    "sample_ids": [r["_id"] for r in results]
                }
                
                print(f"✓ {collection}: {count} documents")
                print(f"  - Has content: {node_analysis[collection]['has_content']}")
                print(f"  - Has embedding refs: {node_analysis[collection]['has_embedding_refs']}")
                print(f"  - Avg content length: {node_analysis[collection]['avg_content_length']:.0f}")
            else:
                node_analysis[collection] = {"count": 0}
                print(f"❌ {collection}: empty")
                
        except Exception as e:
            print(f"❌ Error analyzing {collection}: {e}")
            node_analysis[collection] = {"error": str(e)}
    
    return node_analysis


def analyze_edge_potential(client: ArangoClient, db_name: str) -> Dict[str, Any]:
    """Analyze potential for creating edges between nodes."""
    print(f"\n--- EDGE POTENTIAL ANALYSIS for {db_name} ---")
    
    edge_analysis = {
        "existing_edges": {},
        "potential_edges": {},
        "directory_relationships": {},
        "embedding_relationships": {}
    }
    
    # Check existing edge collections
    edge_collections = ["cross_modal_edges", "intra_modal_edges"]
    for collection in edge_collections:
        try:
            count = client.get_collection_count(collection)
            edge_analysis["existing_edges"][collection] = count
            
            if count > 0:
                # Sample edge structure
                query = f"""
                FOR edge IN {collection}
                LIMIT 3
                RETURN {{
                    _from: edge._from,
                    _to: edge._to,
                    edge_type: edge.edge_type,
                    weight: edge.weight,
                    confidence: edge.confidence
                }}
                """
                results = list(client.execute_aql(query))
                edge_analysis["existing_edges"][f"{collection}_sample"] = results
                print(f"✓ {collection}: {count} edges")
                for edge in results:
                    print(f"  - {edge['_from']} -> {edge['_to']} ({edge.get('edge_type', 'unknown')})")
            else:
                print(f"❌ {collection}: empty")
                
        except Exception as e:
            print(f"❌ Error analyzing {collection}: {e}")
    
    # Analyze directory-based relationships
    try:
        query = """
        FOR doc IN (
            UNION(
                (FOR c IN code_files RETURN {_id: c._id, directory: c.directory, type: "code"}),
                (FOR d IN documentation_files RETURN {_id: d._id, directory: d.directory, type: "doc"}),
                (FOR f IN config_files RETURN {_id: f._id, directory: f.directory, type: "config"})
            )
        )
        COLLECT directory = doc.directory INTO groups = doc
        RETURN {
            directory: directory,
            files: groups,
            count: LENGTH(groups),
            types: UNIQUE(groups[*].type)
        }
        """
        results = list(client.execute_aql(query))
        
        potential_edges = 0
        for result in results:
            file_count = result["count"]
            if file_count > 1:
                # Potential edges = n*(n-1)/2 for complete graph
                potential_edges += file_count * (file_count - 1) // 2
        
        edge_analysis["directory_relationships"] = {
            "directories": len(results),
            "potential_directory_edges": potential_edges,
            "sample_directories": results[:3]
        }
        
        print(f"✓ Directory analysis: {len(results)} directories")
        print(f"  - Potential co-location edges: {potential_edges}")
        
    except Exception as e:
        print(f"❌ Error analyzing directory relationships: {e}")
    
    # Analyze embedding-based similarity potential
    try:
        embedding_count = client.get_collection_count("embeddings")
        if embedding_count > 1:
            # Estimate potential similarity edges (top-k per node)
            k = min(10, embedding_count - 1)  # Top-10 similar for each node
            potential_similarity_edges = embedding_count * k
            
            edge_analysis["embedding_relationships"] = {
                "total_embeddings": embedding_count,
                "potential_similarity_edges": potential_similarity_edges,
                "top_k": k
            }
            
            print(f"✓ Embedding similarity potential: {embedding_count} embeddings")
            print(f"  - Potential similarity edges (top-{k}): {potential_similarity_edges}")
        else:
            print(f"❌ Insufficient embeddings for similarity analysis")
            
    except Exception as e:
        print(f"❌ Error analyzing embedding relationships: {e}")
    
    return edge_analysis


def create_isne_training_matrix(client: ArangoClient, db_name: str) -> Dict[str, Any]:
    """Create a preliminary graph structure for ISNE training."""
    print(f"\n--- ISNE TRAINING MATRIX CREATION for {db_name} ---")
    
    matrix_info = {
        "node_mapping": {},
        "adjacency_info": {},
        "feature_matrix_info": {},
        "edge_index_info": {}
    }
    
    try:
        # Step 1: Create node mapping (assign indices to all documents)
        query = """
        LET all_nodes = UNION(
            (FOR c IN code_files RETURN {_id: c._id, type: "code", embedding_id: c.embedding_id}),
            (FOR d IN documentation_files RETURN {_id: d._id, type: "doc", embedding_id: d.embedding_id}),
            (FOR f IN config_files RETURN {_id: f._id, type: "config", embedding_id: f.embedding_id}),
            (FOR ch IN chunks RETURN {_id: ch._id, type: "chunk", embedding_id: ch.embedding_id})
        )
        FOR node IN all_nodes
        SORT node._id
        RETURN node
        """
        nodes = list(client.execute_aql(query))
        
        # Create node mapping
        node_to_idx = {}
        idx_to_node = {}
        for idx, node in enumerate(nodes):
            node_to_idx[node["_id"]] = idx
            idx_to_node[idx] = node["_id"]
        
        matrix_info["node_mapping"] = {
            "total_nodes": len(nodes),
            "node_to_idx_sample": dict(list(node_to_idx.items())[:5]),
            "node_types": {node["type"]: sum(1 for n in nodes if n["type"] == node["type"]) for node in nodes}
        }
        
        print(f"✓ Created node mapping: {len(nodes)} total nodes")
        for node_type, count in matrix_info["node_mapping"]["node_types"].items():
            print(f"  - {node_type}: {count} nodes")
        
        # Step 2: Analyze embedding availability
        embeddings_with_nodes = sum(1 for node in nodes if node.get("embedding_id"))
        matrix_info["feature_matrix_info"] = {
            "nodes_with_embeddings": embeddings_with_nodes,
            "nodes_without_embeddings": len(nodes) - embeddings_with_nodes,
            "embedding_coverage": embeddings_with_nodes / len(nodes) if nodes else 0
        }
        
        print(f"✓ Feature matrix analysis:")
        print(f"  - Nodes with embeddings: {embeddings_with_nodes}/{len(nodes)} ({matrix_info['feature_matrix_info']['embedding_coverage']:.2%})")
        
        # Step 3: Analyze existing edges
        existing_edge_count = 0
        try:
            query = """
            FOR edge IN cross_modal_edges
            RETURN {_from: edge._from, _to: edge._to}
            """
            edges = list(client.execute_aql(query))
            
            valid_edges = []
            for edge in edges:
                from_node = edge["_from"]
                to_node = edge["_to"]
                if from_node in node_to_idx and to_node in node_to_idx:
                    valid_edges.append((node_to_idx[from_node], node_to_idx[to_node]))
            
            existing_edge_count = len(valid_edges)
            matrix_info["edge_index_info"] = {
                "existing_edges": existing_edge_count,
                "valid_edge_sample": valid_edges[:5]
            }
            
            print(f"✓ Edge analysis: {existing_edge_count} valid edges found")
            
        except Exception as e:
            print(f"⚠ No existing edges found: {e}")
            matrix_info["edge_index_info"] = {"existing_edges": 0}
        
        # Step 4: Assess ISNE readiness
        min_nodes_needed = 10
        min_edges_needed = 5
        min_embedding_coverage = 0.5
        
        isne_ready = (
            len(nodes) >= min_nodes_needed and
            existing_edge_count >= min_edges_needed and
            matrix_info["feature_matrix_info"]["embedding_coverage"] >= min_embedding_coverage
        )
        
        matrix_info["isne_readiness"] = {
            "ready": isne_ready,
            "node_requirement": f"{len(nodes)}/{min_nodes_needed}",
            "edge_requirement": f"{existing_edge_count}/{min_edges_needed}",
            "embedding_requirement": f"{matrix_info['feature_matrix_info']['embedding_coverage']:.2%}/{min_embedding_coverage:.0%}",
            "bottlenecks": []
        }
        
        # Identify bottlenecks
        if len(nodes) < min_nodes_needed:
            matrix_info["isne_readiness"]["bottlenecks"].append("Insufficient nodes")
        if existing_edge_count < min_edges_needed:
            matrix_info["isne_readiness"]["bottlenecks"].append("Insufficient edges")
        if matrix_info["feature_matrix_info"]["embedding_coverage"] < min_embedding_coverage:
            matrix_info["isne_readiness"]["bottlenecks"].append("Poor embedding coverage")
        
        print(f"\n🎯 ISNE Training Readiness: {'✅ READY' if isne_ready else '❌ NOT READY'}")
        if not isne_ready:
            print("Bottlenecks:")
            for bottleneck in matrix_info["isne_readiness"]["bottlenecks"]:
                print(f"  - {bottleneck}")
        
    except Exception as e:
        print(f"❌ Error creating training matrix: {e}")
        matrix_info["error"] = str(e)
    
    return matrix_info


def generate_recommendations(db_analysis: Dict[str, Any]) -> List[str]:
    """Generate specific recommendations for improving ISNE readiness."""
    recommendations = []
    
    for db_name, analysis in db_analysis.items():
        if "error" in analysis:
            continue
            
        print(f"\n🔧 RECOMMENDATIONS FOR {db_name}")
        
        # Check ISNE readiness
        if "isne_readiness" in analysis:
            readiness = analysis["isne_readiness"]
            if not readiness["ready"]:
                print("Priority actions needed:")
                
                # Edge creation recommendations
                if "Insufficient edges" in readiness.get("bottlenecks", []):
                    recommendations.append(f"{db_name}: Create cross-modal edges between related documents")
                    recommendations.append(f"{db_name}: Implement directory co-location edge creation")
                    recommendations.append(f"{db_name}: Add semantic similarity edges between embeddings")
                    print("  1. Create cross-modal edges:")
                    print("     - Link code files to their documentation")
                    print("     - Connect config files to relevant code")
                    print("     - Add directory co-location relationships")
                
                # Embedding coverage recommendations
                if "Poor embedding coverage" in readiness.get("bottlenecks", []):
                    recommendations.append(f"{db_name}: Generate embeddings for all nodes")
                    print("  2. Improve embedding coverage:")
                    print("     - Generate embeddings for documents without them")
                    print("     - Ensure all chunks have corresponding embeddings")
                
                # Node recommendations
                if "Insufficient nodes" in readiness.get("bottlenecks", []):
                    recommendations.append(f"{db_name}: Add more documents to increase graph size")
                    print("  3. Increase graph size:")
                    print("     - Ingest more source files")
                    print("     - Create more fine-grained chunks")
            else:
                recommendations.append(f"{db_name}: Ready for ISNE training!")
                print("✅ Database is ready for ISNE training")
        
        # Specific technical recommendations
        print("\nTechnical next steps:")
        print("  - Implement edge creation pipeline in src/isne/graph_population/")
        print("  - Create ISNE training data loader that interfaces with ArangoDB")
        print("  - Set up feature matrix extraction from embeddings collection")
        print("  - Implement edge_index tensor creation from cross_modal_edges")
    
    return recommendations


def main():
    """Main analysis function."""
    databases = [
        "sequential_isne_performance_test",
        "sequential_isne_testdata", 
        "test_embedding_pipeline"
    ]
    
    client = ArangoClient(
        host="127.0.0.1",
        port=8529,
        username="root",
        password="",
        database="hades"
    )
    
    # Overall analysis results
    full_analysis = {}
    
    for db_name in databases:
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE ISNE ANALYSIS: {db_name}")
        print(f"{'='*80}")
        
        if not client.connect_to_database(db_name):
            print(f"❌ Cannot connect to {db_name}")
            continue
        
        db_analysis = {}
        
        # Run all analyses
        db_analysis["embeddings"] = analyze_embedding_data(client, db_name)
        db_analysis["nodes"] = analyze_node_structure(client, db_name)
        db_analysis["edges"] = analyze_edge_potential(client, db_name)
        db_analysis.update(create_isne_training_matrix(client, db_name))
        
        full_analysis[db_name] = db_analysis
    
    # Generate overall recommendations
    recommendations = generate_recommendations(full_analysis)
    
    # Save detailed results
    output_file = Path(__file__).parent / "isne_readiness_analysis.json"
    with open(output_file, 'w') as f:
        json.dump({
            "analysis": full_analysis,
            "recommendations": recommendations,
            "summary": {
                "total_databases_analyzed": len(full_analysis),
                "ready_for_training": sum(1 for db in full_analysis.values() 
                                        if db.get("isne_readiness", {}).get("ready", False)),
                "key_bottlenecks": list(set(
                    bottleneck for db in full_analysis.values()
                    for bottleneck in db.get("isne_readiness", {}).get("bottlenecks", [])
                ))
            }
        }, f, indent=2, default=str)
    
    print(f"\n📊 COMPREHENSIVE ANALYSIS COMPLETE")
    print(f"Detailed results saved to: {output_file}")
    print(f"\n🎯 SUMMARY:")
    print(f"  - Databases analyzed: {len(full_analysis)}")
    
    ready_count = sum(1 for db in full_analysis.values() 
                     if db.get("isne_readiness", {}).get("ready", False))
    print(f"  - Ready for ISNE training: {ready_count}/{len(full_analysis)}")
    
    if ready_count == 0:
        print(f"\n⚠️  ACTION REQUIRED:")
        for rec in recommendations[:5]:  # Show top 5 recommendations
            print(f"    {rec}")


if __name__ == "__main__":
    main()