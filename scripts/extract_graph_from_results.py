#!/usr/bin/env python3
"""Extract graph data from bootstrap results JSON."""

import json
import torch
import numpy as np
from pathlib import Path

output_dir = Path("/home/todd/ML-Lab/Olympus/HADES/output/isne_comprehensive_training")

# Find the most recent bootstrap results
result_files = list(output_dir.glob("*_monitored_bootstrap_results.json"))
if not result_files:
    print("No bootstrap results found!")
    exit(1)

# Use the most recent file
latest_result = sorted(result_files)[-1]
print(f"Loading results from: {latest_result}")
print(f"File size: {latest_result.stat().st_size / 1024 / 1024:.1f} MB")

# Load and check structure
with open(latest_result) as f:
    # Read first 10000 chars to check structure
    sample = f.read(10000)
    print("\nFile structure sample:")
    print(sample[:500] + "...")
    
# Try to extract graph data
print("\nChecking for graph data in results...")
with open(latest_result) as f:
    data = json.load(f)
    
if "stages" in data:
    print(f"Found stages: {list(data['stages'].keys())}")
    
    if "graph_construction" in data["stages"]:
        graph_stage = data["stages"]["graph_construction"]
        print(f"\nGraph construction stage keys: {list(graph_stage.keys())}")
        
        if "graph_data" in graph_stage:
            print("Found graph_data!")
            graph_data = graph_stage["graph_data"]
            
            # Convert to PyTorch tensors and save
            if "node_features" in graph_data and "edge_index" in graph_data:
                node_features = torch.tensor(graph_data["node_features"], dtype=torch.float32)
                edge_index = torch.tensor(graph_data["edge_index"], dtype=torch.long)
                
                save_path = output_dir / "extracted_graph_data.pt"
                torch.save({
                    "node_features": node_features,
                    "edge_index": edge_index,
                    "node_metadata": graph_data.get("node_metadata", [])
                }, save_path)
                
                print(f"\nGraph extracted and saved to: {save_path}")
                print(f"Nodes: {node_features.shape[0]}")
                print(f"Features: {node_features.shape[1]}")
                print(f"Edges: {edge_index.shape[1]}")
            else:
                print("Graph data is missing node_features or edge_index!")
        else:
            print("No graph_data found in graph_construction stage")
            print(f"Available keys: {list(graph_stage.keys())[:10]}...")
else:
    print("No stages found in results!")
    print(f"Top-level keys: {list(data.keys())[:10]}...")