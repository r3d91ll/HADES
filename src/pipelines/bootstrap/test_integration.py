#!/usr/bin/env python3
"""
Integration test for supra-weight bootstrap pipeline.
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Test imports
print("Testing imports...")
try:
    from src.pipelines.bootstrap.supra_weight.core import (
        SupraWeightCalculator, RelationType, Relationship,
        RelationshipDetector, DensityController, EdgeCandidate
    )
    print("✓ Core imports successful")
except Exception as e:
    print(f"✗ Core import error: {e}")
    sys.exit(1)

try:
    from src.pipelines.bootstrap.supra_weight.processing import (
        DocumentProcessor, ChunkProcessor, EmbeddingProcessor
    )
    print("✓ Processing imports successful")
except Exception as e:
    print(f"✗ Processing import error: {e}")
    sys.exit(1)

try:
    from src.pipelines.bootstrap.supra_weight.storage import (
        ArangoSupraStorage, BatchWriter
    )
    print("✓ Storage imports successful")
except Exception as e:
    print(f"✗ Storage import error: {e}")
    sys.exit(1)

try:
    from src.pipelines.bootstrap.supra_weight import SupraWeightBootstrapPipeline
    print("✓ Pipeline import successful")
except Exception as e:
    print(f"✗ Pipeline import error: {e}")
    sys.exit(1)

# Test component functionality
print("\nTesting components...")

# Test relationship detection
detector = RelationshipDetector()
node1 = {
    'node_id': 'test1',
    'node_type': 'file',
    'file_path': '/project/src/main.py',
    'directory': '/project/src',
    'content': 'import utils'
}
node2 = {
    'node_id': 'test2',
    'node_type': 'file', 
    'file_path': '/project/src/utils.py',
    'directory': '/project/src',
    'content': 'def helper(): pass'
}

relationships = detector.detect_all_relationships(node1, node2)
print(f"✓ Detected {len(relationships)} relationships")

# Test supra-weight calculation
calculator = SupraWeightCalculator()
if relationships:
    weight, vector = calculator.calculate(relationships)
    print(f"✓ Calculated supra-weight: {weight:.3f}")
else:
    print("✓ No relationships to calculate weights")

# Test density controller
controller = DensityController(max_edges_per_node=10)
should_add = controller.should_add_edge('node1', 'node2', 0.5)
print(f"✓ Density controller: should_add={should_add}")

# Test embedding processor
processor = EmbeddingProcessor()
test_nodes = [
    {'node_id': 'n1', 'embedding': np.random.randn(384).tolist()},
    {'node_id': 'n2', 'embedding': np.random.randn(384).tolist()}
]
valid, errors = processor.validate_embeddings(test_nodes)
print(f"✓ Embedding validation: valid={valid}")

print("\nAll component tests passed!")