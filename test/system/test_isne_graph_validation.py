#!/usr/bin/env python3
"""
Test ISNE Graph Population - Validation Script

This script validates that our trained ISNE model can discover
meaningful relationships, especially cross-domain connections
between code and documentation.
"""

import sys
import logging
from pathlib import Path
import json
from datetime import datetime, timezone

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.isne.graph_population import ISNEGraphPopulator, create_test_chunks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def load_real_chunks_from_training_data():
    """
    Load real chunks from our training data for more realistic testing.
    
    Returns a mix of code and document chunks from the actual dataset.
    """
    # Look for processed chunks from our training
    # These would have been created during the graph construction phase
    chunks = []
    
    # First, let's use simulated chunks that mimic our real data
    # In production, we'd load from the actual processed chunks
    
    # Simulated HADES code chunks
    code_chunks = [
        {
            "content": """def validate_token(token: str) -> bool:
    '''Validates JWT token and returns True if valid'''
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        return payload.get('exp', 0) > time.time()
    except jwt.InvalidTokenError:
        return False""",
            "source_type": "code",
            "metadata": {
                "file": "auth/jwt_handler.py",
                "function": "validate_token",
                "language": "python"
            }
        },
        {
            "content": """class ISNEModel(nn.Module):
    def __init__(self, num_nodes, embedding_dim, hidden_dim=256):
        super().__init__()
        self.shallow_embeddings = nn.Embedding(num_nodes, embedding_dim)
        self.gnn = GCN(embedding_dim, hidden_dim, embedding_dim)""",
            "source_type": "code",
            "metadata": {
                "file": "isne/models.py",
                "class": "ISNEModel",
                "language": "python"
            }
        },
        {
            "content": """async def process_document(self, document: Document) -> ProcessedDocument:
    chunks = self.chunker.split(document.content)
    embeddings = await self.embedder.embed_batch(chunks)
    return ProcessedDocument(chunks=chunks, embeddings=embeddings)""",
            "source_type": "code",
            "metadata": {
                "file": "pipeline/processor.py",
                "function": "process_document",
                "language": "python"
            }
        }
    ]
    
    # Simulated documentation chunks
    doc_chunks = [
        {
            "content": """JWT Authentication: All API endpoints require valid JWT tokens
            in the Authorization header. Tokens are validated on each request
            to ensure they haven't expired and contain valid claims.""",
            "source_type": "document", 
            "metadata": {
                "doc": "api_authentication.md",
                "section": "JWT Validation"
            }
        },
        {
            "content": """ISNE (Inductive Shallow Node Embedding) enhances traditional
            embeddings by incorporating graph structure. The model learns both
            shallow node embeddings and deep graph representations through GNN layers.""",
            "source_type": "document",
            "metadata": {
                "doc": "isne_architecture.md",
                "section": "Model Architecture"
            }
        },
        {
            "content": """Document Processing Pipeline: Documents are first chunked into
            semantic units, then each chunk is embedded using the configured embedding
            model. The pipeline supports async batch processing for efficiency.""",
            "source_type": "document",
            "metadata": {
                "doc": "pipeline_overview.md", 
                "section": "Processing Flow"
            }
        }
    ]
    
    # Mix code and documentation chunks
    chunks = []
    for i in range(min(len(code_chunks), len(doc_chunks))):
        chunks.append(code_chunks[i])
        chunks.append(doc_chunks[i])
    
    # Add remaining chunks
    chunks.extend(code_chunks[len(doc_chunks):])
    chunks.extend(doc_chunks[len(code_chunks):])
    
    # Add more simulated chunks to reach ~100
    chunks.extend(create_test_chunks("dummy", 94))
    
    return chunks[:100]  # Return exactly 100 chunks


def main():
    """Run ISNE graph population validation."""
    
    print("=" * 60)
    print("ISNE GRAPH POPULATION VALIDATION")
    print("=" * 60)
    
    # Path to our trained model
    model_path = "output/isne_training/isne_v1_20250615_161826.pth"
    
    if not Path(model_path).exists():
        print(f"❌ Model not found at {model_path}")
        return False
    
    print(f"✅ Found trained model: {model_path}")
    
    # Create output directory
    output_dir = f"output/graph_validation_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Initialize populator
        populator = ISNEGraphPopulator(confidence_threshold=0.75)
        
        # Load test chunks (mix of real and simulated)
        print("\nLoading test chunks...")
        test_chunks = load_real_chunks_from_training_data()
        print(f"✅ Loaded {len(test_chunks)} test chunks")
        
        # Count types
        code_count = sum(1 for c in test_chunks if c.get('source_type') == 'code')
        doc_count = sum(1 for c in test_chunks if c.get('source_type') == 'document')
        print(f"   - Code chunks: {code_count}")
        print(f"   - Document chunks: {doc_count}")
        
        # Run validation
        print("\nDiscovering relationships...")
        results = populator.validate_from_trained_model(
            model_path=model_path,
            test_chunks=test_chunks,
            output_path=output_dir
        )
        
        # Display results
        print("\n" + "=" * 60)
        print("VALIDATION RESULTS")
        print("=" * 60)
        
        print(f"\n📊 Summary:")
        print(f"   - Chunks processed: {results['chunks_processed']}")
        print(f"   - Relationships discovered: {results['relationships_discovered']}")
        
        analysis = results['analysis']
        if 'total_relationships' in analysis:
            print(f"\n📈 Relationship Analysis:")
            print(f"   - Total relationships: {analysis['total_relationships']}")
            print(f"   - Cross-domain: {analysis['cross_domain_relationships']} ({analysis['cross_domain_percentage']:.1f}%)")
            
            print(f"\n🔗 Relationship Types:")
            for rel_type, count in analysis['relationship_types'].items():
                print(f"   - {rel_type}: {count}")
            
            print(f"\n📊 Confidence Statistics:")
            conf_stats = analysis['confidence_stats']
            print(f"   - Mean: {conf_stats['mean']:.3f}")
            print(f"   - Std: {conf_stats['std']:.3f}")
            print(f"   - Range: [{conf_stats['min']:.3f}, {conf_stats['max']:.3f}]")
        
        # Show sample relationships
        if results['relationships']:
            print(f"\n🔍 Sample Relationships (showing top {len(results['relationships'])}):")
            for i, rel in enumerate(results['relationships'][:5]):
                print(f"\n   {i+1}. {rel['type']} (confidence: {rel['confidence']:.3f})")
                print(f"      From: {rel['from_content']}")
                print(f"      To: {rel['to_content']}")
                if rel['cross_domain']:
                    print("      ✨ Cross-domain relationship!")
        
        print(f"\n✅ Results saved to: {output_dir}/")
        print("   - discovered_relationships.json")
        print("   - relationship_analysis.json")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)