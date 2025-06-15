#!/usr/bin/env python3
"""
Test Model Evaluation Stage

Quick test of the new model evaluation stage on our existing trained model.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.isne.bootstrap.stages.model_evaluation import ModelEvaluationStage, ModelEvaluationConfig


def test_evaluation():
    """Test the evaluation stage on our existing model."""
    
    print("🧪 Testing ISNE Model Evaluation Stage")
    print("="*50)
    
    # Paths to our existing model and output
    model_path = "output/olympus_production/isne_model_final.pth"
    output_dir = Path("output/olympus_production")
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        return False
    
    # We need to create the graph_data.json first since our pipeline didn't save it
    graph_data_path = output_dir / "graph_data.json"
    if not graph_data_path.exists():
        print(f"⚠️  Graph data file missing: {graph_data_path}")
        print("   Creating graph data from model...")
        
        # Run our reconstruction script
        import subprocess
        cmd = [
            sys.executable, "scripts/reconstruct_graph_data.py",
            "--model-path", model_path,
            "--output", str(graph_data_path)
        ]
        
        print(f"   Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Failed to create graph data:")
            print(result.stderr)
            return False
        
        print("✅ Graph data created successfully")
    
    # Create evaluation stage and config
    eval_stage = ModelEvaluationStage()
    eval_config = ModelEvaluationConfig(
        run_evaluation=True,
        save_visualizations=True,
        sample_size_for_visualization=500  # Smaller for testing
    )
    
    print(f"\n🚀 Running evaluation...")
    print(f"   Model: {model_path}")
    print(f"   Graph data: {graph_data_path}")
    print(f"   Output: {output_dir}")
    
    # Run evaluation
    try:
        result = eval_stage.run(
            model_path=str(model_path),
            graph_data_path=str(graph_data_path),
            output_dir=output_dir,
            config=eval_config
        )
        
        if result.success:
            print(f"\n✅ Evaluation completed successfully!")
            print(f"📊 Results saved to: {result.stats.get('evaluation_output_dir', 'unknown')}")
            
            # Print key metrics
            if result.evaluation_metrics:
                print(f"\n📋 Key Results:")
                
                # Model info
                if 'model_info' in result.evaluation_metrics:
                    info = result.evaluation_metrics['model_info']
                    print(f"   Model: {info['num_nodes']:,} nodes, {info['num_edges']:,} edges")
                    print(f"   Parameters: {info['model_parameters']:,}")
                
                # Inductive performance (key metric)
                if 'inductive_performance' in result.evaluation_metrics:
                    inductive = result.evaluation_metrics['inductive_performance']
                    if 'relative_performance_percent' in inductive:
                        perf = inductive['relative_performance_percent']
                        target = inductive.get('achieves_90_percent_target', False)
                        print(f"   🎯 Inductive Performance: {perf:.2f}% ({'✅ ACHIEVES' if target else '❌ BELOW'} 90% target)")
                
                # Overall assessment
                if 'overall_status' in result.stats:
                    status = result.stats['overall_status']
                    print(f"   📈 Overall Status: {status.upper()}")
            
            return True
        else:
            print(f"❌ Evaluation failed: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_evaluation()
    sys.exit(0 if success else 1)