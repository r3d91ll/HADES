#!/usr/bin/env python3
"""
Test Bootstrap Pipeline with a small dataset

Tests the Bootstrap Pipeline with just a few files to identify issues.
"""

import sys
import tempfile
import shutil
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipelines.bootstrap import BootstrapPipeline, BootstrapConfig
from src.pipelines.bootstrap.config import FileTypeFilter


def create_test_dataset():
    """Create a small test dataset."""
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create directory structure
    (temp_dir / "src").mkdir()
    (temp_dir / "docs").mkdir()
    (temp_dir / "config").mkdir()
    
    # Create a simple Python file
    (temp_dir / "src" / "main.py").write_text("""
import os

def main():
    '''Main function'''
    print("Hello, World!")
    
if __name__ == "__main__":
    main()
""")
    
    # Create a simple documentation file
    (temp_dir / "docs" / "README.md").write_text("""
# Test Project

This is a test project for Bootstrap Pipeline.

## Usage

Run `python src/main.py` to execute.
""")
    
    # Create a simple config file
    (temp_dir / "config" / "settings.json").write_text("""
{
    "app_name": "test_app",
    "version": "1.0.0"
}
""")
    
    return temp_dir


def main():
    """Run small test."""
    print("🧪 Testing Bootstrap Pipeline with small dataset")
    print("=" * 50)
    
    # Create test dataset
    test_dir = create_test_dataset()
    print(f"✅ Created test dataset at: {test_dir}")
    
    try:
        # Create configuration
        config = BootstrapConfig(
            input_directory=test_dir,
            output_database="test_bootstrap_small",
            file_type_filter=FileTypeFilter.ALL,
            max_workers=1,  # Single threaded for debugging
            batch_size=10,
            enable_parallel_processing=False,
            enable_cross_modal_discovery=True
        )
        
        # Run pipeline
        print("\n🚀 Running Bootstrap Pipeline...")
        pipeline = BootstrapPipeline(config)
        result = pipeline.execute()
        
        # Check results
        print("\n📊 Results:")
        print(f"Success: {result.success}")
        print(f"Nodes created: {result.node_count}")
        print(f"Edges created: {result.edge_count}")
        print(f"Files processed: {result.metrics.total_files_processed}")
        print(f"Processing time: {result.metrics.total_processing_time:.2f}s")
        
        if result.errors:
            print("\n❌ Errors:")
            for error in result.errors:
                print(f"  - {error}")
        
        if result.warnings:
            print("\n⚠️  Warnings:")
            for warning in result.warnings:
                print(f"  - {warning}")
                
    except Exception as e:
        print(f"\n❌ Pipeline failed with exception: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        shutil.rmtree(test_dir)
        print(f"\n🧹 Cleaned up test dataset")


if __name__ == "__main__":
    main()