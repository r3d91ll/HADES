#!/usr/bin/env python3
"""
Debug Bootstrap Pipeline

Tests the Bootstrap Pipeline with extensive debug logging.
"""

import sys
import tempfile
import shutil
import logging
from pathlib import Path

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipelines.bootstrap import BootstrapPipeline, BootstrapConfig
from src.pipelines.bootstrap.config import FileTypeFilter


def create_minimal_test():
    """Create a single file test."""
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create just one simple file
    (temp_dir / "test.py").write_text("""
def hello():
    return "world"
""")
    
    return temp_dir


def main():
    """Run debug test."""
    print("🔍 Debug Bootstrap Pipeline Test")
    print("=" * 50)
    
    # Create minimal test
    test_dir = create_minimal_test()
    print(f"✅ Created test file at: {test_dir}")
    
    try:
        # Create configuration with minimal settings
        config = BootstrapConfig(
            input_directory=test_dir,
            output_database="test_debug",
            file_type_filter=FileTypeFilter.CODE_ONLY,
            max_workers=1,
            batch_size=1,
            enable_parallel_processing=False,
            enable_cross_modal_discovery=False,
            edge_discovery_methods=[]  # No edge discovery
        )
        
        print("\n📋 Configuration created")
        print(f"  - Input: {config.input_directory}")
        print(f"  - Database: {config.output_database}")
        print(f"  - File filter: {config.file_type_filter}")
        
        # Create pipeline
        print("\n🏗️  Creating pipeline...")
        pipeline = BootstrapPipeline(config)
        print("✅ Pipeline created")
        
        # Run dry run first
        print("\n🧪 Running dry run...")
        dry_run_result = pipeline.dry_run()
        print(f"✅ Dry run complete")
        print(f"  - Files found: {dry_run_result['analysis']['total_files']}")
        print(f"  - File types: {dry_run_result['analysis']['file_types']}")
        
        # Now run actual pipeline
        print("\n🚀 Running actual pipeline...")
        print("Phase 1: Storage initialization...")
        
        # We'll call execute but with a timeout
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Pipeline execution timed out")
        
        # Set a 30 second timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        
        try:
            result = pipeline.execute()
            signal.alarm(0)  # Cancel the alarm
            
            print("\n✅ Pipeline completed!")
            print(f"Success: {result.success}")
            
            if result.errors:
                print("\n❌ Errors:")
                for error in result.errors:
                    print(f"  - {error}")
                    
        except TimeoutError:
            print("\n⏱️  Pipeline execution timed out after 30 seconds")
            
    except Exception as e:
        print(f"\n❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        try:
            shutil.rmtree(test_dir)
            print(f"\n🧹 Cleaned up test dataset")
        except:
            pass


if __name__ == "__main__":
    main()