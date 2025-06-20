#!/usr/bin/env python
"""
Debug Pipeline Runner for HADES

This script runs the pipeline with detailed debugging output, saving intermediate
JSON files after each stage transformation for inspection and debugging.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add the project root to sys.path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.alerts import AlertLevel, AlertManager
from src.orchestration.pipelines.data_ingestion.stages.document_processor import DocumentProcessorStage
from src.orchestration.pipelines.data_ingestion.stages.chunking import ChunkingStage
from src.orchestration.pipelines.data_ingestion.stages.embedding import EmbeddingStage
from src.orchestration.pipelines.data_ingestion.stages.isne import ISNEStage
from src.orchestration.pipelines.schema import DocumentSchema

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def save_stage_output(data: Any, stage_name: str, output_dir: Path, run_id: str, file_idx: int) -> Path:
    """Save the output of a pipeline stage to a JSON file."""
    stage_dir = output_dir / f"stage_{stage_name}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = stage_dir / f"{run_id}_file{file_idx:03d}_{stage_name}_output.json"
    
    # Convert data to serializable format
    if isinstance(data, list):
        # Handle list of DocumentSchema or similar objects
        serializable_data = []
        for item in data:
            if hasattr(item, 'model_dump'):
                serializable_data.append(item.model_dump())
            elif hasattr(item, 'dict'):
                serializable_data.append(item.dict())
            elif hasattr(item, '__dict__'):
                serializable_data.append(item.__dict__)
            else:
                serializable_data.append(str(item))
    elif hasattr(data, 'model_dump'):
        serializable_data = data.model_dump()
    elif hasattr(data, 'dict'):
        serializable_data = data.dict()
    else:
        serializable_data = data
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=2, default=str, ensure_ascii=False)
    
    logger.info(f"Saved {stage_name} output to {output_file}")
    return output_file


def process_single_file(file_path: Path, stages: List[Tuple[str, Any]], output_dir: Path, run_id: str, file_idx: int) -> Dict[str, Any]:
    """Process a single file through all pipeline stages, saving outputs at each stage."""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing file {file_idx + 1}: {file_path}")
    logger.info(f"{'='*60}")
    
    # Track results and timing
    stage_results = {}
    current_data = file_path
    
    for stage_name, stage_obj in stages:
        logger.info(f"\n--- Stage: {stage_name} ---")
        start_time = time.time()
        
        try:
            # Execute the stage
            if stage_name == "document_processor":
                # Document processor expects file paths
                stage_input = [str(current_data)]
            else:
                # Other stages expect the output from previous stage
                stage_input = current_data if isinstance(current_data, list) else [current_data]
            
            logger.info(f"Stage input type: {type(stage_input)}")
            if isinstance(stage_input, list):
                logger.info(f"Stage input count: {len(stage_input)}")
            
            # Run the stage
            stage_output = stage_obj.run(stage_input)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Save the output
            output_file = save_stage_output(stage_output, stage_name, output_dir, run_id, file_idx)
            
            # Store results
            stage_results[stage_name] = {
                "status": "success",
                "processing_time": processing_time,
                "output_file": str(output_file),
                "output_count": len(stage_output) if isinstance(stage_output, list) else 1
            }
            
            # Log stage success
            logger.info(f"Stage {stage_name} completed in {processing_time:.2f}s")
            if isinstance(stage_output, list):
                logger.info(f"Output: {len(stage_output)} items")
                if stage_output and hasattr(stage_output[0], 'chunks'):
                    total_chunks = sum(len(item.chunks) for item in stage_output)
                    logger.info(f"Total chunks: {total_chunks}")
            
            # Update current data for next stage
            current_data = stage_output
            
        except Exception as e:
            logger.error(f"Stage {stage_name} failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Save error information
            stage_results[stage_name] = {
                "status": "failed",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
            
            # Save error details
            error_file = output_dir / f"stage_{stage_name}" / f"{run_id}_file{file_idx:03d}_{stage_name}_error.json"
            error_file.parent.mkdir(parents=True, exist_ok=True)
            with open(error_file, 'w') as f:
                json.dump({
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "input_type": type(stage_input).__name__
                }, f, indent=2)
            
            break  # Stop processing on error
    
    return stage_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug Pipeline Runner for HADES")
    parser.add_argument("-i", "--input-dir", required=True, help="Input directory containing files to process")
    parser.add_argument("-o", "--output-dir", required=True, help="Output directory for results and debug files")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum number of files to process")
    parser.add_argument("--file-pattern", default="*.pdf", help="File pattern to match (default: *.pdf)")
    
    args = parser.parse_args()
    
    # Setup directories
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate run ID
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    
    # Find input files
    if args.file_pattern:
        files = list(input_dir.glob(args.file_pattern))
    else:
        files = list(input_dir.iterdir())
    
    files = [f for f in files if f.is_file()]
    
    if args.max_files:
        files = files[:args.max_files]
    
    if not files:
        logger.error(f"No files found in {input_dir}")
        return 1
    
    logger.info(f"Found {len(files)} files to process")
    
    # Initialize pipeline stages
    logger.info("Initializing pipeline stages...")
    
    stages = [
        ("document_processor", DocumentProcessorStage()),
        ("chunking", ChunkingStage()),
        ("embedding", EmbeddingStage()),
        ("isne", ISNEStage())
    ]
    
    # Process each file
    all_results = []
    overall_start = time.time()
    
    for file_idx, file_path in enumerate(files):
        file_results = process_single_file(file_path, stages, output_dir, run_id, file_idx)
        all_results.append({
            "file": str(file_path),
            "stages": file_results
        })
    
    overall_time = time.time() - overall_start
    
    # Save summary
    summary = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_files": len(files),
        "total_time": overall_time,
        "files_processed": all_results
    }
    
    summary_file = output_dir / f"{run_id}_pipeline_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Pipeline completed in {overall_time:.2f}s")
    logger.info(f"Summary saved to {summary_file}")
    logger.info(f"Stage outputs saved to {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())