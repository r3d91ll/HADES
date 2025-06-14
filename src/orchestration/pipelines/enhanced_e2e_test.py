#!/usr/bin/env python
"""
Enhanced end-to-end test of the HADES pipeline with multi-format support and debug mode.

This script demonstrates the complete workflow from multi-format document processing 
to ISNE enhancement with comprehensive debugging, validation, and alerts.

Supports all Docling formats: PDF, Word, PowerPoint, Excel, Python code files, 
Markdown, HTML, CSV, JSON, YAML, and more.

Debug mode saves intermediate JSON files at each pipeline stage for troubleshooting.

Usage:
    # Basic run with all files in a directory
    python -m src.orchestration.pipelines.enhanced_e2e_test -i ./test-data -o ./e2e-output
    
    # Debug mode with stage-by-stage output
    python -m src.orchestration.pipelines.enhanced_e2e_test -i ./test-data -o ./e2e-output --debug
    
    # Run only specific stages
    python -m src.orchestration.pipelines.enhanced_e2e_test -i ./test-data -o ./e2e-output --stage-only chunking
    
    # Compare outputs between runs
    python -m src.orchestration.pipelines.enhanced_e2e_test -i ./test-data -o ./e2e-output --diff-with ./previous-run
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import difflib

# Pipeline imports
from .orchestrator import Pipeline, PipelineConfiguration
from .stages import (
    DocumentProcessorStage, 
    ChunkingStage, 
    EmbeddingStage, 
    ISNEStage
)
from .stages.base import PipelineStage
from .schema import PipelineExecutionResult

# Core imports
from src.alerts import AlertManager, AlertLevel


# Supported file extensions (based on Docling capabilities + Python AST processing)
SUPPORTED_EXTENSIONS = {
    # Document formats
    '.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls', 
    '.html', '.htm', '.xml', '.epub', '.rtf', '.odt', '.csv',
    # Text formats
    '.txt', '.md', '.markdown', '.json', '.yaml', '.yml',
    # Code formats (Python only - requires AST processing for symbol tables)
    '.py'
}


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None, debug_mode: bool = False) -> None:
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # More verbose logging in debug mode
    if debug_mode and numeric_level > logging.DEBUG:
        numeric_level = logging.DEBUG
    
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers
    )


def discover_files(input_dir: Path, max_files: Optional[int] = None) -> List[Path]:
    """Discover all supported files in the input directory."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} not found")
    
    files = []
    for file_path in input_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(file_path)
    
    # Sort by modification time (newest first) for consistent ordering
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    
    if max_files:
        files = files[:max_files]
    
    logging.info(f"Discovered {len(files)} supported files in {input_dir}")
    return files


def save_debug_output(data: Any, stage_name: str, debug_dir: Path, run_id: str) -> Path:
    """Save intermediate pipeline output for debugging."""
    stage_dir = debug_dir / f"{stage_name.replace(' ', '_').lower()}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the main data
    safe_stage_name = stage_name.lower().replace(' ', '_')
    output_file = stage_dir / f"{run_id}_{safe_stage_name}_output.json"
    
    # Handle different data types
    if hasattr(data, 'model_dump'):  # New Pydantic models
        json_data = data.model_dump()
    elif hasattr(data, 'dict'):  # Legacy Pydantic models
        json_data = data.dict()
    elif isinstance(data, list):
        # Handle list of objects (e.g., documents)
        json_data = []
        for item in data:
            if hasattr(item, 'model_dump'):
                json_data.append(item.model_dump())
            elif hasattr(item, 'dict'):
                json_data.append(item.dict())
            elif hasattr(item, '__dict__'):
                json_data.append(item.__dict__)
            else:
                json_data.append(str(item))
    elif isinstance(data, dict):
        json_data = data
    elif hasattr(data, '__dict__'):  # Regular objects
        json_data = data.__dict__
    else:
        json_data = {"data": str(data), "type": str(type(data))}
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, default=str, ensure_ascii=False)
    
    logging.debug(f"Saved {stage_name} debug output to {output_file}")
    return output_file


def save_stage_stats(stats: Dict[str, Any], stage_name: str, debug_dir: Path, run_id: str) -> Path:
    """Save stage processing statistics."""
    stage_dir = debug_dir / f"{stage_name.replace(' ', '_').lower()}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    
    stats_file = stage_dir / f"{run_id}_{stage_name.lower()}_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, default=str)
    
    return stats_file


def compare_outputs(current_dir: Path, previous_dir: Path) -> Dict[str, Any]:
    """Compare outputs between two pipeline runs."""
    comparison_results: Dict[str, Any] = {"differences": [], "summary": {}}
    
    if not previous_dir.exists():
        comparison_results["summary"]["error"] = f"Previous run directory {previous_dir} not found"
        return comparison_results
    
    # Compare stage outputs
    for stage_dir in current_dir.iterdir():
        if not stage_dir.is_dir():
            continue
            
        prev_stage_dir = previous_dir / stage_dir.name
        if not prev_stage_dir.exists():
            comparison_results["differences"].append({
                "type": "missing_stage",
                "stage": stage_dir.name,
                "message": f"Stage {stage_dir.name} not found in previous run"
            })
            continue
        
        # Compare JSON files in the stage
        for json_file in stage_dir.glob("*.json"):
            prev_json_file = prev_stage_dir / json_file.name
            if not prev_json_file.exists():
                comparison_results["differences"].append({
                    "type": "missing_file", 
                    "file": str(json_file),
                    "message": f"File {json_file.name} not found in previous run"
                })
                continue
            
            # Load and compare JSON content
            try:
                with open(json_file, 'r') as f1, open(prev_json_file, 'r') as f2:
                    current_data = json.load(f1)
                    previous_data = json.load(f2)
                
                if current_data != previous_data:
                    # Generate diff
                    current_str = json.dumps(current_data, indent=2, sort_keys=True)
                    previous_str = json.dumps(previous_data, indent=2, sort_keys=True)
                    
                    diff = list(difflib.unified_diff(
                        previous_str.splitlines(keepends=True),
                        current_str.splitlines(keepends=True),
                        fromfile=f"previous/{json_file.name}",
                        tofile=f"current/{json_file.name}"
                    ))
                    
                    comparison_results["differences"].append({
                        "type": "content_diff",
                        "file": str(json_file),
                        "diff_lines": len(diff),
                        "diff": ''.join(diff[:100])  # Limit diff size
                    })
            
            except Exception as e:
                comparison_results["differences"].append({
                    "type": "comparison_error",
                    "file": str(json_file),
                    "error": str(e)
                })
    
    comparison_results["summary"] = {
        "total_differences": len(comparison_results["differences"]),
        "missing_stages": len([d for d in comparison_results["differences"] if d["type"] == "missing_stage"]),
        "missing_files": len([d for d in comparison_results["differences"] if d["type"] == "missing_file"]),
        "content_diffs": len([d for d in comparison_results["differences"] if d["type"] == "content_diff"]),
        "errors": len([d for d in comparison_results["differences"] if d["type"] == "comparison_error"])
    }
    
    return comparison_results


def create_pipeline_stages(args: argparse.Namespace) -> List[Tuple[str, PipelineStage]]:
    """Create pipeline stages based on configuration.
    
    Uses the existing specialized methods for Python file processing:
    - PythonAdapter/PythonCodeAdapter for AST processing
    - PythonCodeChunker for AST-aware chunking  
    - create_graph_from_documents for AST-based ISNE graphs
    """
    stages: List[Tuple[str, PipelineStage]] = []
    
    # Stage 1: Document Processing 
    # Uses PythonAdapter._process_python_file() and EntityExtractor for Python files
    if not args.stage_only or args.stage_only == "docproc":
        doc_config = {
            "supported_formats": list(ext.lstrip('.') for ext in SUPPORTED_EXTENSIONS),
            "extract_metadata": True,
            "extract_entities": True,
            # Ensure Python files use PythonAdapter/PythonCodeAdapter
            "force_python_adapter": True
        }
        stages.append(("01_docproc", DocumentProcessorStage(config=doc_config)))
    
    # Stage 2: Chunking
    # Uses PythonCodeChunker for .py files and chunk_python_code() function
    if not args.stage_only or args.stage_only == "chunking":
        chunk_config = {
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "strategy": args.chunk_strategy,
            "min_chunk_size": 50,
            "max_chunk_size": args.chunk_size * 2,
            # Ensure Python files use PythonCodeChunker
            "force_python_chunker": True
        }
        stages.append(("02_chunking", ChunkingStage(config=chunk_config)))
    
    # Stage 3: Embedding
    # Uses standard embedding adapters (treat Python code as enhanced text)
    if not args.stage_only or args.stage_only == "embedding":
        embed_config = {
            "adapter_name": args.embedding_adapter,
            "batch_size": args.embedding_batch_size,
            "normalize_embeddings": True
        }
        stages.append(("03_embedding", EmbeddingStage(config=embed_config)))
    
    # Stage 4: ISNE Enhancement
    # Uses create_graph_from_documents() which processes AST relationships
    if not args.stage_only or args.stage_only == "isne":
        isne_config = {
            "generate_relationships": True,
            "similarity_threshold": args.similarity_threshold,
            "max_relationships": args.max_relationships,
            # create_graph_from_documents() automatically processes Python relationships
            "process_code_relationships": True
        }
        if args.model_path:
            isne_config["model_path"] = args.model_path
        stages.append(("04_isne", ISNEStage(config=isne_config)))
    
    return stages


def run_enhanced_pipeline(args: argparse.Namespace) -> int:
    """Run the enhanced multi-format pipeline with debug mode."""
    
    # Setup directories
    output_dir = Path(args.output_dir)
    debug_dir = output_dir / "debug" if args.debug else None
    alerts_dir = output_dir / "alerts"
    results_dir = output_dir / "results"
    
    for dir_path in [output_dir, alerts_dir, results_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate run ID for this execution
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup alert manager
    alert_manager = AlertManager(
        alert_dir=str(alerts_dir),
        min_level=getattr(AlertLevel, args.alert_threshold.upper())
    )
    
    # Discover input files
    input_dir = Path(args.input_dir)
    try:
        files = discover_files(input_dir, args.max_files)
        if not files:
            logging.error(f"No supported files found in {input_dir}")
            alert_manager.alert(
                message=f"No supported files found in {input_dir}",
                level=AlertLevel.HIGH,
                source="enhanced_e2e_test"
            )
            return 1
    except Exception as e:
        logging.error(f"Error discovering files: {e}")
        return 1
    
    # Create pipeline configuration
    pipeline_config = PipelineConfiguration(
        execution_mode="parallel" if args.parallel else "sequential",
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        enable_monitoring=True
    )
    
    # Create pipeline stages
    stages = create_pipeline_stages(args)
    if not stages:
        logging.error("No pipeline stages configured")
        return 1
    
    # Create and execute pipeline
    pipeline_stages = [stage for _, stage in stages]
    logging.info(f"Created {len(pipeline_stages)} pipeline stages: {[s.name for s in pipeline_stages]}")
    
    pipeline: Pipeline = Pipeline(
        stages=pipeline_stages, 
        config=pipeline_config, 
        name="enhanced_e2e_test"
    )
    
    logging.info(f"Starting pipeline with {len(stages)} stages for {len(files)} files")
    start_time = time.time()
    
    try:
        # Convert file paths to strings for pipeline input
        file_paths = [str(f) for f in files]
        logging.info(f"Executing pipeline with file paths: {file_paths}")
        logging.info(f"Pipeline has {len(pipeline.stages)} stages configured")
        results: PipelineExecutionResult = pipeline.execute_batch(file_paths)
        
        processing_time = time.time() - start_time
        
        # Save debug output if enabled - create output file for each pipeline stage
        if debug_dir:
            # Save stage-by-stage debug output
            if hasattr(results, 'stage_results') and results.stage_results:
                # If we have individual stage results, save each one
                for i, (stage_name, _) in enumerate(stages):
                    if i < len(results.stage_results):
                        stage_result = results.stage_results[i]
                        save_debug_output(stage_result, stage_name, debug_dir, run_id)
                        
                        # Also save stage-specific statistics
                        stage_stats = {
                            "stage_name": stage_name,
                            "stage_index": i,
                            "input_count": getattr(stage_result, 'input_count', 0),
                            "output_count": getattr(stage_result, 'output_count', 0),
                            "processing_time": getattr(stage_result, 'processing_time', 0),
                            "errors": getattr(stage_result, 'errors', []),
                            "warnings": getattr(stage_result, 'warnings', [])
                        }
                        save_stage_stats(stage_stats, stage_name, debug_dir, run_id)
            
            elif hasattr(results, 'documents') and results.documents:
                # Fallback: save documents for each stage (assuming they contain stage info)
                for i, (stage_name, _) in enumerate(stages):
                    # Create stage-specific document subset if possible
                    stage_docs = []
                    for doc in results.documents:
                        if hasattr(doc, 'processing_stages') and stage_name in doc.processing_stages:
                            stage_docs.append(doc)
                    
                    if stage_docs:
                        save_debug_output(stage_docs, stage_name, debug_dir, run_id)
                    else:
                        # Save all documents with stage marker
                        save_debug_output(results.documents, f"{stage_name}_all_docs", debug_dir, run_id)
            
            else:
                # Ultimate fallback: save the raw results
                save_debug_output(results, "final_results", debug_dir, run_id)
            
            # Save a debug summary with information about each stage
            debug_summary: Dict[str, Any] = {
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "stages_processed": [stage_name for stage_name, _ in stages],
                "files_input": file_paths,
                "debug_files_created": []
            }
            
            # List all debug files created
            for stage_name, _ in stages:
                stage_dir = debug_dir / f"{stage_name.replace(' ', '_').lower()}"
                if stage_dir.exists():
                    debug_files = list(stage_dir.glob(f"{run_id}_*.json"))
                    debug_summary["debug_files_created"].extend([str(f) for f in debug_files])
            
            # Save debug summary
            debug_summary_file = debug_dir / f"{run_id}_debug_summary.json"
            with open(debug_summary_file, 'w', encoding='utf-8') as f:
                json.dump(debug_summary, f, indent=2, default=str, ensure_ascii=False)
        
        # Save final results
        results_file = results_dir / f"{run_id}_final_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            if hasattr(results, 'model_dump'):
                json.dump(results.model_dump(), f, indent=2, default=str, ensure_ascii=False)
            elif hasattr(results, 'dict'):
                json.dump(results.dict(), f, indent=2, default=str, ensure_ascii=False)
            else:
                json.dump({"results": str(results)}, f, indent=2, default=str, ensure_ascii=False)
        
        # Generate summary statistics
        stats = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "input_dir": str(input_dir),
            "files_processed": len(files),
            "stages_completed": len(stages),
            "processing_time_seconds": processing_time,
            "files": [str(f) for f in files],
            "stages": [stage_name for stage_name, _ in stages]
        }
        
        # Calculate per-stage statistics
        for i, (stage_name, _) in enumerate(stages):
            stage_stats = {
                "stage_index": i,
                "documents_processed": len(files),  # Simplified - all files go through all stages
                "stage_name": stage_name
            }
            stats[f"{stage_name}_stats"] = stage_stats
            
            if debug_dir:
                save_stage_stats(stage_stats, stage_name, debug_dir, run_id)
        
        # Compare with previous run if requested
        if args.diff_with and debug_dir:
            comparison = compare_outputs(debug_dir, Path(args.diff_with) / "debug")
            stats["comparison"] = comparison
            
            # Save comparison results
            comparison_file = output_dir / f"{run_id}_comparison.json"
            with open(comparison_file, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, indent=2, default=str)
        
        # Save run statistics
        stats_file = output_dir / f"{run_id}_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, default=str)
        
        # Print summary
        print(f"\n===== Enhanced Pipeline Summary =====")
        print(f"Run ID: {run_id}")
        print(f"Files processed: {len(files)}")
        print(f"Stages completed: {len(stages)}")
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Results saved to: {results_file}")
        
        if debug_dir:
            print(f"Debug output saved to: {debug_dir}")
        
        if args.diff_with:
            comparison_data = stats.get("comparison", {})
            comp_summary = comparison_data.get("summary", {}) if isinstance(comparison_data, dict) else {}
            print(f"\n===== Comparison Summary =====")
            print(f"Total differences: {comp_summary.get('total_differences', 0)}")
            if comp_summary.get('total_differences', 0) > 0:
                print(f"Content differences: {comp_summary.get('content_diffs', 0)}")
                print(f"Missing files: {comp_summary.get('missing_files', 0)}")
        
        # Check for alerts
        alert_stats = alert_manager.get_alert_stats()
        critical_count = alert_stats.get("CRITICAL", 0) + alert_stats.get("HIGH", 0)
        if critical_count > 0:
            print(f"\n⚠️ WARNING: {critical_count} critical/high alerts generated")
            print(f"Review alerts in {alerts_dir}")
        
        return 0
        
    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}", exc_info=True)
        alert_manager.alert(
            message=f"Pipeline execution failed: {e}",
            level=AlertLevel.CRITICAL,
            source="enhanced_e2e_test"
        )
        return 1


def main() -> int:
    """Main entry point for the enhanced end-to-end test script."""
    parser = argparse.ArgumentParser(
        description="Enhanced end-to-end HADES pipeline test with multi-format support and debug mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input/Output
    parser.add_argument("-i", "--input-dir", required=True, help="Input directory containing files to process")
    parser.add_argument("-o", "--output-dir", default="./e2e-output", help="Output directory for all results")
    
    # Debug and troubleshooting
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with intermediate file saving")
    parser.add_argument("--stage-only", choices=["docproc", "chunking", "embedding", "isne"], 
                       help="Run only a specific pipeline stage")
    parser.add_argument("--diff-with", help="Compare outputs with a previous run directory")
    
    # Processing options
    parser.add_argument("--max-files", type=int, help="Maximum number of files to process")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel processing")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of worker processes")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for processing")
    
    # Stage-specific configuration
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for document chunking")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Overlap between chunks")
    parser.add_argument("--chunk-strategy", default="paragraph", help="Chunking strategy")
    parser.add_argument("--embedding-adapter", default="cpu", help="Embedding adapter to use")
    parser.add_argument("--embedding-batch-size", type=int, default=32, help="Embedding batch size")
    parser.add_argument("--similarity-threshold", type=float, default=0.7, help="Similarity threshold for ISNE")
    parser.add_argument("--max-relationships", type=int, default=10, help="Maximum relationships per chunk")
    
    # Model and alerting
    parser.add_argument("-m", "--model-path", help="Path to ISNE model file")
    parser.add_argument("-a", "--alert-threshold", choices=["low", "medium", "high", "critical"], 
                       default="medium", help="Alert threshold level")
    
    # Logging
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="Logging level")
    parser.add_argument("--log-file", help="Log file path")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level, args.log_file, args.debug)
    
    try:
        return run_enhanced_pipeline(args)
    except Exception as e:
        logging.error(f"Error in enhanced end-to-end test: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())