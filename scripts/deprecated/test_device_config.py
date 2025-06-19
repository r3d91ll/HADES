#!/usr/bin/env python3
"""
Test script for comparing CPU and GPU configurations.

This script runs the pipeline test twice:
1. With CUDA_VISIBLE_DEVICES="" (forced CPU mode)
2. With default CUDA_VISIBLE_DEVICES (GPU mode if available)

Usage:
    python scripts/test_device_config.py --num-files 1
"""
import os
import sys
import argparse
import subprocess
import time


def run_test_with_config(cuda_devices: str, num_files: int) -> None:
    """Run the pipeline test with specific CUDA_VISIBLE_DEVICES setting."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    
    config_name = "CPU-only" if cuda_devices == "" else "GPU-enabled"
    print(f"\n\n{'='*80}")
    print(f"Running test with {config_name} configuration (CUDA_VISIBLE_DEVICES='{cuda_devices}')")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    result = subprocess.run(
        [sys.executable, "-m", "tests.integration.pipeline_multiprocess_test", "--num-files", str(num_files)],
        env=env,
        text=True,
        capture_output=True
    )
    elapsed = time.time() - start_time
    
    print(f"\nTest completed in {elapsed:.2f} seconds")
    print(f"Exit code: {result.returncode}")
    
    # Print a summary of the output, focusing on key information
    output_lines = result.stdout.splitlines()
    device_info_lines = [line for line in output_lines if "Using CPU configuration" in line 
                        or "Using GPU configuration" in line 
                        or "CUDA_VISIBLE_DEVICES" in line]
    
    print("\nDevice configuration:")
    for line in device_info_lines:
        print(f"  {line.strip()}")
    
    # Extract timing information
    timing_lines = [line for line in output_lines if "Timing Breakdown:" in line 
                   or "Parallel Processing:" in line
                   or "Document Processing:" in line
                   or "Chunking:" in line
                   or "Embedding:" in line]
    
    print("\nPerformance summary:")
    for line in output_lines:
        if "Parallel Processing:" in line or "Document Processing:" in line or "Chunking:" in line or "Embedding:" in line:
            print(f"  {line.strip()}")


def main():
    parser = argparse.ArgumentParser(description="Test pipeline with different device configurations")
    parser.add_argument("--num-files", type=int, default=1, help="Number of files to process")
    args = parser.parse_args()
    
    # Run with CPU-only configuration
    run_test_with_config("", args.num_files)
    
    # Run with GPU configuration (if available)
    run_test_with_config("0", args.num_files)  # Use first GPU


if __name__ == "__main__":
    main()
