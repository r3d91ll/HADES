#!/usr/bin/env python3
"""
Analyze Full Dataset

Quick script to analyze the contents of test-data3 before bootstrap.
"""

from pathlib import Path
import json

def analyze_dataset():
    """Analyze the full dataset contents."""
    
    input_dir = Path("/home/todd/ML-Lab/Olympus/test-data3")
    
    if not input_dir.exists():
        print(f"❌ Directory not found: {input_dir}")
        return
    
    # Count files by type
    file_types = {
        'pdf': list(input_dir.rglob("*.pdf")),
        'py': list(input_dir.rglob("*.py")),
        'md': list(input_dir.rglob("*.md")),
        'txt': list(input_dir.rglob("*.txt")),
        'json': list(input_dir.rglob("*.json")),
        'html': list(input_dir.rglob("*.html")),
        'yaml': list(input_dir.rglob("*.yaml")),
        'yml': list(input_dir.rglob("*.yml")),
    }
    
    # Find directories that look like code projects
    code_dirs = []
    for item in input_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            py_files = list(item.rglob("*.py"))
            if len(py_files) > 3:  # Likely a code project
                code_dirs.append(item)
    
    print("=" * 60)
    print("FULL DATASET ANALYSIS")
    print("=" * 60)
    print(f"📁 Base directory: {input_dir}")
    print()
    
    total_files = 0
    for file_type, files in file_types.items():
        if files:
            print(f"📄 {file_type.upper()} files: {len(files):,}")
            total_files += len(files)
            
            # Show some examples
            if len(files) <= 5:
                for f in files:
                    print(f"    └─ {f.name}")
            else:
                for f in files[:3]:
                    print(f"    ├─ {f.name}")
                print(f"    └─ ... and {len(files)-3} more")
            print()
    
    print(f"📊 Total files: {total_files:,}")
    print()
    
    if code_dirs:
        print("🐍 Code Project Directories:")
        for code_dir in code_dirs:
            py_files = list(code_dir.rglob("*.py"))
            print(f"    📁 {code_dir.name}: {len(py_files)} Python files")
        print()
    
    # Estimate processing time
    print("⏱️  ESTIMATED PROCESSING TIME:")
    print(f"    Small files (py, md, txt, json): ~{(total_files - len(file_types['pdf'])) * 0.1:.0f} seconds")
    print(f"    PDF files: ~{len(file_types['pdf']) * 5:.0f} seconds")
    print(f"    Embedding generation: ~{total_files * 2:.0f} seconds")
    print(f"    Graph construction: ~30-60 minutes")
    print(f"    ISNE training: ~2-6 hours")
    print(f"    📊 Total estimated: 3-8 hours")
    print()
    
    # Calculate rough dataset scale
    estimated_chunks = total_files * 15  # Rough estimate
    print("📈 ESTIMATED SCALE:")
    print(f"    Documents: ~{total_files:,}")
    print(f"    Chunks: ~{estimated_chunks:,}")
    print(f"    Graph nodes: ~{estimated_chunks:,}")
    print(f"    Graph edges: ~{estimated_chunks * 20:,}")
    print("=" * 60)


if __name__ == "__main__":
    analyze_dataset()