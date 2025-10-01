#!/usr/bin/env python3
"""
Build Graph Edges (Phase D)
============================

Extracts edges from ingested documents to create the code graph:

1. imports edges: code → code (from metadata.symbols.imports)
2. contains edges: directories → files (from filesystem structure)
3. references edges: code → code (from function calls in symbols)

This completes the ingestion pipeline (Phases A-D) and enables PathRAG traversal.

Usage:
    poetry run python scripts/build_graph_edges.py --write
    poetry run python scripts/build_graph_edges.py --dry-run  # Preview only
"""

from __future__ import annotations

import argparse
import pprint
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.database.arango.memory_client import (
    ArangoMemoryClient,
    CollectionDefinition,
    resolve_memory_config,
)


def sanitize_key(relpath: str, max_len: int = 254) -> str:
    """Sanitize string for ArangoDB _key (matches ingestion workflow)."""
    s = relpath.replace("/", "_").replace(" ", "_")
    s = "".join(ch if ch.isalnum() or ch in "-_.:@()+" else "_" for ch in s)
    s = s.strip("._-") or "item"
    return s[:max_len]


def resolve_import(import_stmt: str, current_path: str) -> Optional[str]:
    """
    Resolve an import statement to a target document _key.

    Args:
        import_stmt: Python import statement (e.g., "from core.embedders import Jina")
        current_path: Path of the file containing the import

    Returns:
        Target document _key if resolvable, else None
    """
    # Extract module name from import statement
    # Patterns: "from X import Y" or "import X"
    from_match = re.search(r'from\s+([\w.]+)', import_stmt)
    import_match = re.search(r'import\s+([\w.]+)', import_stmt)

    module = None
    if from_match:
        module = from_match.group(1)
    elif import_match:
        module = import_match.group(1)

    if not module:
        return None

    # Skip standard library and third-party imports
    stdlib_prefixes = ['os', 'sys', 're', 'json', 'typing', 'pathlib', 'collections',
                       'dataclasses', 'logging', 'datetime', 'time', 'random', 'math',
                       '__future__', 'abc', 'enum', 'functools', 'itertools', 'io']
    thirdparty_prefixes = ['torch', 'numpy', 'transformers', 'sentence_transformers',
                           'httpx', 'grpc', 'pytest', 'tree_sitter']

    for prefix in stdlib_prefixes + thirdparty_prefixes:
        if module.startswith(prefix):
            return None

    # Handle relative imports (e.g., ".embedders_base" or "..config")
    if module.startswith('.'):
        # Count leading dots
        dots = len(module) - len(module.lstrip('.'))
        rel_module = module.lstrip('.')

        # Get current file's directory
        current_dir = Path(current_path).parent

        # Go up 'dots-1' levels
        for _ in range(dots - 1):
            current_dir = current_dir.parent

        # Build absolute module path
        if rel_module:
            # from .embedders_base → core/embedders/embedders_base.py
            target_path = current_dir / (rel_module + '.py')
        else:
            # from . import X → core/embedders/__init__.py
            target_path = current_dir / '__init__.py'

        # Convert to key
        target_key = sanitize_key(str(target_path))
        return target_key

    # Absolute imports within project (e.g., "core.embedders.embedders_base")
    # Convert module path to file path
    # e.g., "core.embedders.embedders_jina" → "core/embedders/embedders_jina.py"
    parts = module.split('.')

    # Try direct file match
    file_path = '/'.join(parts) + '.py'
    target_key = sanitize_key(file_path)

    return target_key


def extract_directory_path(file_path: str) -> str:
    """Extract directory path from file path."""
    parts = Path(file_path).parts
    if len(parts) <= 1:
        return "."
    return str(Path(*parts[:-1]))


def build_edges_from_symbols(
    client: ArangoMemoryClient,
    dry_run: bool = False
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Extract edges from ingested documents.

    Returns:
        (import_edges, contains_edges, reference_edges)
    """
    print("[Phase 1] Loading documents from repo_docs...")

    # Load all documents
    docs = list(client.execute_query("FOR doc IN repo_docs RETURN doc"))
    print(f"  Loaded {len(docs)} documents")

    # Track which documents exist (for import resolution)
    existing_keys = {doc['_key'] for doc in docs}

    # Track directories
    directories: Dict[str, dict] = {}

    import_edges: List[dict] = []
    contains_edges: List[dict] = []
    reference_edges: List[dict] = []

    print("[Phase 2] Extracting import edges...")

    for doc in docs:
        doc_key = doc['_key']
        doc_path = doc.get('path', '')
        metadata = doc.get('metadata', {})
        symbols = metadata.get('symbols', {})

        # Extract directory for contains edges
        dir_path = extract_directory_path(doc_path)
        if dir_path not in directories:
            dir_key = sanitize_key(dir_path)
            directories[dir_path] = {
                '_key': dir_key,
                'path': dir_path,
                'type': 'directory',
                'file_count': 0
            }
        directories[dir_path]['file_count'] += 1

        # Extract import edges
        imports = symbols.get('imports', [])
        for imp in imports:
            import_stmt = imp.get('statement', '')
            if not import_stmt:
                continue

            target_key = resolve_import(import_stmt, doc_path)

            # Only create edge if target exists
            if target_key and target_key in existing_keys:
                import_edges.append({
                    '_from': f'repo_docs/{doc_key}',
                    '_to': f'repo_docs/{target_key}',
                    'type': 'imports',
                    'weight': 1.0,
                    'import_statement': import_stmt[:200],  # Truncate long imports
                    'source_file': doc_path
                })

    print(f"  Found {len(import_edges)} import edges")

    print("[Phase 3] Creating contains edges (directories → files)...")

    for doc in docs:
        doc_key = doc['_key']
        doc_path = doc.get('path', '')
        dir_path = extract_directory_path(doc_path)
        dir_key = sanitize_key(dir_path)

        contains_edges.append({
            '_from': f'directories/{dir_key}',
            '_to': f'repo_docs/{doc_key}',
            'type': 'contains',
            'weight': 1.0,
            'directory': dir_path,
            'file': doc_path
        })

    print(f"  Found {len(contains_edges)} contains edges")

    print("[Phase 4] Extracting reference edges (function calls)...")

    # Build function name index for reference detection
    function_index: Dict[str, List[str]] = defaultdict(list)
    for doc in docs:
        doc_key = doc['_key']
        symbols = doc.get('metadata', {}).get('symbols', {})
        functions = symbols.get('functions', [])

        for func in functions:
            func_name = func.get('name', '')
            if func_name and not func_name.startswith('_'):  # Skip private functions
                function_index[func_name].append(doc_key)

    # Look for function references in code text
    for doc in docs:
        doc_key = doc['_key']
        doc_text = doc.get('text', '')

        # Simple heuristic: look for function_name( patterns
        for func_name, target_keys in function_index.items():
            # Skip if referencing own functions
            if doc_key in target_keys:
                continue

            # Look for function call patterns
            pattern = rf'\b{re.escape(func_name)}\s*\('
            if re.search(pattern, doc_text):
                # Create reference edges to all files defining this function
                for target_key in target_keys:
                    reference_edges.append({
                        '_from': f'repo_docs/{doc_key}',
                        '_to': f'repo_docs/{target_key}',
                        'type': 'references',
                        'weight': 0.8,
                        'function': func_name,
                        'source_file': doc['path']
                    })

    # Deduplicate reference edges
    seen = set()
    unique_refs = []
    for edge in reference_edges:
        key = (edge['_from'], edge['_to'], edge['function'])
        if key not in seen:
            seen.add(key)
            unique_refs.append(edge)
    reference_edges = unique_refs

    print(f"  Found {len(reference_edges)} reference edges")

    return import_edges, contains_edges, reference_edges, directories


def main():
    parser = argparse.ArgumentParser(description="Build graph edges from ingested documents")
    parser.add_argument("--write", action="store_true", help="Write edges to database")
    parser.add_argument("--dry-run", action="store_true", help="Preview edges without writing")
    args = parser.parse_args()

    if not args.write and not args.dry_run:
        parser.error("Must specify either --write or --dry-run")

    cfg = resolve_memory_config()
    client = ArangoMemoryClient(cfg)

    try:
        # Extract edges
        import_edges, contains_edges, reference_edges, directories = build_edges_from_symbols(
            client,
            dry_run=args.dry_run
        )

        if args.dry_run:
            print("\n[DRY RUN] Would create:")
            print(f"  - {len(directories)} directories")
            print(f"  - {len(import_edges)} import edges")
            print(f"  - {len(contains_edges)} contains edges")
            print(f"  - {len(reference_edges)} reference edges")
            print("\nSample import edge:")
            if import_edges:
                import pprint
                pprint.pprint(import_edges[0])
            print("\nSample contains edge:")
            if contains_edges:
                pprint.pprint(contains_edges[0])
            print("\nSample reference edge:")
            if reference_edges:
                pprint.pprint(reference_edges[0])
            return 0

        print("\n[Phase 5] Creating collections...")

        # Ensure collections exist
        collections_to_create = [
            CollectionDefinition(name="directories", type="document"),
            CollectionDefinition(name="code_edges", type="edge"),
            CollectionDefinition(name="contains_edges", type="edge"),
        ]

        for coll_def in collections_to_create:
            try:
                if coll_def.type == "edge":
                    client._write_client.request(
                        "POST",
                        f"/_db/{cfg.database}/_api/collection",
                        json={"name": coll_def.name, "type": 3}  # 3 = edge collection
                    )
                else:
                    client._write_client.request(
                        "POST",
                        f"/_db/{cfg.database}/_api/collection",
                        json={"name": coll_def.name, "type": 2}  # 2 = document collection
                    )
                print(f"  Created collection: {coll_def.name}")
            except Exception as e:
                if "409" in str(e) or "duplicate" in str(e).lower():
                    print(f"  Collection {coll_def.name} already exists")
                else:
                    raise

        print("\n[Phase 6] Writing directories...")
        dir_docs = list(directories.values())
        if dir_docs:
            client.bulk_import("directories", dir_docs, on_duplicate="update")
            print(f"  Wrote {len(dir_docs)} directories")

        print("\n[Phase 7] Writing import edges...")
        if import_edges:
            # Batch write edges (ArangoDB bulk import for edges)
            client.bulk_import("code_edges", import_edges, on_duplicate="ignore")
            print(f"  Wrote {len(import_edges)} import edges")

        print("\n[Phase 8] Writing contains edges...")
        if contains_edges:
            client.bulk_import("contains_edges", contains_edges, on_duplicate="ignore")
            print(f"  Wrote {len(contains_edges)} contains edges")

        print("\n[Phase 9] Writing reference edges...")
        if reference_edges:
            # Add reference edges to code_edges collection
            client.bulk_import("code_edges", reference_edges, on_duplicate="ignore")
            print(f"  Wrote {len(reference_edges)} reference edges")

        print("\n[SUCCESS] Graph edges created!")
        print(f"\nSummary:")
        print(f"  - {len(dir_docs)} directories")
        print(f"  - {len(import_edges)} import edges")
        print(f"  - {len(contains_edges)} contains edges")
        print(f"  - {len(reference_edges)} reference edges")
        print(f"  - Total edges: {len(import_edges) + len(contains_edges) + len(reference_edges)}")

        # Test query
        print("\n[Phase 10] Testing graph traversal...")
        test_query = """
            FOR doc IN repo_docs
                FILTER doc.path LIKE '%embedders_jina%'
                LIMIT 1
                LET imports = (
                    FOR v, e IN 1..1 OUTBOUND doc code_edges
                        FILTER e.type == 'imports'
                        RETURN v.path
                )
                RETURN {
                    file: doc.path,
                    imports: imports
                }
        """
        try:
            result = list(client.execute_query(test_query))
            if result:
                print("  Graph traversal working!")
                print(f"  Test result: {result[0]}")
            else:
                print("  No test results (this is OK if embedders_jina not in dataset)")
        except Exception as e:
            print(f"  Warning: Graph traversal test failed: {e}")

    finally:
        client.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
