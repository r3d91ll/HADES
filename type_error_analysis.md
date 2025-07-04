# Type Error Analysis and Remediation Tasks

## Summary
Total errors: 256 across 21 files

## Error Distribution by Category

### 1. Missing Type Definitions (49 errors)
- **RAG-related types**: RAGResult (13), RAGMode (10), RAGStrategyInput (9), PathInfo (6), RAGStrategyOutput (4)
- **Component types**: ComponentType (4), ComponentMetadata (3)
- **Location**: Primarily in `pathrag_rag_strategy.py`, `pathrag_engine.py`, `pathrag_graph_engine.py`

### 2. Attribute Errors (39 errors)
- Classes missing expected attributes/methods
- Most common in: `directory_aware_trainer.py`, `server.py`, `storage.py`

### 3. Missing Imports (34 errors)
- **External libraries without stubs**: arango (7), vllm (1), pathspec (1), docling (4)
- **Internal modules not found**: src.components.*, src.isne.models.*, src.isne.bootstrap.*
- **Already configured to ignore**: fastapi, torch_geometric, transformers, etc.

### 4. Missing Type Annotations (33 errors)
- Functions without return type annotations
- Untyped function definitions
- Most in: `server.py`, `pipeline.py`

### 5. Incorrect Function Arguments (28 errors)
- Unexpected keyword arguments
- Missing required arguments
- Type mismatches in function calls

## Proposed Task Groups

### Task Group 1: Core Type Definitions (Priority: HIGH)
**Files affected**: `src/types/pathrag/strategy.py` (to be created/updated)
**Errors to fix**: ~30-40 errors

1. Create missing RAG types:
   - RAGResult
   - RAGMode (enum)
   - RAGStrategyInput
   - RAGStrategyOutput
   - PathInfo

2. Update existing types that are missing attributes

### Task Group 2: Storage System Types (Priority: HIGH)
**Files affected**: `src/storage/arangodb/storage.py`, `src/types/storage/interfaces.py`
**Errors to fix**: ~30 errors

1. Fix StorageOutput/QueryInput/QueryOutput to include missing attributes
2. Fix method signature mismatches with protocols
3. Add missing attributes to ComponentMetadata

### Task Group 3: External Library Integration (Priority: MEDIUM)
**Files affected**: Multiple
**Errors to fix**: ~30 errors

1. Install missing packages or add type stubs:
   - python-arango (for arango imports)
   - pathspec
   - Add more ignore rules for libraries without stubs

2. Fix import paths for internal modules

### Task Group 4: ISNE Module Structure (Priority: MEDIUM)
**Files affected**: `src/isne/training/*`
**Errors to fix**: ~50 errors

1. Fix DirectoryMetadata missing attributes
2. Fix BatchSample constructor arguments
3. Add missing model imports or create stub modules

### Task Group 5: API and Server Types (Priority: MEDIUM)
**Files affected**: `src/api/server.py`, `src/api/pathrag_engine.py`
**Errors to fix**: ~30 errors

1. Add return type annotations to FastAPI endpoints
2. Fix method names (initialize vs is_initialized)
3. Add missing process_directory/process_file methods

### Task Group 6: Jina Integration (Priority: LOW)
**Files affected**: `src/jina_v4/*`
**Errors to fix**: ~15 errors

1. Add type annotations for class attributes
2. Fix return statements
3. Handle optional imports gracefully

### Task Group 7: Minor Fixes (Priority: LOW)
**Files affected**: Various utility files
**Errors to fix**: ~20 errors

1. Fix unreachable code
2. Fix variable redefinitions
3. Add missing type annotations for collections

## Recommended Execution Order

1. **Start with Task Group 1**: Creating core type definitions will resolve many errors across multiple files
2. **Then Task Group 2**: Fix storage types to resolve protocol compliance issues
3. **Task Group 3**: Install missing dependencies and fix imports
4. **Task Groups 4-7**: Can be done in parallel or as needed

## Quick Wins
- Add `python-arango` to dependencies
- Create `src/types/pathrag/strategy.py` with RAG types
- Update mypy.ini to ignore more third-party libraries without stubs