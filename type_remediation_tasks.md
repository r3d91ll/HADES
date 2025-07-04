# Type Error Remediation Tasks

## Overview
Total: 256 type errors across 21 files  
Estimated effort: 4-6 hours for complete remediation

## Task Breakdown by Priority

### 🔴 Priority 1: Core Type Definitions (Est. 1 hour)
**Impact**: Will resolve ~60 errors across multiple files

#### Task 1.1: Create RAG Strategy Types
- **File to create**: `src/types/pathrag/rag_types.py`
- **Types needed**:
  ```python
  class RAGMode(Enum)
  class RAGStrategyInput(BaseSchema)
  class RAGStrategyOutput(BaseSchema)
  class RAGResult(BaseSchema)
  class PathInfo(BaseSchema)
  ```
- **Affected files**: 
  - `pathrag_rag_strategy.py` (20+ errors)
  - `pathrag_engine.py` (9 errors)

#### Task 1.2: Import ComponentType/ComponentMetadata
- **Issue**: These exist in `src/types/common.py` but aren't imported
- **Fix**: Add imports to `pathrag_graph_engine.py`
- **Resolves**: 7 errors

### 🟠 Priority 2: Storage Type Fixes (Est. 1 hour)
**Impact**: Will resolve ~34 errors in storage system

#### Task 2.1: Update Storage Interface Types
- **File**: `src/types/storage/interfaces.py`
- **Add attributes to**:
  - QueryInput: `query_embedding`, `top_k`, `search_options`
  - StorageOutput: `document_id`, `documents`, `error_message`
  - ComponentMetadata: `processed_at`, `metrics`

#### Task 2.2: Fix Protocol Compliance
- **File**: `src/storage/arangodb/storage.py`
- **Issues**:
  - Return type mismatches (NodeID vs str)
  - Method signature mismatches
  - Optional parameter handling

### 🟡 Priority 3: Missing Dependencies (Est. 30 min)
**Impact**: Will resolve ~34 import errors

#### Task 3.1: Install Missing Packages
```bash
poetry add python-arango pathspec
poetry add --dev types-tqdm
```

#### Task 3.2: Create Missing Internal Modules
- Create stub files or proper modules for:
  - `src/components/registry.py`
  - `src/api/base.py`
  - `src/isne/models/` (multiple files)

### 🟢 Priority 4: Type Annotations (Est. 1 hour)
**Impact**: Will resolve ~33 errors

#### Task 4.1: FastAPI Endpoints
- **File**: `src/api/server.py`
- Add return type annotations to all endpoints
- Fix: 7 missing annotations

#### Task 4.2: ISNE Training Functions
- **Files**: `pipeline.py`, `directory_aware_trainer.py`
- Add return types and parameter types
- Fix: ~15 missing annotations

### 🔵 Priority 5: ISNE Type Fixes (Est. 1 hour)
**Impact**: Will resolve ~47 errors

#### Task 5.1: Fix DirectoryMetadata
- **Issue**: Missing attributes: `directory_path`, `sibling_count`, `is_leaf`
- **File**: Update type definition or fix usage

#### Task 5.2: Fix BatchSample Constructor
- **Issue**: Missing required arguments
- **Fix**: Update constructor calls with all required params

### ⚪ Priority 6: Minor Fixes (Est. 30 min)
**Impact**: Will resolve remaining ~30 errors

- Fix unreachable code
- Handle None types properly
- Fix method name mismatches
- Add collection type annotations

## Execution Plan

### Phase 1: Quick Wins (30 minutes)
1. Create `src/types/pathrag/rag_types.py` with all RAG types
2. Add missing imports for ComponentType/ComponentMetadata
3. Install python-arango and pathspec

### Phase 2: Core Fixes (2 hours)
1. Update storage interface types
2. Fix protocol compliance issues
3. Add missing type annotations to critical paths

### Phase 3: Comprehensive Fix (2 hours)
1. Fix ISNE types and training code
2. Create missing internal modules
3. Handle remaining edge cases

## Verification Commands
```bash
# After each phase, run:
poetry run mypy src/ --config-file mypy.ini 2>&1 | grep "Found"

# Check specific modules:
poetry run mypy src/pathrag/ --config-file mypy.ini
poetry run mypy src/storage/ --config-file mypy.ini
```

## Notes
- Some errors may cascade - fixing type definitions often resolves multiple related errors
- Consider relaxing mypy strictness temporarily if needed for gradual adoption
- Document any intentional type ignores with clear comments