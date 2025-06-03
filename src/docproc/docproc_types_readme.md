# Document Processing Module Type Fixes

This document tracks the type error fixes for the HADES document processing module.

## Current Issues

The module has several type errors identified by mypy:

```shell
src/docproc/adapters/markdown_adapter.py:272: error: Incompatible types in assignment (expression has type "EntityExtractor", variable has type "MarkdownEntityExtractor")
src/docproc/adapters/markdown_adapter.py:275: error: Incompatible types in assignment (expression has type "EntityExtractor", variable has type "MarkdownEntityExtractor")
src/docproc/adapters/markdown_adapter.py:312: error: Unsupported target for indexed assignment ("Collection[Collection[str]]")
src/docproc/schemas/utils.py:105: error: Returning Any from function declared to return "dict[str, Any]"
src/docproc/adapters/yaml_adapter.py:133: error: Returning Any from function declared to return "dict[str, Any]"
src/docproc/adapters/yaml_adapter.py:211: error: Incompatible types in assignment (expression has type "list[dict[str, Any]]", target has type "Collection[str]")
src/docproc/adapters/yaml_adapter.py:214: error: Unsupported target for indexed assignment ("Collection[str]")
src/docproc/adapters/yaml_adapter.py:215: error: Unsupported target for indexed assignment ("Collection[str]")
src/docproc/adapters/yaml_adapter.py:309: error: Argument 1 to "update" of "MutableMapping" has incompatible type "dict[str, YAMLNodeInfo]"; expected "SupportsKeysAndGetItem[str, dict[str, Any]]"
src/docproc/adapters/yaml_adapter.py:355: error: Argument 1 to "update" of "MutableMapping" has incompatible type "dict[str, YAMLNodeInfo]"; expected "SupportsKeysAndGetItem[str, dict[str, Any]]"
src/docproc/adapters/json_adapter.py:99: error: Incompatible types in assignment (expression has type "list[dict[str, Any]]", target has type "Collection[str]")
src/docproc/adapters/json_adapter.py:102: error: Unsupported target for indexed assignment ("Collection[str]")
src/docproc/adapters/json_adapter.py:103: error: Unsupported target for indexed assignment ("Collection[str]")
src/docproc/adapters/json_adapter.py:246: error: Argument 1 to "update" of "MutableMapping" has incompatible type "dict[str, JSONNodeInfo]"; expected "SupportsKeysAndGetItem[str, dict[str, Any]]"
src/docproc/adapters/json_adapter.py:292: error: Argument 1 to "update" of "MutableMapping" has incompatible type "dict[str, JSONNodeInfo]"; expected "SupportsKeysAndGetItem[str, dict[str, Any]]"
```

## Fix Strategy

1. Fix base classes and interfaces first
2. Fix entity_extractor type annotations (✅ completed)
3. Add proper type casts to dictionary update operations
4. Fix Collection assignment and indexing issues
5. Add proper return type handling for functions

## Progress Tracking

- [x] Fixed entity_extractor type annotations in markdown_adapter.py
- [ ] Fix dictionary update operations in yaml_adapter.py
- [ ] Fix dictionary update operations in json_adapter.py
- [ ] Fix Collection assignments and indexing
- [ ] Fix return type issues

## Testing

After each set of fixes, run mypy to verify:

```shell
mypy src/docproc/
```
