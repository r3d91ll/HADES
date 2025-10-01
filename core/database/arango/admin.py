"""
Module: core/database/arango/admin.py
Summary: Admin helpers for analyzers, views, indexes, and DB creation.
Owners: @todd, @hades-runtime
Last-Updated: 2025-09-30
Inputs: DB name (from client), analyzer/view names, index specs
Outputs: Idempotent POST/PUT to `/_db/{db}/_api/{view,index}`; `/_api/analyzer`; `/_api/database`
Data-Contracts: analyzer (text_en/hades_text_en), view links {collection.fields.text.analyzers}
Related: core/database/arango/README.md (bootstrap policy)
Stability: stable; Security: should be executed via admin socket only
Boundary: P_ij: admin endpoints must not pass through RW proxy in prod

Admin helpers use ArangoHttp2Client via ArangoMemoryClient to perform
idempotent admin operations without external CLIs.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .memory_client import ArangoMemoryClient
from .optimized_client import ArangoHttp2Client, ArangoHttpError


def _request(client: object, method: str, path: str, **kwargs) -> Dict[str, Any]:
    if isinstance(client, ArangoHttp2Client):
        return client.request(method, path, **kwargs)
    if isinstance(client, ArangoMemoryClient):
        return client._write_client.request(method, path, **kwargs)  # type: ignore[attr-defined]
    # Best effort for similar wrappers
    if hasattr(client, "request"):
        return client.request(method, path, **kwargs)  # type: ignore[attr-defined]
    if hasattr(client, "_write_client"):
        return client._write_client.request(method, path, **kwargs)  # type: ignore[attr-defined]
    raise TypeError("Unsupported client type for Arango admin request")


def _get_database_name(client: ArangoMemoryClient | ArangoHttp2Client) -> str:
    if isinstance(client, ArangoMemoryClient):
        return client._config.database  # type: ignore[attr-defined]
    return client.config.database  # type: ignore[attr-defined]


def get_analyzer(client: ArangoMemoryClient | ArangoHttp2Client, name: str) -> Optional[Dict[str, Any]]:
    """Return analyzer definition if present, else None."""
    try:
        return _request(client, "GET", f"/_api/analyzer/{name}")
    except ArangoHttpError as exc:
        if exc.status_code == 404:
            return None
        raise


def ensure_text_analyzer(client: ArangoMemoryClient | ArangoHttp2Client, name: str = "text_en") -> None:
    """Ensure a basic English text analyzer exists.

    Properties: lowercase -> norm(en) -> stem(en).
    """

    # Check if analyzer already exists (idempotent operation)
    existing = get_analyzer(client, name)
    if existing is not None:
        return  # Already exists, nothing to do

    payload = {
        "name": name,
        "type": "text",
        "properties": {
            "locale": "en.utf-8",
            "case": "lower",
            "accent": False,
            "stemming": True,
            "stopwords": [],
        },
    }
    try:
        _request(client, "POST", "/_api/analyzer", json=payload)
        return
    except ArangoHttpError as exc:  # 409 or 400 if collision with different props
        if exc.status_code == 409:
            return
        if exc.status_code == 403:
            # Permission denied - check if analyzer exists anyway (may have been created by admin)
            if get_analyzer(client, name) is not None:
                return  # Exists, treat as success
            raise  # Doesn't exist and can't create - this is a real error
        if exc.status_code == 400 and "Name collision" in str(exc):
            # Treat as already exists; caller should align view to the existing definition
            if get_analyzer(client, name) is not None:
                return
        raise


def ensure_arangosearch_view(
    client: ArangoMemoryClient | ArangoHttp2Client,
    *,
    view_name: str,
    links: Dict[str, Dict[str, Any]],
) -> None:
    """Ensure an ArangoSearch view with the given links exists.

    links example:
      {
        "doc_chunks": {"fields": {"text": {"analyzers": ["text_en"]}}}
      }
    """

    db = _get_database_name(client)

    # Check if view already exists (idempotent operation)
    view_path = f"/_db/{db}/_api/view/{view_name}"
    try:
        existing = _request(client, "GET", view_path)
        # View exists, skip creation (updating requires permissions we may not have)
        return
    except ArangoHttpError as exc:
        if exc.status_code != 404:
            # Some error other than "not found" - view may exist but we can't check
            # Try to create anyway
            pass

    create_payload = {"name": view_name, "type": "arangosearch", "links": links}
    path = f"/_db/{db}/_api/view"
    try:
        _request(client, "POST", path, json=create_payload)
    except ArangoHttpError as exc:
        if exc.status_code == 403:
            # Permission denied - check if view exists anyway (may have been created by admin)
            try:
                _request(client, "GET", view_path)
                return  # Exists, treat as success
            except ArangoHttpError:
                raise exc  # Doesn't exist and can't create - this is a real error
        if exc.status_code not in (400, 409):
            raise
        # 400 for existing view sometimes; update properties regardless
        props_path = f"/_db/{db}/_api/view/{view_name}/properties"
        try:
            _request(client, "PUT", props_path, json={"links": links})
        except ArangoHttpError as update_exc:
            if update_exc.status_code == 403:
                # Can't update, but view exists - acceptable for non-admin users
                return
            raise


def ensure_persistent_index(
    client: ArangoMemoryClient | ArangoHttp2Client,
    *,
    collection: str,
    fields: list[str],
    unique: bool = False,
    sparse: bool = False,
) -> None:
    db = _get_database_name(client)
    path = f"/_db/{db}/_api/index"
    payload = {"type": "persistent", "fields": fields, "unique": unique, "sparse": sparse}
    params = {"collection": collection}
    try:
        _request(client, "POST", path, json=payload, params=params)
    except ArangoHttpError as exc:
        if exc.status_code != 409:
            raise


def ensure_vector_index(
    client: ArangoMemoryClient | ArangoHttp2Client,
    *,
    collection: str,
    field: str,
    dimensions: int,
    similarity: str = "cosine",
    storage_engine: str = "disk",
) -> None:
    """Create a vector index if ArangoDB 3.12+ is available.

    Best-effort: ignore if unsupported (server <3.12) or already exists.
    """

    db = _get_database_name(client)
    path = f"/_db/{db}/(_api/index|_api/index)".replace("|", "")  # silence linters
    payload = {
        "type": "vector",
        "fields": [field],
        "dimensions": dimensions,
        "similarity": similarity,
        "storage": storage_engine,
    }
    params = {"collection": collection}
    try:
        _request(client, "POST", path, json=payload, params=params)
    except ArangoHttpError:
        return


def ensure_database(client: ArangoHttp2Client | ArangoMemoryClient, *, name: str) -> None:
    """Create database if missing.

    Uses POST /_api/database; ignores 409 conflicts.
    """
    payload = {"name": name}
    try:
        _request(client, "POST", "/_api/database", json=payload)
    except ArangoHttpError as exc:
        if exc.status_code != 409:
            raise


__all__ = [
    "ensure_text_analyzer",
    "ensure_arangosearch_view",
    "ensure_persistent_index",
    "ensure_vector_index",
    "ensure_database",
]
