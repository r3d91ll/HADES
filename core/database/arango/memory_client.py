"""
Module: core/database/arango/memory_client.py
Summary: High-level ArangoDB client on HTTP/2 transport (gRPC-like surface).
Owners: @todd, @hades-runtime
Last-Updated: 2025-09-30
Inputs: ARANGO_* env (RO/RW sockets, auth, timeouts), base_url
Outputs: Document CRUD, bulk import, AQL queries; creates collections/indexes
Data-Contracts: collections {repo_docs, doc_chunks}, indexes (persistent, vector)
Related: core/database/arango/optimized_client.py, core/database/arango/README.md
Stability: stable; Security: RO for reads, RW for writes; admin endpoints blocked at proxy
Boundary: C_ext=local UDS; P_ij=DB-scoped endpoints and JSON payload limits respected

Implements Phase 3 of Issue #51 by integrating the optimized HTTP/2 client
with application workflows. Provides an interface compatible with the previous
gRPC client so existing code requires minimal changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import grpc
from grpc import StatusCode

from .optimized_client import ArangoHttp2Client, ArangoHttp2Config, ArangoHttpError

DEFAULT_ARANGO_SOCKET = "/run/arangodb3/arangodb.sock"

HTTP_TO_GRPC_STATUS: dict[int, StatusCode] = {
    400: StatusCode.INVALID_ARGUMENT,
    401: StatusCode.UNAUTHENTICATED,
    403: StatusCode.PERMISSION_DENIED,
    404: StatusCode.NOT_FOUND,
    408: StatusCode.DEADLINE_EXCEEDED,
    409: StatusCode.ALREADY_EXISTS,
    412: StatusCode.FAILED_PRECONDITION,
    413: StatusCode.RESOURCE_EXHAUSTED,
    425: StatusCode.FAILED_PRECONDITION,
    429: StatusCode.RESOURCE_EXHAUSTED,
    500: StatusCode.INTERNAL,
    503: StatusCode.UNAVAILABLE,
}


class MemoryServiceError(grpc.RpcError):
    """gRPC-compatible error raised by the memory client."""

    def __init__(self, message: str, *, status: StatusCode = StatusCode.UNKNOWN, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self._message = message
        self._status = status
        self._details = details or {}

    # grpc.RpcError API -------------------------------------------------
    def code(self) -> StatusCode:  # pragma: no cover - trivial delegation
        return self._status

    def details(self) -> str:  # pragma: no cover - trivial delegation
        return self._message

    def trailing_metadata(self):  # pragma: no cover - compatibility stub
        return None

    def debug_error_string(self) -> str:  # pragma: no cover - compatibility stub
        return self._message


@dataclass(slots=True)
class ArangoMemoryClientConfig:
    """Configuration values resolved for the memory client."""

    database: str
    username: str
    password: str
    base_url: str
    read_socket: str
    write_socket: str
    connect_timeout: float
    read_timeout: float
    write_timeout: float

    def build_read_config(self) -> ArangoHttp2Config:
        return ArangoHttp2Config(
            database=self.database,
            socket_path=self.read_socket,
            base_url=self.base_url,
            username=self.username,
            password=self.password,
            connect_timeout=self.connect_timeout,
            read_timeout=self.read_timeout,
            write_timeout=self.write_timeout,
        )

    def build_write_config(self) -> ArangoHttp2Config:
        return ArangoHttp2Config(
            database=self.database,
            socket_path=self.write_socket,
            base_url=self.base_url,
            username=self.username,
            password=self.password,
            connect_timeout=self.connect_timeout,
            read_timeout=self.read_timeout,
            write_timeout=self.write_timeout,
        )


@dataclass(slots=True)
class CollectionDefinition:
    """Collection metadata used when creating collections."""

    name: str
    type: str = "document"
    options: dict[str, Any] | None = None
    indexes: Sequence[dict[str, Any]] | None = None


def _http_status_to_grpc(status_code: int) -> StatusCode:
    return HTTP_TO_GRPC_STATUS.get(status_code, StatusCode.UNKNOWN)


def _parse_timeout(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def resolve_memory_config(
    *,
    database: str = "hades_memories",
    username: str | None = None,
    password: str | None = None,
    socket_path: str | None = None,
    read_socket: str | None = None,
    write_socket: str | None = None,
    use_proxies: bool | None = None,
    base_url: str | None = None,
    connect_timeout: float | None = None,
    read_timeout: float | None = None,
    write_timeout: float | None = None,
) -> ArangoMemoryClientConfig:
    """Resolve configuration using explicit parameters and environment values."""

    env = os.environ

    # Username defaults to env or 'root'
    if username is None:
        username = env.get("ARANGO_USERNAME", "root")

    # Allow skipping auth entirely for local dev when server auth is disabled
    skip_auth = (env.get("ARANGO_SKIP_AUTH", "").strip().lower() in {"1", "true", "yes", "on"})

    if not skip_auth and password is None:
        password = env.get("ARANGO_PASSWORD")
        if password is None or password == "":
            raise ValueError(
                "ArangoDB password required (set ARANGO_PASSWORD) or export ARANGO_SKIP_AUTH=1 for dev"
            )

    if base_url is None:
        base_url = env.get("ARANGO_HTTP_BASE_URL", "http://localhost")

    # Allow explicit sockets to override environment
    if socket_path:
        read_socket = read_socket or socket_path
        write_socket = write_socket or socket_path

    env_ro = env.get("ARANGO_RO_SOCKET")
    env_rw = env.get("ARANGO_RW_SOCKET")
    env_direct = env.get("ARANGO_SOCKET")

    proxies_requested = True if use_proxies is None else use_proxies

    if proxies_requested:
        default_ro = "/run/hades/readonly/arangod.sock"
        default_rw = "/run/hades/readwrite/arangod.sock"

        read_socket = read_socket or env_ro or default_ro
        write_socket = write_socket or env_rw or default_rw

        # If only one socket could be resolved, mirror it so we still have a
        # functioning configuration instead of passing ``None`` into httpx.
        if read_socket is None:
            read_socket = write_socket or default_ro
        if write_socket is None:
            write_socket = read_socket or default_rw
    else:
        direct_socket = socket_path or env_direct or DEFAULT_ARANGO_SOCKET
        read_socket = read_socket or direct_socket
        write_socket = write_socket or direct_socket

    connect_timeout = connect_timeout if connect_timeout is not None else _parse_timeout(env.get("ARANGO_CONNECT_TIMEOUT"), 5.0)
    read_timeout = read_timeout if read_timeout is not None else _parse_timeout(env.get("ARANGO_READ_TIMEOUT"), 30.0)
    write_timeout = write_timeout if write_timeout is not None else _parse_timeout(env.get("ARANGO_WRITE_TIMEOUT"), 30.0)

    return ArangoMemoryClientConfig(
        database=database,
        username=username if not skip_auth else "",
        password=password if not skip_auth else "",
        base_url=base_url,
        read_socket=read_socket,
        write_socket=write_socket,
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        write_timeout=write_timeout,
    )


class ArangoMemoryClient:
    """High-level client matching the previous MemoryService gRPC surface."""

    def __init__(self, config: ArangoMemoryClientConfig) -> None:
        self._config = config
        self._read_client = ArangoHttp2Client(config.build_read_config())
        if config.read_socket == config.write_socket:
            self._write_client = self._read_client
            self._shared_clients = True
        else:
            self._write_client = ArangoHttp2Client(config.build_write_config())
            self._shared_clients = False
        self._closed = False

    # Context management ------------------------------------------------
    def close(self) -> None:
        if self._closed:
            return
        self._read_client.close()
        if not self._shared_clients:
            self._write_client.close()
        self._closed = True

    def __enter__(self) -> "ArangoMemoryClient":  # pragma: no cover - helper
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - helper
        self.close()

    # Public API --------------------------------------------------------
    def execute_query(
        self,
        aql: str,
        bind_vars: dict[str, Any] | None = None,
        *,
        batch_size: int | None = None,
        full_count: bool = False,
    ) -> list[dict[str, Any]]:
        try:
            return self._read_client.query(
                aql,
                bind_vars=bind_vars,
                batch_size=batch_size or 1000,
                full_count=full_count,
            )
        except ArangoHttpError as exc:  # pragma: no cover - thin wrapper
            raise self._wrap_error(exc) from exc

    def bulk_insert(
        self,
        collection: str,
        documents: Iterable[dict[str, Any]],
        *,
        chunk_size: int = 1000,
    ) -> int:
        total_inserted = 0
        batch: list[dict[str, Any]] = []

        def flush() -> int:
            if not batch:
                return 0
            try:
                response = self._write_client.insert_documents(collection, batch, on_duplicate="ignore")
            except ArangoHttpError as exc:  # pragma: no cover - thin wrapper
                raise self._wrap_error(exc) from exc
            created = response.get("created")
            return int(created) if isinstance(created, int) else len(batch)

        for doc in documents:
            batch.append(doc)
            if len(batch) >= chunk_size:
                total_inserted += flush()
                batch.clear()

        if batch:
            total_inserted += flush()
            batch.clear()

        return total_inserted

    def bulk_import(
        self,
        collection: str,
        documents: Iterable[dict[str, Any]],
        *,
        chunk_size: int = 1000,
        on_duplicate: str = "ignore",
    ) -> int:
        """Bulk import with configurable duplicate policy.

        on_duplicate: one of 'ignore', 'update', 'replace', 'error'.
        """
        total_inserted = 0
        batch: list[dict[str, Any]] = []

        def flush() -> int:
            if not batch:
                return 0
            try:
                response = self._write_client.insert_documents(collection, batch, on_duplicate=on_duplicate)
            except ArangoHttpError as exc:  # pragma: no cover - thin wrapper
                raise self._wrap_error(exc) from exc
            # import API reports created/updated/ignored; treat any mutation as progress
            created = int(response.get("created", 0))
            updated = int(response.get("updated", 0))
            return created + updated

        for doc in documents:
            batch.append(doc)
            if len(batch) >= chunk_size:
                total_inserted += flush()
                batch.clear()

        if batch:
            total_inserted += flush()
            batch.clear()

        return total_inserted

    def get_document(self, collection: str, key: str) -> dict[str, Any]:
        try:
            return self._read_client.get_document(collection, key)
        except ArangoHttpError as exc:  # pragma: no cover - thin wrapper
            raise self._wrap_error(exc) from exc

    def execute_transaction(
        self,
        *,
        write: Sequence[str],
        read: Sequence[str] | None = None,
        exclusive: Sequence[str] | None = None,
        action: str,
        params: dict[str, Any] | None = None,
        wait_for_sync: bool | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "collections": {
                "write": list(write),
            },
            "action": action,
            "params": params or {},
        }

        if read:
            payload["collections"]["read"] = list(read)
        if exclusive:
            payload["collections"]["exclusive"] = list(exclusive)
        if wait_for_sync is not None:
            payload["waitForSync"] = bool(wait_for_sync)

        path = f"/_db/{self._config.database}/_api/transaction"
        try:
            return self._write_client.request("POST", path, json=payload)
        except ArangoHttpError as exc:  # pragma: no cover - thin wrapper
            raise self._wrap_error(exc) from exc

    def drop_collections(self, names: Iterable[str], *, ignore_missing: bool = True) -> None:
        for name in names:
            path = f"/_db/{self._config.database}/_api/collection/{name}"
            try:
                self._write_client.request("DELETE", path)
            except ArangoHttpError as exc:
                if ignore_missing and exc.status_code == 404:
                    continue
                raise self._wrap_error(exc) from exc

    def create_collections(self, definitions: Iterable[CollectionDefinition]) -> None:
        created: list[str] = []
        try:
            for definition in definitions:
                options = dict(definition.options or {})
                collection_type = 3 if definition.type.lower() == "edge" else 2
                options.setdefault("type", collection_type)
                options.setdefault("name", definition.name)
                path = f"/_db/{self._config.database}/_api/collection"

                try:
                    self._write_client.request("POST", path, json=options)
                except ArangoHttpError as exc:
                    if exc.status_code != 409:
                        raise
                else:
                    created.append(definition.name)

                if definition.indexes:
                    for index in definition.indexes:
                        index_path = f"/_db/{self._config.database}/_api/index"
                        params = {"collection": definition.name}
                        try:
                            self._write_client.request("POST", index_path, json=index, params=params)
                        except ArangoHttpError as exc:
                            if exc.status_code != 409:
                                raise
        except ArangoHttpError as exc:
            self._rollback_created(created)
            raise self._wrap_error(exc) from exc
        except Exception:
            self._rollback_created(created)
            raise

    def _rollback_created(self, created: Sequence[str]) -> None:
        """Attempt to drop collections created earlier in this call."""
        for name in reversed(created):
            path = f"/_db/{self._config.database}/_api/collection/{name}"
            try:
                self._write_client.request("DELETE", path)
            except ArangoHttpError:
                # Suppress rollback errors; best-effort cleanup
                continue

    # Internal helpers --------------------------------------------------
    def _wrap_error(self, error: ArangoHttpError) -> MemoryServiceError:
        status = _http_status_to_grpc(error.status_code)
        message = error.details.get("errorMessage") or error.details.get("message") or str(error)
        return MemoryServiceError(message, status=status, details=error.details)


__all__ = [
    "ArangoMemoryClient",
    "ArangoMemoryClientConfig",
    "CollectionDefinition",
    "MemoryServiceError",
    "resolve_memory_config",
]
