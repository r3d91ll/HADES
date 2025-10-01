"""
Module: core/database/arango/optimized_client.py
Summary: HTTP/2 optimized ArangoDB client over Unix sockets (get, import, query).
Owners: @todd, @hades-runtime
Last-Updated: 2025-09-30
Inputs: ARANGO_* env (username/password, timeouts); UDS path via config
Outputs: DB-scoped REST calls (`/_db/{db}/_api/{document,index,import,cursor}`)
Data-Contracts: NDJSON import payload; responses with {created, updated, ignored}
Related: core/database/arango/README.md, Docs/PRD_Sockets_and_Proxy_Perf.md
Stability: stable; Security: honor RO/RW boundary; no admin endpoints here
Boundary: C_ext assumes local UDS latency; P_ij requires DB‑scoped endpoints only

Phase 1 deliverable for Issue #51. Minimal operations:
- get_document
- insert_documents (NDJSON bulk import)
- query (AQL cursor)

Additional features (pooling, streaming cursors) are layered later.
"""

from __future__ import annotations

import io
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import httpx


class ArangoHttpError(RuntimeError):
    """Raised when the ArangoDB HTTP API reports an error."""

    def __init__(self, status_code: int, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(f"ArangoDB HTTP {status_code}: {message}")
        self.status_code = status_code
        self.details = details or {}


@dataclass
class ArangoHttp2Config:
    """Configuration for the HTTP/2 client."""

    database: str = "_system"
    socket_path: str = "/tmp/arangodb.sock"
    base_url: str = "http://localhost"
    username: str | None = None
    password: str | None = None
    connect_timeout: float = 5.0
    read_timeout: float = 30.0
    write_timeout: float = 30.0
    pool_limits: httpx.Limits | None = None


class ArangoHttp2Client:
    """Minimal HTTP/2 ArangoDB client speaking over Unix sockets."""

    def __init__(self, config: ArangoHttp2Config) -> None:
        self._config = config
        transport = httpx.HTTPTransport(
            uds=config.socket_path,
            retries=0,
        )
        timeout = httpx.Timeout(
            connect=config.connect_timeout,
            read=config.read_timeout,
            write=config.write_timeout,
            pool=config.connect_timeout,
        )
        auth = None
        if config.username and config.password:
            auth = (config.username, config.password)

        self._client = httpx.Client(
            http2=True,
            base_url=config.base_url,
            transport=transport,
            timeout=timeout,
            limits=config.pool_limits,
            auth=auth,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> ArangoHttp2Client:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - context helper
        self.close()

    @property
    def config(self) -> ArangoHttp2Config:
        """Expose the underlying configuration (database name, socket, etc.)."""

        return self._config

    def get_document(self, collection: str, key: str) -> dict[str, Any]:
        """Fetch a single document by collection/key."""

        path = f"/_db/{self._config.database}/_api/document/{collection}/{key}"
        response = self._client.get(path)
        return self._handle_response(response)

    def insert_documents(
        self,
        collection: str,
        documents: Iterable[dict[str, Any]],
        *,
        on_duplicate: str = "ignore",  # ignore|update|replace|error
    ) -> dict[str, Any]:
        """Bulk insert documents using NDJSON import.

        on_duplicate controls how duplicate _key values are handled. Default 'ignore'
        makes ingestion idempotent for stable keys.
        """

        document_iter = iter(documents)
        try:
            first_doc = next(document_iter)
        except StopIteration:
            return {"created": 0}

        buffer = io.BytesIO()
        first_line = json.dumps(first_doc, separators=(",", ":")).encode("utf-8")
        buffer.write(first_line)

        for doc in document_iter:
            line = json.dumps(doc, separators=(",", ":")).encode("utf-8")
            buffer.write(b"\n")
            buffer.write(line)

        payload = buffer.getvalue()
        path = (
            f"/_db/{self._config.database}/_api/import"
            f"?collection={collection}&type=documents&complete=true&overwrite=false&onDuplicate={on_duplicate}"
        )
        user_agent = os.environ.get("HADES_HTTP_USER_AGENT", "hades-arango-http2/1.0")
        trace_id = os.environ.get("HADES_TRACE_ID")
        headers = {
            "Content-Type": "application/x-ndjson",
            "Content-Length": str(len(payload)),
            "User-Agent": user_agent,
        }
        if trace_id:
            headers["x-hades-trace"] = trace_id
        response = self._client.post(path, content=payload, headers=headers)
        return self._handle_response(response)

    def query(
        self,
        aql: str,
        bind_vars: dict[str, Any] | None = None,
        batch_size: int = 1000,
        full_count: bool = False,
    ) -> list[dict[str, Any]]:
        """Execute an AQL query and return the full result set."""

        payload = {
            "query": aql,
            "batchSize": batch_size,
            "bindVars": bind_vars or {},
            "options": {"fullCount": full_count},
        }
        path = f"/_db/{self._config.database}/_api/cursor"
        response = self._client.post(path, json=payload)
        data = self._handle_response(response)

        results = data.get("result", [])
        cursor_id = data.get("id")
        while data.get("hasMore") and cursor_id:
            follow_path = f"/_db/{self._config.database}/_api/cursor/{cursor_id}"
            data = self._handle_response(self._client.put(follow_path))
            results.extend(data.get("result", []))
            cursor_id = data.get("id")

        return results

    def request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        content: bytes | None = None,
    ) -> dict[str, Any]:
        """Perform an arbitrary HTTP request against the ArangoDB REST API."""

        response = self._client.request(
            method,
            path,
            json=json,
            params=params,
            headers=headers,
            content=content,
        )
        return self._handle_response(response)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _handle_response(self, response: httpx.Response) -> dict[str, Any]:
        if response.http_version not in {"HTTP/2", "HTTP/1.1"}:
            raise RuntimeError(
                f"Unexpected HTTP version {response.http_version!r} "
                f"for {response.request.method} {response.request.url}"
            )

        if response.status_code >= 400:
            try:
                details = response.json()
            except ValueError:
                details = {"message": response.text}
            message = details.get("errorMessage") or details.get("message") or response.text
            raise ArangoHttpError(response.status_code, message, details)

        if response.status_code == 204:
            return {}

        if response.headers.get("content-type", "").startswith("application/json"):
            return response.json()

        # For NDJSON import Arango responds with JSON content type, but as a safeguard:
        try:
            return json.loads(response.text)
        except ValueError:
            return {"raw": response.text}


__all__ = [
    "ArangoHttp2Client",
    "ArangoHttp2Config",
    "ArangoHttpError",
]
