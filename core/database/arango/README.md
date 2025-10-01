# ArangoDB Optimized HTTP/2 Client

## Benchmarks

| Operation                                   | PHP Subprocess | HTTP/2 (direct) | HTTP/2 via proxies |
|---------------------------------------------|----------------|-----------------|--------------------|
| GET single doc (hot cache, p50)             | ~100 ms        | ~0.4 ms         | ~0.6 ms            |
| GET single doc (hot cache, p95 target)      | n/a            | 1.0 ms          | 1.0 ms             |
| Insert 1000 docs (waitForSync=false, p50)   | ~400–500 ms    | ~6 ms           | ~7 ms              |
| Query (LIMIT 1000, batch size 1000, p50)    | ~200 ms        | ~0.7 ms         | ~0.8 ms            |

## Usage

### Client

```python
from core.database.arango.optimized_client import ArangoHttp2Client, ArangoHttp2Config

config = ArangoHttp2Config(
    database="arxiv_repository",
    socket_path="/run/hades/readonly/arangod.sock",
    username="arxiv_reader",
    password="...",
)
with ArangoHttp2Client(config) as client:
    doc = client.get_document("arxiv_metadata", "0704_0003")
    print(doc)
```

### Workflow Integration

```python
from core.database.database_factory import DatabaseFactory

memory_client = DatabaseFactory.get_arango_memory_service()
try:
    documents = memory_client.execute_query(
        "FOR doc IN @@collection LIMIT 5 RETURN doc",
        {"@collection": "arxiv_metadata"},
    )
finally:
    memory_client.close()
```

### Proxy Binaries

1. Build: `cd core/database/arango/proxies && go build ./...`
2. Run RO proxy: `go run ./cmd/roproxy`
3. Run RW proxy: `go run ./cmd/rwproxy`

Sockets default to `/run/hades/readonly/arangod.sock` and `/run/hades/readwrite/arangod.sock` (systemd-managed). Ensure permissions (0640/0600) and adjust via env vars `LISTEN_SOCKET`, `UPSTREAM_SOCKET`.

### Benchmark CLI (Phase 4)

`tests/benchmarks/arango_connection_test.py` now supports:

- TTFB and E2E timing (full body consumption).
- Cache-busting via multiple `--key` values or varying bind variables.
- Adjustable payload size (`--doc-bytes`), `waitForSync`, and concurrency (`--concurrency`).
- JSON report emission (`--report-json`) for regression tracking.

Example:

```
poetry run python tests/benchmarks/arango_connection_test.py \
    --socket /run/hades/readonly/arangod.sock \
    --database arxiv_repository \
    --collection arxiv_metadata \
    --key 0704_0001 --key 0704_0002 \
    --iterations 20 --concurrency 4 \
    --report-json reports/get_hot.json
```

### Testing Infrastructure

- The HTTP/2 memory client is now the default access path for automated tests.
- Run `poetry run pytest tests/core/database/test_memory_client_config.py` for a quick sanity check.
- Future regression suites should share the proxy-aware fixtures so workflows exercise the same transport stack.

### Production Hardening Notes

- Treat the RO (`/run/hades/readonly/arangod.sock`) and RW (`/run/hades/readwrite/arangod.sock`) proxies as the security boundary.
- Run them as systemd services so socket permissions persist across reboots.

#### Systemd Services (reboot‑survivable perms)

Unit templates live in `infra/systemd/`:

- `infra/systemd/hades-roproxy.service` → RO proxy with group‑connect mode 0660.
- `infra/systemd/hades-rwproxy.service` → RW proxy running as `arangodb:hades`; group‑connect mode 0660 (dev). For prod, change group to `arangodb` and set mode 0600.
- `infra/systemd/tmpfiles.d/hades-proxy.conf` → creates `/run/hades/{readonly,readwrite}` at boot (readwrite dir 0770 so group can traverse).

Install and enable:

```
# Install binaries
sudo install -m 0755 core/database/arango/proxies/bin/roproxy /usr/local/bin/hades-roproxy
sudo install -m 0755 core/database/arango/proxies/bin/rwproxy /usr/local/bin/hades-rwproxy

# Install unit files
sudo install -m 0644 infra/systemd/hades-roproxy.service /etc/systemd/system/
sudo install -m 0644 infra/systemd/hades-rwproxy.service /etc/systemd/system/

# Ensure runtime dirs on boot
sudo install -m 0644 infra/systemd/tmpfiles.d/hades-proxy.conf /etc/tmpfiles.d/
sudo systemd-tmpfiles --create

# Optional: environment overrides (sockets)
echo 'LISTEN_SOCKET=/run/hades/readonly/arangod.sock' | sudo tee -a /etc/default/hades-arango-proxy
echo 'UPSTREAM_SOCKET=/run/arangodb3/arangodb.sock' | sudo tee -a /etc/default/hades-arango-proxy

sudo systemctl daemon-reload
sudo systemctl enable --now hades-roproxy hades-rwproxy
```

Verification (permissions and latency):

```
ls -al /run/hades/readonly/arangod.sock   # srw-rw---- arangodb:hades (0660)
ls -al /run/hades/readwrite/arangod.sock  # srw-rw---- arangodb:hades (0660 dev) or srw------- arangodb:arangodb (0600 prod)

poetry run python tests/benchmarks/uds_microbench.py \
  --upstream /run/arangodb3/arangodb.sock \
  --ro /run/hades/readonly/arangod.sock \
  --rw /run/hades/readwrite/arangod.sock \
  --database _system --iters 1000
```

The RO proxy binary also forces 0660 on its socket at runtime; the systemd service makes this **persistent across reboots** even if defaults differ on the host.
- Arango HTTP responses are enforced to negotiate HTTP/2; mismatches raise immediately.
- Reference benchmark summary: see `docs/benchmarks/arango_phase4_summary.md` for the latest latency table.
- Systemd templates for the proxies live in `docs/deploy/arango_proxy_systemd.md`.

### Bootstrap & Admin Access (Temporary Note)

- The RW proxy intentionally blocks admin endpoints such as `/_api/database`, `/_api/view`, and `/_api/analyzer`. This is by design: the runtime should not be able to create/alter databases or global resources.
- For initial provisioning (creating the `hades_memories` database, analyzers, and views), point tools at the upstream admin socket via the `ARANGO_ADMIN_SOCKET` env var (e.g., `/run/arangodb3/arangodb.sock`).
- Avoid changing socket permissions globally. Using `chmod 775 /run/arangodb3/arangodb.sock` is an expedient but not a long‑term or safe solution. Prefer:
  - A dedicated systemd socket for admin bootstrap with mode `0660` and a limited group (e.g., `hades-admin`).
  - Add the operator user to that group only for the bootstrap window, or use a one‑shot `setfacl`.
  - Keep the RW proxy locked down for regular operation.

#### Recommended Provisioning Flow (Manual)

Rationale
- Least privilege: agents should not modify cluster‑level resources.
- Separation of duties: DB/view/analyzer creation via ops/IaC; app uses RO/RW sockets.
- Auditability: all admin operations are deliberate and logged.

Steps
1) Pick the admin Unix socket and auth
   - Socket: `/run/arangodb3/arangodb.sock` (or `/var/run/arangodb3/arangodb.sock`).
   - If auth disabled: `export ARANGO_SKIP_AUTH=1`.
   - If auth enabled: `export ARANGO_USERNAME=root` and `export ARANGO_PASSWORD='…'`.

2) Create the database (one‑time)
   - curl (UDS):
     - `curl --unix-socket $ARANGO_ADMIN_SOCKET -u "$ARANGO_USERNAME:$ARANGO_PASSWORD" -X POST http://arangodb/_api/database -H 'Content-Type: application/json' -d '{"name":"hades_memories"}'`
   - or arangosh:
     - `arangosh --server.endpoint unix://$ARANGO_ADMIN_SOCKET --server.username "$ARANGO_USERNAME" --server.password "$ARANGO_PASSWORD" --javascript.execute "db._createDatabase('hades_memories')"`

3) Analyzer and View (choose one)
   - Reuse existing `text_en` analyzer (preferred if cluster already defines it)
     - No action needed to create; link views to `text_en`.
   - Or create a project‑scoped analyzer `hades_text_en` (avoids name collisions):
     - `curl --unix-socket $ARANGO_ADMIN_SOCKET -u "$ARANGO_USERNAME:$ARANGO_PASSWORD" -X POST http://arangodb/_api/analyzer -H 'Content-Type: application/json' -d '{"name":"hades_text_en","type":"text","properties":{"locale":"en","case":"lower","accent":false,"stemming":true},"features":["frequency","position","norm"]}'`
   - Create/Update ArangoSearch view to link `doc_chunks.text` to your analyzer:
     - Create: `curl --unix-socket $ARANGO_ADMIN_SOCKET -u "$ARANGO_USERNAME:$ARANGO_PASSWORD" -X POST http://arangodb/_db/hades_memories/_api/view -H 'Content-Type: application/json' -d '{"name":"repo_text","type":"arangosearch","links":{"doc_chunks":{"fields":{"text":{"analyzers":["text_en"]}}}}}'`
     - Update props (idempotent): `curl --unix-socket $ARANGO_ADMIN_SOCKET -u "$ARANGO_USERNAME:$ARANGO_PASSWORD" -X PUT http://arangodb/_db/hades_memories/_api/view/repo_text/properties -H 'Content-Type: application/json' -d '{"links":{"doc_chunks":{"fields":{"text":{"analyzers":["text_en"]}}}}}'`
     - Replace `text_en` with `hades_text_en` if you created a namespaced analyzer.

4) Collections and indexes
   - Run the bootstrap to create collections/edges and indexes (uses RW proxy for doc/edge ops; admin for view/analyzer if configured):
     - `ARANGO_ADMIN_SOCKET=/run/arangodb3/arangodb.sock poetry run python scripts/bootstrap_repo_graph.py --root . --write`
   - The script is idempotent for collections and indexes.

5) Runtime RBAC (optional but recommended)
   - Create service accounts bound only to `hades_memories` (e.g., `mem_ro` and `mem_rw`).
   - Configure RW proxy/systemd to use these accounts, and set runtime `.env` (`ARANGO_USERNAME/ARANGO_PASSWORD`).

Verification
- Version check (UDS): `curl --unix-socket /run/hades/readwrite/arangod.sock http://arangodb/_api/version`
- Python quick probe:
  ```python
  from core.database.arango.memory_client import resolve_memory_config, ArangoMemoryClient
  c=ArangoMemoryClient(resolve_memory_config());
  print('files:', c.execute_query('FOR d IN files LIMIT 1 RETURN d'))
  c.close()
  ```

### Analyzer Name Collision (Heads‑Up)

- During bootstrap, creating the `text_en` analyzer can fail with HTTP 400 if an analyzer named `text_en` already exists with different properties (e.g., existing analyzer has `features: [frequency, position, norm]` and `locale: "en"`, while our script attempts `locale: "en.utf-8"` without features).
- Current behavior: the bootstrap script will error on this mismatch.
- Workarounds until we harden the logic:
  - Prefer reusing the existing `text_en` analyzer by aligning our view to it (no creation). If you control the cluster defaults, standardize on one `text_en` definition.
  - Alternatively, create a project‑scoped analyzer name (e.g., `hades_text_en`) and update the bootstrap to link views to that name.
  - If you must proceed quickly, pre‑create the analyzer with the desired properties or adjust `core/database/arango/admin.py::ensure_text_analyzer` to treat a 400 name collision as “exists, keep going”.
- Follow‑up: we will update the bootstrap to (a) attempt a `GET /_api/analyzer/{name}` and compare properties, (b) log a clear diff, and (c) default to reusing the existing analyzer or a namespaced fallback (e.g., `hades_text_en`).
