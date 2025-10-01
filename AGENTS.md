# Repository Guidelines

## Project Structure & Module Organization
- Root keeps `Docs/` for theory, specs, and agent notes; treat updated PRDs as the source of truth for scope.
- `core/` will host Python services implementing the Conveyance runtime, with subpackages such as `database/`, `embedders/`, and `workflows/` once scaffolding lands.
- ArangoDB proxies in Go live under `core/database/arango/proxies/cmd/{roproxy,rwproxy}`; keep read-only and read-write configs synchronized.
- Connect to ArangoDB via Unix sockets `/run/hades/readonly/arangod.sock` and `/run/hades/readwrite/arangod.sock`, targeting the `hades_memories` database.
- Store experiment artifacts in `benchmarks/` and raw inputs in `datasets/`; never check derived outputs into `Docs/`.

## Build, Test, and Development Commands
- `poetry install` — bootstrap the Python 3.12 environment and lock dependency versions.
- `poetry run pytest` — execute unit and integration suites; add `-k <pattern>` to target modules.
- `poetry run ruff check` / `poetry run ruff format` — lint and auto-format Python code to repository conventions.
- `go build ./core/database/arango/proxies/...` — compile both Arango proxy binaries before pushing changes.
- `make dev` (planned) — bring up local services; document any new targets you add to `Makefile`.

## Coding Style & Naming Conventions
- Python: 4-space indentation, strict type hints on public APIs, `snake_case` for functions, `CapWords` for classes. Prefer dataclasses for structured payloads and keep module names lowercase.
- Go: ensure files are gofmt-clean, exported identifiers use `PascalCase`, and proxy allowlists remain alphabetical. Keep package comments concise.
- Configuration: secrets stay in `.env` or Vault. YAML manifests/coordinator scripts belong in `infra/` once provisioning starts.

## Testing Guidelines
- Mirror the module tree under `tests/`, naming files `test_<topic>.py` and functions `test_<behavior>`.
- Target ≥80% coverage on new Python code (`poetry run pytest --cov=core`). Attach coverage reports to PR descriptions when they introduce major features.
- Mock Arango and LLM interfaces in unit tests; gate real-service checks behind `pytest.mark.integration` so CI can opt in explicitly.

## Commit & Pull Request Guidelines
- Commit subject lines remain imperative (`Add recall cache adapter`), ~72 characters, and append issue IDs such as `#12` when applicable.
- PRs must outline goal, major changes, validation evidence, and impact on Conveyance metrics (W, R, H, T, C_ext, P_ij). Upload benchmark traces under `benchmarks/reports/`.
- Request review from at least one Conveyance-focused agent and track follow-up items in a checklist.

## Security & Configuration Tips
- Enforce late chunking, sanitize logs, and rerun `setup/verify_storage.py` after touching socket or credential plumbing.
- Document updates to α-estimation, protocol compatibility scoring, or boundary definitions inside `Docs/` with versioned notes.
