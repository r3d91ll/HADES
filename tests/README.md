# tests/ — Smoke, Benchmarks

Parent: `../README.md`

Children
- benchmarks/uds_microbench.py — UDS p50/p95/p99 for version and tiny cursor.

Usage
```
poetry run python tests/benchmarks/uds_microbench.py \
  --upstream /run/arangodb3/arangodb.sock \
  --ro /run/hades/readonly/arangod.sock \
  --rw /run/hades/readwrite/arangod.sock \
  --database _system --iters 1000
```
