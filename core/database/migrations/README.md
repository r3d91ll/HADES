This directory contains idempotent AQL migration scripts applied to the `hades_memories` database.

Order and naming
- Use zero-padded prefixes to define order, e.g., `0001_*`, `0002_*`.
- Migrations must be idempotent: check existence before create/update.

Tracking
- Maintain an `_applied` collection to record applied migration filenames and timestamps.

Apply
- Use `scripts/dev_bootstrap.sh` or your deployment runner to apply migrations in order.
