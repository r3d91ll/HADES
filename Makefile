SHELL := /bin/bash

PY := poetry run

.PHONY: lint format test migrate db-bootstrap dev

lint:
	$(PY) ruff check

format:
	$(PY) ruff format

test:
	$(PY) pytest -q

# Placeholder: apply AQL migrations in core/database/migrations
migrate:
	@echo "Apply ArangoDB migrations (stub). See scripts/dev_bootstrap.sh"

db-bootstrap:
	bash scripts/dev_bootstrap.sh

dev:
	@echo "Start dev services (stub). Add your process manager here."
