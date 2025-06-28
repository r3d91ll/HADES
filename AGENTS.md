# AGENTS.md - Development Guidelines for AI Coding Agents

## Build/Test Commands
- **Install dependencies**: `poetry install`
- **Run all tests**: `poetry run pytest`
- **Run single test**: `poetry run pytest tests/path/to/test_file.py::test_function_name -v`
- **Run with coverage**: `poetry run pytest --cov=src --cov-report=term`
- **Type checking**: `poetry run mypy src/`
- **Format code**: `poetry run black src/` and `poetry run isort src/`
- **Lint**: `poetry run flake8 src/` or `trunk check`

## Code Style Guidelines
- **Line length**: 88 characters (Black standard)
- **Type annotations**: Required for all functions/methods (strict mypy config)
- **Imports**: Use isort with Black profile, group by: stdlib, third-party, local
- **Docstrings**: Required for all public functions/classes, use Google style
- **Error handling**: Use specific exception types, avoid bare `except:`
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **File structure**: Follow src/ layout, use __init__.py for packages

## Test Organization
- **Test markers**: Use pytest markers: `unit`, `integration`, `slow`, `models`, `parsers`, `adapters`, `storage`
- **Coverage requirement**: Minimum 85% for all new code
- **Test location**: All tests in `tests/` directory, mirror `src/` structure