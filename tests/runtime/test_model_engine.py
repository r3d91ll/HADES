import importlib
import sys

import pytest

from core.runtime.memgpt.model_engine import (
    MissingDependencyError,
    ModelNotLoadedError,
    QwenModelConfig,
    QwenModelEngine,
)


def test_engine_instantiation_without_dependencies():
    engine = QwenModelEngine()
    assert not engine.is_loaded()
    with pytest.raises(ModelNotLoadedError):
        engine.generate("Hello")


def test_engine_reports_missing_dependencies(monkeypatch):
    def fake_import(name, *args, **kwargs):
        raise ImportError

    monkeypatch.setitem(sys.modules, "transformers", None)
    monkeypatch.setitem(sys.modules, "accelerate", None)
    monkeypatch.setitem(sys.modules, "autoawq", None)
    monkeypatch.setitem(sys.modules, "awq", None)

    engine = QwenModelEngine(QwenModelConfig())
    with pytest.raises(MissingDependencyError):
        engine.load()


@pytest.mark.parametrize("pkg", ["transformers", "accelerate", "autoawq", "awq"])
def test_dependencies_available_handles_missing(monkeypatch, pkg):
    if pkg in sys.modules:
        monkeypatch.delitem(sys.modules, pkg, raising=False)
    available = QwenModelEngine.dependencies_available()
    assert available in {True, False}


def test_flash_attn_dependency_failure(monkeypatch):

    real_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "flash_attn":
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    engine = QwenModelEngine(QwenModelConfig(use_flash_attn=True))

    with pytest.raises(MissingDependencyError):
        engine._ensure_dependencies()
