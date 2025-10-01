"""
Model Engine - Qwen3-30B-A3B-Thinking Interface

WHAT: Model loading and generation wrapper for Qwen3 30B with thinking tokens
WHERE: core/runtime/memgpt/model_engine.py - manages GPU resources and generation
WHO: MemGPT agents requiring LLM completions
TIME: First-token latency <50ms after warmup, generation throughput >20 tok/s

Manages the Qwen3-30B-A3B-Thinking-2507 model with AWQ quantization and
FlashAttention-2 optimization. Supports 262k context window with balanced
multi-GPU distribution (2x44GiB).

Boundary Notes:
- H (agent capability) directly tied to model quality
- Generation latency contributes to T dimension in conveyance
- Thinking tokens tracked separately for analysis
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, Optional

from .turn_manager import ConversationTurn


class ModelNotLoadedError(RuntimeError):
    """Raised when generation is attempted before the model is loaded."""


class MissingDependencyError(RuntimeError):
    """Raised when required model packages are unavailable in the environment."""


REQUIRED_PACKAGES = ("transformers", "accelerate")
AWQ_PACKAGES = ("autoawq", "awq")
FLASH_ATTN_PACKAGE = "flash_attn"


@dataclass(slots=True)
class QwenModelConfig:
    """Configuration for the Qwen3-30B Thinking model."""

    model_id: str = "Qwen/Qwen3-30B-A3B-Thinking-2507"
    awq_weights: Optional[str] = None
    device: str = "cuda:0"
    device_map: Optional[Any] = None
    max_memory: Optional[Dict[str, str]] = None
    dtype: str = "bfloat16"
    max_new_tokens: int = 2048
    max_context_length: int = 65536
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    streaming: bool = True
    use_flash_attn: bool = False


class QwenModelEngine:
    """Handles loading and generation for the production Qwen model."""

    def __init__(self, config: QwenModelConfig | None = None) -> None:
        self.config = config or QwenModelConfig()
        self._tokenizer = None
        self._model = None
        self._generation_kwargs: Dict[str, Any] = {}
        self._modules: Dict[str, Any] = {}

    @staticmethod
    def dependencies_available() -> bool:
        import importlib

        for package in REQUIRED_PACKAGES:
            try:
                importlib.import_module(package)
            except ImportError:
                return False

        for candidate in AWQ_PACKAGES:
            try:
                importlib.import_module(candidate)
                return True
            except ImportError:
                continue
        return False

    def _ensure_dependencies(self) -> None:
        missing: list[str] = []
        import importlib

        modules = {}
        for package in REQUIRED_PACKAGES:
            try:
                modules[package] = importlib.import_module(package)
            except ImportError:
                missing.append(package)

        awq_module = None
        for candidate in AWQ_PACKAGES:
            try:
                awq_module = importlib.import_module(candidate)
                modules[candidate] = awq_module
                modules["awq_active"] = candidate
                break
            except ImportError:
                continue

        if missing or awq_module is None:
            awq_hint = "autoawq/awq" if awq_module is None else None
            missing_bits = list(missing)
            if awq_hint:
                missing_bits.append(awq_hint)
            raise MissingDependencyError(
                "Missing model dependencies: "
                + ", ".join(missing_bits)
                + ". Install with `poetry install --extras qwen` or manual AWQ setup."
            )

        if self.config.use_flash_attn:
            try:
                modules[FLASH_ATTN_PACKAGE] = importlib.import_module(FLASH_ATTN_PACKAGE)
            except ImportError as exc:
                raise MissingDependencyError(
                    "Flash attention requested but `flash_attn` is not installed."
                ) from exc

        self._modules = modules

    def load(self, *, device_map: str | None = None) -> None:
        """Load tokenizer and model into memory."""

        self._ensure_dependencies()
        transformers = self._modules["transformers"]
        accelerate = self._modules["accelerate"]
        try:
            import torch
        except Exception:
            torch = None  # type: ignore[assignment]

        AutoTokenizer = transformers.AutoTokenizer
        model_id = self.config.model_id
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        cfg_device_map = self._normalize_device_map(device_map or self.config.device_map)
        normalized_max_memory = self._normalize_max_memory(self.config.max_memory)
        flash_kwargs: Dict[str, Any] = {}
        if self.config.use_flash_attn:
            flash_kwargs["attn_implementation"] = "flash_attention_2"

        if self.config.awq_weights:
            awq_pkg = self._modules.get("awq_active", AWQ_PACKAGES[0])
            AutoAWQForCausalLM = getattr(self._modules[awq_pkg], "AutoAWQForCausalLM")  # type: ignore[attr-defined]

            awq_device_map = cfg_device_map or "balanced"
            self._model = AutoAWQForCausalLM.from_quantized(
                model_id,
                quantize_config=None,
                trust_remote_code=True,
                safetensors=True,
                fuse_layers=True,
                device_map=awq_device_map,
                max_memory=normalized_max_memory,
                max_new_tokens=self.config.max_new_tokens,
                cache_dir=self.config.awq_weights,
                **flash_kwargs,
            )
        else:
            AutoModelForCausalLM = transformers.AutoModelForCausalLM
            dtype = None
            if torch is not None:
                # Map 'float16'/'bfloat16'/'float32' -> torch.<dtype>
                try:
                    dtype = getattr(torch, self.config.dtype)
                except AttributeError:
                    dtype = None
            load_kwargs: Dict[str, Any] = {
                "trust_remote_code": True,
                "torch_dtype": dtype,
            }

            if cfg_device_map or normalized_max_memory:
                load_kwargs["device_map"] = cfg_device_map or self.config.device
                if normalized_max_memory:
                    load_kwargs["max_memory"] = normalized_max_memory
            else:
                load_kwargs["device_map"] = self.config.device

            load_kwargs.update(flash_kwargs)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **load_kwargs,
            )

        accelerate.init_empty_weights  # touch to avoid lint unused import
        if hasattr(self._tokenizer, "model_max_length"):
            self._tokenizer.model_max_length = self.config.max_context_length
        if hasattr(self._model, "config") and hasattr(self._model.config, "max_position_embeddings"):
            try:
                self._model.config.max_position_embeddings = max(
                    self._model.config.max_position_embeddings,
                    self.config.max_context_length,
                )
            except TypeError:
                pass

        self._configure_flash_attention()

        self._generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "repetition_penalty": self.config.repetition_penalty,
        }

    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def _normalize_max_memory(self, max_memory: Optional[Dict[Any, str]]) -> Optional[Dict[Any, str]]:
        if not max_memory:
            return None

        normalized: Dict[Any, str] = {}
        for raw_key, raw_value in max_memory.items():
            if raw_value is None:
                continue

            key = str(raw_key).strip()
            value = str(raw_value).strip()
            if not key or not value:
                continue

            lowered = key.lower()
            if lowered in {"cpu", "disk", "mps"}:
                normalized[lowered] = value
                continue

            # Translate identifiers like "cuda:0" or "gpu1" into integers
            lowered = lowered.replace("cuda", "").replace("gpu", "")
            lowered = lowered.lstrip(":")

            try:
                normalized[int(lowered)] = value
            except ValueError:
                normalized[lowered] = value

        return normalized or None

    def _normalize_device_map(self, device_map: Any) -> Any:
        if device_map is None:
            return None

        if isinstance(device_map, str):
            lowered = device_map.strip().lower()
            if lowered.startswith(("cuda", "gpu")):
                lowered = lowered.replace("cuda", "").replace("gpu", "").lstrip(":")
                try:
                    return int(lowered)
                except ValueError:
                    return device_map
            return device_map

        if isinstance(device_map, dict):
            normalized: Dict[Any, Any] = {}
            for key, value in device_map.items():
                if isinstance(key, str):
                    lowered = key.strip().lower()
                    if lowered in {"cpu", "disk", "mps"}:
                        normalized[lowered] = value
                        continue
                    lowered = lowered.replace("cuda", "").replace("gpu", "").lstrip(":")
                    try:
                        normalized[int(lowered)] = value
                        continue
                    except ValueError:
                        normalized[key] = value
                else:
                    normalized[key] = value
            return normalized

        return device_map

    def _configure_flash_attention(self) -> None:
        if not self.config.use_flash_attn or not self._model:
            return

        try:
            import torch

            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "enable_flash_sdp"):
                torch.backends.cuda.enable_flash_sdp(True)
        except Exception:  # pragma: no cover - best effort enabling
            pass

        configs: list[Any] = []
        model = self._model
        if hasattr(model, "config") and getattr(model, "config", None) is not None:
            configs.append(model.config)
        inner = getattr(model, "model", None)
        if inner is not None and hasattr(inner, "config") and getattr(inner, "config", None) is not None:
            configs.append(inner.config)

        for cfg in configs:
            if hasattr(cfg, "_attn_implementation"):
                cfg._attn_implementation = "flash_attention_2"
            if hasattr(cfg, "attn_implementation"):
                setattr(cfg, "attn_implementation", "flash_attention_2")
            if hasattr(cfg, "use_flash_attn"):
                setattr(cfg, "use_flash_attn", True)
            if hasattr(cfg, "use_flash_attention"):
                setattr(cfg, "use_flash_attention", True)
            if hasattr(cfg, "use_flash_attention_2"):
                setattr(cfg, "use_flash_attention_2", True)

    def build_prompt(self, turns: Iterable[ConversationTurn]) -> str:
        lines = []
        for turn in turns:
            prefix = turn.role.upper()
            lines.append(f"[{prefix}] {turn.content}")
        return "\n".join(lines)

    def generate(self, prompt: str, **overrides: Any) -> str:
        if not self.is_loaded():
            raise ModelNotLoadedError("Qwen model is not loaded; call load() first")

        transformers = self._modules["transformers"]
        tokenizer = self._tokenizer
        model = self._model
        kwargs = dict(self._generation_kwargs)
        kwargs.update(overrides)

        try:
            model_device = model.device  # type: ignore[attr-defined]
        except AttributeError:  # sharded models may not expose .device
            model_device = next(model.parameters()).device  # type: ignore[call-arg]

        input_ids = tokenizer(prompt, return_tensors="pt").to(model_device)
        if kwargs.pop("stream", False):
            return "".join(self.generate_stream(prompt, **kwargs))

        output_ids = model.generate(**input_ids, **kwargs)
        response = tokenizer.decode(output_ids[0][input_ids["input_ids"].shape[-1] :], skip_special_tokens=True)
        return response.strip()

    def generate_stream(self, prompt: str, **overrides: Any) -> Iterator[str]:
        if not self.config.streaming:
            raise RuntimeError("Streaming disabled in configuration")
        if not self.is_loaded():
            raise ModelNotLoadedError("Qwen model is not loaded; call load() first")

        transformers = self._modules["transformers"]
        tokenizer = self._tokenizer
        model = self._model
        kwargs = dict(self._generation_kwargs)
        kwargs.update(overrides)

        try:
            model_device = model.device  # type: ignore[attr-defined]
        except AttributeError:
            model_device = next(model.parameters()).device  # type: ignore[call-arg]

        input_ids = tokenizer(prompt, return_tensors="pt").to(model_device)
        streamer = transformers.TextIteratorStreamer(tokenizer, skip_prompt=True)
        kwargs["streamer"] = streamer

        def worker():
            model.generate(**input_ids, **kwargs)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        for token in streamer:
            yield token
        thread.join()
