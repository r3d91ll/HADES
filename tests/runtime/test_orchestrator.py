
from core.runtime.memgpt.model_engine import QwenModelEngine
from core.runtime.memgpt.orchestrator import MemGPTOrchestrator, OrchestratorConfig
from core.runtime.memgpt.telemetry import TelemetryClient


class DummyEngine(QwenModelEngine):
    def __init__(self):
        super().__init__()
        self._tokenizer = object()
        self._model = object()
        self._prompt = None

    def build_prompt(self, turns):
        self._prompt = " | ".join(t.content for t in turns)
        return self._prompt

    def generate(self, prompt: str, **overrides):
        self._prompt = prompt
        return "echo:" + prompt

    def generate_stream(self, prompt: str, **overrides):  # type: ignore[override]
        self._prompt = prompt
        for chunk in ["echo:", prompt]:
            yield chunk


class CaptureTelemetryClient(TelemetryClient):
    def __init__(self) -> None:
        self.spans: list[tuple[str, dict]] = []

    def emit_span(self, name: str, attributes: dict) -> None:
        self.spans.append((name, attributes))



def test_orchestrator_generates_reply():
    engine = DummyEngine()
    orchestrator = MemGPTOrchestrator(
        config=OrchestratorConfig(max_turns=3, token_budget=None),
        model_engine=engine,
    )
    orchestrator.ingest_turn(role="user", content="hello")
    reply = orchestrator.generate_reply()

    assert reply.role == "assistant"
    assert reply.content.startswith("echo:")
    assert engine._prompt is not None


def test_orchestrator_turn_limit():
    engine = DummyEngine()
    orchestrator = MemGPTOrchestrator(
        config=OrchestratorConfig(max_turns=2, token_budget=None),
        model_engine=engine,
    )
    orchestrator.ingest_turn(role="user", content="one")
    orchestrator.ingest_turn(role="assistant", content="two")
    orchestrator.ingest_turn(role="user", content="three")

    turns = orchestrator.turn_manager.turns
    assert len(turns) == 2
    assert turns[0].content == "two"
    assert turns[1].content == "three"


def test_orchestrator_emits_telemetry():
    engine = DummyEngine()
    telemetry = CaptureTelemetryClient()
    orchestrator = MemGPTOrchestrator(
        config=OrchestratorConfig(max_turns=2, token_budget=None),
        model_engine=engine,
        telemetry=telemetry,
    )
    orchestrator.ingest_turn(role="user", content="probe")
    orchestrator.generate_reply()

    assert telemetry.spans, "expected telemetry span to be recorded"
    name, attrs = telemetry.spans[-1]
    assert name == "memgpt.model_generate"
    assert attrs["success"] is True
    assert "duration_ms" in attrs
    assert attrs["prompt_chars"] > 0
