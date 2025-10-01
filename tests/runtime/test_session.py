from core.runtime.memgpt.model_engine import QwenModelEngine
from core.runtime.memgpt.orchestrator import MemGPTOrchestrator, OrchestratorConfig
from core.runtime.memgpt.session import MemGPTSession


class DummyEngine(QwenModelEngine):
    def __init__(self):
        super().__init__()
        self._tokenizer = object()
        self._model = object()

    def build_prompt(self, turns):
        return " | ".join(t.content for t in turns)

    def generate(self, prompt: str, **overrides):
        return "ECHO:" + prompt


class FakeStore:
    def __init__(self):
        self.turns = []

    def ensure_schema(self):
        return None

    def append_turn(self, *, session_id, turn):
        self.turns.append((session_id, turn.role, turn.content))
        return "k"

    def list_recent_turns(self, *, session_id, limit=50):
        return []

    def log_recall(self, **kwargs):  # pragma: no cover - not used here
        return "r"


def test_session_send_user_message():
    engine = DummyEngine()
    orch = MemGPTOrchestrator(config=OrchestratorConfig(max_turns=5, token_budget=None), model_engine=engine)
    store = FakeStore()
    session = MemGPTSession.new(orch, store, session_id="s1")

    reply = session.send_user_message("hello")
    assert reply.role == "assistant"
    assert reply.content.startswith("ECHO:")
    assert len(store.turns) == 2  # user + assistant persisted

