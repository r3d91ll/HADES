from datetime import datetime, timezone

from core.runtime.memgpt.memory_store import ArangoMemoryStore
from core.runtime.memgpt.turn_manager import ConversationTurn


class DummyClient:
    def __init__(self) -> None:
        self.inserted = []
        self.queries = []
        self.collections = []

    def bulk_insert(self, collection: str, docs):  # type: ignore[override]
        self.inserted.append((collection, list(docs)))
        return len(list(docs))

    def execute_query(self, aql: str, bind_vars=None, *, batch_size=None, full_count=False):  # type: ignore[override]
        self.queries.append((aql, dict(bind_vars or {})))
        # Return empty by default
        return []

    def create_collections(self, definitions):  # type: ignore[override]
        self.collections.extend([d.name for d in definitions])


def test_memory_store_append_and_list_recent():
    client = DummyClient()
    store = ArangoMemoryStore(client=client, database="hades_memories")
    store.ensure_schema()
    assert set(client.collections) >= {store.MESSAGES, store.RECALL_LOG}

    now = datetime.now(timezone.utc)
    turn = ConversationTurn.create(role="user", content="hello", tokens=3, metadata={"x": 1})
    turn.timestamp = now
    key = store.append_turn(session_id="s1", turn=turn)
    assert key
    assert client.inserted

    # Simulate query result for list_recent_turns
    def fake_query(aql, bind_vars=None, **_):
        return [
            {
                "_key": key,
                "session_id": "s1",
                "role": "user",
                "content": "hello",
                "tokens": 3,
                "created_at": now.isoformat(),
                "metadata": {"x": 1},
            }
        ]

    client.execute_query = fake_query  # type: ignore[assignment]
    rows = store.list_recent_turns(session_id="s1", limit=10)
    assert rows and rows[0]["content"] == "hello"


def test_memory_store_log_recall():
    client = DummyClient()
    store = ArangoMemoryStore(client=client, database="hades_memories")
    rid = store.log_recall(session_id="s2", query="term", result_count=5, duration_ms=1.23)
    assert rid
    assert client.inserted

