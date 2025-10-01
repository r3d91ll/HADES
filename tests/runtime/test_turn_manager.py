from datetime import datetime, timezone

from core.runtime.memgpt.turn_manager import ConversationTurn, TurnManager


def test_turn_manager_enforces_max_turns():
    manager = TurnManager(max_turns=2)
    manager.add_turn(ConversationTurn.create(role="user", content="hi"))
    manager.add_turn(ConversationTurn.create(role="assistant", content="hello"))
    manager.add_turn(ConversationTurn.create(role="user", content="again"))

    turns = manager.turns
    assert len(turns) == 2
    assert turns[0].role == "assistant"
    assert turns[1].content == "again"


def test_turn_manager_token_budget():
    now = datetime.now(timezone.utc)
    turn1 = ConversationTurn(turn_id="1", role="user", content="a", timestamp=now, tokens=80)
    turn2 = ConversationTurn(turn_id="2", role="assistant", content="b", timestamp=now, tokens=40)
    turn3 = ConversationTurn(turn_id="3", role="user", content="c", timestamp=now, tokens=30)

    manager = TurnManager(max_turns=5, token_budget=100)
    manager.extend([turn1, turn2, turn3])

    turns = manager.turns
    assert len(turns) == 2
    assert turns[0].turn_id == "2"
    assert turns[1].turn_id == "3"


def test_turn_manager_summary():
    manager = TurnManager(max_turns=3, token_budget=200)
    manager.add_turn(ConversationTurn.create(role="system", content="base", tokens=5))
    manager.add_turn(ConversationTurn.create(role="user", content="hi", tokens=12))

    summary = manager.summarize()
    assert summary["turn_count"] == 2
    assert summary["roles"] == ["system", "user"]
    assert summary["tokens"] == 17
