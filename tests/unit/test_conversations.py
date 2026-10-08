import time

import pytest

from kassist.conversations import ConversationNotFoundError, ConversationTurn, SqliteConversationStore


@pytest.fixture
def store(tmp_path) -> SqliteConversationStore:
    return SqliteConversationStore(tmp_path / "c.sqlite", ttl_hours=1, max_turns=3)


def turn(i: int) -> ConversationTurn:
    return ConversationTurn(question=f"q{i}", answer=f"a{i}", status="answered")


def test_turns_are_returned_oldest_first_and_limited_to_most_recent(store):
    cid = store.create()
    for i in range(3):
        store.append(cid, turn(i))
    assert [t.question for t in store.turns(cid)] == ["q0", "q1", "q2"]
    assert [t.question for t in store.turns(cid, limit=2)] == ["q1", "q2"]


def test_only_max_turns_are_kept(store):
    cid = store.create()
    for i in range(5):
        store.append(cid, turn(i))
    assert [t.question for t in store.turns(cid)] == ["q2", "q3", "q4"]


def test_conversations_are_isolated_and_persist_across_instances(store, tmp_path):
    a, b = store.create(), store.create()
    store.append(a, turn(1))
    reopened = SqliteConversationStore(tmp_path / "c.sqlite", ttl_hours=1, max_turns=3)
    assert [t.question for t in reopened.turns(a)] == ["q1"]
    assert reopened.turns(b) == []


def test_unknown_conversation(store):
    assert not store.exists("00000000-0000-4000-8000-000000000000")
    with pytest.raises(ConversationNotFoundError):
        store.turns("missing")
    with pytest.raises(ConversationNotFoundError):
        store.append("missing", turn(1))


def test_delete_removes_history(store):
    cid = store.create()
    store.append(cid, turn(1))
    assert store.delete(cid) and not store.exists(cid)
    assert not store.delete(cid)


def test_idle_conversations_expire(tmp_path):
    s = SqliteConversationStore(tmp_path / "c.sqlite", ttl_hours=0.5 / 3600, max_turns=3)  # 0.5 s
    cid = s.create()
    time.sleep(0.6)
    assert not s.exists(cid)
