"""Unit tests for session memory.

The ``InMemorySessionStore`` is a strict in-memory mirror of the
Postgres store. These tests pin down the semantics that both
implementations must share — if either diverges the agent will behave
differently between local and prod.

Covered:
- ``create_session`` returns a unique UUID.
- ``append_turn`` enforces unique ``(session_id, turn_idx)``.
- ``append_turn`` rejects invalid roles.
- ``get_history`` returns turns in ascending ``turn_idx`` order.
- ``get_history(max_turns=k)`` returns the *last* k turns.
- ``next_turn_idx`` returns ``max(idx) + 1`` starting at 0.
- Accumulated-filters round-trip preserves the dict verbatim.
- ``DDL_STATEMENTS`` shape (sanity — we don't run real Postgres here).
"""

from __future__ import annotations

import pytest

from nhtsa_curator.memory import (
    DDL_STATEMENTS,
    InMemorySessionStore,
    Turn,
    VALID_ROLES,
)


@pytest.fixture
def store() -> InMemorySessionStore:
    s = InMemorySessionStore()
    s.init_schema()  # no-op — confirms the method exists & returns.
    return s


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


def test_create_session_returns_unique_uuid(store: InMemorySessionStore) -> None:
    ids = {store.create_session() for _ in range(10)}
    assert len(ids) == 10
    for sid in ids:
        assert len(sid) == 36  # UUID4 canonical length


def test_create_session_records_user_id(store: InMemorySessionStore) -> None:
    sid = store.create_session(user_id="pralay")
    store.append_turn(Turn(session_id=sid, turn_idx=0, role="user", content="hi"))
    # user_id isn't read back by the protocol, but it must not error.
    assert store.get_history(sid)[0].content == "hi"


# ---------------------------------------------------------------------------
# append_turn invariants
# ---------------------------------------------------------------------------


def test_append_turn_rejects_duplicate_idx(store: InMemorySessionStore) -> None:
    sid = store.create_session()
    store.append_turn(Turn(session_id=sid, turn_idx=0, role="user", content="a"))
    with pytest.raises(ValueError, match="turn_idx 0"):
        store.append_turn(
            Turn(session_id=sid, turn_idx=0, role="assistant", content="b")
        )


def test_append_turn_rejects_unknown_session(store: InMemorySessionStore) -> None:
    with pytest.raises(KeyError):
        store.append_turn(
            Turn(session_id="no-such-sid", turn_idx=0, role="user", content="a")
        )


def test_turn_rejects_invalid_role(store: InMemorySessionStore) -> None:
    sid = store.create_session()
    with pytest.raises(ValueError, match="invalid role"):
        Turn(session_id=sid, turn_idx=0, role="robot", content="oops")


def test_valid_roles_are_frozen() -> None:
    assert "user" in VALID_ROLES
    assert "assistant" in VALID_ROLES
    assert "tool" in VALID_ROLES
    assert len(VALID_ROLES) == 3


# ---------------------------------------------------------------------------
# History ordering + windowing
# ---------------------------------------------------------------------------


def test_history_ordered_ascending_regardless_of_insert_order(
    store: InMemorySessionStore,
) -> None:
    sid = store.create_session()
    store.append_turn(Turn(session_id=sid, turn_idx=2, role="user", content="c"))
    store.append_turn(Turn(session_id=sid, turn_idx=0, role="user", content="a"))
    store.append_turn(Turn(session_id=sid, turn_idx=1, role="assistant", content="b"))
    hist = store.get_history(sid)
    assert [t.content for t in hist] == ["a", "b", "c"]


def test_history_window_returns_last_k(store: InMemorySessionStore) -> None:
    sid = store.create_session()
    for i in range(5):
        store.append_turn(
            Turn(session_id=sid, turn_idx=i, role="user", content=str(i))
        )
    hist = store.get_history(sid, max_turns=2)
    assert [t.content for t in hist] == ["3", "4"]


def test_next_turn_idx_starts_at_zero(store: InMemorySessionStore) -> None:
    sid = store.create_session()
    assert store.next_turn_idx(sid) == 0
    store.append_turn(Turn(session_id=sid, turn_idx=0, role="user", content="x"))
    assert store.next_turn_idx(sid) == 1
    store.append_turn(Turn(session_id=sid, turn_idx=1, role="assistant", content="y"))
    assert store.next_turn_idx(sid) == 2


# ---------------------------------------------------------------------------
# Accumulated filters
# ---------------------------------------------------------------------------


def test_accumulated_filters_default_empty(store: InMemorySessionStore) -> None:
    sid = store.create_session()
    assert store.get_accumulated_filters(sid) == {}


def test_accumulated_filters_roundtrip(store: InMemorySessionStore) -> None:
    sid = store.create_session()
    bag = {"make_norm": "Tesla", "model_year": 2023, "components": ["brakes", "ecu"]}
    store.set_accumulated_filters(sid, bag)
    assert store.get_accumulated_filters(sid) == bag


def test_accumulated_filters_unknown_session_raises(store: InMemorySessionStore) -> None:
    with pytest.raises(KeyError):
        store.get_accumulated_filters("not-a-sid")
    with pytest.raises(KeyError):
        store.set_accumulated_filters("not-a-sid", {})


# ---------------------------------------------------------------------------
# DDL shape sanity
# ---------------------------------------------------------------------------


def test_ddl_mentions_expected_tables() -> None:
    """Fail loudly if someone renames a table without updating the agent."""
    joined = " ".join(DDL_STATEMENTS).lower()
    assert "agent_sessions" in joined
    assert "agent_turns" in joined
    assert "jsonb" in joined
    assert "on delete cascade" in joined


def test_ddl_statements_are_all_create_or_alter() -> None:
    for stmt in DDL_STATEMENTS:
        head = stmt.strip().split()[0].upper()
        assert head in {"CREATE", "ALTER"}, f"unexpected DDL head: {head}"
