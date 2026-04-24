"""Lakebase-backed session memory for the agent.

Why Lakebase and not Delta: every turn appends a small row, and the
agent reads + writes this row synchronously in the critical path. A
managed Postgres (Lakebase) gives us:

- **Low-latency row ops** — Delta's commit model is optimised for
  append-many-read-many batch analytics, not single-row upserts during a
  live agent turn.
- **Native JSONB** for ``tool_calls`` and ``accumulated_filters`` —
  queryable without writing a UDF.
- **Per-session primary key** — simple retrieval by ``session_id`` with
  a btree index.

Two tables:

    agent_sessions  — one row per conversation
        session_id       UUID PRIMARY KEY
        created_at       TIMESTAMPTZ
        last_turn_at     TIMESTAMPTZ
        user_id          TEXT              -- opaque; no PII expected
        accumulated_filters JSONB          -- running refinement bag

    agent_turns     — one row per LLM turn (user / assistant / tool)
        session_id       UUID REFERENCES agent_sessions
        turn_idx         INTEGER
        role             TEXT CHECK (role IN ('user','assistant','tool'))
        content          TEXT
        tool_calls       JSONB
        created_at       TIMESTAMPTZ
        PRIMARY KEY (session_id, turn_idx)

Two implementations of the ``SessionStore`` protocol:

1. ``PostgresSessionStore`` — real Lakebase connection via psycopg.
2. ``InMemorySessionStore`` — tests + local notebook debugging without
   a Lakebase project.

The agent only holds a ``SessionStore`` — swap between them with no
code changes.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

VALID_ROLES: frozenset[str] = frozenset({"user", "assistant", "tool"})


@dataclass
class Turn:
    """One LLM turn. Rows in ``agent_turns`` map 1:1 to this shape."""

    session_id: str
    turn_idx: int
    role: str
    content: str
    tool_calls: list[dict] = field(default_factory=list)
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.role not in VALID_ROLES:
            raise ValueError(
                f"invalid role {self.role!r}; expected one of {sorted(VALID_ROLES)}"
            )


@dataclass
class Session:
    """One conversation. Maps to a row in ``agent_sessions``."""

    session_id: str
    created_at: datetime
    last_turn_at: datetime
    user_id: str | None = None
    accumulated_filters: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class SessionStore(Protocol):
    """Minimum surface the agent needs from a session store."""

    def init_schema(self) -> None:
        """Create tables + indexes if they don't exist. Idempotent."""
        ...

    def create_session(self, user_id: str | None = None) -> str:
        """Create a new session and return its id."""
        ...

    def ensure_session(self, session_id: str, user_id: str | None = None) -> None:
        """Idempotently ensure a row exists in the sessions table for
        ``session_id``. Needed when the caller (Responses API client,
        eval harness) supplies a pre-generated id — `append_turn`
        requires the parent row by FK."""
        ...

    def append_turn(self, turn: Turn) -> None:
        """Persist a turn. Caller sets ``turn_idx`` explicitly."""
        ...

    def get_history(self, session_id: str, max_turns: int | None = None) -> list[Turn]:
        """Return turns ordered by ``turn_idx`` ascending."""
        ...

    def next_turn_idx(self, session_id: str) -> int:
        """Return the next unused ``turn_idx`` for this session."""
        ...

    def get_accumulated_filters(self, session_id: str) -> dict:
        """Read the current accumulated-filters bag."""
        ...

    def set_accumulated_filters(self, session_id: str, filters: dict) -> None:
        """Overwrite the accumulated-filters bag."""
        ...


# ---------------------------------------------------------------------------
# DDL — one place, used by both real-Postgres init and the setup notebook.
# ---------------------------------------------------------------------------

DDL_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS agent_sessions (
        session_id          UUID PRIMARY KEY,
        created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        last_turn_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        user_id             TEXT,
        accumulated_filters JSONB NOT NULL DEFAULT '{}'::jsonb
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS agent_turns (
        session_id  UUID        NOT NULL REFERENCES agent_sessions(session_id)
                                 ON DELETE CASCADE,
        turn_idx    INTEGER     NOT NULL,
        role        TEXT        NOT NULL
                                 CHECK (role IN ('user','assistant','tool')),
        content     TEXT        NOT NULL,
        tool_calls  JSONB       NOT NULL DEFAULT '[]'::jsonb,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (session_id, turn_idx)
    )
    """,
    # Speeds up history reads when sessions grow past a few turns. Also
    # used by the ops dashboard to find long conversations.
    """
    CREATE INDEX IF NOT EXISTS idx_agent_turns_session_created
        ON agent_turns (session_id, created_at)
    """,
)


# ---------------------------------------------------------------------------
# Postgres / Lakebase implementation
# ---------------------------------------------------------------------------


class PostgresSessionStore:
    """``SessionStore`` backed by a Lakebase-managed Postgres.

    Pass a **connection factory** rather than a live connection so the
    store plays nice with Databricks auth rotation — the factory fetches
    a fresh token each call. Example::

        from databricks.sdk import WorkspaceClient
        def _conn():
            cred = WorkspaceClient().database.generate_database_credential(...)
            return psycopg.connect(host=..., user=..., password=cred.token, ...)

        store = PostgresSessionStore(_conn)
        store.init_schema()
    """

    def __init__(self, conn_factory: Callable[[], Any]) -> None:
        self._conn_factory = conn_factory

    def _run(self, fn: Callable[[Any], Any]) -> Any:
        """Open a short-lived connection, run ``fn(conn)``, commit + close."""
        conn = self._conn_factory()
        try:
            with conn:
                return fn(conn)
        finally:
            conn.close()

    def init_schema(self) -> None:
        def _do(conn: Any) -> None:
            with conn.cursor() as cur:
                for ddl in DDL_STATEMENTS:
                    cur.execute(ddl)

        self._run(_do)

    def create_session(self, user_id: str | None = None) -> str:
        sid = str(uuid.uuid4())

        def _do(conn: Any) -> None:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO agent_sessions (session_id, user_id)
                    VALUES (%s, %s)
                    """,
                    (sid, user_id),
                )

        self._run(_do)
        return sid

    def ensure_session(self, session_id: str, user_id: str | None = None) -> None:
        def _do(conn: Any) -> None:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO agent_sessions (session_id, user_id)
                    VALUES (%s, %s)
                    ON CONFLICT (session_id) DO NOTHING
                    """,
                    (session_id, user_id),
                )

        self._run(_do)

    def append_turn(self, turn: Turn) -> None:
        def _do(conn: Any) -> None:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO agent_turns
                        (session_id, turn_idx, role, content, tool_calls)
                    VALUES (%s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        turn.session_id,
                        turn.turn_idx,
                        turn.role,
                        turn.content,
                        json.dumps(turn.tool_calls),
                    ),
                )
                cur.execute(
                    """
                    UPDATE agent_sessions SET last_turn_at = NOW()
                    WHERE session_id = %s
                    """,
                    (turn.session_id,),
                )

        self._run(_do)

    def get_history(self, session_id: str, max_turns: int | None = None) -> list[Turn]:
        def _do(conn: Any) -> list[Turn]:
            with conn.cursor() as cur:
                q = (
                    "SELECT session_id, turn_idx, role, content, "
                    "tool_calls, created_at "
                    "FROM agent_turns WHERE session_id = %s "
                    "ORDER BY turn_idx ASC"
                )
                cur.execute(q, (session_id,))
                rows = cur.fetchall()
            turns = [
                Turn(
                    session_id=str(r[0]),
                    turn_idx=r[1],
                    role=r[2],
                    content=r[3],
                    tool_calls=r[4]
                    if isinstance(r[4], list)
                    else json.loads(r[4] or "[]"),
                    created_at=r[5],
                )
                for r in rows
            ]
            if max_turns is not None:
                turns = turns[-max_turns:]
            return turns

        return self._run(_do)

    def next_turn_idx(self, session_id: str) -> int:
        def _do(conn: Any) -> int:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COALESCE(MAX(turn_idx), -1) + 1 "
                    "FROM agent_turns WHERE session_id = %s",
                    (session_id,),
                )
                return int(cur.fetchone()[0])

        return self._run(_do)

    def get_accumulated_filters(self, session_id: str) -> dict:
        def _do(conn: Any) -> dict:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT accumulated_filters FROM agent_sessions "
                    "WHERE session_id = %s",
                    (session_id,),
                )
                row = cur.fetchone()
                if not row:
                    raise KeyError(f"session {session_id} not found")
                val = row[0]
                return val if isinstance(val, dict) else json.loads(val or "{}")

        return self._run(_do)

    def set_accumulated_filters(self, session_id: str, filters: dict) -> None:
        def _do(conn: Any) -> None:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE agent_sessions
                    SET accumulated_filters = %s::jsonb
                    WHERE session_id = %s
                    """,
                    (json.dumps(filters), session_id),
                )

        self._run(_do)


# ---------------------------------------------------------------------------
# In-memory implementation — used by tests + notebook debugging.
# ---------------------------------------------------------------------------


class InMemorySessionStore:
    """Thread-safe in-memory ``SessionStore`` stand-in.

    Mirrors the Postgres store's semantics (turn_idx collisions raise,
    ``get_history`` returns ascending order) so tests catch contract
    violations that would bite later against Lakebase.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._turns: dict[str, list[Turn]] = {}
        self._lock = threading.Lock()

    def init_schema(self) -> None:
        # No-op — the in-memory "schema" is the Python objects themselves.
        return

    def create_session(self, user_id: str | None = None) -> str:
        sid = str(uuid.uuid4())
        now = datetime.now(UTC)
        with self._lock:
            self._sessions[sid] = Session(
                session_id=sid,
                created_at=now,
                last_turn_at=now,
                user_id=user_id,
            )
            self._turns[sid] = []
        return sid

    def ensure_session(self, session_id: str, user_id: str | None = None) -> None:
        now = datetime.now(UTC)
        with self._lock:
            if session_id in self._sessions:
                return
            self._sessions[session_id] = Session(
                session_id=session_id,
                created_at=now,
                last_turn_at=now,
                user_id=user_id,
            )
            self._turns[session_id] = []

    def append_turn(self, turn: Turn) -> None:
        with self._lock:
            if turn.session_id not in self._sessions:
                raise KeyError(f"session {turn.session_id} not found")
            turns = self._turns[turn.session_id]
            if any(t.turn_idx == turn.turn_idx for t in turns):
                raise ValueError(
                    f"turn_idx {turn.turn_idx} already exists in "
                    f"session {turn.session_id}"
                )
            # Stamp a created_at if the caller didn't.
            if turn.created_at is None:
                turn.created_at = datetime.now(UTC)
            turns.append(turn)
            turns.sort(key=lambda t: t.turn_idx)
            self._sessions[turn.session_id].last_turn_at = turn.created_at

    def get_history(self, session_id: str, max_turns: int | None = None) -> list[Turn]:
        with self._lock:
            turns = list(self._turns.get(session_id, []))
        if max_turns is not None:
            return turns[-max_turns:]
        return turns

    def next_turn_idx(self, session_id: str) -> int:
        with self._lock:
            turns = self._turns.get(session_id, [])
            return (max((t.turn_idx for t in turns), default=-1)) + 1

    def get_accumulated_filters(self, session_id: str) -> dict:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"session {session_id} not found")
            return dict(self._sessions[session_id].accumulated_filters)

    def set_accumulated_filters(self, session_id: str, filters: dict) -> None:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"session {session_id} not found")
            self._sessions[session_id].accumulated_filters = dict(filters)
