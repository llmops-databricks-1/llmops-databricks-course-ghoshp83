"""Tool-routing tests for :class:`NhtsaAgent`.

These tests do *not* talk to a real LLM. We inject a ``ScriptedLLM``
that returns a deterministic sequence of responses (tool calls, then a
final assistant text). This lets us pin down:

- The agent executes the tool calls emitted by the LLM.
- Tool results are appended back into the message list.
- Multi-step tool loops terminate with a final text answer.
- ``max_tool_steps`` hard-stops a runaway model.
- The session store receives a user turn + assistant turn per call.
- Vector-search filters surface into ``accumulated_filters`` across turns.
- The system prompt is carried in the first message.
- Malformed tool args (non-JSON) don't crash — they dispatch with ``{}``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from nhtsa_curator.agent import NhtsaAgent
from nhtsa_curator.config import ProjectConfig, VectorSearchConfig
from nhtsa_curator.mcp import (
    FETCH_TSB_TOOL_NAME,
    GENIE_TOOL_NAME,
    VECTOR_SEARCH_TOOL_NAME,
    ToolContext,
)
from nhtsa_curator.memory import InMemorySessionStore


# ---------------------------------------------------------------------------
# ScriptedLLM — tiny fake of openai.OpenAI.chat.completions.create
# ---------------------------------------------------------------------------


@dataclass
class _FakeToolCall:
    id: str
    name: str
    arguments: str  # JSON-encoded

    @property
    def function(self) -> "_FakeToolCall":
        # The openai SDK wraps name+arguments in .function; we mimic that.
        return self

    @property
    def type(self) -> str:
        return "function"


@dataclass
class _FakeMessage:
    content: str | None = None
    tool_calls: list[_FakeToolCall] | None = None


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeResp:
    choices: list[_FakeChoice]


class _Completions:
    def __init__(self, script: list[_FakeMessage]) -> None:
        self._script = list(script)
        self.calls: list[dict] = []

    def create(self, **kwargs: Any) -> _FakeResp:
        self.calls.append(kwargs)
        if not self._script:
            # Default to a plain-text answer to end the loop.
            return _FakeResp(choices=[_FakeChoice(_FakeMessage(content="done"))])
        msg = self._script.pop(0)
        return _FakeResp(choices=[_FakeChoice(msg)])


class _ScriptedLLM:
    def __init__(self, script: list[_FakeMessage]) -> None:
        self._completions = _Completions(script)

    @property
    def chat(self) -> Any:
        return self

    @property
    def completions(self) -> _Completions:
        return self._completions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool_call(tc_id: str, name: str, args: dict) -> _FakeToolCall:
    return _FakeToolCall(id=tc_id, name=name, arguments=json.dumps(args))


def _make_cfg() -> ProjectConfig:
    return ProjectConfig(
        catalog="cat", schema="sch", volume="vol", llm_endpoint="llm",
        embedding_endpoint="emb", warehouse_id="wh", vector_search_endpoint="ep",
        genie_space_id="real-space",
    )


class _FakeVSIndex:
    def similarity_search(self, **_: Any) -> dict:
        return {
            "result": {
                "manifest": {"columns": [{"name": "chunk_id"}, {"name": "content"}]},
                "data_array": [["abc", "Tesla brakes"], ["def", "rotor wear"]],
            }
        }


class _FakeVSClient:
    def get_index(self, **_: Any) -> _FakeVSIndex:
        return _FakeVSIndex()


class _FakeGenieClient:
    def start_conversation_and_wait(self, space_id: str, content: str) -> dict:
        return {"conversation_id": "c", "message_id": "m"}

    def get_message(self, **_: Any) -> dict:
        return {"attachments": [{"query": {"query": "SELECT 1"}}]}

    def get_message_query_result(self, **_: Any) -> dict:
        return {
            "statement_response": {
                "manifest": {"schema": {"columns": [{"name": "n"}]}},
                "result": {"data_array": [[42]]},
            }
        }


@pytest.fixture
def ctx(monkeypatch: pytest.MonkeyPatch) -> ToolContext:
    return ToolContext(
        cfg=_make_cfg(),
        vs_client=_FakeVSClient(),
        genie_client=_FakeGenieClient(),
        sql_executor=lambda _q: [{"nhtsa_item_number": "TSB-1", "summary": "OK"}],
        vs_cfg=VectorSearchConfig(num_results=2),
    )


def _agent(
    script: list[_FakeMessage],
    ctx: ToolContext,
    *,
    max_steps: int = 6,
) -> tuple[NhtsaAgent, InMemorySessionStore]:
    store = InMemorySessionStore()
    agent = NhtsaAgent(
        cfg=ctx.cfg,
        llm_client=_ScriptedLLM(script),
        tool_context=ctx,
        session_store=store,
        system_prompt="SYSTEM",
        max_tool_steps=max_steps,
    )
    return agent, store


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_single_tool_then_answer(ctx: ToolContext) -> None:
    script = [
        _FakeMessage(
            tool_calls=[_tool_call("1", GENIE_TOOL_NAME, {"question": "how many?"})]
        ),
        _FakeMessage(content="42 campaigns."),
    ]
    agent, store = _agent(script, ctx)
    result = agent.run_turn("How many Tesla recalls in 2023?")
    assert result.answer == "42 campaigns."
    assert result.stopped_reason == "ok"
    assert [t["name"] for t in result.tool_trace] == [GENIE_TOOL_NAME]
    assert result.n_llm_calls == 2
    # Store: user + assistant
    hist = store.get_history(result.session_id)
    assert [t.role for t in hist] == ["user", "assistant"]


def test_multiple_parallel_tool_calls_single_step(ctx: ToolContext) -> None:
    """When the LLM emits two tool calls in one response, both run."""
    script = [
        _FakeMessage(
            tool_calls=[
                _tool_call("1", GENIE_TOOL_NAME, {"question": "counts"}),
                _tool_call(
                    "2", VECTOR_SEARCH_TOOL_NAME,
                    {"query": "fire", "filters": {"make_norm": "Tesla"}},
                ),
            ]
        ),
        _FakeMessage(content="combined answer"),
    ]
    agent, _ = _agent(script, ctx)
    result = agent.run_turn("Combined question")
    names = [t["name"] for t in result.tool_trace]
    assert names == [GENIE_TOOL_NAME, VECTOR_SEARCH_TOOL_NAME]
    # VS filters should land in accumulated_filters.
    assert result.accumulated_filters == {"make_norm": "Tesla"}


def test_serial_tool_loop(ctx: ToolContext) -> None:
    """Two separate tool steps, then a final answer."""
    script = [
        _FakeMessage(
            tool_calls=[_tool_call("1", GENIE_TOOL_NAME, {"question": "q1"})]
        ),
        _FakeMessage(
            tool_calls=[
                _tool_call("2", VECTOR_SEARCH_TOOL_NAME, {"query": "fire"})
            ]
        ),
        _FakeMessage(content="final"),
    ]
    agent, _ = _agent(script, ctx)
    result = agent.run_turn("q")
    assert result.n_llm_calls == 3
    assert result.stopped_reason == "ok"
    assert [t["step"] for t in result.tool_trace] == [0, 1]


def test_fetch_tsb_routing(ctx: ToolContext) -> None:
    script = [
        _FakeMessage(
            tool_calls=[
                _tool_call("1", FETCH_TSB_TOOL_NAME, {"nhtsa_item_number": "10160095"})
            ]
        ),
        _FakeMessage(content="TSB found."),
    ]
    agent, _ = _agent(script, ctx)
    result = agent.run_turn("Tell me about TSB 10160095")
    assert result.tool_trace[0]["name"] == FETCH_TSB_TOOL_NAME
    assert result.tool_trace[0]["error"] is False


# ---------------------------------------------------------------------------
# Safety rails
# ---------------------------------------------------------------------------


def test_max_steps_terminates_runaway_loop(ctx: ToolContext) -> None:
    """A model that never stops calling tools is hard-stopped."""
    call = _tool_call("x", GENIE_TOOL_NAME, {"question": "q"})
    script = [_FakeMessage(tool_calls=[call]) for _ in range(10)]
    agent, _ = _agent(script, ctx, max_steps=3)
    result = agent.run_turn("spin")
    assert result.stopped_reason == "max_steps"
    # max_steps=3 means we allow 3 tool-execution rounds, then the 4th
    # LLM call is forced to terminate.
    assert len(result.tool_trace) == 3


def test_malformed_tool_args_do_not_crash(ctx: ToolContext) -> None:
    bad_tc = _FakeToolCall(id="1", name=GENIE_TOOL_NAME, arguments="{not json}")
    script = [
        _FakeMessage(tool_calls=[bad_tc]),
        _FakeMessage(content="recovered"),
    ]
    agent, _ = _agent(script, ctx)
    result = agent.run_turn("q")
    # Dispatched with {} args → Genie raises KeyError on missing 'question',
    # which is caught by execute_tool and surfaced as error=True.
    assert result.tool_trace[0]["error"] is True
    assert result.answer == "recovered"
    assert result.stopped_reason == "tool_error"


def test_system_prompt_injected_first(ctx: ToolContext) -> None:
    script = [_FakeMessage(content="hi")]
    agent, _ = _agent(script, ctx)
    agent.run_turn("hello")
    first_call = agent.llm_client.chat.completions.calls[0]
    messages = first_call["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "SYSTEM"


def test_tools_passed_to_every_llm_call(ctx: ToolContext) -> None:
    script = [
        _FakeMessage(
            tool_calls=[_tool_call("1", GENIE_TOOL_NAME, {"question": "q"})]
        ),
        _FakeMessage(content="done"),
    ]
    agent, _ = _agent(script, ctx)
    agent.run_turn("q")
    for call in agent.llm_client.chat.completions.calls:
        assert "tools" in call
        assert len(call["tools"]) == 4
        assert call["tool_choice"] == "auto"


# ---------------------------------------------------------------------------
# Multi-turn memory
# ---------------------------------------------------------------------------


def test_filters_accumulate_across_turns(ctx: ToolContext) -> None:
    # Turn 1: LLM asks VS with Tesla filter.
    script1 = [
        _FakeMessage(
            tool_calls=[
                _tool_call("1", VECTOR_SEARCH_TOOL_NAME,
                           {"query": "fire", "filters": {"make_norm": "Tesla"}})
            ]
        ),
        _FakeMessage(content="Fires on Teslas."),
    ]
    store = InMemorySessionStore()
    agent = NhtsaAgent(
        cfg=ctx.cfg,
        llm_client=_ScriptedLLM(script1),
        tool_context=ctx,
        session_store=store,
        system_prompt="SYS",
    )
    r1 = agent.run_turn("Tesla fires")
    assert store.get_accumulated_filters(r1.session_id) == {"make_norm": "Tesla"}

    # Turn 2: same session. Filters from turn 1 should show up as a
    # system message injected by _build_messages.
    agent.llm_client = _ScriptedLLM([_FakeMessage(content="still Tesla")])
    r2 = agent.run_turn("any more?", session_id=r1.session_id)
    injected = agent.llm_client.chat.completions.calls[0]["messages"]
    system_contents = [m["content"] for m in injected if m["role"] == "system"]
    assert any("make_norm" in c and "Tesla" in c for c in system_contents)
    assert r2.session_id == r1.session_id


def test_prior_user_turns_replayed(ctx: ToolContext) -> None:
    """Turn 2's LLM call should receive turn 1's user+assistant turns."""
    agent = NhtsaAgent(
        cfg=ctx.cfg,
        llm_client=_ScriptedLLM([_FakeMessage(content="a1")]),
        tool_context=ctx,
        session_store=InMemorySessionStore(),
        system_prompt="SYS",
    )
    r1 = agent.run_turn("first question")

    agent.llm_client = _ScriptedLLM([_FakeMessage(content="a2")])
    agent.run_turn("second question", session_id=r1.session_id)

    second_call_msgs = agent.llm_client.chat.completions.calls[0]["messages"]
    roles_and_content = [(m["role"], m["content"]) for m in second_call_msgs]
    # System prompt + prior user "first question" + prior assistant "a1" +
    # current user "second question".
    assert ("user", "first question") in roles_and_content
    assert ("assistant", "a1") in roles_and_content
    assert ("user", "second question") in roles_and_content


def test_tool_turns_not_replayed(ctx: ToolContext) -> None:
    """Prior tool-call rounds aren't re-sent — they'd lack matching IDs."""
    # Turn 1: one tool call.
    turn1 = [
        _FakeMessage(
            tool_calls=[_tool_call("1", GENIE_TOOL_NAME, {"question": "q"})]
        ),
        _FakeMessage(content="done1"),
    ]
    store = InMemorySessionStore()
    agent = NhtsaAgent(
        cfg=ctx.cfg,
        llm_client=_ScriptedLLM(turn1),
        tool_context=ctx,
        session_store=store,
        system_prompt="SYS",
    )
    r1 = agent.run_turn("q1")

    # Turn 2: re-check — no "tool" role should be in the outgoing messages.
    agent.llm_client = _ScriptedLLM([_FakeMessage(content="done2")])
    agent.run_turn("q2", session_id=r1.session_id)
    msgs = agent.llm_client.chat.completions.calls[0]["messages"]
    assert not any(m["role"] == "tool" for m in msgs)
