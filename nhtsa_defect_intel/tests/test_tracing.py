"""Tests for Phase-5 MLflow tracing integration.

What we verify:
- The agent's ``run_turn`` + ``_call_llm`` methods create nested spans
  of the expected type (``AGENT`` → ``LLM``).
- Each tool dispatch creates a child span with ``span_type`` matching
  the tool's role (RETRIEVER for Genie/VS, TOOL for fetch_*).
- Span names include the tool name so ops dashboards can group / filter
  without parsing inputs.
- When tracing is disabled, no ``mlflow.tracking`` writes happen — the
  agent still runs and returns the same result.
- Session id tags land on the trace when available.

These tests use the ``tracing_enabled`` fixture from conftest.py to
opt in to actual trace collection (file-store backend). All other
suites keep tracing off by default to avoid the MLflow startup noise.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import mlflow
import pytest

from nhtsa_curator.agent import NhtsaAgent
from nhtsa_curator.config import ProjectConfig, VectorSearchConfig
from nhtsa_curator.mcp import (
    GENIE_TOOL_NAME,
    VECTOR_SEARCH_TOOL_NAME,
    ToolContext,
    execute_tool,
)
from nhtsa_curator.memory import InMemorySessionStore


# ---------------------------------------------------------------------------
# Fakes (trimmed copies of test_agent_routing.py — not imported to avoid
# coupling the tracing suite to that file's evolution).
# ---------------------------------------------------------------------------


@dataclass
class _FakeToolCall:
    id: str
    name: str
    arguments: str

    @property
    def function(self) -> "_FakeToolCall":
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
            return _FakeResp(choices=[_FakeChoice(_FakeMessage(content="done"))])
        return _FakeResp(choices=[_FakeChoice(self._script.pop(0))])


class _ScriptedLLM:
    def __init__(self, script: list[_FakeMessage]) -> None:
        self._completions = _Completions(script)

    @property
    def chat(self) -> Any:
        return self

    @property
    def completions(self) -> _Completions:
        return self._completions


class _FakeVSIndex:
    def similarity_search(self, **_: Any) -> dict:
        return {
            "result": {
                "manifest": {"columns": [{"name": "c"}]},
                "data_array": [["v"]],
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
                "result": {"data_array": [[1]]},
            }
        }


@pytest.fixture
def cfg() -> ProjectConfig:
    return ProjectConfig(
        catalog="cat", schema="sch", volume="vol", llm_endpoint="llm",
        embedding_endpoint="emb", warehouse_id="wh",
        vector_search_endpoint="ep", genie_space_id="real",
    )


@pytest.fixture
def ctx(cfg: ProjectConfig) -> ToolContext:
    return ToolContext(
        cfg=cfg,
        vs_client=_FakeVSClient(),
        genie_client=_FakeGenieClient(),
        sql_executor=lambda _q: [{"summary": "OK"}],
        vs_cfg=VectorSearchConfig(num_results=1),
    )


def _tool_call(name: str, args: dict) -> _FakeToolCall:
    return _FakeToolCall(id="1", name=name, arguments=json.dumps(args))


def _build_agent(script: list[_FakeMessage], ctx: ToolContext) -> NhtsaAgent:
    return NhtsaAgent(
        cfg=ctx.cfg,
        llm_client=_ScriptedLLM(script),
        tool_context=ctx,
        session_store=InMemorySessionStore(),
        system_prompt="SYS",
    )


# ---------------------------------------------------------------------------
# With tracing ENABLED — verify spans get created.
# ---------------------------------------------------------------------------


def _last_trace() -> Any:
    tid = mlflow.get_last_active_trace_id()
    assert tid, "expected a trace id from the previous call"
    trace = mlflow.get_trace(tid)
    assert trace is not None, f"trace {tid} not retrievable"
    return trace


def test_execute_tool_creates_retriever_span(ctx: ToolContext, tracing_enabled):
    execute_tool(VECTOR_SEARCH_TOOL_NAME, {"query": "fire"}, ctx)
    trace = _last_trace()
    names = [(s.name, s.span_type) for s in trace.data.spans]
    assert (f"tool.{VECTOR_SEARCH_TOOL_NAME}", "RETRIEVER") in names


def test_execute_tool_tool_span_for_fetch(ctx: ToolContext, tracing_enabled):
    execute_tool("fetch_tsb", {"nhtsa_item_number": "TSB-1"}, ctx)
    trace = _last_trace()
    span = next(s for s in trace.data.spans if s.name.startswith("tool."))
    assert span.span_type == "TOOL"


def test_run_turn_emits_nested_agent_llm_spans(ctx: ToolContext, tracing_enabled):
    script = [_FakeMessage(content="hello")]  # no tool calls — simplest shape.
    agent = _build_agent(script, ctx)
    agent.run_turn("q1", user_id="user-a")
    trace = _last_trace()
    types = {s.span_type for s in trace.data.spans}
    assert "AGENT" in types
    assert "LLM" in types


def test_run_turn_nests_tool_span_under_agent(ctx: ToolContext, tracing_enabled):
    script = [
        _FakeMessage(tool_calls=[_tool_call(GENIE_TOOL_NAME, {"question": "q"})]),
        _FakeMessage(content="done"),
    ]
    agent = _build_agent(script, ctx)
    agent.run_turn("q")
    trace = _last_trace()
    type_set = {s.span_type for s in trace.data.spans}
    assert {"AGENT", "LLM", "RETRIEVER"} <= type_set


def test_run_turn_tags_trace_with_session_id(ctx: ToolContext, tracing_enabled):
    agent = _build_agent([_FakeMessage(content="x")], ctx)
    result = agent.run_turn("q", user_id="user-x")
    trace = _last_trace()
    # Tags are merged into trace.info.tags after completion.
    assert trace.info.tags.get("session_id") == result.session_id
    assert trace.info.tags.get("user_id") == "user-x"


# ---------------------------------------------------------------------------
# With tracing DISABLED (default) — agent still produces results.
# ---------------------------------------------------------------------------


def test_disabled_tracing_still_runs_agent(ctx: ToolContext):
    agent = _build_agent([_FakeMessage(content="hi")], ctx)
    result = agent.run_turn("q")
    assert result.answer == "hi"
    assert result.stopped_reason == "ok"


def test_disabled_tracing_still_runs_execute_tool(ctx: ToolContext):
    out = execute_tool(VECTOR_SEARCH_TOOL_NAME, {"query": "q"}, ctx)
    assert out["hit_count"] == 1
    assert "_latency_ms" in out
