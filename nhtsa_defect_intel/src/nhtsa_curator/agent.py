"""The NHTSA defect-intelligence agent.

Responsibilities:
    1. Keep a long-running tool-call loop between the LLM and the four
       tools defined in :mod:`nhtsa_curator.mcp`.
    2. Persist every turn to the session store so multi-turn refinements
       can be resumed (Lakebase in prod, in-memory in tests).
    3. Merge each turn's filters into the session's ``accumulated_filters``
       bag so the LLM gets prior constraints back on its next call.
    4. Return a structured result (answer text + tool trace + final
       filters) so callers — notebook driver, MLflow wrapper, eval
       harness — all speak the same shape.

What this file is NOT:

    - An MLflow-logged model. That's :mod:`nhtsa_curator.serving`.
    - An async loop. The tool calls are sequential on purpose: Genie and
      Vector Search both have perceivable latency, and the LLM needs
      each tool's output before planning the next. Parallel fan-out is
      a later optimisation once we have traces to measure with.

Tracing (Phase 5):
    ``run_turn`` is decorated with ``@mlflow.trace(span_type="AGENT")``,
    and ``_execute_llm_call`` is decorated with ``span_type="LLM"``. The
    per-tool spans come from :mod:`nhtsa_curator.mcp`. Together they
    produce a nested trace per user turn: AGENT → LLM → TOOL → (RETRIEVER).
    Tracing is enabled only when an MLflow tracking URI is configured —
    at import time it's a no-op that adds ~2 µs of overhead per call.

LLM calling contract:
    The LLM client is OpenAI-compatible (``openai.OpenAI``) but pointed
    at a Databricks serving endpoint. Tests pass a ``FakeLLM`` that
    satisfies the same ``chat.completions.create(...)`` signature.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

import mlflow
from loguru import logger

from .config import ProjectConfig, load_system_prompt
from .mcp import (
    ALL_TOOL_NAMES,
    VECTOR_SEARCH_TOOL_NAME,
    ToolContext,
    execute_tool,
    merge_filters,
    serialise_tool_result,
    tool_specs,
)
from .memory import SessionStore, Turn

# ---------------------------------------------------------------------------
# LLM client Protocol — so tests can inject a fake without importing openai.
# ---------------------------------------------------------------------------


class _ChatCompletions(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


class _ChatNamespace(Protocol):
    completions: _ChatCompletions


class LLMClient(Protocol):
    """Minimal slice of ``openai.OpenAI`` the agent uses."""

    chat: _ChatNamespace


# ---------------------------------------------------------------------------
# Result type — stable shape for the notebook, eval harness, and serving.
# ---------------------------------------------------------------------------


@dataclass
class AgentResult:
    """What a single call to :meth:`NhtsaAgent.run_turn` returns."""

    session_id: str
    answer: str
    tool_trace: list[dict] = field(default_factory=list)
    accumulated_filters: dict = field(default_factory=dict)
    n_llm_calls: int = 0
    stopped_reason: str = "ok"  # ok | max_steps | tool_error

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "answer": self.answer,
            "tool_trace": self.tool_trace,
            "accumulated_filters": self.accumulated_filters,
            "n_llm_calls": self.n_llm_calls,
            "stopped_reason": self.stopped_reason,
        }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class NhtsaAgent:
    """Tool-calling agent around the four NHTSA tools.

    Construction takes every collaborator (LLM client, tool context,
    session store, system prompt) so a test / notebook / serving wrapper
    all build the agent the same way — no hidden singletons, no
    environment-dependent module-level state.
    """

    def __init__(
        self,
        cfg: ProjectConfig,
        llm_client: LLMClient,
        tool_context: ToolContext,
        session_store: SessionStore,
        system_prompt: str | None = None,
        *,
        max_tool_steps: int = 6,
        history_window: int = 20,
    ) -> None:
        """
        Args:
            max_tool_steps: safety rail so a confused LLM can't spin
                forever. 6 is chosen because the worst-case legitimate
                trajectory is `genie + vs_search + fetch_tsb +
                fetch_investigation + final_answer` with one retry —
                ~5 steps.
            history_window: how many prior turns to replay on each call.
                Covers the typical 5-10 turn session without blowing the
                model's context on very long ones.
        """
        self.cfg = cfg
        self.llm_client = llm_client
        self.tool_context = tool_context
        self.session_store = session_store
        self.system_prompt = system_prompt or load_system_prompt()
        self.max_tool_steps = max_tool_steps
        self.history_window = history_window
        self._tools = tool_specs(cfg)

    # -------------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------------

    @mlflow.trace(span_type="AGENT", name="NhtsaAgent.run_turn")
    def run_turn(
        self,
        user_message: str,
        session_id: str | None = None,
        *,
        user_id: str | None = None,
    ) -> AgentResult:
        """Execute one user turn end-to-end.

        Creates a session if ``session_id`` is None. Persists the user
        message first so a mid-turn crash still records what was asked.
        """
        if session_id is None:
            session_id = self.session_store.create_session(user_id=user_id)
        else:
            # Client supplied an id (Responses API, eval harness). The
            # parent row in `agent_sessions` may not exist yet — create
            # it idempotently so `append_turn` doesn't hit the FK.
            self.session_store.ensure_session(session_id, user_id=user_id)

        # Tag the running trace so downstream ops dashboards can group by
        # session / user without hitting the Lakebase store. ``update_
        # current_trace`` is a no-op if tracing is disabled, so this is
        # safe in tests.
        _tag_current_trace(session_id=session_id, user_id=user_id)

        user_idx = self.session_store.next_turn_idx(session_id)
        self.session_store.append_turn(
            Turn(
                session_id=session_id,
                turn_idx=user_idx,
                role="user",
                content=user_message,
            )
        )

        prior_filters = self.session_store.get_accumulated_filters(session_id)
        messages = self._build_messages(session_id, user_message, prior_filters)

        tool_trace: list[dict] = []
        n_llm_calls = 0
        stopped = "ok"

        for step in range(self.max_tool_steps + 1):
            n_llm_calls += 1
            resp = self._call_llm(messages=messages)
            choice = _first_choice(resp)
            msg = _message_of(choice)

            tool_calls = _tool_calls_of(msg)

            # Llama-class fallback. Some Databricks foundation models
            # (notably ``llama-4-maverick``) intermittently emit tool
            # calls as Python-style text content
            # (``genie_recalls(question="...")``) instead of populating
            # the structured ``tool_calls`` field. Without recovery the
            # loop would treat the leaked syntax as the final answer
            # and return junk to the user. Detected via the per-tool
            # name registry so we never accidentally re-interpret real
            # narrative answers as calls.
            if not tool_calls:
                raw_content = (
                    getattr(msg, "content", None)
                    if not isinstance(msg, dict)
                    else msg.get("content")
                )
                recovered = _recover_textual_tool_call(raw_content, ALL_TOOL_NAMES)
                if recovered is not None:
                    logger.warning(
                        "Recovered textual tool call: {}({})",
                        recovered["function"]["name"],
                        recovered["function"]["arguments"][:80],
                    )
                    tool_calls = [recovered]
                    messages.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [recovered],
                        }
                    )
                else:
                    # Genuine final answer — append unmodified and stop.
                    messages.append(_assistant_message_dict(msg))
                    break
            else:
                messages.append(_assistant_message_dict(msg))

            if step == self.max_tool_steps:
                stopped = "max_steps"
                logger.warning(
                    "Agent hit max_tool_steps={} — forcing termination",
                    self.max_tool_steps,
                )
                break

            # Dispatch every requested tool call in order, appending each
            # result back into the message list so the next LLM hop has it.
            new_filters: dict = {}
            for tc in tool_calls:
                name = _tool_name(tc)
                raw_args = _tool_args(tc)
                args = _parse_args(raw_args)

                result = execute_tool(name, args, self.tool_context)
                tool_trace.append(
                    {
                        "step": step,
                        "name": name,
                        "args": args,
                        "result_preview": _preview(result),
                        "latency_ms": result.get("_latency_ms"),
                        "error": bool(result.get("error")),
                    }
                )

                # If the LLM called vector_search with structured filters,
                # merge them into the session-level bag so subsequent
                # turns inherit them.
                if name == VECTOR_SEARCH_TOOL_NAME:
                    new_filters = merge_filters(new_filters, args.get("filters"))

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": _tool_call_id(tc),
                        "name": name,
                        "content": serialise_tool_result(result),
                    }
                )

            if new_filters:
                prior_filters = merge_filters(prior_filters, new_filters)

        answer = _final_text(messages)
        self._persist_assistant_turn(session_id, answer, tool_trace)
        self.session_store.set_accumulated_filters(session_id, prior_filters)

        if any(t["error"] for t in tool_trace) and stopped == "ok":
            stopped = "tool_error"

        return AgentResult(
            session_id=session_id,
            answer=answer,
            tool_trace=tool_trace,
            accumulated_filters=prior_filters,
            n_llm_calls=n_llm_calls,
            stopped_reason=stopped,
        )

    # -------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------

    @mlflow.trace(span_type="LLM", name="NhtsaAgent._call_llm")
    def _call_llm(self, messages: list[dict]) -> Any:
        """Thin wrapper that exists so the LLM call gets its own span.

        Making it a method (rather than inlining the client call) means
        the trace tree is AGENT → LLM per round-trip, which is what ops
        dashboards want when triaging tool-loop cost.
        """
        return self.llm_client.chat.completions.create(
            model=self.cfg.llm_endpoint,
            messages=messages,
            tools=self._tools,
            tool_choice="auto",
            temperature=0.2,
        )

    def _build_messages(
        self,
        session_id: str,
        user_message: str,
        prior_filters: dict,
    ) -> list[dict]:
        """Assemble the LLM message list from system + history + user."""
        messages: list[dict] = [{"role": "system", "content": self.system_prompt}]

        if prior_filters:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Accumulated filters from earlier turns (apply when "
                        "relevant, override if the user contradicts): "
                        + json.dumps(prior_filters)
                    ),
                }
            )

        history = self.session_store.get_history(session_id, self.history_window)
        for turn in history:
            # Skip the just-appended user turn — we'll add it last.
            if turn.role == "user" and turn.content == user_message:
                continue
            if turn.role == "tool":
                # Tool turns from previous runs are logged for audit,
                # but re-sending them confuses the LLM (no matching
                # tool_call_id in the current request). Drop them.
                continue
            messages.append({"role": turn.role, "content": turn.content})

        messages.append({"role": "user", "content": user_message})
        return messages

    def _persist_assistant_turn(
        self, session_id: str, answer: str, tool_trace: list[dict]
    ) -> None:
        idx = self.session_store.next_turn_idx(session_id)
        self.session_store.append_turn(
            Turn(
                session_id=session_id,
                turn_idx=idx,
                role="assistant",
                content=answer,
                tool_calls=tool_trace,
            )
        )


# ---------------------------------------------------------------------------
# Tracing helper — wrapper around mlflow.update_current_trace that swallows
# "no active trace" errors so the agent doesn't need to know whether
# tracing is enabled at the call site.
# ---------------------------------------------------------------------------


def _tag_current_trace(**tags: Any) -> None:
    """Attach k/v tags to the active trace; no-op when tracing is off.

    ``mlflow.update_current_trace`` logs a warning when there is no
    active span — which happens every time tracing is disabled, which is
    every test run. We short-circuit by checking for an active span
    first to keep test output clean.
    """
    try:
        if mlflow.get_current_active_span() is None:
            return
        mlflow.update_current_trace(
            tags={k: str(v) for k, v in tags.items() if v is not None}
        )
    except Exception:  # noqa: BLE001 — tracing must never break the agent.
        pass


# ---------------------------------------------------------------------------
# Response-shape helpers — tolerant of both openai SDK objects and plain
# dicts (fakes in tests).
# ---------------------------------------------------------------------------


def _first_choice(resp: Any) -> Any:
    choices = getattr(resp, "choices", None) or resp["choices"]
    return choices[0]


def _message_of(choice: Any) -> Any:
    return getattr(choice, "message", None) or choice["message"]


def _tool_calls_of(msg: Any) -> list[Any]:
    tc = getattr(msg, "tool_calls", None)
    if tc is None and isinstance(msg, dict):
        tc = msg.get("tool_calls")
    return tc or []


def _tool_name(tc: Any) -> str:
    fn = getattr(tc, "function", None) or tc["function"]
    return getattr(fn, "name", None) or fn["name"]


def _tool_args(tc: Any) -> str:
    fn = getattr(tc, "function", None) or tc["function"]
    return getattr(fn, "arguments", None) or fn.get("arguments", "{}")


def _tool_call_id(tc: Any) -> str:
    return getattr(tc, "id", None) or tc["id"]


def _parse_args(raw: str | dict) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        logger.warning("Tool args not valid JSON: {}", raw[:120])
        return {}


# Matches a single Python-style call expression as the entire content
# (optional whitespace either side). DOTALL so multi-line argument
# strings still match. Anchored end-to-end on purpose: a paragraph that
# happens to contain a parenthesised phrase must not be misclassified.
_TEXTUAL_TOOL_CALL_RE = re.compile(r"^\s*(\w+)\s*\((.*)\)\s*$", re.DOTALL)
_KWARG_RE = re.compile(r'(\w+)\s*=\s*(?:"((?:[^"\\]|\\.)*)"|\'((?:[^\'\\]|\\.)*)\')')


def _recover_textual_tool_call(
    content: Any, allowed_names: tuple[str, ...] | set[str] | frozenset[str]
) -> dict | None:
    """Best-effort parse of a Python-style call out of assistant content.

    Why: ``databricks-llama-4-maverick`` (and other Llama-class endpoints)
    intermittently emit calls like ``genie_recalls(question="...")`` in
    the assistant ``content`` field instead of populating ``tool_calls``.
    Without recovery the agent loop sees no tool calls, breaks, and
    returns the leaked syntax to the user as the final answer.

    Returns a tool-call dict in OpenAI shape, or None if the content
    does not look like a single call to a known tool.
    """
    if not isinstance(content, str) or not content.strip():
        return None
    m = _TEXTUAL_TOOL_CALL_RE.match(content)
    if not m:
        return None
    name = m.group(1)
    body = m.group(2).strip()
    if name not in allowed_names:
        return None

    args: dict | None = None
    if body.startswith("{"):
        try:
            args = json.loads(body)
        except json.JSONDecodeError:
            args = None
    if args is None:
        args = {}
        for km in _KWARG_RE.finditer(body):
            key = km.group(1)
            val = km.group(2) if km.group(2) is not None else km.group(3)
            args[key] = val
        if not args:
            return None

    return {
        "id": f"recovered_{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _assistant_message_dict(msg: Any) -> dict:
    """Normalise an assistant message (object or dict) for replay."""
    if isinstance(msg, dict):
        out = {"role": "assistant", "content": msg.get("content")}
        if msg.get("tool_calls"):
            out["tool_calls"] = msg["tool_calls"]
        return out

    out = {"role": "assistant", "content": getattr(msg, "content", None)}
    tc = getattr(msg, "tool_calls", None)
    if tc:
        out["tool_calls"] = [
            {
                "id": _tool_call_id(t),
                "type": "function",
                "function": {
                    "name": _tool_name(t),
                    "arguments": _tool_args(t),
                },
            }
            for t in tc
        ]
    return out


def _final_text(messages: list[dict]) -> str:
    """Find the last assistant message with non-empty content."""
    for m in reversed(messages):
        if m.get("role") == "assistant" and m.get("content"):
            return m["content"]
    return ""


def _preview(result: dict, max_chars: int = 400) -> str:
    """Short string preview of a tool result for the trace log."""
    try:
        s = json.dumps(result, default=str, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        s = str(result)
    return s[:max_chars]


def log_register_agent(
    cfg: Any,
    git_sha: str,
    run_id: str,
    agent_code_path: str,
    model_name: str,
    evaluation_metrics: dict | None = None,
    gold_delta_version: str | None = None,
    experiment_name: str | None = None,
    code_paths: list[str] | None = None,
    pip_requirements: list[str] | None = None,
) -> Any:
    """Log ``agent_code_path`` as an MLflow model and register to Unity Catalog.

    This is the "turn a file into a Champion candidate" step of Phase 6.
    Called once by ``resources/deployment_scripts/log_register_agent.py``.

    What it does:

    1. Declares the set of Databricks resources the agent needs at serve
       time (LLM endpoint, Genie space, VS index, gold tables, SQL
       warehouse). MLflow attaches these to the logged model so
       ``databricks.agents.deploy`` can pre-authorise them on the
       serving principal — no runtime 403s.
    2. Writes a ``model_config`` dict matching the ``ModelConfig``
       shape in ``nhtsa_agent_pg.py`` so the logged model loads the
       right env.
    3. Logs, registers, sets ``latest-model`` alias.

    ``cfg`` is typed as ``Any`` rather than ``ProjectConfig`` to avoid a
    circular import with ``.config`` — the attribute access is what
    matters, and typecheckers verify it.
    """
    from datetime import datetime  # noqa: PLC0415

    from mlflow import MlflowClient  # noqa: PLC0415
    from mlflow.models.resources import (  # noqa: PLC0415
        DatabricksGenieSpace,
        DatabricksServingEndpoint,
        DatabricksSQLWarehouse,
        DatabricksTable,
        DatabricksVectorSearchIndex,
    )

    full_schema = f"{cfg.catalog}.{cfg.db_schema}"
    vs_index_name = f"{full_schema}.gold_narrative_chunks_index"

    resources = [
        DatabricksServingEndpoint(endpoint_name=cfg.llm_endpoint),
        DatabricksServingEndpoint(endpoint_name=cfg.embedding_endpoint),
        DatabricksSQLWarehouse(warehouse_id=cfg.warehouse_id),
        DatabricksVectorSearchIndex(index_name=vs_index_name),
        # Gold tables the agent's Genie tool reads. Declaring each one
        # explicitly means serving-principal authorisation is tight;
        # granting the whole schema would be easier but over-broad.
        DatabricksTable(table_name=f"{full_schema}.gold_recalls_fact"),
        DatabricksTable(table_name=f"{full_schema}.gold_complaints_fact"),
        DatabricksTable(table_name=f"{full_schema}.gold_investigations_fact"),
        # TBD for the below tables
        # DatabricksTable(table_name=f"{full_schema}.gold_tsb_meta"),
        # DatabricksTable(table_name=f"{full_schema}.gold_sgo_av_crashes"),
        DatabricksTable(table_name=f"{full_schema}.gold_narrative_chunks"),
    ]
    if cfg.genie_space_id and not cfg.genie_space_id.startswith("PLACEHOLDER"):
        resources.append(DatabricksGenieSpace(genie_space_id=cfg.genie_space_id))
    # Lakebase is intentionally NOT declared as an MLflow resource. The
    # ``DatabricksLakebase`` class only supports the older ``Database
    # Instance`` API; our project is created via the newer ``PostgresAPI``
    # (Project model) in ``4.2_lakebase_setup.py``, so the deploy-time
    # auto-grant lookup fails with ``NOT_FOUND: Database instance``.
    # Serving-pod Lakebase auth uses the dedicated ``dev_SPN`` instead,
    # injected as ``LAKEBASE_SP_*`` env vars in ``deploy_agent.py`` and
    # granted a Postgres role in ``4.3_grant_dev_spn_lakebase.py``.

    model_config = {
        "env": "dev",  # overwritten per-target below when deploy_agent runs
        "config_path": "project_config.yml",
        "catalog": cfg.catalog,
        "schema": cfg.db_schema,
        "llm_endpoint": cfg.llm_endpoint,
        "genie_space_id": cfg.genie_space_id,
        "lakebase_project_id": cfg.lakebase_project_id,
    }

    # The input example is used by MLflow's ``validate_serving_input``
    # to smoke-test the model before it's deployed. Keep the question
    # intentionally broad so it touches both Genie and VS tools.
    test_request = {
        "input": [
            {
                "role": "user",
                "content": (
                    "Summarise the top 3 recall campaigns involving "
                    "braking system components in 2023, and cite each "
                    "campaign number."
                ),
            }
        ],
        "custom_inputs": {
            # Must be a valid UUID — the Lakebase `agent_sessions` table
            # types `session_id` as UUID, so `"demo-session"` is rejected
            # by Postgres during the MLflow log-time input_example probe.
            "session_id": str(uuid.uuid4()),
            "request_id": f"demo-request-{uuid.uuid4().hex[:8]}",
        },
    }

    experiment = experiment_name or cfg.experiment_name
    mlflow.set_experiment(experiment)
    ts = datetime.now().strftime("%Y-%m-%d")

    run_tags = {
        "git_sha": git_sha,
        "run_id": run_id,
        "project": "nhtsa-defect-intel",
        "phase": "6",
    }
    if gold_delta_version:
        # Databricks registered-model tags reject `.`, `=`, `/`, `>`, `<`,
        # `%`, `&`, `?`, `\` in *keys*. Values are unrestricted, so we
        # keep the per-table "name=version" list but use an underscore
        # in the key itself.
        run_tags["gold_delta_version"] = gold_delta_version

    with mlflow.start_run(
        run_name=f"nhtsa-agent-{ts}",
        tags=run_tags,
    ):
        # ``code_paths`` ships the ``nhtsa_curator`` source tree inside the
        # model artifacts — otherwise MLflow auto-infers
        # ``nhtsa-curator==0.0.1`` as a pip requirement (the bundle-built
        # wheel is installed in the calling job env) and the serving
        # container fails at build time because the wheel isn't on any
        # pip index.
        # ``pip_requirements`` is pinned explicitly for the same reason —
        # it replaces MLflow's auto-inferred set, which would otherwise
        # include the local package. Keep this list in sync with
        # ``pyproject.toml`` dependencies.
        model_info = mlflow.pyfunc.log_model(
            name="agent",
            python_model=agent_code_path,
            resources=resources,
            input_example=test_request,
            model_config=model_config,
            code_paths=code_paths,
            pip_requirements=pip_requirements,
        )
        if evaluation_metrics:
            # ``log_metrics`` refuses non-float values, so filter.
            flat = {
                k: float(v)
                for k, v in evaluation_metrics.items()
                if isinstance(v, (int, float))
            }
            if flat:
                mlflow.log_metrics(flat)

    logger.info("Registering model: {}", model_name)
    registered_model = mlflow.register_model(
        model_uri=model_info.model_uri,
        name=model_name,
        tags=run_tags,
    )
    logger.info("Registered version: {}", registered_model.version)

    client = MlflowClient()
    client.set_registered_model_alias(
        name=model_name,
        alias="latest-model",
        version=registered_model.version,
    )
    # Champion alias is moved separately by the promotion step — we
    # only stamp ``latest-model`` here, never ``champion``.
    return registered_model


__all__ = [
    "NhtsaAgent",
    "AgentResult",
    "LLMClient",
    "ALL_TOOL_NAMES",
    "log_register_agent",
]
