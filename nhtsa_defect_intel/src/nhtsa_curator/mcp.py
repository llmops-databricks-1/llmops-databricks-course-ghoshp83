"""Tool layer for the NHTSA Defect Intelligence agent.

The agent exposes exactly **four** tools (per ``docs/04_agent_design.md``).
Narrow tool surfaces produce far better LLM routing decisions — every
extra tool widens the prompt and dilutes the model's signal on which
tool to call.

    1. ``genie_recalls``             → Managed MCP against the Genie space
                                        for structured SQL over gold facts.
    2. ``vector_search_narrative``   → Managed MCP against the Vector
                                        Search index over narratives.
    3. ``fetch_tsb``                 → UC lookup by ``nhtsa_item_number``.
    4. ``fetch_investigation``       → UC lookup by ``nhtsa_action_number``.

Why this file is "mcp.py" even though ``fetch_tsb`` /
``fetch_investigation`` aren't MCP servers: the agent loop doesn't care
what's on the other side of a tool call. From the LLM's perspective all
four are just function specs. We keep them together so the dispatcher
is a single switch, and so adding a 5th tool means touching one file.

Testability is the other driver. We pass a ``ToolContext`` holding the
VS client, Genie client, and a SQL executor. Tests inject fakes; prod
notebooks inject the real Databricks SDK clients. The tool functions
themselves never import Databricks modules directly.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import mlflow
from loguru import logger

from .config import ProjectConfig, VectorSearchConfig
from .genie import GENIE_TABLES
from .vector_search import INDEX_COLUMNS, similarity_search


# ---------------------------------------------------------------------------
# Tool names — exported so the agent + tests can reference them.
# ---------------------------------------------------------------------------

GENIE_TOOL_NAME = "genie_recalls"
VECTOR_SEARCH_TOOL_NAME = "vector_search_narrative"
FETCH_TSB_TOOL_NAME = "fetch_tsb"
FETCH_INVESTIGATION_TOOL_NAME = "fetch_investigation"

ALL_TOOL_NAMES = (
    GENIE_TOOL_NAME,
    VECTOR_SEARCH_TOOL_NAME,
    FETCH_TSB_TOOL_NAME,
    FETCH_INVESTIGATION_TOOL_NAME,
)


# ---------------------------------------------------------------------------
# Context — the dependency bundle every tool needs.
# ---------------------------------------------------------------------------

class SqlExecutor(Protocol):
    """Minimal protocol any SQL-runner needs to implement.

    Real impls: ``spark.sql`` (returns a DataFrame we ``.collect()``) or
    the Databricks SDK's ``statement_execution.execute_statement``. We
    ultimately just want ``list[dict]``.
    """

    def __call__(self, query: str) -> list[dict]: ...


@dataclass
class ToolContext:
    """Dependency bundle carried through tool dispatch.

    Keeping this as a dataclass (rather than individual function
    arguments) means the agent loop passes a single opaque object to
    :func:`execute_tool`, and tests can construct a context without
    instantiating any Databricks SDK client.
    """

    cfg: ProjectConfig
    vs_client: Any | None = None
    genie_client: Any | None = None  # databricks.sdk.WorkspaceClient().genie
    sql_executor: SqlExecutor | None = None
    vs_cfg: VectorSearchConfig = field(default_factory=VectorSearchConfig)

    # Optional cap on NL→SQL wait in seconds; Genie runs can be slow on
    # cold warehouses.
    genie_timeout_s: int = 60


# ---------------------------------------------------------------------------
# Tool specs — the OpenAI-compatible function descriptions passed to the LLM.
# ---------------------------------------------------------------------------

def _vs_filter_schema() -> dict:
    """The filter subset we expose to the LLM.

    Kept deliberately small: exposing every column in the index would
    invite the model to invent filter keys. Only the ones we know the
    index indexes + we want the agent to routinely use.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "source_dataset": {
                "type": "string",
                "enum": ["complaints", "tsbs", "investigations", "sgo"],
                "description": "Restrict to one narrative source.",
            },
            "make_norm": {
                "type": "string",
                "description": "Canonical manufacturer brand, title-cased.",
            },
            "model_year": {
                "type": "integer",
                "description": "4-digit model year.",
            },
            "component_group": {
                "type": "string",
                "description": (
                    "Top-level NHTSA component category, e.g. "
                    "'Engine And Engine Cooling'."
                ),
            },
            "oem_group": {
                "type": "string",
                "description": "Corporate parent, e.g. 'General Motors'.",
            },
        },
    }


def tool_specs(cfg: ProjectConfig) -> list[dict]:
    """Return the 4 OpenAI-function tool specs for this agent.

    ``cfg`` is accepted (but mostly unused) so the spec list can be
    specialised later — e.g. to pre-bind the Genie space id into the
    description, so the LLM sees which gold tables are in scope.
    """
    genie_scope = ", ".join(GENIE_TABLES)
    return [
        {
            "type": "function",
            "function": {
                "name": GENIE_TOOL_NAME,
                "description": (
                    "Answer structured questions over NHTSA gold fact "
                    "tables via Genie text-to-SQL. Use this for counts, "
                    "rankings, time-series, and any question that mentions "
                    "'how many', 'top N', 'by year', or an aggregate. "
                    f"In-scope tables: {genie_scope}. Returns rows + "
                    "the generated SQL."
                ),
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": (
                                "The natural-language question to run "
                                "through Genie. Include any filters "
                                "(make, year, component) inline."
                            ),
                        }
                    },
                    "required": ["question"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": VECTOR_SEARCH_TOOL_NAME,
                "description": (
                    "Semantic retrieval over NHTSA complaint narratives, "
                    "TSB descriptions, and investigation document chunks. "
                    "Use this for 'what kinds of issues', 'find similar', "
                    "'examples of', and any qualitative-evidence question. "
                    "Returns top chunks with content + metadata."
                ),
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Free-text query to embed.",
                        },
                        "filters": _vs_filter_schema(),
                        "num_results": {
                            "type": "integer",
                            "description": (
                                "How many chunks to return (1-50). Defaults "
                                "to the configured vector-search num_results."
                            ),
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": FETCH_TSB_TOOL_NAME,
                "description": (
                    "Retrieve the parsed content + metadata for one TSB "
                    "given its NHTSA item number. Use only when the user "
                    "references a specific TSB or asks for full detail."
                ),
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "nhtsa_item_number": {
                            "type": "string",
                            "description": "NHTSA TSB item number.",
                        }
                    },
                    "required": ["nhtsa_item_number"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": FETCH_INVESTIGATION_TOOL_NAME,
                "description": (
                    "Retrieve one investigation case file by NHTSA action "
                    "number. Use only when the user references a specific "
                    "investigation."
                ),
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "nhtsa_action_number": {
                            "type": "string",
                            "description": "NHTSA investigation action number.",
                        }
                    },
                    "required": ["nhtsa_action_number"],
                },
            },
        },
    ]


# ---------------------------------------------------------------------------
# Tool implementations.
# ---------------------------------------------------------------------------

def _run_genie(ctx: ToolContext, question: str) -> dict:
    """Submit a question to the configured Genie space and return results.

    Uses the Databricks SDK's Genie API. The happy path:

        c = WorkspaceClient().genie
        conv = c.start_conversation_and_wait(space_id, content=question)
        msg  = c.get_message(space_id, conv.conversation_id, conv.message_id)
        attachment = msg.attachments[0]
        query      = attachment.query          # generated SQL
        result     = c.get_message_query_result(
            space_id, conv.conversation_id, conv.message_id
        )

    Returns ``{"sql": str, "rows": list[dict], "row_count": int}``. If
    the space id is unset, raises so the caller can surface a clear
    error (agent.py catches and reports back to the LLM as a tool error).
    """
    if not ctx.cfg.genie_space_id or ctx.cfg.genie_space_id.startswith("PLACEHOLDER"):
        raise RuntimeError(
            "genie_space_id is unset. Create the space and update "
            "project_config.yml before calling the Genie tool."
        )
    if ctx.genie_client is None:
        raise RuntimeError("ToolContext.genie_client is not wired.")

    client = ctx.genie_client
    logger.info("Genie question: {}", question)
    conv = client.start_conversation_and_wait(
        space_id=ctx.cfg.genie_space_id, content=question
    )

    # Different SDK shapes return conv differently — normalise.
    conv_id = getattr(conv, "conversation_id", None) or conv["conversation_id"]
    msg_id = getattr(conv, "message_id", None) or conv["message_id"]

    msg = client.get_message(
        space_id=ctx.cfg.genie_space_id, conversation_id=conv_id, message_id=msg_id
    )
    attachments = getattr(msg, "attachments", None) or (
        msg.get("attachments") if isinstance(msg, dict) else None
    ) or []
    sql = ""
    if attachments:
        first = attachments[0]
        # ``first`` is a ``GenieAttachment`` object (dataclass-like) in the
        # live SDK; older SDK shapes / mocks return a dict. Handle both
        # without assuming ``.get`` — on a text-only response ``.query`` is
        # simply ``None`` and we fall through to ``sql=""``.
        q = getattr(first, "query", None)
        if q is None and isinstance(first, dict):
            q = first.get("query")
        if q is not None:
            inner = getattr(q, "query", None)
            if inner is None and isinstance(q, dict):
                inner = q.get("query")
            sql = inner or ""

    res = client.get_message_query_result(
        space_id=ctx.cfg.genie_space_id, conversation_id=conv_id, message_id=msg_id
    )
    rows = _normalise_statement_result(res)
    return {"sql": sql, "rows": rows, "row_count": len(rows)}


def _normalise_statement_result(res: Any) -> list[dict]:
    """Turn a Databricks statement-execution result into ``list[dict]``.

    Accepts both the pydantic-ish object shape from the SDK and the
    raw-dict shape so tests can pass plain dicts without mocking the
    whole SDK.
    """
    sr = getattr(res, "statement_response", None)
    if sr is None:
        sr = res["statement_response"] if isinstance(res, dict) and "statement_response" in res else res
    if isinstance(sr, dict):
        manifest = sr.get("manifest", {})
        cols = [c["name"] for c in manifest.get("schema", {}).get("columns", [])]
        data = sr.get("result", {}).get("data_array") or []
    else:
        manifest = getattr(sr, "manifest", None)
        cols = [c.name for c in manifest.schema.columns] if manifest else []
        data = getattr(getattr(sr, "result", None), "data_array", []) or []
    return [dict(zip(cols, row, strict=False)) for row in data]


def _run_vector_search(ctx: ToolContext, query: str, filters: dict | None,
                      num_results: int | None) -> dict:
    """Run a filtered similarity search and return flattened hits."""
    if ctx.vs_client is None:
        raise RuntimeError("ToolContext.vs_client is not wired.")

    index_name = f"{ctx.cfg.full_schema_name}.gold_narrative_chunks_index"
    vs_cfg = ctx.vs_cfg
    if num_results is not None:
        vs_cfg = VectorSearchConfig(
            embedding_dimension=vs_cfg.embedding_dimension,
            similarity_metric=vs_cfg.similarity_metric,
            num_results=num_results,
        )

    hits = similarity_search(
        client=ctx.vs_client,
        endpoint_name=ctx.cfg.vector_search_endpoint,
        index_name=index_name,
        query_text=query,
        vs_cfg=vs_cfg,
        filters=filters or {},
        columns=INDEX_COLUMNS,
    )
    return {"hits": hits, "hit_count": len(hits)}


def _run_fetch_tsb(ctx: ToolContext, nhtsa_item_number: str) -> dict:
    """Fetch a single TSB row from ``silver_tsbs``.

    NHTSA's May-2024 MfrComms rewrite inlined the bulletin text into
    ``summary`` (up to 4000 chars) and dropped per-TSB PDF URLs, so
    there is no ``silver_tsbs_parsed`` table anymore — content and
    metadata both live on ``silver_tsbs``.
    """
    if ctx.sql_executor is None:
        raise RuntimeError("ToolContext.sql_executor is not wired.")
    sql = (
        f"SELECT nhtsa_item_number, tsb_id, make_norm, model_norm, model_year, "
        f"component_raw, component_group, communication_type, summary, "
        f"mfr_component_system, mfr_component_subsystem, "
        f"original_date, changed_date "
        f"FROM {ctx.cfg.full_schema_name}.silver_tsbs "
        f"WHERE nhtsa_item_number = '{_sql_quote(nhtsa_item_number)}' "
        f"LIMIT 1"
    )
    rows = ctx.sql_executor(sql)
    if not rows:
        return {"found": False, "nhtsa_item_number": nhtsa_item_number}
    row = rows[0]
    row["found"] = True
    return row


def _run_fetch_investigation(ctx: ToolContext, nhtsa_action_number: str) -> dict:
    """Fetch a single investigation row + related recalls."""
    if ctx.sql_executor is None:
        raise RuntimeError("ToolContext.sql_executor is not wired.")
    sql = (
        f"SELECT nhtsa_action_number, investigation_type, status, "
        f"open_date, close_date, subject, summary, "
        f"make, model, model_year, component "
        f"FROM {ctx.cfg.full_schema_name}.silver_investigations_parsed "
        f"WHERE nhtsa_action_number = '{_sql_quote(nhtsa_action_number)}' "
        f"LIMIT 1"
    )
    rows = ctx.sql_executor(sql)
    if not rows:
        return {"found": False, "nhtsa_action_number": nhtsa_action_number}
    row = rows[0]
    row["found"] = True
    return row


def _sql_quote(raw: Any) -> str:
    """Minimal single-quote escaping for literal SQL interpolation.

    We prefer parameterised queries when the executor supports them, but
    the Databricks statement-execution SDK and ``spark.sql`` both take a
    literal string. NHTSA item / action numbers are short ASCII tokens
    (``14V123000``, ``PE22017``), so this escape plus a length cap is
    sufficient. The LLM-provided value is also validated upstream in
    agent.py before reaching this function.

    Coerces to str first — the LLM often emits ``nhtsa_item_number`` as
    an integer when the id looks numeric (e.g. ``10240998``), which
    would otherwise break ``len(raw)`` with a TypeError.
    """
    s = str(raw)
    if len(s) > 64:
        raise ValueError(f"id too long: {len(s)} chars")
    return s.replace("'", "''")


# ---------------------------------------------------------------------------
# Dispatcher.
# ---------------------------------------------------------------------------

def execute_tool(name: str, args: dict, ctx: ToolContext) -> dict:
    """Run the named tool and return a JSON-serialisable dict.

    Errors are caught and returned inside the dict (with ``error=True``)
    rather than raised — the agent loop needs to feed a tool-result
    message back to the LLM regardless of success, so a thrown exception
    would break the loop.

    Tracing: this function opens a ``TOOL`` span named ``tool.<name>`` so
    the trace tree records which tool fired, the arguments the LLM
    passed, and the result (post-serialisation — not the full payload,
    to keep trace storage bounded). ``ToolContext`` itself is NOT attached
    as a span input because it holds live Databricks clients that would
    serialise to something useless.
    """
    t0 = time.perf_counter()
    span_name = f"tool.{name}" if name in ALL_TOOL_NAMES else f"tool.unknown:{name}"
    span_type = _span_type_for_tool(name)

    with mlflow.start_span(name=span_name, span_type=span_type) as span:
        _span_set_inputs(span, {"tool": name, "args": args})
        try:
            if name == GENIE_TOOL_NAME:
                out = _run_genie(ctx, args["question"])
            elif name == VECTOR_SEARCH_TOOL_NAME:
                out = _run_vector_search(
                    ctx,
                    query=args["query"],
                    filters=args.get("filters"),
                    num_results=args.get("num_results"),
                )
            elif name == FETCH_TSB_TOOL_NAME:
                out = _run_fetch_tsb(ctx, args["nhtsa_item_number"])
            elif name == FETCH_INVESTIGATION_TOOL_NAME:
                out = _run_fetch_investigation(ctx, args["nhtsa_action_number"])
            else:
                out = {"error": True, "message": f"unknown tool: {name}"}
        except Exception as exc:  # noqa: BLE001 — surface all failures to the LLM
            logger.exception("Tool {} failed: {}", name, exc)
            out = {"error": True, "message": str(exc), "tool": name}

        dt_ms = (time.perf_counter() - t0) * 1000.0
        out.setdefault("_latency_ms", round(dt_ms, 2))
        _span_set_outputs(span, _span_output_preview(out))
        return out


def _span_type_for_tool(name: str) -> str:
    """Map our tool names to MLflow span types.

    Genie + Vector Search are retrievers (they fetch evidence). The
    fetch-by-id tools are plain TOOL spans because they're exact lookups.
    """
    if name in (GENIE_TOOL_NAME, VECTOR_SEARCH_TOOL_NAME):
        return "RETRIEVER"
    return "TOOL"


def _span_set_inputs(span: Any, inputs: dict) -> None:
    """Guarded wrapper around ``span.set_inputs`` — silently no-ops on fakes.

    In tests ``mlflow.start_span`` with tracing disabled returns a stub
    that doesn't implement ``set_inputs`` on every MLflow build. The
    agent must never blow up because instrumentation is degraded.
    """
    try:
        span.set_inputs(inputs)
    except Exception:  # noqa: BLE001
        pass


def _span_set_outputs(span: Any, outputs: Any) -> None:
    try:
        span.set_outputs(outputs)
    except Exception:  # noqa: BLE001
        pass


def _span_output_preview(out: dict, max_rows: int = 3) -> dict:
    """Trim tool outputs before attaching to span.

    Full VS hit lists + Genie result sets bloat the trace store. We keep
    scalar fields (row_count, hit_count, sql, latency) verbatim, and
    show only the first few rows/hits with a ``_truncated`` marker. This
    is a trace-only preview — it does NOT affect what the LLM sees.
    """
    preview: dict = {}
    for k, v in out.items():
        if isinstance(v, list) and len(v) > max_rows:
            preview[k] = v[:max_rows] + [{"_truncated": len(v) - max_rows}]
        else:
            preview[k] = v
    return preview


def serialise_tool_result(result: dict, max_chars: int = 8000) -> str:
    """JSON-encode a tool result for feeding back into the LLM.

    Long results (big VS hits lists, wide Genie result sets) get
    truncated. The LLM is told what was truncated so it can ask for
    narrower filters instead of hallucinating a complete view.
    """
    payload = json.dumps(result, default=str, ensure_ascii=False)
    if len(payload) <= max_chars:
        return payload
    trimmed = payload[:max_chars]
    return (
        trimmed
        + f'... [TRUNCATED {len(payload) - max_chars} chars — '
        + 'narrow your filters or lower num_results]"'
    )


# ---------------------------------------------------------------------------
# Managed MCP URL builders (for when the agent is wired via MCP instead of
# direct SDK). Kept here for symmetry; agent.py doesn't require them in
# Phase 4 but the deployment notebook can use these to enumerate tools
# from a Managed MCP server instead of from ``tool_specs``.
# ---------------------------------------------------------------------------

def genie_mcp_url(workspace_host: str, space_id: str) -> str:
    """Return the Databricks-managed MCP endpoint URL for a Genie space."""
    return f"{workspace_host.rstrip('/')}/api/2.0/mcp/genie/{space_id}"


def vector_search_mcp_url(workspace_host: str, index_name: str) -> str:
    """Return the Databricks-managed MCP endpoint URL for a VS index."""
    return f"{workspace_host.rstrip('/')}/api/2.0/mcp/vector-search/{index_name}"


# ---------------------------------------------------------------------------
# Filter merge — used by the agent when accumulating state across turns.
# ---------------------------------------------------------------------------

def merge_filters(existing: dict | None, new: dict | None) -> dict:
    """Merge an ``accumulated_filters`` bag with a new turn's filters.

    Rules:
    - ``None`` + x → x (bag start / no change cases).
    - Scalar value overwrites (last-writer-wins — most recent user turn
      is authoritative).
    - List values are unioned and de-duplicated, preserving order.
    - Keys explicitly set to ``None`` in ``new`` clear the key.
    """
    out: dict = dict(existing or {})
    for k, v in (new or {}).items():
        if v is None:
            out.pop(k, None)
            continue
        if isinstance(v, list) and isinstance(out.get(k), list):
            merged = list(out[k])
            for item in v:
                if item not in merged:
                    merged.append(item)
            out[k] = merged
        else:
            out[k] = v
    return out
