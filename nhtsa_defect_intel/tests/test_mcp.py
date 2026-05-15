"""Unit tests for the tool layer.

Covers:
- Tool specs are valid JSON-schema-ish shapes the OpenAI SDK will accept.
- The dispatcher routes to the right implementation by name.
- Each tool wraps errors rather than raising.
- ``merge_filters`` semantics: overwrite-scalar, union-list, None-clears.
- ``_normalise_statement_result`` handles both SDK object + raw-dict
  shapes — these are the two shapes production code actually encounters.

Spark-free. No Databricks SDK imported.
"""

from __future__ import annotations

from typing import Any

import pytest

from nhtsa_curator.config import ProjectConfig, VectorSearchConfig
from nhtsa_curator.mcp import (
    ALL_TOOL_NAMES,
    FETCH_INVESTIGATION_TOOL_NAME,
    FETCH_TSB_TOOL_NAME,
    GENIE_TOOL_NAME,
    VECTOR_SEARCH_TOOL_NAME,
    ToolContext,
    _normalise_statement_result,
    execute_tool,
    genie_mcp_url,
    merge_filters,
    serialise_tool_result,
    tool_specs,
    vector_search_mcp_url,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg() -> ProjectConfig:
    return ProjectConfig(
        catalog="cat",
        schema="sch",
        volume="vol",
        llm_endpoint="llm",
        embedding_endpoint="emb",
        warehouse_id="wh",
        vector_search_endpoint="ep",
        genie_space_id="real-space-id",
    )


class _FakeVSIndex:
    def __init__(self, hits: list[list]) -> None:
        self._hits = hits

    def similarity_search(self, **_: Any) -> dict:
        return {
            "result": {
                "manifest": {
                    "columns": [
                        {"name": "chunk_id"},
                        {"name": "make_norm"},
                        {"name": "content"},
                    ]
                },
                "data_array": self._hits,
            }
        }


class _FakeVSClient:
    def __init__(self, hits: list[list]) -> None:
        self._idx = _FakeVSIndex(hits)

    def get_index(self, **_: Any) -> _FakeVSIndex:
        return self._idx


class _FakeGenieClient:
    """Minimal stand-in for ``WorkspaceClient().genie``."""

    def __init__(self, sql: str, rows: list[dict]) -> None:
        self._sql = sql
        self._rows = rows

    def start_conversation_and_wait(self, space_id: str, content: str) -> dict:
        return {"conversation_id": "conv1", "message_id": "msg1"}

    def get_message(self, space_id: str, conversation_id: str, message_id: str) -> dict:
        return {"attachments": [{"query": {"query": self._sql}}]}

    def get_message_query_result(self, **_: Any) -> dict:
        cols = list(self._rows[0].keys()) if self._rows else []
        return {
            "statement_response": {
                "manifest": {"schema": {"columns": [{"name": c} for c in cols]}},
                "result": {"data_array": [list(r.values()) for r in self._rows]},
            }
        }


# ---------------------------------------------------------------------------
# Tool specs
# ---------------------------------------------------------------------------


def test_tool_specs_has_all_four(cfg: ProjectConfig) -> None:
    specs = tool_specs(cfg)
    assert len(specs) == 4
    names = {s["function"]["name"] for s in specs}
    assert names == set(ALL_TOOL_NAMES)


def test_tool_specs_well_formed(cfg: ProjectConfig) -> None:
    """Every spec has ``type=function`` and a non-empty parameters schema."""
    for spec in tool_specs(cfg):
        assert spec["type"] == "function"
        fn = spec["function"]
        assert fn["name"]
        assert fn["description"]
        params = fn["parameters"]
        assert params["type"] == "object"
        assert "required" in params


def test_vs_tool_exposes_filter_enum(cfg: ProjectConfig) -> None:
    """The filter schema must constrain source_dataset to the known 4."""
    vs = next(
        s for s in tool_specs(cfg) if s["function"]["name"] == VECTOR_SEARCH_TOOL_NAME
    )
    enum = vs["function"]["parameters"]["properties"]["filters"]["properties"][
        "source_dataset"
    ]["enum"]
    assert set(enum) == {"complaints", "tsbs", "investigations", "sgo"}


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def test_dispatcher_rejects_unknown_tool(cfg: ProjectConfig) -> None:
    ctx = ToolContext(cfg=cfg)
    out = execute_tool("not_a_tool", {}, ctx)
    assert out["error"] is True
    assert "unknown tool" in out["message"]


def test_vector_search_tool_returns_flat_hits(cfg: ProjectConfig) -> None:
    hits_raw = [["abc", "Tesla", "burned out"], ["def", "Ford", "stalled"]]
    ctx = ToolContext(
        cfg=cfg,
        vs_client=_FakeVSClient(hits_raw),
        vs_cfg=VectorSearchConfig(num_results=2),
    )
    out = execute_tool(
        VECTOR_SEARCH_TOOL_NAME,
        {"query": "fire", "filters": {"make_norm": "Tesla"}},
        ctx,
    )
    assert out["hit_count"] == 2
    assert out["hits"][0]["make_norm"] == "Tesla"
    assert "_latency_ms" in out


def test_genie_tool_returns_sql_and_rows(cfg: ProjectConfig) -> None:
    rows = [{"make_norm": "Tesla", "n": 42}]
    ctx = ToolContext(cfg=cfg, genie_client=_FakeGenieClient("SELECT 42", rows))
    out = execute_tool(GENIE_TOOL_NAME, {"question": "how many?"}, ctx)
    assert out["sql"] == "SELECT 42"
    assert out["rows"] == rows
    assert out["row_count"] == 1


def test_genie_tool_fails_fast_on_placeholder_id() -> None:
    bad = ProjectConfig(
        catalog="cat",
        schema="sch",
        volume="vol",
        llm_endpoint="llm",
        embedding_endpoint="emb",
        warehouse_id="wh",
        vector_search_endpoint="ep",
        genie_space_id="PLACEHOLDER_DEV_GENIE_SPACE_ID",
    )
    ctx = ToolContext(cfg=bad, genie_client=_FakeGenieClient("", []))
    out = execute_tool(GENIE_TOOL_NAME, {"question": "x"}, ctx)
    assert out["error"] is True
    assert "genie_space_id" in out["message"]


def test_fetch_tsb_hits_silver(cfg: ProjectConfig) -> None:
    captured: list[str] = []

    def _sql(q: str) -> list[dict]:
        captured.append(q)
        return [{"nhtsa_item_number": "10160095", "summary": "OK"}]

    ctx = ToolContext(cfg=cfg, sql_executor=_sql)
    out = execute_tool(FETCH_TSB_TOOL_NAME, {"nhtsa_item_number": "10160095"}, ctx)
    assert out["found"] is True
    assert out["summary"] == "OK"
    assert captured and "silver_tsbs" in captured[0]
    assert "silver_tsbs_parsed" not in captured[0]


def test_fetch_tsb_not_found_returns_structured(cfg: ProjectConfig) -> None:
    ctx = ToolContext(cfg=cfg, sql_executor=lambda _q: [])
    out = execute_tool(FETCH_TSB_TOOL_NAME, {"nhtsa_item_number": "NOPE"}, ctx)
    assert out["found"] is False
    assert out["nhtsa_item_number"] == "NOPE"


def test_fetch_investigation_hits_silver(cfg: ProjectConfig) -> None:
    captured: list[str] = []

    def _sql(q: str) -> list[dict]:
        captured.append(q)
        return [{"nhtsa_action_number": "EA22-002", "status": "open"}]

    ctx = ToolContext(cfg=cfg, sql_executor=_sql)
    out = execute_tool(
        FETCH_INVESTIGATION_TOOL_NAME,
        {"nhtsa_action_number": "EA22-002"},
        ctx,
    )
    assert out["found"] is True
    assert "silver_investigations_parsed" in captured[0]


def test_fetch_rejects_overlong_id(cfg: ProjectConfig) -> None:
    ctx = ToolContext(cfg=cfg, sql_executor=lambda _q: [])
    out = execute_tool(
        FETCH_TSB_TOOL_NAME,
        {"nhtsa_item_number": "x" * 65},
        ctx,
    )
    assert out["error"] is True
    assert "too long" in out["message"]


# ---------------------------------------------------------------------------
# Result serialisation + truncation
# ---------------------------------------------------------------------------


def test_serialise_tool_result_truncates() -> None:
    big = {"rows": [{"x": "a" * 1000} for _ in range(20)]}
    s = serialise_tool_result(big, max_chars=1000)
    assert len(s) <= 1100  # trim + trailing hint
    assert "TRUNCATED" in s


def test_serialise_tool_result_passthrough_small() -> None:
    out = serialise_tool_result({"a": 1})
    assert out == '{"a": 1}'


# ---------------------------------------------------------------------------
# Filter merging
# ---------------------------------------------------------------------------


def test_merge_filters_scalar_overwrite() -> None:
    out = merge_filters({"make_norm": "Tesla"}, {"make_norm": "Ford"})
    assert out == {"make_norm": "Ford"}


def test_merge_filters_list_unions_preserving_order() -> None:
    out = merge_filters(
        {"make_norm": ["Tesla", "Ford"]},
        {"make_norm": ["Ford", "Rivian"]},
    )
    assert out == {"make_norm": ["Tesla", "Ford", "Rivian"]}


def test_merge_filters_none_clears_key() -> None:
    out = merge_filters({"make_norm": "Tesla", "model_year": 2023}, {"make_norm": None})
    assert out == {"model_year": 2023}


def test_merge_filters_handles_none_inputs() -> None:
    assert merge_filters(None, None) == {}
    assert merge_filters(None, {"a": 1}) == {"a": 1}
    assert merge_filters({"a": 1}, None) == {"a": 1}


# ---------------------------------------------------------------------------
# Statement-result normalisation
# ---------------------------------------------------------------------------


def test_normalise_statement_result_from_dict() -> None:
    res = {
        "statement_response": {
            "manifest": {"schema": {"columns": [{"name": "a"}, {"name": "b"}]}},
            "result": {"data_array": [[1, 2], [3, 4]]},
        }
    }
    out = _normalise_statement_result(res)
    assert out == [{"a": 1, "b": 2}, {"a": 3, "b": 4}]


def test_normalise_statement_result_empty_data() -> None:
    res = {
        "statement_response": {
            "manifest": {"schema": {"columns": [{"name": "a"}]}},
            "result": {"data_array": None},
        }
    }
    assert _normalise_statement_result(res) == []


# ---------------------------------------------------------------------------
# Managed MCP URL builders
# ---------------------------------------------------------------------------


def test_genie_mcp_url_shape() -> None:
    url = genie_mcp_url("https://workspace.cloud.databricks.com/", "spc-123")
    assert url == "https://workspace.cloud.databricks.com/api/2.0/mcp/genie/spc-123"


def test_vector_search_mcp_url_shape() -> None:
    url = vector_search_mcp_url("https://workspace.cloud.databricks.com", "cat.sch.idx")
    assert url.endswith("/api/2.0/mcp/vector-search/cat.sch.idx")
