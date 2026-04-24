"""Phase 6 tests — feedback scorers, Responses wrapper, and deploy wiring.

Covers:
- ``cite_id_present``, ``word_count_under``, ``mentions_oem`` scorers
  on the half-dozen answer shapes the agent emits (plain str, dict with
  ``text``, Responses envelope with ``output``).
- ``NhtsaResponsesAgent`` request-parsing invariants: pulls the last
  user message, reads ``custom_inputs.session_id``, returns a Responses
  ``message`` event, honours dict / pydantic / plain-string inputs.
- ``log_register_agent`` resources list shape — the one place that
  enumerates every UC object the serving principal must be authorised
  against; a missed table here = a runtime 403 on prod.
- Trace tag stamping propagates ``GIT_SHA`` / ``MODEL_VERSION`` /
  ``session_id`` without blowing up when tracing is disabled.

No live Databricks clients. Live SDK wiring is skipped at import time
by mocking ``load_context``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from nhtsa_curator.agent import AgentResult
from nhtsa_curator.config import ProjectConfig
from nhtsa_curator.evaluation import (
    OEM_KEYWORDS,
    cite_id_present,
    mentions_oem,
    word_count_under,
)
from nhtsa_curator.serving import (
    NhtsaResponsesAgent,
    _extract_responses_request,
    _last_user_text,
    _responses_message_event,
)

# ---------------------------------------------------------------------------
# Feedback scorers
# ---------------------------------------------------------------------------


class TestCiteIdPresent:
    """cite_id_present returns True iff a NHTSA source-id appears."""

    @pytest.mark.parametrize(
        "text",
        [
            "Recall 23V123456 affects 12,000 vehicles.",
            "See TSB 10160095 for repair steps.",
            "TSB-10160095 covers the fix.",
            "Investigation PE22-001 is ongoing.",
            "Complaint ODI 11567823 describes the defect.",
            "22V-456-789 describes the remedy.",
            "Campaign 23V123 applies.",
        ],
    )
    def test_positive(self, text: str) -> None:
        assert cite_id_present(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "The answer is unknown at this time.",
            "Tesla issued a recall — see their website.",
            "",
            "1234567",  # 7-digit, below ODI's 6+ floor for the TSB/ODI pattern
        ],
    )
    def test_negative(self, text: str) -> None:
        assert cite_id_present(text) is False

    def test_handles_responses_envelope(self) -> None:
        envelope = {
            "output": [{"type": "message", "content": [{"text": "See TSB 10160095."}]}]
        }
        assert cite_id_present(envelope) is True

    def test_handles_plain_list(self) -> None:
        assert cite_id_present(["No citations here."]) is False
        assert cite_id_present(["Recall 23V999888."]) is True


class TestWordCountUnder:
    def test_under_threshold(self) -> None:
        assert word_count_under("three short words") is True

    def test_exactly_threshold(self) -> None:
        text = " ".join(["word"] * 400)
        assert word_count_under(text, max_words=400) is True

    def test_over_threshold(self) -> None:
        text = " ".join(["word"] * 401)
        assert word_count_under(text, max_words=400) is False

    def test_responses_envelope(self) -> None:
        payload = {"text": "short answer."}
        assert word_count_under(payload) is True


class TestMentionsOem:
    def test_direct_hit(self) -> None:
        assert mentions_oem("Recall for Ford trucks.") is True

    def test_case_insensitive(self) -> None:
        assert mentions_oem("rivian emergency brake") is True

    def test_multi_word_oem(self) -> None:
        assert mentions_oem("General Motors issued a recall.") is True

    def test_negative(self) -> None:
        assert mentions_oem("The recall covers several vehicles.") is False

    def test_oem_keywords_nonempty(self) -> None:
        # Sanity — the dashboard's OEM breakdown depends on this list.
        assert len(OEM_KEYWORDS) > 10
        assert "Tesla" in OEM_KEYWORDS


# ---------------------------------------------------------------------------
# Responses-request parsing
# ---------------------------------------------------------------------------


class TestExtractResponsesRequest:
    def test_dict_with_input(self) -> None:
        req = {
            "input": [
                {"role": "user", "content": "question one"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "final question"},
            ],
            "custom_inputs": {"session_id": "s1", "request_id": "r1"},
        }
        msg, custom = _extract_responses_request(req)
        assert msg == "final question"
        assert custom == {"session_id": "s1", "request_id": "r1"}

    def test_dict_with_messages_alias(self) -> None:
        req = {"messages": [{"role": "user", "content": "hello"}]}
        msg, custom = _extract_responses_request(req)
        assert msg == "hello"
        assert custom == {}

    def test_plain_string_probe(self) -> None:
        msg, custom = _extract_responses_request("smoke check")
        assert msg == "smoke check"
        assert custom == {}

    def test_last_user_text_skips_non_user(self) -> None:
        items = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "mid"},
        ]
        assert _last_user_text(items) == "first"

    def test_last_user_text_handles_content_list(self) -> None:
        items = [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "multi-part"}],
            }
        ]
        assert _last_user_text(items) == "multi-part"

    def test_pydantic_like_object(self) -> None:
        @dataclass
        class Item:
            role: str
            content: str

            def model_dump(self) -> dict:
                return {"role": self.role, "content": self.content}

        @dataclass
        class Req:
            input: list
            custom_inputs: dict

        req = Req(
            input=[Item(role="user", content="pydantic question")],
            custom_inputs={"session_id": "s2"},
        )
        msg, custom = _extract_responses_request(req)
        assert msg == "pydantic question"
        assert custom["session_id"] == "s2"


# ---------------------------------------------------------------------------
# Responses event shape
# ---------------------------------------------------------------------------


def test_responses_message_event_shape() -> None:
    result = AgentResult(
        session_id="s-42",
        answer="The answer.",
        n_llm_calls=3,
        stopped_reason="ok",
    )
    event = _responses_message_event(result)
    assert event["type"] == "response.output_item.done"
    item = event["item"]
    assert item["type"] == "message"
    assert item["role"] == "assistant"
    assert item["content"][0]["text"] == "The answer."
    assert item["custom_outputs"]["session_id"] == "s-42"
    assert item["custom_outputs"]["n_llm_calls"] == 3


# ---------------------------------------------------------------------------
# NhtsaResponsesAgent — end-to-end with a scripted NhtsaAgent
# ---------------------------------------------------------------------------


@dataclass
class _FakeAgent:
    """Stand-in for NhtsaAgent. Records inputs, returns a canned result."""

    canned: AgentResult
    captured: dict = None  # type: ignore[assignment]

    def run_turn(
        self,
        user_message: str,
        session_id: str | None = None,
        *,
        user_id: str | None = None,
    ) -> AgentResult:
        self.captured = {
            "user_message": user_message,
            "session_id": session_id,
            "user_id": user_id,
        }
        # Echo the session back so the test can assert propagation.
        if session_id:
            self.canned.session_id = session_id
        return self.canned


def _make_responses_agent_with_fake(fake: _FakeAgent) -> NhtsaResponsesAgent:
    """Construct an NhtsaResponsesAgent without booting live SDKs.

    Patches ``NhtsaAgentModel.load_context`` to be a no-op and manually
    sets the inner ``_agent`` to the fake. Avoids importing the full
    Databricks SDK stack during tests.
    """
    with patch("nhtsa_curator.serving.NhtsaAgentModel.load_context", return_value=None):
        wrapper = NhtsaResponsesAgent(config_path="project_config.yml", env="dev")
    wrapper._inner._agent = fake  # type: ignore[assignment]
    return wrapper


def test_responses_agent_predict_returns_text() -> None:
    fake = _FakeAgent(
        canned=AgentResult(session_id="", answer="The answer.", n_llm_calls=1)
    )
    wrapper = _make_responses_agent_with_fake(fake)
    result = wrapper.predict(
        {
            "input": [{"role": "user", "content": "How many recalls in 2023?"}],
            "custom_inputs": {"session_id": "abc"},
        }
    )
    assert isinstance(result, dict)
    assert result["output"][0]["content"][0]["text"] == "The answer."
    assert fake.captured["user_message"] == "How many recalls in 2023?"
    assert fake.captured["session_id"] == "abc"


def test_responses_agent_stream_yields_one_event() -> None:
    fake = _FakeAgent(canned=AgentResult(session_id="", answer="hi.", n_llm_calls=1))
    wrapper = _make_responses_agent_with_fake(fake)
    events = list(wrapper.predict_stream({"input": [{"role": "user", "content": "hi"}]}))
    assert len(events) == 1
    assert events[0]["type"] == "response.output_item.done"


def test_responses_agent_propagates_user_id() -> None:
    fake = _FakeAgent(canned=AgentResult(session_id="", answer="x", n_llm_calls=0))
    wrapper = _make_responses_agent_with_fake(fake)
    wrapper.predict(
        {
            "input": [{"role": "user", "content": "q"}],
            "custom_inputs": {"user_id": "demo-user"},
        }
    )
    assert fake.captured["user_id"] == "demo-user"


# ---------------------------------------------------------------------------
# log_register_agent — resources enumeration
# ---------------------------------------------------------------------------


def test_log_register_agent_resources_contain_all_gold_tables() -> None:
    """Every gold fact table referenced by the Genie tool must be in the
    registered resources — missing one manifests as a UC 403 at prod
    serve time, not at deploy time. We assert the full list here.
    """
    from nhtsa_curator.agent import log_register_agent

    cfg = ProjectConfig(
        catalog="cat",
        schema="sch",  # alias field
        volume="vol",
        llm_endpoint="llm-ep",
        embedding_endpoint="emb-ep",
        warehouse_id="wh-123",
        vector_search_endpoint="vs-ep",
        genie_space_id="PLACEHOLDER_X",
        usage_policy_id=None,
        lakebase_project_id=None,
        experiment_name="/Shared/test",
    )

    captured: dict[str, Any] = {}

    def _fake_log_model(**kwargs: Any) -> Any:
        captured["log_model_kwargs"] = kwargs
        obj = MagicMock()
        obj.model_uri = "runs:/abc/agent"
        return obj

    with (
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run") as _start_run,
        patch("mlflow.pyfunc.log_model", side_effect=_fake_log_model),
        patch("mlflow.log_metrics"),
        patch("mlflow.register_model") as _register,
        patch("mlflow.MlflowClient") as _client,
    ):
        _start_run.return_value.__enter__ = MagicMock(return_value=None)
        _start_run.return_value.__exit__ = MagicMock(return_value=False)
        _register.return_value.version = "7"

        log_register_agent(
            cfg=cfg,
            git_sha="sha",
            run_id="run",
            agent_code_path="nhtsa_agent_pg.py",
            model_name="cat.sch.nhtsa_agent_pg",
            evaluation_metrics={"cite_id_present": 0.8, "not_numeric": "skip"},
        )

    resources = captured["log_model_kwargs"]["resources"]
    # MLflow resource objects don't stringify helpfully — inspect to_dict().
    import json

    resource_blob = json.dumps([r.to_dict() for r in resources])
    # Serving endpoints for LLM + embedding.
    assert "llm-ep" in resource_blob
    assert "emb-ep" in resource_blob
    # Warehouse.
    assert "wh-123" in resource_blob
    # Every gold table referenced by the agent's Genie tool.
    required_tables = {
        "gold_recalls_fact",
        "gold_complaints_fact",
        "gold_investigations_fact",
        "gold_tsb_meta",
        "gold_sgo_av_crashes",
        "gold_narrative_chunks",
    }
    for tbl in required_tables:
        assert tbl in resource_blob, tbl
    # Vector search index.
    assert "gold_narrative_chunks_index" in resource_blob
    # Placeholder genie_space_id must NOT be registered as a resource —
    # UC rejects a placeholder string.
    assert "PLACEHOLDER" not in resource_blob


def test_log_register_agent_numeric_metrics_only_logged() -> None:
    """mlflow.log_metrics refuses non-numeric values; we must filter."""
    from nhtsa_curator.agent import log_register_agent

    cfg = ProjectConfig(
        catalog="cat",
        schema="sch",
        volume="vol",
        llm_endpoint="llm-ep",
        embedding_endpoint="emb-ep",
        warehouse_id="wh-123",
        vector_search_endpoint="vs-ep",
    )

    captured_metrics: dict[str, Any] = {}

    def _fake_log_metrics(metrics: dict) -> None:
        captured_metrics.update(metrics)

    def _fake_log_model(**kwargs: Any) -> Any:
        obj = MagicMock()
        obj.model_uri = "runs:/abc/agent"
        return obj

    with (
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run") as _start_run,
        patch("mlflow.pyfunc.log_model", side_effect=_fake_log_model),
        patch("mlflow.log_metrics", side_effect=_fake_log_metrics),
        patch("mlflow.register_model") as _register,
        patch("mlflow.MlflowClient"),
    ):
        _start_run.return_value.__enter__ = MagicMock(return_value=None)
        _start_run.return_value.__exit__ = MagicMock(return_value=False)
        _register.return_value.version = "1"
        log_register_agent(
            cfg=cfg,
            git_sha="sha",
            run_id="run",
            agent_code_path="nhtsa_agent_pg.py",
            model_name="cat.sch.nhtsa_agent_pg",
            evaluation_metrics={
                "cite_id_present": 0.9,
                "word_count_under": 1.0,
                "stays_in_scope": "Pass",  # non-numeric — must be filtered
                "n_questions": 25,
            },
        )

    assert captured_metrics == {
        "cite_id_present": 0.9,
        "word_count_under": 1.0,
        "n_questions": 25.0,
    }


# ---------------------------------------------------------------------------
# Deploy-trace tag stamping
# ---------------------------------------------------------------------------


def test_stamp_trace_deploy_tags_noop_when_tracing_off() -> None:
    """When ``get_current_active_span`` returns None we must not raise."""
    from nhtsa_curator.serving import _stamp_trace_deploy_tags

    with patch("mlflow.get_current_active_span", return_value=None):
        # Should complete silently.
        _stamp_trace_deploy_tags(session_id="s1", request_id="r1")


def test_stamp_trace_deploy_tags_forwards_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GIT_SHA", "deadbeef")
    monkeypatch.setenv("MODEL_VERSION", "12")
    monkeypatch.setenv("MODEL_SERVING_ENDPOINT_NAME", "nhtsa-agent-endpoint-dev-pg")

    from nhtsa_curator.serving import _stamp_trace_deploy_tags

    captured: dict[str, Any] = {}

    def _fake_update(**kwargs: Any) -> None:
        captured.update(kwargs)

    with (
        patch("mlflow.get_current_active_span", return_value=MagicMock()),
        patch("mlflow.update_current_trace", side_effect=_fake_update),
    ):
        _stamp_trace_deploy_tags(session_id="s-1", request_id="r-1")

    assert captured["tags"]["git_sha"] == "deadbeef"
    assert captured["tags"]["model_version"] == "12"
    assert (
        captured["tags"]["model_serving_endpoint_name"] == "nhtsa-agent-endpoint-dev-pg"
    )
    assert captured["tags"]["session_id"] == "s-1"
    assert captured["client_request_id"] == "r-1"


# ---------------------------------------------------------------------------
# update_traces_aggregated SQL surface area (static assertions)
# ---------------------------------------------------------------------------


def test_update_traces_script_uses_nhtsa_tool_span_names() -> None:
    """Guard against renaming tool spans in mcp.py without updating the
    aggregated-view SQL — the mismatch would manifest as all tool counts
    dropping to 0 silently on the dashboard.
    """
    from pathlib import Path

    script = (
        Path(__file__).resolve().parent.parent
        / "resources"
        / "deployment_scripts"
        / "update_traces_aggregated.py"
    )
    text = script.read_text(encoding="utf-8")
    # These must match the tool names in nhtsa_curator.mcp.
    for needle in (
        "tool.genie_recalls",
        "tool.vector_search_narrative",
        "tool.fetch_tsb",
        "tool.fetch_investigation",
    ):
        assert needle in text, needle


def test_update_traces_script_exposes_nhtsa_columns() -> None:
    """Dashboard depends on these exact column names — keep them stable."""
    from pathlib import Path

    script = (
        Path(__file__).resolve().parent.parent
        / "resources"
        / "deployment_scripts"
        / "update_traces_aggregated.py"
    )
    text = script.read_text(encoding="utf-8")
    for col in (
        "cite_id_present",
        "word_count_under",
        "mentions_oem",
        "factual_defect",
        "cite_every_claim",
        "stays_in_scope",
        "genie_call_count",
        "vs_call_count",
        "fetch_tsb_count",
        "fetch_investigation_count",
        "session_id",
    ):
        assert col in text, col


# ---------------------------------------------------------------------------
# Dashboard JSON shape
# ---------------------------------------------------------------------------


def test_dashboard_json_parses_and_points_at_view() -> None:
    """The dashboard must load and reference the aggregated view dataset."""
    from pathlib import Path

    json_path = (
        Path(__file__).resolve().parent.parent
        / "resources"
        / "dashboard"
        / "nhtsa_agent_monitoring_dashboard.lvdash.json"
    )
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["datasets"], "dashboard must declare a dataset"
    query = " ".join(data["datasets"][0]["queryLines"])
    assert "nhtsa_traces_aggregated_pg" in query


def test_dashboard_json_widgets_cover_key_kpis() -> None:
    """Smoke check that the KPIs we promise in docs exist in the JSON."""
    from pathlib import Path

    json_path = (
        Path(__file__).resolve().parent.parent
        / "resources"
        / "dashboard"
        / "nhtsa_agent_monitoring_dashboard.lvdash.json"
    )
    text = json_path.read_text(encoding="utf-8")
    for needle in (
        "kpi_cite_rate",
        "kpi_oem_rate",
        "kpi_p95_latency",
        "tool_mix",
        "cite_rate_over_time",
    ):
        assert needle in text, needle
