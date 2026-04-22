"""Unit tests for the Phase-5 eval harness.

Covers:
- TSV loading per tier (header validation, TBD-flagging, error on
  malformed rows).
- Each scorer's key branches (exact-match, list-Jaccard, numeric,
  citation-matching, judge score thresholds, 5-point rubric average).
- ``run_question`` end-to-end against a scripted agent + fake judge.
- ``aggregate_metrics`` shape including tier-specific keys.
- Promotion-gate-adjacent assertions — pass rate reflects (passes /
  scored), pending rows excluded from both numerator and denominator.

No network, no MLflow server, no Databricks SDK. The tests use a
hand-rolled ScriptedLLM and InMemorySessionStore.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from nhtsa_curator.agent import NhtsaAgent
from nhtsa_curator.config import ProjectConfig, VectorSearchConfig
from nhtsa_curator.evaluation import (
    DEFAULT_EVAL_FILES,
    TIER1_LIST_JACCARD_MIN,
    TIER2_JUDGE_MIN,
    TIER3_JUDGE_MIN,
    TIER_DETERMINISTIC,
    TIER_GROUNDED,
    TIER_SYNTHESIS,
    VALID_TIERS,
    EvalQuestion,
    aggregate_metrics,
    load_all,
    load_eval_set,
    run_question,
    score_tier1,
    score_tier2,
    score_tier3,
)
from nhtsa_curator.mcp import GENIE_TOOL_NAME, ToolContext
from nhtsa_curator.memory import InMemorySessionStore


# ---------------------------------------------------------------------------
# TSV loading
# ---------------------------------------------------------------------------


def _write_tsv(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_load_tier1_parses_rows(tmp_path: Path) -> None:
    tsv = _write_tsv(
        tmp_path / "t1.tsv",
        "# header comment\n"
        "question\treference_sql\texpected_value\n"
        "Q1?\tSELECT 1\t42\n"
        "Q2?\t\tTBD:pending_truth\n",
    )
    qs = load_eval_set(TIER_DETERMINISTIC, tsv)
    assert [q.question for q in qs] == ["Q1?", "Q2?"]
    assert qs[0].expected_value == "42"
    assert qs[0].reference_sql == "SELECT 1"
    assert qs[0].is_pending is False
    assert qs[1].is_pending is True


def test_load_tier2_parses_rows(tmp_path: Path) -> None:
    tsv = _write_tsv(
        tmp_path / "t2.tsv",
        "question\tsource_id\texpected_claim\n"
        "What?\t23V-085\tThe remedy is X.\n",
    )
    qs = load_eval_set(TIER_GROUNDED, tsv)
    assert qs[0].source_id == "23V-085"
    assert qs[0].expected_claim == "The remedy is X."


def test_load_tier3_parses_rows(tmp_path: Path) -> None:
    tsv = _write_tsv(
        tmp_path / "t3.tsv",
        "question\treference_answer\n"
        "Theme?\tSome synthesis answer.\n",
    )
    qs = load_eval_set(TIER_SYNTHESIS, tsv)
    assert qs[0].reference_answer == "Some synthesis answer."


def test_load_unknown_tier_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown tier"):
        load_eval_set("tier99", tmp_path / "x.tsv")


def test_load_rejects_wrong_header(tmp_path: Path) -> None:
    tsv = _write_tsv(
        tmp_path / "bad.tsv",
        "q\trsql\texpected\n"
        "Q\tSQL\t1\n",
    )
    with pytest.raises(ValueError, match="expected header"):
        load_eval_set(TIER_DETERMINISTIC, tsv)


def test_load_rejects_wrong_column_count(tmp_path: Path) -> None:
    tsv = _write_tsv(
        tmp_path / "bad.tsv",
        "question\treference_sql\texpected_value\n"
        "only-two-cols\tSELECT 1\n",
    )
    with pytest.raises(ValueError, match="columns"):
        load_eval_set(TIER_DETERMINISTIC, tsv)


def test_load_rejects_empty_file(tmp_path: Path) -> None:
    tsv = _write_tsv(
        tmp_path / "empty.tsv",
        "question\treference_sql\texpected_value\n",
    )
    with pytest.raises(ValueError, match="no questions"):
        load_eval_set(TIER_DETERMINISTIC, tsv)


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_eval_set(TIER_DETERMINISTIC, tmp_path / "does-not-exist.tsv")


def test_load_all_reads_three_tiers() -> None:
    """The canonical Phase-5 eval files in notebooks/eval/ should load."""
    eval_dir = Path(__file__).resolve().parents[1] / "notebooks" / "eval"
    sets = load_all(eval_dir)
    assert set(sets.keys()) == set(VALID_TIERS)
    for tier in VALID_TIERS:
        assert len(sets[tier]) >= 20, f"{tier}: expected ~25 questions"


# ---------------------------------------------------------------------------
# Tier 1 scorer
# ---------------------------------------------------------------------------


def test_tier1_numeric_exact_match() -> None:
    score, breakdown = score_tier1("42", "There were 42 recall campaigns.")
    assert score == 1.0
    assert breakdown["kind"] == "numeric"


def test_tier1_numeric_mismatch() -> None:
    score, breakdown = score_tier1("42", "There were 41 recall campaigns.")
    assert score == 0.0
    assert breakdown["actual"] == "41"


def test_tier1_numeric_missing_from_answer() -> None:
    score, breakdown = score_tier1("42", "I could not determine a count.")
    assert score == 0.0
    assert breakdown["actual"] is None


def test_tier1_list_high_jaccard_passes() -> None:
    score, breakdown = score_tier1(
        "Tesla|Ford|General Motors",
        "The top 3 are Tesla, Ford, and General Motors.",
    )
    assert score == 1.0
    assert breakdown["kind"] == "list"
    assert breakdown["jaccard"] >= TIER1_LIST_JACCARD_MIN


def test_tier1_list_low_jaccard_fails() -> None:
    score, breakdown = score_tier1(
        "Tesla|Ford|General Motors",
        "The top make is Rivian.",
    )
    assert score == 0.0
    assert breakdown["jaccard"] < TIER1_LIST_JACCARD_MIN
    assert "Tesla" in breakdown["missing"]


def test_tier1_string_contains_substring() -> None:
    score, breakdown = score_tier1(
        "model year 2023",
        "The peak year is model year 2023 for Toyota.",
    )
    assert score == 1.0
    assert breakdown["kind"] == "string"


def test_tier1_string_not_found() -> None:
    score, _ = score_tier1("automatic transmission", "Manual gearbox only.")
    assert score == 0.0


# ---------------------------------------------------------------------------
# Tier 2 scorer
# ---------------------------------------------------------------------------


class _FakeJudge:
    """Hand-picked judge output for deterministic tests."""

    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    def score(self, prompt: str, *, context: dict | None = None) -> dict:
        self.prompts.append(prompt)
        return self._responses.pop(0)


def test_tier2_passes_with_citation_and_high_judge() -> None:
    judge = _FakeJudge([{"score": 5, "notes": "matches"}])
    score, breakdown = score_tier2(
        source_id="23V-085",
        expected_claim="Dealers reprogrammed the firmware.",
        actual_answer="According to campaign 23V-085, dealers reprogrammed the firmware.",
        judge=judge,
    )
    assert score == 1.0
    assert breakdown["cite_ok"] is True
    assert breakdown["judge_score"] == 5


def test_tier2_normalises_hyphen_in_citation() -> None:
    """Agent says 'TSB 10160095', ground truth says 'TSB-10160095'."""
    judge = _FakeJudge([{"score": 4, "notes": "ok"}])
    score, breakdown = score_tier2(
        source_id="TSB-10160095",
        expected_claim="Applies to F-150.",
        actual_answer="TSB 10160095 applies to F-150.",
        judge=judge,
    )
    assert score == 1.0
    assert breakdown["cite_ok"] is True


def test_tier2_fails_without_citation() -> None:
    judge = _FakeJudge([{"score": 5}])  # never called
    score, breakdown = score_tier2(
        source_id="23V-085",
        expected_claim="claim",
        actual_answer="Some answer without the id.",
        judge=judge,
    )
    assert score == 0.0
    assert breakdown["cite_ok"] is False
    assert judge.prompts == []  # short-circuits


def test_tier2_fails_on_low_judge_score() -> None:
    judge = _FakeJudge([{"score": TIER2_JUDGE_MIN - 1, "notes": "off-topic"}])
    score, breakdown = score_tier2(
        source_id="23V-085",
        expected_claim="The remedy is X.",
        actual_answer="Campaign 23V-085 is about something else entirely.",
        judge=judge,
    )
    assert score == 0.0
    assert breakdown["judge_score"] == TIER2_JUDGE_MIN - 1


def test_tier2_no_judge_falls_back_to_citation_only() -> None:
    score, breakdown = score_tier2(
        source_id="23V-085",
        expected_claim="claim",
        actual_answer="I read 23V-085 and it said X.",
        judge=None,
    )
    assert score == 1.0
    assert breakdown["judge_score"] is None


# ---------------------------------------------------------------------------
# Tier 3 scorer
# ---------------------------------------------------------------------------


def test_tier3_averages_subscores_and_passes_above_threshold() -> None:
    judge = _FakeJudge([{
        "faithfulness": 5, "coverage": 4, "calibration": 4,
        "citation": 4, "conciseness": 4, "notes": "solid synthesis",
    }])
    score, breakdown = score_tier3(
        reference_answer="ref",
        actual_answer="answer",
        retrieved_context="ctx",
        judge=judge,
    )
    assert breakdown["sub_scores"]["faithfulness"] == 5.0
    assert score == pytest.approx(4.2)
    assert breakdown["passed"] is True


def test_tier3_fails_below_threshold() -> None:
    judge = _FakeJudge([{
        "faithfulness": 2, "coverage": 3, "calibration": 3,
        "citation": 2, "conciseness": 3,
    }])
    score, breakdown = score_tier3(
        reference_answer="ref", actual_answer="a", retrieved_context="",
        judge=judge,
    )
    assert score < TIER3_JUDGE_MIN
    assert breakdown["passed"] is False


def test_tier3_falls_back_to_top_level_score_when_subscores_absent() -> None:
    judge = _FakeJudge([{"score": 4.0, "notes": "n/a"}])
    score, breakdown = score_tier3(
        reference_answer="r", actual_answer="a", retrieved_context="c",
        judge=judge,
    )
    assert score == 4.0
    assert breakdown["passed"] is True


def test_tier3_requires_judge() -> None:
    with pytest.raises(ValueError, match="judge"):
        score_tier3(
            reference_answer="r", actual_answer="a", retrieved_context="c",
            judge=None,
        )


# ---------------------------------------------------------------------------
# End-to-end run_question + aggregate_metrics using a scripted agent
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
    def __init__(self, answer: str) -> None:
        self._answer = answer
        self.calls: list[dict] = []

    def create(self, **kwargs: Any) -> _FakeResp:
        self.calls.append(kwargs)
        return _FakeResp(choices=[_FakeChoice(_FakeMessage(content=self._answer))])


class _ScriptedLLM:
    def __init__(self, answer: str) -> None:
        self._completions = _Completions(answer)

    @property
    def chat(self) -> Any:
        return self

    @property
    def completions(self) -> _Completions:
        return self._completions


def _make_agent(answer: str) -> NhtsaAgent:
    cfg = ProjectConfig(
        catalog="cat", schema="sch", volume="vol", llm_endpoint="llm",
        embedding_endpoint="emb", warehouse_id="wh",
        vector_search_endpoint="ep", genie_space_id="real",
    )
    return NhtsaAgent(
        cfg=cfg,
        llm_client=_ScriptedLLM(answer),
        tool_context=ToolContext(cfg=cfg, vs_cfg=VectorSearchConfig()),
        session_store=InMemorySessionStore(),
        system_prompt="SYS",
    )


def test_run_question_tier1_pass() -> None:
    agent = _make_agent("The answer is 42.")
    q = EvalQuestion(tier=TIER_DETERMINISTIC, question="how many?", expected_value="42")
    result = run_question(agent, q, judge=None)
    assert result.passed is True
    assert result.score == 1.0
    assert result.pending is False


def test_run_question_tier1_pending_is_not_passed() -> None:
    agent = _make_agent("anything")
    q = EvalQuestion(
        tier=TIER_DETERMINISTIC, question="how many?", expected_value="TBD:ungrounded"
    )
    result = run_question(agent, q, judge=None)
    assert result.pending is True
    assert result.passed is False
    assert result.score == 0.0


def test_run_question_tier2_uses_judge() -> None:
    agent = _make_agent("Per campaign 23V-085, the remedy reprograms firmware.")
    q = EvalQuestion(
        tier=TIER_GROUNDED,
        question="remedy?",
        source_id="23V-085",
        expected_claim="Reprogrammed firmware.",
    )
    judge = _FakeJudge([{"score": 5, "notes": "match"}])
    result = run_question(agent, q, judge=judge)
    assert result.passed is True
    assert judge.prompts, "judge should be called on cite-match"


def test_run_question_tier3_uses_judge() -> None:
    agent = _make_agent("Synthesised answer with IDs 23V-085 and 11543210.")
    q = EvalQuestion(
        tier=TIER_SYNTHESIS, question="themes?", reference_answer="ref",
    )
    judge = _FakeJudge([{
        "faithfulness": 4, "coverage": 4, "calibration": 4,
        "citation": 4, "conciseness": 4, "notes": "ok",
    }])
    result = run_question(agent, q, judge=judge)
    assert result.score == 4.0
    assert result.passed is True


def test_aggregate_metrics_tier1_ignores_pending_in_pass_rate() -> None:
    agent = _make_agent("42")
    qs = [
        EvalQuestion(tier=TIER_DETERMINISTIC, question="q1", expected_value="42"),
        EvalQuestion(tier=TIER_DETERMINISTIC, question="q2", expected_value="99"),
        EvalQuestion(tier=TIER_DETERMINISTIC, question="q3", expected_value="TBD:x"),
    ]
    results = [run_question(agent, q, judge=None) for q in qs]
    agg = aggregate_metrics(TIER_DETERMINISTIC, results)
    # 2 scored (q1 pass, q2 fail) → pass_rate = 0.5, n_pending = 1.
    assert agg["n"] == 3
    assert agg["n_scored"] == 2
    assert agg["n_pending"] == 1
    assert agg["pass_rate"] == 0.5
    assert "tier1_exact_match_rate" in agg


def test_aggregate_metrics_tier3_tracks_mean_judge_score() -> None:
    agent = _make_agent("answer")
    q = EvalQuestion(tier=TIER_SYNTHESIS, question="q", reference_answer="r")
    judge = _FakeJudge(
        [
            {"faithfulness": 5, "coverage": 5, "calibration": 5,
             "citation": 5, "conciseness": 5},
            {"faithfulness": 3, "coverage": 3, "calibration": 3,
             "citation": 3, "conciseness": 3},
        ]
    )
    results = [run_question(agent, q, judge) for _ in range(2)]
    agg = aggregate_metrics(TIER_SYNTHESIS, results)
    assert agg["tier3_avg_judge_score"] == pytest.approx(4.0)


def test_aggregate_metrics_empty_returns_zero() -> None:
    agg = aggregate_metrics(TIER_DETERMINISTIC, [])
    assert agg == {"n": 0, "pass_rate": 0.0}


def test_default_eval_files_cover_all_tiers() -> None:
    assert set(DEFAULT_EVAL_FILES.keys()) == set(VALID_TIERS)
