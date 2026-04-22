"""Eval harness for the NHTSA defect-intelligence agent.

Implements the 3-tier strategy described in ``docs/05_evaluation_strategy.md``:

    Tier 1 — Deterministic (exact-match / Jaccard over list values).
    Tier 2 — Citation-grounded (cite-id match + LLM judge on claim).
    Tier 3 — Judged synthesis (5-point rubric averaged).

Responsibilities kept narrow on purpose — this module is the harness,
not the scorers' intelligence. The LLM judge itself is an injected
``LLMJudge`` protocol object (see the notebook driver for the default
wiring against the Databricks serving endpoint). Tests inject a fake
judge that returns canned scores.

The harness writes:

    - Per-question MLflow tags + a row in the aggregate results table.
    - One MLflow run per ``run_eval`` call, tagged with the delta
      version of the gold tables (when supplied) and the system-prompt
      hash.
    - A Python dict of aggregate metrics — this is the value returned
      by the CLI / notebook so callers can gate on it without needing
      an MLflow round-trip.

Usage (CLI):

    uv run python -m nhtsa_curator.evaluation \\
        --tier all --target dev \\
        --eval-dir notebooks/eval \\
        --experiment /Shared/nhtsa-curator-course-pg

Usage (library — inside notebook 5.1):

    from nhtsa_curator.evaluation import run_eval, load_eval_set
    questions = load_eval_set("tier1", eval_dir / "tier1_deterministic.tsv")
    metrics = run_eval("tier1", questions, agent, judge=my_judge)

This module has no Databricks SDK imports — it's Spark-free and
testable on any machine with mlflow installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import mlflow
from loguru import logger

from .agent import AgentResult, NhtsaAgent


# ---------------------------------------------------------------------------
# Tier names — exported so callers don't hard-code strings.
# ---------------------------------------------------------------------------

TIER_DETERMINISTIC = "tier1"
TIER_GROUNDED = "tier2"
TIER_SYNTHESIS = "tier3"

VALID_TIERS: tuple[str, ...] = (
    TIER_DETERMINISTIC,
    TIER_GROUNDED,
    TIER_SYNTHESIS,
)

DEFAULT_EVAL_FILES: dict[str, str] = {
    TIER_DETERMINISTIC: "tier1_deterministic.tsv",
    TIER_GROUNDED: "tier2_grounded.tsv",
    TIER_SYNTHESIS: "tier3_synthesis.tsv",
}

# Pass thresholds per tier. Kept as module-level constants because the
# promotion gate in ``docs/05_evaluation_strategy.md`` is phrased as
# *relative* to these, so centralising them means a single edit shifts
# the whole pipeline.
TIER1_PASS_SCORE = 1.0          # exact match required (or Jaccard ≥ 0.8 for lists)
TIER1_LIST_JACCARD_MIN = 0.8
TIER2_JUDGE_MIN = 4             # 1-5 scale; ≥4 required
TIER3_JUDGE_MIN = 3.5           # 1-5 mean across 5 sub-scores


# ---------------------------------------------------------------------------
# Question + result shapes
# ---------------------------------------------------------------------------

@dataclass
class EvalQuestion:
    """One row from an eval TSV.

    We keep tier-specific fields as optional rather than building three
    dataclasses — downstream aggregations treat all tiers uniformly and
    the mutex-by-tier is enforced at load time.
    """

    tier: str
    question: str
    reference_sql: str | None = None         # tier1
    expected_value: str | None = None        # tier1
    source_id: str | None = None             # tier2
    expected_claim: str | None = None        # tier2
    reference_answer: str | None = None      # tier3

    @property
    def is_pending(self) -> bool:
        """True when this question's ground truth is unresolved.

        Tier-1 loader accepts ``TBD:<name>`` placeholders so the file
        can be committed before a prod data-refresh produces the real
        numbers. Pending questions are scored as 0 but flagged so they
        contribute to coverage but not to the pass rate.
        """
        if self.tier == TIER_DETERMINISTIC:
            return bool(self.expected_value and self.expected_value.startswith("TBD:"))
        return False


@dataclass
class QuestionResult:
    """Single-question outcome."""

    question: EvalQuestion
    agent_result: AgentResult
    score: float                             # 0.0 .. 1.0 for tier1/2; mean 1-5 for tier3
    passed: bool
    metric_breakdown: dict = field(default_factory=dict)
    judge_notes: str | None = None
    latency_s: float = 0.0
    pending: bool = False                    # truth-value TBD (tier1 only)

    def to_dict(self) -> dict:
        return {
            "tier": self.question.tier,
            "question": self.question.question,
            "answer": self.agent_result.answer,
            "score": self.score,
            "passed": self.passed,
            "pending": self.pending,
            "latency_s": round(self.latency_s, 3),
            "n_tool_calls": len(self.agent_result.tool_trace),
            "n_llm_calls": self.agent_result.n_llm_calls,
            "stopped_reason": self.agent_result.stopped_reason,
            "metric_breakdown": self.metric_breakdown,
            "judge_notes": self.judge_notes,
            "tool_trace": self.agent_result.tool_trace,
        }


# ---------------------------------------------------------------------------
# LLM judge protocol
# ---------------------------------------------------------------------------

class LLMJudge(Protocol):
    """Scoring LLM used by tier-2 and tier-3 scorers.

    Implementations return a dict with at minimum a ``score`` key. Extra
    keys are captured as breakdown + notes for later review.
    """

    def score(self, prompt: str, *, context: dict | None = None) -> dict: ...


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_eval_set(tier: str, path: Path | str) -> list[EvalQuestion]:
    """Parse a TSV eval file into ``EvalQuestion``s.

    The loader is strict about tier shape:

    - Tier 1 expects ``question\\treference_sql\\texpected_value``.
    - Tier 2 expects ``question\\tsource_id\\texpected_claim``.
    - Tier 3 expects ``question\\treference_answer``.

    Lines starting with ``#`` or blank lines are skipped. The first
    non-comment line is treated as the header and validated against the
    expected column names — a mismatch fails fast rather than silently
    misaligning columns.
    """
    if tier not in VALID_TIERS:
        raise ValueError(f"unknown tier {tier!r}; expected one of {VALID_TIERS}")

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"eval file not found: {p}")

    header_expected = {
        TIER_DETERMINISTIC: ("question", "reference_sql", "expected_value"),
        TIER_GROUNDED: ("question", "source_id", "expected_claim"),
        TIER_SYNTHESIS: ("question", "reference_answer"),
    }[tier]

    questions: list[EvalQuestion] = []
    seen_header = False

    with p.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, 1):
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split("\t")
            if not seen_header:
                if tuple(f.strip() for f in fields) != header_expected:
                    raise ValueError(
                        f"{p}: line {line_no}: expected header "
                        f"{header_expected}, got {tuple(fields)}"
                    )
                seen_header = True
                continue
            if len(fields) != len(header_expected):
                raise ValueError(
                    f"{p}: line {line_no}: expected {len(header_expected)} "
                    f"columns, got {len(fields)}"
                )
            questions.append(_build_question(tier, fields))

    if not questions:
        raise ValueError(f"{p}: no questions parsed (file empty after header?)")
    return questions


def _build_question(tier: str, fields: list[str]) -> EvalQuestion:
    if tier == TIER_DETERMINISTIC:
        return EvalQuestion(
            tier=tier,
            question=fields[0].strip(),
            reference_sql=fields[1].strip() or None,
            expected_value=fields[2].strip(),
        )
    if tier == TIER_GROUNDED:
        return EvalQuestion(
            tier=tier,
            question=fields[0].strip(),
            source_id=fields[1].strip(),
            expected_claim=fields[2].strip(),
        )
    # tier3
    return EvalQuestion(
        tier=tier,
        question=fields[0].strip(),
        reference_answer=fields[1].strip(),
    )


def load_all(eval_dir: Path | str) -> dict[str, list[EvalQuestion]]:
    """Convenience loader — returns one list per tier."""
    d = Path(eval_dir)
    return {
        tier: load_eval_set(tier, d / fname)
        for tier, fname in DEFAULT_EVAL_FILES.items()
    }


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_NUMERIC_ONLY_RE = re.compile(r"-?\d+(?:\.\d+)?")
_ID_NORMALISE_RE = re.compile(r"[^a-z0-9]")


def score_tier1(expected: str, actual_answer: str) -> tuple[float, dict]:
    """Exact-match / Jaccard scorer for deterministic questions.

    Decides list-vs-scalar by presence of a ``|`` separator in
    ``expected``. List answers are scored by substring membership: each
    expected token is checked against the answer (case-insensitive), and
    the coverage ratio (found / expected) is treated as Jaccard. Anything
    under :data:`TIER1_LIST_JACCARD_MIN` is 0, at-or-above is 1.0 — the
    gate is pass/fail, no smooth gradient.

    Substring membership (not tokenization on the answer side) is used
    because multi-word list values like "General Motors" get shredded
    by naive comma-splitting of natural-language answers.

    Scalar answers are normalised:
    - Numbers: if ``expected`` is pure numeric, extract the first number
      from the answer and compare as floats. Handles "42 campaigns"
      vs "42". Mixed text+number (e.g. "model year 2023") falls through
      to the string path so the surrounding context has to match too.
    - Strings: lowercased substring match.
    """
    expected = (expected or "").strip()
    actual_answer = actual_answer or ""

    if "|" in expected:
        exp_tokens = [t.strip() for t in expected.split("|") if t.strip()]
        if not exp_tokens:
            return 0.0, {"kind": "list", "reason": "empty_expected"}
        norm_answer = actual_answer.lower()
        intersection = [t for t in exp_tokens if t.lower() in norm_answer]
        missing = [t for t in exp_tokens if t.lower() not in norm_answer]
        jaccard = len(intersection) / len(exp_tokens)
        passed = jaccard >= TIER1_LIST_JACCARD_MIN
        return (
            1.0 if passed else 0.0,
            {
                "kind": "list",
                "jaccard": round(jaccard, 3),
                "intersection": intersection,
                "missing": missing,
            },
        )

    if _NUMERIC_ONLY_RE.fullmatch(expected):
        act_num = _NUMBER_RE.findall(actual_answer)
        if act_num and abs(float(act_num[0]) - float(expected)) < 1e-6:
            return 1.0, {"kind": "numeric", "expected": expected, "actual": act_num[0]}
        return 0.0, {
            "kind": "numeric",
            "expected": expected,
            "actual": act_num[0] if act_num else None,
        }

    norm_exp = expected.lower()
    norm_act = actual_answer.lower()
    if norm_exp and norm_exp in norm_act:
        return 1.0, {"kind": "string", "expected": expected}
    return 0.0, {"kind": "string", "expected": expected}


def score_tier2(
    source_id: str,
    expected_claim: str,
    actual_answer: str,
    judge: LLMJudge | None,
) -> tuple[float, dict]:
    """Citation + faithfulness scorer.

    A tier-2 answer passes iff:
        (a) the ``source_id`` appears in the answer (case-insensitive,
            ignoring hyphens/spaces so the agent's "TSB 10160095" matches
            ground truth "TSB-10160095"), AND
        (b) the LLM judge's claim-faithfulness score is
            >= :data:`TIER2_JUDGE_MIN`.

    Score is 1.0 on pass, 0.0 otherwise. Breakdown records both sub-scores
    so failed rows are diagnosable.

    When ``judge`` is None the claim side is skipped — the score
    degrades to citation-only. This is used in CI smoke runs where the
    judge round-trip would slow things unacceptably.
    """
    cite_ok = _cite_match(source_id, actual_answer)
    if not cite_ok:
        return 0.0, {
            "cite_ok": False,
            "judge_score": None,
            "reason": "source_id not present in answer",
        }

    if judge is None:
        return 1.0, {"cite_ok": True, "judge_score": None, "reason": "no judge"}

    judge_out = judge.score(
        _tier2_judge_prompt(source_id, expected_claim, actual_answer)
    )
    judge_score = _extract_score(judge_out)
    passed = judge_score >= TIER2_JUDGE_MIN
    return (
        1.0 if passed else 0.0,
        {
            "cite_ok": True,
            "judge_score": judge_score,
            "judge_notes": judge_out.get("notes"),
            "min_required": TIER2_JUDGE_MIN,
        },
    )


def score_tier3(
    reference_answer: str,
    actual_answer: str,
    retrieved_context: str,
    judge: LLMJudge,
) -> tuple[float, dict]:
    """5-point rubric scorer.

    The judge returns five sub-scores (faithfulness, coverage,
    calibration, citation, conciseness). We average them. Pass is mean
    >= :data:`TIER3_JUDGE_MIN` (3.5 on a 1-5 scale, leaving headroom
    above a mediocre 3).
    """
    if judge is None:
        raise ValueError("tier3 scoring requires a judge (LLMJudge protocol)")

    judge_out = judge.score(
        _tier3_judge_prompt(reference_answer, actual_answer, retrieved_context)
    )
    sub_scores = {
        k: float(judge_out.get(k, 0.0))
        for k in ("faithfulness", "coverage", "calibration", "citation", "conciseness")
    }
    # If the judge didn't return sub-scores, fall back to a top-level score.
    if sum(sub_scores.values()) == 0.0 and "score" in judge_out:
        mean = float(judge_out["score"])
    else:
        mean = statistics.fmean(sub_scores.values())
    passed = mean >= TIER3_JUDGE_MIN
    return (
        mean,
        {
            "sub_scores": sub_scores,
            "mean": round(mean, 3),
            "judge_notes": judge_out.get("notes"),
            "min_required": TIER3_JUDGE_MIN,
            "passed": passed,
        },
    )


def _cite_match(source_id: str, actual_answer: str) -> bool:
    """Return True iff the normalised source_id appears in the answer."""
    if not source_id:
        return False
    norm_id = _ID_NORMALISE_RE.sub("", source_id.lower())
    norm_answer = _ID_NORMALISE_RE.sub("", actual_answer.lower())
    return norm_id in norm_answer


def _extract_score(judge_out: dict) -> float:
    """Pull a numeric score out of the judge response, with fallbacks."""
    for key in ("score", "judge_score", "value"):
        if key in judge_out:
            try:
                return float(judge_out[key])
            except (TypeError, ValueError):
                continue
    return 0.0


def _tier2_judge_prompt(source_id: str, expected_claim: str, actual_answer: str) -> str:
    """Prompt wrapper for tier-2 judging (claim vs. source)."""
    return (
        "You are evaluating a defect-intelligence agent's answer.\n\n"
        f"Cited source id: {source_id}\n"
        f"Expected claim (ground truth):\n{expected_claim}\n\n"
        f"Agent answer:\n{actual_answer}\n\n"
        "On a 1-5 scale, how well does the agent's claim about this "
        "source match the expected claim? Return JSON:\n"
        "  {\"score\": <1-5 int>, \"notes\": \"<one-sentence rationale>\"}"
    )


def _tier3_judge_prompt(
    reference_answer: str, actual_answer: str, retrieved_context: str
) -> str:
    """Prompt wrapper for tier-3 judging (synthesis rubric)."""
    return (
        "Score the agent's synthesis on a 5-point rubric. For each "
        "dimension return an integer 1-5.\n\n"
        "Dimensions:\n"
        "  - faithfulness: claims supported by retrieved context\n"
        "  - coverage: relevant themes captured vs. reference\n"
        "  - calibration: uncertainty flagged where warranted\n"
        "  - citation: every claim backed by a source id\n"
        "  - conciseness: information-dense, no padding\n\n"
        f"Reference answer:\n{reference_answer}\n\n"
        f"Retrieved context (agent-visible):\n{retrieved_context}\n\n"
        f"Agent answer:\n{actual_answer}\n\n"
        "Return JSON:\n"
        "  {\n"
        "    \"faithfulness\": <1-5>, \"coverage\": <1-5>,\n"
        "    \"calibration\": <1-5>, \"citation\": <1-5>,\n"
        "    \"conciseness\": <1-5>, \"notes\": \"<one-sentence rationale>\"\n"
        "  }"
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_question(
    agent: NhtsaAgent,
    question: EvalQuestion,
    judge: LLMJudge | None,
) -> QuestionResult:
    """Run one question end-to-end and score it.

    Uses a fresh session per question so one bad turn can't poison the
    next one's accumulated_filters. Eval is stateless by design.
    """
    t0 = time.perf_counter()
    agent_result = agent.run_turn(
        user_message=question.question,
        user_id=f"eval-{question.tier}",
    )
    latency = time.perf_counter() - t0

    if question.is_pending:
        return QuestionResult(
            question=question,
            agent_result=agent_result,
            score=0.0,
            passed=False,
            pending=True,
            metric_breakdown={"reason": "expected_value is TBD:*"},
            latency_s=latency,
        )

    if question.tier == TIER_DETERMINISTIC:
        score, breakdown = score_tier1(
            expected=question.expected_value or "",
            actual_answer=agent_result.answer,
        )
        passed = score >= TIER1_PASS_SCORE
    elif question.tier == TIER_GROUNDED:
        score, breakdown = score_tier2(
            source_id=question.source_id or "",
            expected_claim=question.expected_claim or "",
            actual_answer=agent_result.answer,
            judge=judge,
        )
        passed = score >= TIER1_PASS_SCORE
    else:  # tier3
        retrieved = _build_retrieved_context(agent_result)
        score, breakdown = score_tier3(
            reference_answer=question.reference_answer or "",
            actual_answer=agent_result.answer,
            retrieved_context=retrieved,
            judge=judge,
        )
        passed = bool(breakdown.get("passed", score >= TIER3_JUDGE_MIN))

    return QuestionResult(
        question=question,
        agent_result=agent_result,
        score=score,
        passed=passed,
        metric_breakdown=breakdown,
        latency_s=latency,
        judge_notes=breakdown.get("judge_notes"),
    )


def _build_retrieved_context(agent_result: AgentResult) -> str:
    """Flatten tool results into a single string for the judge's view.

    We deliberately only include the *preview* from the tool trace
    (tool results are long; the judge only needs an evidentiary
    foothold). If nothing was retrieved, returns empty string — the
    judge knows to penalise citation in that case.
    """
    parts: list[str] = []
    for t in agent_result.tool_trace:
        preview = t.get("result_preview") or ""
        parts.append(f"[{t['name']}] {preview}")
    return "\n".join(parts)


def aggregate_metrics(tier: str, results: list[QuestionResult]) -> dict:
    """Compute per-tier aggregate metrics.

    The metric names match ``docs/05_evaluation_strategy.md`` so the
    dashboard and promotion gate reference the same strings.
    """
    if not results:
        return {"n": 0, "pass_rate": 0.0}

    scored = [r for r in results if not r.pending]
    total_scored = len(scored) or 1
    passes = sum(1 for r in scored if r.passed)
    latencies = [r.latency_s for r in results]
    tool_calls = [len(r.agent_result.tool_trace) for r in results]
    llm_calls = [r.agent_result.n_llm_calls for r in results]

    metrics = {
        "tier": tier,
        "n": len(results),
        "n_scored": len(scored),
        "n_pending": sum(1 for r in results if r.pending),
        "pass_rate": round(passes / total_scored, 4),
        "mean_latency_seconds": round(statistics.fmean(latencies), 3),
        "p95_latency_seconds": round(_p95(latencies), 3),
        "mean_tool_calls": round(statistics.fmean(tool_calls), 2),
        "mean_llm_calls": round(statistics.fmean(llm_calls), 2),
        "error_rate": round(
            sum(1 for r in results if r.agent_result.stopped_reason != "ok")
            / len(results),
            4,
        ),
    }

    if tier == TIER_DETERMINISTIC:
        metrics["tier1_exact_match_rate"] = metrics["pass_rate"]
    elif tier == TIER_GROUNDED:
        metrics["tier2_grounded_pass_rate"] = metrics["pass_rate"]
    else:
        metrics["tier3_avg_judge_score"] = round(
            statistics.fmean(r.score for r in scored) if scored else 0.0, 3
        )
    return metrics


def _p95(values: list[float]) -> float:
    """Lightweight p95 without numpy."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return ordered[idx]


def run_eval(
    tier: str,
    questions: list[EvalQuestion],
    agent: NhtsaAgent,
    judge: LLMJudge | None = None,
    *,
    experiment_name: str | None = None,
    run_name: str | None = None,
    extra_tags: dict[str, str] | None = None,
    system_prompt: str | None = None,
    gold_delta_version: str | None = None,
) -> dict:
    """Run the agent over a question list and log results to MLflow.

    Returns a dict with ``metrics`` + ``results`` (list of per-question
    dicts) + ``mlflow_run_id`` (None when MLflow isn't configured).
    """
    if tier not in VALID_TIERS:
        raise ValueError(f"unknown tier {tier!r}")

    if experiment_name:
        mlflow.set_experiment(experiment_name)

    tags = {
        "tier": tier,
        "agent.model": agent.cfg.llm_endpoint,
        "agent.catalog": agent.cfg.catalog,
        "agent.schema": agent.cfg.db_schema,
    }
    if gold_delta_version:
        tags["gold.delta_version"] = gold_delta_version
    if system_prompt:
        tags["system_prompt.sha256"] = _sha256(system_prompt)
    if extra_tags:
        tags.update(extra_tags)

    results: list[QuestionResult] = []
    with mlflow.start_run(run_name=run_name or f"eval-{tier}"):
        mlflow.set_tags(tags)
        for q in questions:
            res = run_question(agent, q, judge)
            results.append(res)
            _log_question(res)

        metrics = aggregate_metrics(tier, results)
        _log_aggregate(metrics)
        run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None

    return {
        "mlflow_run_id": run_id,
        "metrics": metrics,
        "results": [r.to_dict() for r in results],
    }


def _log_question(result: QuestionResult) -> None:
    """Log per-question outcome as a metric + a tag-free run parameter.

    MLflow doesn't have a per-row concept natively, so we emit the score
    as a step-indexed metric and dump the full row as a JSON artifact
    per-question. This keeps the aggregate metrics queryable while
    preserving per-row debuggability.
    """
    idx = hash(result.question.question) & 0xFFFF
    mlflow.log_metric(f"{result.question.tier}.score", result.score, step=idx)
    mlflow.log_metric(f"{result.question.tier}.latency_s", result.latency_s, step=idx)
    try:
        mlflow.log_dict(result.to_dict(), f"per_question/{idx}.json")
    except Exception as exc:  # noqa: BLE001 — artifacts are best-effort in CI
        logger.warning("failed to log per-question artifact: {}", exc)


def _log_aggregate(metrics: dict) -> None:
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(f"agg.{k}", float(v))
    try:
        mlflow.log_dict(metrics, "aggregate_metrics.json")
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to log aggregate artifact: {}", exc)


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the NHTSA agent eval harness.")
    p.add_argument(
        "--tier", choices=(*VALID_TIERS, "all"), default="all",
        help="Which tier to run (default: all).",
    )
    p.add_argument(
        "--target", default="dev",
        help="databricks.yml target (dev/acc/prd). Selects env config.",
    )
    p.add_argument(
        "--eval-dir", default="notebooks/eval",
        help="Path to the directory holding the eval TSVs.",
    )
    p.add_argument(
        "--config", default="project_config.yml",
        help="Path to project_config.yml.",
    )
    p.add_argument(
        "--experiment",
        help="MLflow experiment path. Defaults to ProjectConfig.experiment_name.",
    )
    p.add_argument(
        "--run-name",
        help="MLflow run name. Default: 'eval-<tier>'.",
    )
    p.add_argument(
        "--no-judge", action="store_true",
        help="Skip tier-2/3 judging (citation-only for tier-2, fails tier-3).",
    )
    return p.parse_args(argv)


def _cli_main(argv: list[str] | None = None) -> int:
    """CLI entry point — kept thin so the library API is the canonical one.

    Imports Databricks dependencies lazily so ``python -m nhtsa_curator.
    evaluation --help`` works on a clean machine without the full
    Databricks stack installed.
    """
    args = _parse_args(argv)
    from .config import load_config, load_system_prompt  # noqa: PLC0415

    cfg = load_config(args.config, args.target)
    system_prompt = load_system_prompt(args.config)

    # Build a live agent + judge via the same wiring notebook 5.1 uses.
    agent, judge = _build_live_agent_and_judge(cfg, system_prompt, use_judge=not args.no_judge)

    eval_dir = Path(args.eval_dir)
    tiers = VALID_TIERS if args.tier == "all" else (args.tier,)
    experiment = args.experiment or cfg.experiment_name

    summary: dict[str, dict] = {}
    for tier in tiers:
        questions = load_eval_set(tier, eval_dir / DEFAULT_EVAL_FILES[tier])
        logger.info("running tier={} n={}", tier, len(questions))
        out = run_eval(
            tier=tier,
            questions=questions,
            agent=agent,
            judge=judge,
            experiment_name=experiment,
            run_name=args.run_name,
            system_prompt=system_prompt,
        )
        summary[tier] = out["metrics"]

    print(json.dumps(summary, indent=2))
    return 0


def _build_live_agent_and_judge(
    cfg: Any, system_prompt: str, use_judge: bool
) -> tuple[NhtsaAgent, LLMJudge | None]:
    """Wire up the production agent + judge for the CLI path.

    Kept in this module (rather than the notebook) so both the CLI and
    scheduled workflow share the same wiring. Fails loudly if the
    Databricks environment isn't available — the CLI is not meant for
    laptop smoke tests, those use the notebook.
    """
    from databricks.sdk import WorkspaceClient  # noqa: PLC0415
    from databricks.vector_search.client import VectorSearchClient  # noqa: PLC0415

    from .config import VectorSearchConfig  # noqa: PLC0415
    from .mcp import ToolContext, _normalise_statement_result  # noqa: PLC0415
    from .memory import InMemorySessionStore  # noqa: PLC0415

    ws = WorkspaceClient()
    vs_client = VectorSearchClient(disable_notice=True)

    def _sql(query: str) -> list[dict]:
        res = ws.statement_execution.execute_statement(
            statement=query, warehouse_id=cfg.warehouse_id, wait_timeout="30s"
        )
        return _normalise_statement_result(res)

    tool_ctx = ToolContext(
        cfg=cfg,
        vs_client=vs_client,
        genie_client=ws.genie,
        sql_executor=_sql,
        vs_cfg=VectorSearchConfig(),
    )

    llm_client = ws.serving_endpoints.get_open_ai_client()
    agent = NhtsaAgent(
        cfg=cfg,
        llm_client=llm_client,
        tool_context=tool_ctx,
        session_store=InMemorySessionStore(),
        system_prompt=system_prompt,
    )

    judge = OpenAICompatJudge(llm_client, model=cfg.llm_endpoint) if use_judge else None
    return agent, judge


# ---------------------------------------------------------------------------
# Default LLM judge implementation
# ---------------------------------------------------------------------------

class OpenAICompatJudge:
    """Concrete ``LLMJudge`` using the OpenAI-compatible chat API.

    Returns JSON output parsed to a dict. On parse failure returns a
    zero-score response so the harness degrades gracefully — a crashed
    judge should not nuke the whole eval run.
    """

    def __init__(
        self,
        llm_client: Any,
        model: str,
        *,
        temperature: float = 0.0,
    ) -> None:
        self._llm = llm_client
        self._model = model
        self._temperature = temperature

    def score(self, prompt: str, *, context: dict | None = None) -> dict:
        msgs = [
            {"role": "system", "content": "You are a strict but fair evaluator."},
            {"role": "user", "content": prompt},
        ]
        try:
            resp = self._llm.chat.completions.create(
                model=self._model,
                messages=msgs,
                temperature=self._temperature,
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content or "{}"
            return json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning("judge returned non-JSON: {}", exc)
            return {"score": 0, "notes": "judge output was not valid JSON"}
        except Exception as exc:  # noqa: BLE001
            logger.exception("judge call failed: {}", exc)
            return {"score": 0, "notes": f"judge error: {exc}"}


# ---------------------------------------------------------------------------
# Phase 6 — text-only feedback scorers + evaluate_agent pre-register wrapper
#
# These are the scorers applied to *live traces* by the
# ``update_traces_aggregated`` workflow, AND to the eval_inputs.txt
# smoke set by the ``log_register_agent`` workflow. Kept deliberately
# simple (regex / small LLM-judge guidelines) because they run on
# every trace, every hour — cost matters.
# ---------------------------------------------------------------------------

# Matches NHTSA-style source ids emitted by the agent. Kept as a union
# so a single regex can detect any citation shape without per-type code.
#   - Recall campaign numbers: 23V123456 / 22V-123-456 / RECALL 23V123456
#   - Complaint ODI ids: 11512345 (8-digit)
#   - Investigation numbers: PE22-001 / EA23-002 / DP22-003 / RQ20-004
#   - TSB numbers: TSB 10160095 / TSB-10160095 / 10160095 (8+ digit)
_CITE_ID_RE = re.compile(
    r"(?:"
    r"\b\d{2}[VEIT]-?\d{3,}-?\d*\b"      # recall campaign #
    r"|\b(?:PE|EA|DP|RQ|AQ)\d{2}-\d{3}\b"  # investigation #
    r"|\bTSB[- ]?\d{6,}\b"                 # TSB #
    r"|\bODI[- ]?\d{6,}\b"                 # ODI complaint #
    r")",
    re.IGNORECASE,
)

# A short list of known OEM signals we care about for dashboarding.
# Kept as a constant so the dashboard's "top queried OEMs" chart matches
# the scorer's label set exactly.
OEM_KEYWORDS: tuple[str, ...] = (
    "Tesla", "Ford", "General Motors", "Chevrolet", "GMC", "Cadillac",
    "Toyota", "Lexus", "Honda", "Acura", "Nissan", "Infiniti",
    "Hyundai", "Kia", "Genesis", "BMW", "Mercedes-Benz", "Audi",
    "Volkswagen", "Porsche", "Stellantis", "Jeep", "Ram", "Dodge",
    "Chrysler", "Subaru", "Mazda", "Volvo", "Rivian", "Lucid",
    "Polestar", "Cruise", "Waymo", "Zoox",
)


def _response_text(outputs: Any) -> str:
    """Normalise the variant ``outputs`` shapes MLflow hands to scorers.

    ``mlflow.genai.evaluate`` passes whatever the model's predict fn
    returned — which may be a raw string, a dict with ``text``, a list
    of strings, or a Responses API ``output_text`` envelope. We flatten
    all of those to a single string so the scorer bodies stay boring.
    """
    if isinstance(outputs, str):
        return outputs
    if isinstance(outputs, dict):
        if "text" in outputs:
            return str(outputs["text"])
        if "content" in outputs and isinstance(outputs["content"], str):
            return outputs["content"]
        if "output" in outputs:
            return _response_text(outputs["output"])
    if isinstance(outputs, list) and outputs:
        head = outputs[0]
        if isinstance(head, str):
            return head
        if isinstance(head, dict):
            if "text" in head:
                return str(head["text"])
            if "content" in head:
                return _response_text(head["content"])
    return str(outputs or "")


def cite_id_present(outputs: Any) -> bool:
    """Scorer: at least one NHTSA source id appears in the answer.

    Cheapest possible trust signal — the agent's instructions require
    every claim to carry a source id. If this fails, we know the answer
    is ungrounded without paying an LLM judge.
    """
    return bool(_CITE_ID_RE.search(_response_text(outputs)))


def word_count_under(outputs: Any, max_words: int = 400) -> bool:
    """Scorer: answer stays under ``max_words`` words.

    NHTSA questions reward tight, evidence-linked summaries. Raise the
    ceiling higher than arxiv (350) because defect-intel answers often
    list multiple campaigns. A capped length also prevents the agent
    from stealing dashboard space with very long narratives.
    """
    return len(_response_text(outputs).split()) <= max_words


def mentions_oem(outputs: Any) -> bool:
    """Scorer: answer references at least one known OEM.

    This is an NHTSA-specific scorer — tells the dashboard how often
    the agent actually grounds its answer in a manufacturer, vs giving
    a generic industry comment. Not every question mentions an OEM, so
    this is treated as an informational metric, not a hard gate.
    """
    text = _response_text(outputs)
    norm = text.lower()
    return any(oem.lower() in norm for oem in OEM_KEYWORDS)


def _build_mlflow_guidelines(model: str | None = None) -> dict:
    """Construct the MLflow ``Guidelines`` scorers lazily.

    Defined inside a function so a test import doesn't try to connect to
    a Databricks-hosted judge. The caller passes a ``model`` string;
    outside Databricks this will be None and the guidelines scorer is
    skipped.
    """
    from mlflow.genai.scorers import Guidelines  # noqa: PLC0415

    judge_model = model or "databricks:/databricks-gpt-oss-120b"
    # MLflow Guidelines scorers require a fully-qualified URI
    # (`<provider>:/<model-name>`). Bare endpoint names like
    # `databricks-llama-4-maverick` — which is what `cfg.llm_endpoint`
    # carries — get rejected with "Malformed model uri". Normalise here
    # so every caller gets a valid URI without caring about the prefix.
    if judge_model and ":/" not in judge_model:
        judge_model = f"databricks:/{judge_model}"

    factual_defect_guideline = Guidelines(
        name="factual_defect",
        guidelines=[
            "The response must only make factual claims about vehicle "
            "defects, recalls, investigations, or TSBs that are supported "
            "by the source id(s) it cites.",
            "The response must not speculate about unreported defects.",
            "If the evidence is insufficient, the response must say so "
            "rather than inventing details.",
        ],
        model=judge_model,
    )

    cite_every_claim_guideline = Guidelines(
        name="cite_every_claim",
        guidelines=[
            "Every quantitative claim (counts, dates, model years) must "
            "be accompanied by a source id (recall campaign number, "
            "complaint ODI id, investigation number, or TSB number).",
            "The response must not give a count or date without a citation.",
        ],
        model=judge_model,
    )

    scope_guideline = Guidelines(
        name="stays_in_scope",
        guidelines=[
            "The response must only discuss topics about vehicle safety, "
            "defects, recalls, complaints, investigations, or TSBs.",
            "If asked about unrelated topics, the response must politely "
            "redirect to defect-intelligence questions.",
        ],
        model=judge_model,
    )

    return {
        "factual_defect": factual_defect_guideline,
        "cite_every_claim": cite_every_claim_guideline,
        "stays_in_scope": scope_guideline,
    }


def evaluate_agent(
    cfg: Any,
    eval_inputs_path: str | Path,
    *,
    use_guidelines: bool = True,
) -> Any:
    """Pre-register smoke-eval used by ``log_register_agent``.

    Runs the same live agent that Phase 4/5 builds, feeds it the
    questions in ``eval_inputs_path`` (one per line — same file format
    as the reference project), and scores each trace with the
    Phase 6 feedback scorers. The result's ``metrics`` dict is fed into
    :func:`log_register_agent` as the ``evaluation_metrics`` argument so
    every model version carries its smoke score at registration time.

    Returns an ``mlflow.models.EvaluationResult``. Guidelines scoring is
    opt-out (``use_guidelines=False``) because it calls a
    Databricks-hosted judge endpoint — fine in the scheduled workflow
    but a bad default for CI / unit tests.
    """
    from .agent import NhtsaAgent  # noqa: PLC0415 — avoid self-import
    from .config import VectorSearchConfig, load_system_prompt  # noqa: PLC0415
    from .mcp import ToolContext, _normalise_statement_result  # noqa: PLC0415
    from .memory import InMemorySessionStore  # noqa: PLC0415

    from databricks.sdk import WorkspaceClient  # noqa: PLC0415
    from databricks.vector_search.client import VectorSearchClient  # noqa: PLC0415

    ws = WorkspaceClient()
    vs_client = VectorSearchClient(disable_notice=True)

    def _sql(query: str) -> list[dict]:
        res = ws.statement_execution.execute_statement(
            statement=query, warehouse_id=cfg.warehouse_id, wait_timeout="30s"
        )
        return _normalise_statement_result(res)

    tool_ctx = ToolContext(
        cfg=cfg,
        vs_client=vs_client,
        genie_client=ws.genie,
        sql_executor=_sql,
        vs_cfg=VectorSearchConfig(),
    )

    llm_client = ws.serving_endpoints.get_open_ai_client()
    agent = NhtsaAgent(
        cfg=cfg,
        llm_client=llm_client,
        tool_context=tool_ctx,
        session_store=InMemorySessionStore(),
        system_prompt=load_system_prompt(),
    )

    with open(eval_inputs_path) as fh:
        eval_data = [
            {"inputs": {"question": line.strip()}}
            for line in fh
            if line.strip() and not line.lstrip().startswith("#")
        ]

    def _predict_fn(question: str) -> str:
        result = agent.run_turn(user_message=question, user_id="register-eval")
        return result.answer

    scorers: list[Any] = [
        mlflow.genai.scorer(cite_id_present),
        mlflow.genai.scorer(word_count_under),
        mlflow.genai.scorer(mentions_oem),
    ]
    if use_guidelines:
        scorers.extend(_build_mlflow_guidelines(model=cfg.llm_endpoint).values())

    return mlflow.genai.evaluate(
        predict_fn=_predict_fn,
        data=eval_data,
        scorers=scorers,
    )


__all__ = [
    "TIER_DETERMINISTIC",
    "TIER_GROUNDED",
    "TIER_SYNTHESIS",
    "VALID_TIERS",
    "DEFAULT_EVAL_FILES",
    "TIER1_LIST_JACCARD_MIN",
    "TIER2_JUDGE_MIN",
    "TIER3_JUDGE_MIN",
    "EvalQuestion",
    "QuestionResult",
    "LLMJudge",
    "OpenAICompatJudge",
    "OEM_KEYWORDS",
    "load_eval_set",
    "load_all",
    "score_tier1",
    "score_tier2",
    "score_tier3",
    "run_question",
    "run_eval",
    "aggregate_metrics",
    "cite_id_present",
    "word_count_under",
    "mentions_oem",
    "evaluate_agent",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli_main())
