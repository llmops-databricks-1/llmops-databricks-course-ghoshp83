# Databricks notebook source
# MAGIC %md
# MAGIC # 5.2 — Eval triage (post-run)
# MAGIC
# MAGIC Reads the most-recent per-tier MLflow runs produced by `5.1_run_eval`
# MAGIC and prints:
# MAGIC   1. Per-tier run URLs + aggregate metrics (for the demo deck).
# MAGIC   2. Tier-1 scored-row breakdown (expected vs. actual answer) to
# MAGIC      diagnose why the agent's answers miss the TSV ground truth.
# MAGIC   3. Tier-2 failure breakdown (cite_ok, judge_score, reason) to
# MAGIC      bucket failures into TSV-fix vs. judge-calibration categories.
# MAGIC   4. Tier-3 spot-check on the two worst-scoring rows.
# MAGIC
# MAGIC **Why this is a separate notebook, not cells appended to 5.1:**
# MAGIC - 5.1 runs the eval (26+ minutes). This notebook only reads MLflow
# MAGIC   artifacts — seconds. Running it doesn't cost cluster time on the
# MAGIC   expensive agent calls, so you can iterate on triage logic freely.
# MAGIC - After a job run of 5.1, the in-memory `summary` dict is gone. This
# MAGIC   notebook rehydrates state from MLflow so the triage is reproducible.

# COMMAND ----------
import json
import os
from pathlib import Path

import mlflow
from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession

from nhtsa_curator.config import get_env, load_config

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()

dbutils.widgets.text("env", "dev")
dbutils.widgets.text("run_id_filter", "", "Only triage runs tagged with this run_id (blank = latest).")
dbutils.widgets.text("max_tier1_scored", "25", "How many tier-1 scored rows to print.")
dbutils.widgets.text("max_tier2_failures", "10", "How many tier-2 failures to print.")

env = get_env(spark)
cfg = load_config("../project_config.yml", env)

run_id_filter = dbutils.widgets.get("run_id_filter").strip()
max_tier1_scored = int(dbutils.widgets.get("max_tier1_scored"))
max_tier2_failures = int(dbutils.widgets.get("max_tier2_failures"))

print(f"env={env} | experiment={cfg.experiment_name}")
print(f"run_id_filter={run_id_filter or '(latest)'}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Locate the experiment + resolve the latest per-tier runs
# MAGIC Search by `tags.tier = 'tier{N}'` (set by `run_eval`). If `run_id_filter`
# MAGIC is provided, additionally filter by `tags.run_id`, which makes the
# MAGIC triage deterministic even when parallel jobs write into the same
# MAGIC experiment.

# COMMAND ----------
w = WorkspaceClient()
host = w.config.host.rstrip("/")

exp = mlflow.get_experiment_by_name(cfg.experiment_name)
assert exp is not None, f"experiment missing: {cfg.experiment_name} — did 5.1 ever run?"
print(f"experiment_id = {exp.experiment_id}")
print(f"experiment URL: {host}/ml/experiments/{exp.experiment_id}")

client = mlflow.MlflowClient()


def _latest_run_for_tier(tier: str):
    """Most-recent run tagged with this tier (optionally narrowed by run_id)."""
    filters = [f"tags.tier = '{tier}'"]
    if run_id_filter:
        filters.append(f"tags.run_id = '{run_id_filter}'")
    filter_string = " and ".join(filters)
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=filter_string,
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    return runs[0] if runs else None


runs_by_tier: dict[str, object] = {}
for tier in ("tier1", "tier2", "tier3"):
    run = _latest_run_for_tier(tier)
    if run is None:
        print(f"  {tier}: NO RUN FOUND (skip)")
        continue
    runs_by_tier[tier] = run
    url = f"{host}/ml/experiments/{exp.experiment_id}/runs/{run.info.run_id}"
    print(f"  {tier}: {run.info.run_id}  {url}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Aggregate metrics snapshot — for the demo deck
# MAGIC Pulls every ``agg.*`` metric off each run. These are the same numbers
# MAGIC printed at the end of 5.1 — but they now live in MLflow, not in a
# MAGIC notebook-local variable.

# COMMAND ----------
agg_snapshot: dict[str, dict] = {}
for tier, run in runs_by_tier.items():
    metrics = {
        k.removeprefix("agg."): round(v, 4)
        for k, v in run.data.metrics.items()
        if k.startswith("agg.")
    }
    agg_snapshot[tier] = metrics
    print(f"\n--- {tier} ---")
    print(json.dumps(metrics, indent=2))

print("\n--- compact JSON for deck ---")
print(json.dumps(agg_snapshot, indent=2))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Load per-question artifacts
# MAGIC Every `run_eval` call logs one `per_question/<hash>.json` artifact per
# MAGIC row. We download + parse them once here, then reuse the list across
# MAGIC the triage cells below.

# COMMAND ----------
def _load_per_question(run) -> list[dict]:
    """Download + parse per_question/*.json for one MLflow run."""
    try:
        artifact_dir = client.download_artifacts(run.info.run_id, "per_question")
    except Exception as exc:  # noqa: BLE001 — artifacts may be missing if run crashed
        print(f"  could not load artifacts for {run.info.run_id}: {exc}")
        return []
    rows: list[dict] = []
    for fname in sorted(os.listdir(artifact_dir)):
        if not fname.endswith(".json"):
            continue
        with open(Path(artifact_dir) / fname) as f:
            rows.append(json.load(f))
    return rows


per_tier_results: dict[str, list[dict]] = {
    tier: _load_per_question(run) for tier, run in runs_by_tier.items()
}
for tier, rows in per_tier_results.items():
    print(f"{tier}: loaded {len(rows)} per-question artifacts")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Enrich artifacts with TSV ground-truth fields
# MAGIC The MLflow artifact only stores the question *text* + agent answer
# MAGIC + score (see ``QuestionResult.to_dict`` — it flattens the
# MAGIC ``EvalQuestion``). So ``expected_value`` (tier1), ``source_id`` /
# MAGIC ``expected_claim`` (tier2), and ``reference_answer`` (tier3) live
# MAGIC only in the TSVs. Re-load them here and join by question text so
# MAGIC the triage cells can show expected vs. actual side by side.

# COMMAND ----------
from nhtsa_curator.evaluation import DEFAULT_EVAL_FILES, load_eval_set

eval_dir = Path("./eval")
assert eval_dir.exists(), f"eval dir missing: {eval_dir.resolve()}"

# Build per-tier lookup: question_text -> EvalQuestion.
tsv_lookup: dict[str, dict[str, object]] = {}
for tier, fname in DEFAULT_EVAL_FILES.items():
    questions = load_eval_set(tier, eval_dir / fname)
    tsv_lookup[tier] = {q.question: q for q in questions}
    print(f"{tier}: loaded {len(questions)} TSV rows")


def _enrich(tier: str, row: dict) -> dict:
    """Merge TSV ground-truth fields into one MLflow row, keyed by question."""
    text = row.get("question") if isinstance(row.get("question"), str) else ""
    q_obj = tsv_lookup.get(tier, {}).get(text)
    if q_obj is None:
        # No TSV match — leave the row alone but flag it.
        row["_tsv_match"] = False
        return row
    row["_tsv_match"] = True
    row["expected_value"] = getattr(q_obj, "expected_value", None)
    row["source_id"] = getattr(q_obj, "source_id", None)
    row["expected_claim"] = getattr(q_obj, "expected_claim", None)
    row["reference_answer"] = getattr(q_obj, "reference_answer", None)
    row["is_pending"] = getattr(q_obj, "is_pending", False)
    return row


for tier in per_tier_results:
    per_tier_results[tier] = [_enrich(tier, r) for r in per_tier_results[tier]]
    n_unmatched = sum(1 for r in per_tier_results[tier] if not r.get("_tsv_match"))
    if n_unmatched:
        print(f"  WARN {tier}: {n_unmatched} rows had no matching TSV question")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Tier-1 triage — why scored rows miss the TSV ground truth
# MAGIC For every non-pending tier-1 row, print (expected, actual, score). The
# MAGIC expected value is from the TSV; the actual is what the agent answered.
# MAGIC
# MAGIC **What to look for:**
# MAGIC - **Pattern 1 — scalar mismatch**: agent returns a different number
# MAGIC   than the reference SQL (different date column or dedup in Genie's
# MAGIC   generated SQL). Fix: update the TSV expected value to the agent's
# MAGIC   consistent answer.
# MAGIC - **Pattern 2 — list wording mismatch**: agent shortens
# MAGIC   `"General Motors"` → `"GM"` so the substring check fails. Fix:
# MAGIC   shorten the expected tokens to what the agent actually emits.
# MAGIC - **Pattern 3 — agent apology / no data**: agent couldn't answer.
# MAGIC   Fix: skip or swap the question.

# COMMAND ----------
t1_rows = per_tier_results.get("tier1", [])
t1_scored = [r for r in t1_rows if not r.get("pending")]
t1_passed = [r for r in t1_scored if r.get("passed")]
print(f"tier1: {len(t1_rows)} total, {len(t1_scored)} scored, {len(t1_passed)} passed\n")

for r in t1_scored[:max_tier1_scored]:
    print(f"Q: {r.get('question')}")
    print(f"  expected: {r.get('expected_value')!r}")
    answer = (r.get("answer") or "").replace("\n", " ")
    print(f"  answer:   {answer[:400]!r}")
    print(f"  score:    {r.get('score')}  passed={r.get('passed')}")
    print(f"  reason:   {r.get('metric_breakdown')}")
    print("---")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Tier-2 triage — citation + grounding breakdown
# MAGIC For each failing tier-2 row, print:
# MAGIC   - `cite_ok`: did the normalised source_id appear in the answer?
# MAGIC   - `judge_score`: 1-5 on how well the agent's claim matches the
# MAGIC     expected claim (only scored when `cite_ok` is true).
# MAGIC
# MAGIC **Buckets:**
# MAGIC - **A**: `cite_ok=False` but answer contains the id in a different
# MAGIC   form (prefix mismatch, e.g. `TSB-10160095` vs `10160095`). Fix:
# MAGIC   drop the prefix in the TSV's `source_id` column.
# MAGIC - **B**: `cite_ok=False` and the id is nowhere in the answer. Fix:
# MAGIC   swap for a real id that exists in the silver tables.
# MAGIC - **C**: `cite_ok=True` but `judge_score < 4`. Fix: loosen the
# MAGIC   expected_claim wording or accept as a rubric-calibration gap.
# MAGIC - **D**: `cite_ok=True, judge_score is None`. Transient — judge
# MAGIC   returned invalid JSON. Ignore.

# COMMAND ----------
t2_rows = per_tier_results.get("tier2", [])
t2_failed = [r for r in t2_rows if not r.get("passed")]
print(f"tier2: {len(t2_rows)} total, {len(t2_failed)} failed\n")

bucket_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "other": 0}

for r in t2_failed[:max_tier2_failures]:
    bd = r.get("metric_breakdown", {}) or {}
    answer = (r.get("answer") or "").replace("\n", " ")

    cite_ok = bd.get("cite_ok")
    judge_score = bd.get("judge_score")
    source_id = r.get("source_id") or ""

    # Bucket classification.
    norm_id_no_prefix = "".join(c for c in source_id.lower() if c.isalnum())
    id_alphanum = norm_id_no_prefix
    id_numeric = "".join(c for c in source_id if c.isdigit())
    answer_alphanum = "".join(c for c in answer.lower() if c.isalnum())

    if cite_ok is False:
        if id_numeric and id_numeric in answer_alphanum:
            bucket = "A"
        else:
            bucket = "B"
    elif cite_ok is True and (judge_score is None):
        bucket = "D"
    elif cite_ok is True and isinstance(judge_score, (int, float)) and judge_score < 4:
        bucket = "C"
    else:
        bucket = "other"
    bucket_counts[bucket] += 1

    print(f"[bucket {bucket}] Q: {r.get('question')}")
    print(f"  expected source_id: {source_id}")
    print(f"  expected claim:     {(r.get('expected_claim') or '')[:140]}")
    print(f"  cite_ok:            {cite_ok}")
    print(f"  judge_score:        {judge_score}")
    print(f"  reason:             {bd.get('reason')}")
    print(f"  answer (first 400): {answer[:400]}")
    print("---")

print(f"\nbucket counts (top {max_tier2_failures} failures): {bucket_counts}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Tier-3 spot-check — two worst-scoring synthesis answers
# MAGIC Judge rubric averages 5 sub-scores (1-5). Below ~3.5 means the
# MAGIC synthesis is weak. Read the answer + judge notes to decide whether
# MAGIC it's a retrieval gap (no relevant chunks), a synthesis gap (chunks
# MAGIC present but answer is vague), or a rubric gap (the question itself
# MAGIC is underspecified).

# COMMAND ----------
t3_rows = per_tier_results.get("tier3", [])
t3_sorted = sorted(t3_rows, key=lambda r: r.get("score") or 0.0)
print(f"tier3: {len(t3_rows)} rows; printing 2 worst\n")

for r in t3_sorted[:2]:
    bd = r.get("metric_breakdown", {}) or {}
    answer = (r.get("answer") or "").replace("\n", " ")
    print(f"Q: {r.get('question')}")
    print(f"  score:    {r.get('score')}  passed={r.get('passed')}")
    print(f"  breakdown:{bd.get('breakdown')}")
    print(f"  notes:    {bd.get('notes')}")
    print(f"  answer (first 500): {answer[:500]}")
    print("---")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Phase 5 closure summary
# MAGIC Prints the 3 artefacts you need for the Phase 5 demo checkpoint:
# MAGIC   1. Experiment URL.
# MAGIC   2. Per-tier run URLs.
# MAGIC   3. Compact metrics JSON.
# MAGIC Screenshot / paste this cell's output into the demo deck.

# COMMAND ----------
print("=" * 70)
print("PHASE 5 — EVAL CLOSURE SUMMARY")
print("=" * 70)
print(f"\nExperiment: {host}/ml/experiments/{exp.experiment_id}")
print("\nPer-tier run URLs:")
for tier, run in runs_by_tier.items():
    url = f"{host}/ml/experiments/{exp.experiment_id}/runs/{run.info.run_id}"
    print(f"  {tier}: {url}")

print("\nAggregate metrics:")
print(json.dumps(agg_snapshot, indent=2))

print("\n" + "=" * 70)
