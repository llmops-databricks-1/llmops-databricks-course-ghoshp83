# Databricks notebook source
# MAGIC %md
# MAGIC # 5.1 — Run the NHTSA agent eval harness
# MAGIC
# MAGIC Runs the 3-tier eval set (`tier1_deterministic`, `tier2_grounded`,
# MAGIC `tier3_synthesis`) against a live agent, logs per-question + aggregate
# MAGIC metrics to MLflow, and prints a summary.
# MAGIC
# MAGIC **Why this notebook exists rather than only the CLI:**
# MAGIC - Interactive iteration: change the system prompt, rerun one tier,
# MAGIC   compare runs in MLflow's UI.
# MAGIC - Visible traces: `@mlflow.trace` decorators produce one trace per
# MAGIC   eval question — browseable in the Traces tab of the experiment.
# MAGIC - Debuggability: every question's full tool trace + retrieved chunks
# MAGIC   land as MLflow artifacts under `per_question/<hash>.json`.
# MAGIC
# MAGIC **Phase 5 exit check (at the bottom of the notebook):**
# MAGIC - All 3 tiers produce an MLflow run with aggregate metrics present.
# MAGIC - No tier reports an `error_rate` > 0.2 (allowance for TBD tier-1 rows).
# MAGIC - The per-question artifact count matches the input size.

# COMMAND ----------
import json
import os
from pathlib import Path

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient
from pyspark.sql import SparkSession

from nhtsa_curator.agent import NhtsaAgent
from nhtsa_curator.config import (
    VectorSearchConfig,
    get_env,
    load_config,
    load_system_prompt,
)
from nhtsa_curator.evaluation import (
    DEFAULT_EVAL_FILES,
    VALID_TIERS,
    OpenAICompatJudge,
    load_eval_set,
    run_eval,
)
from nhtsa_curator.mcp import ToolContext, _normalise_statement_result
from nhtsa_curator.memory import InMemorySessionStore

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()

dbutils.widgets.text("env", "dev")
dbutils.widgets.text("run_id", "manual")
# Subset widget: "all" runs all 3 tiers; otherwise runs just the named tier.
# Useful when iterating on one tier without paying the cost of the others.
dbutils.widgets.dropdown("tier", "all", ["all", *VALID_TIERS])
dbutils.widgets.dropdown("use_judge", "true", ["true", "false"])
# Gold delta version: tag-only — used to pin tier-1 ground truth to a
# specific gold snapshot. Leave blank in ad-hoc runs.
dbutils.widgets.text("gold_delta_version", "")

env = get_env(spark)
cfg = load_config("../project_config.yml", env)
system_prompt = load_system_prompt("../project_config.yml")
tier_sel = dbutils.widgets.get("tier")
use_judge = dbutils.widgets.get("use_judge") == "true"
gold_delta_version = dbutils.widgets.get("gold_delta_version") or None

print(f"env={env} | catalog={cfg.catalog}.{cfg.db_schema}")
print(f"tier={tier_sel} | judge={use_judge}")
print(f"experiment={cfg.experiment_name}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Build live clients
# MAGIC Same wiring as notebook 4.1. Eval uses a fresh `InMemorySessionStore`
# MAGIC per question — a bad turn can't contaminate later questions.

# COMMAND ----------
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

# COMMAND ----------
# MAGIC %md
# MAGIC ## Run the eval
# MAGIC Each tier produces its own MLflow run. We run them in sequence rather
# MAGIC than parallel — Databricks serving endpoints have token rate-limits,
# MAGIC and linear execution gives cleaner traces to inspect afterwards.

# COMMAND ----------
mlflow.set_experiment(cfg.experiment_name)
mlflow.tracing.enable()  # critical — tool spans go to this experiment

# Eval TSVs live next to this notebook under ./eval/.
eval_dir = Path("./eval")
assert eval_dir.exists(), f"eval dir missing: {eval_dir.resolve()}"

tiers_to_run = VALID_TIERS if tier_sel == "all" else (tier_sel,)

summary: dict[str, dict] = {}
for tier in tiers_to_run:
    path = eval_dir / DEFAULT_EVAL_FILES[tier]
    questions = load_eval_set(tier, path)
    print(f"\n--- Running {tier}: {len(questions)} questions ---")
    result = run_eval(
        tier=tier,
        questions=questions,
        agent=agent,
        judge=judge,
        experiment_name=cfg.experiment_name,
        run_name=f"eval-{tier}-{dbutils.widgets.get('run_id')}",
        extra_tags={"env": env, "run_id": dbutils.widgets.get("run_id")},
        system_prompt=system_prompt,
        gold_delta_version=gold_delta_version,
    )
    summary[tier] = result["metrics"]
    print(f"mlflow_run_id={result['mlflow_run_id']}")
    print(json.dumps(result["metrics"], indent=2))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Phase 5 exit check

# COMMAND ----------
assert summary, "no tiers ran — check the tier widget"

issues: list[str] = []
for tier, m in summary.items():
    if m["n"] == 0:
        issues.append(f"{tier}: zero questions")
    if m.get("error_rate", 0) > 0.2:
        issues.append(f"{tier}: error_rate={m['error_rate']} > 0.2")
    # Tier-1 tolerates pending ground truth (TBD:* rows) — those aren't
    # counted in pass_rate and shouldn't trip the gate.
    if tier == "tier1" and m["n_scored"] == 0 and m["n_pending"] > 0:
        print(f"{tier}: all rows pending — skipping pass_rate check")

if issues:
    print("Issues detected:")
    for i in issues:
        print(f"  - {i}")
    raise AssertionError("Phase 5 eval exit check failed — see issues above")

print("\nPhase 5 eval checkpoint: PASS")
print(json.dumps(summary, indent=2))
