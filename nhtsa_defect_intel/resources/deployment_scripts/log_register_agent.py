# Databricks notebook source
# MAGIC %md
# MAGIC # Log + Register the NHTSA Defect Intelligence Agent
# MAGIC
# MAGIC Task in the `register_deploy_agent` workflow. Runs:
# MAGIC 1. Smoke eval against the live agent (feedback scorers + guidelines)
# MAGIC 2. `mlflow.pyfunc.log_model` on `../../nhtsa_agent_pg.py` with a full
# MAGIC    resources list so the serving principal is pre-authorised
# MAGIC 3. Register + `latest-model` alias bump in Unity Catalog
# MAGIC
# MAGIC The companion task `deploy_agent.py` picks the new version up via
# MAGIC the `latest-model` alias and rolls the serving endpoint.

# COMMAND ----------
import mlflow

from nhtsa_curator.agent import log_register_agent
from nhtsa_curator.config import load_config
from nhtsa_curator.evaluation import evaluate_agent
from nhtsa_curator.utils.common import (
    get_delta_table_version,
    get_widget,
    set_mlflow_tracking_uri,
)

from pyspark.sql import SparkSession

set_mlflow_tracking_uri()

env = get_widget("env", "dev")
git_sha = get_widget("git_sha", "local")
run_id = get_widget("run_id", "local")

cfg = load_config(config_path="../../project_config.yml", env=env)

mlflow.set_experiment(cfg.experiment_name)

model_name = f"{cfg.catalog}.{cfg.db_schema}.nhtsa_agent_pg"

# COMMAND ----------
# MAGIC %md
# MAGIC ## Pin the gold snapshot
# MAGIC Tag the MLflow run with the Delta version of each gold fact
# MAGIC table. Lets us reproduce a deployed agent's numbers later — if a
# MAGIC prod score regresses, we can diff against the version that
# MAGIC passed registration.

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()

gold_versions: dict[str, str] = {}
# `gold_tsb_meta` and `gold_sgo_av_crashes` transforms were never built
# (silver tables exist, gold transforms TBD). Not on the demo critical
# path — the agent answers TSB/SGO questions via the vector index +
# Genie space.
for tbl in (
    "gold_recalls_fact",
    "gold_complaints_fact",
    "gold_investig_fact",
    "gold_narrative_chunks",
):
    try:
        v = get_delta_table_version(
            spark, f"{cfg.catalog}.{cfg.db_schema}.{tbl}"
        )
        gold_versions[tbl] = v
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] gold table {tbl} version lookup failed: {exc}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Run pre-register smoke eval
# MAGIC Applies `cite_id_present`, `word_count_under`, `mentions_oem`, plus
# MAGIC the three Guidelines scorers to the questions in
# MAGIC `../../eval_inputs.txt`. Blocks registration on **nothing** — the
# MAGIC metrics are *informational*; the real promotion gate is tier-1/2/3
# MAGIC in Phase 5 (`eval_workflow.yml`). We still run it here so every
# MAGIC model version carries a score attached to it.

# COMMAND ----------
results = evaluate_agent(cfg, eval_inputs_path="../../eval_inputs.txt")

eval_metrics = dict(results.metrics) if hasattr(results, "metrics") else {}
print("smoke eval metrics:")
for k, v in eval_metrics.items():
    print(f"  {k}: {v}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Log + register

# COMMAND ----------
# Ship the ``nhtsa_curator`` source tree with the model so the serving
# container doesn't try to ``pip install nhtsa-curator==0.0.1`` (the
# local wheel isn't on any pip index). ``pip_requirements`` is pinned
# to the public deps from ``pyproject.toml`` — MLflow's auto-inferred
# set would include the local package and break container build.
# ``project_config.yml`` ships alongside the source tree so the
# serving container can resolve it via the ``__file__``-relative
# fallback in ``_resolve_config_path`` (CWD in the scoring pod is
# ``/``, which never finds the file otherwise).
agent_code_paths = ["../../src/nhtsa_curator", "../../project_config.yml"]
agent_pip_requirements = [
    "cffi==1.17.1",
    "cloudpickle==3.1.1",
    "numpy==2.4.0",
    "pandas==2.3.0",
    "pyarrow==22.0.0",
    "databricks-sdk==0.85.0",
    "pydantic==2.11.7",
    "loguru==0.7.3",
    "python-dotenv==1.1.1",
    "databricks-vectorsearch==0.63",
    "openai==2.8.0",
    "databricks-mcp==0.4.0",
    "backoff==2.2.1",
    "mlflow==3.8.1",
    "nest-asyncio==1.6.0",
    "databricks-agents==1.8.2",
    "psycopg==3.3.2",
    "psycopg-pool==3.3.0",
    "psycopg[binary]==3.3.2",
    "httpx==0.28.1",
    "tenacity==9.0.0",
]

registered_model = log_register_agent(
    cfg=cfg,
    git_sha=git_sha,
    run_id=run_id,
    agent_code_path="../../nhtsa_agent_pg.py",
    model_name=model_name,
    evaluation_metrics=eval_metrics,
    gold_delta_version=",".join(f"{k}={v}" for k, v in gold_versions.items()),
    experiment_name=cfg.experiment_name,
    code_paths=agent_code_paths,
    pip_requirements=agent_pip_requirements,
)

print(f"\n✓ Registered {model_name} version {registered_model.version}")
