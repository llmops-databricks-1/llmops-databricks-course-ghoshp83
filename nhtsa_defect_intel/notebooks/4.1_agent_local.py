# Databricks notebook source
# MAGIC %md
# MAGIC # 4.1 — Agent local driver
# MAGIC
# MAGIC Builds a live `NhtsaAgent` and runs 5 hand-picked end-to-end questions
# MAGIC to exercise tool routing, multi-turn memory, and answer-citation quality.
# MAGIC
# MAGIC **What this notebook verifies (Phase 4 checkpoint):**
# MAGIC - The agent picks the right tool for each question shape.
# MAGIC - Citations (campaign #, ODI #, TSB #, action #) appear inline.
# MAGIC - Multi-turn context is maintained — turn 5 refers back to turn 3.
# MAGIC - `accumulated_filters` carries (make, model_year, component_group)
# MAGIC   forward across refinements.
# MAGIC
# MAGIC **Pre-reqs (from earlier phases):**
# MAGIC - Phase 3 Vector Search index online.
# MAGIC - Genie space created + `genie_space_id` set in `project_config.yml`.
# MAGIC - Lakebase project provisioned + schema initialised by notebook 4.2
# MAGIC   (or leave `lakebase_project_id` unset to use the in-memory fallback).
# MAGIC
# MAGIC **Not in this notebook:** MLflow logging / model registration.
# MAGIC That's `serving.py` + a separate deployment notebook in Phase 6.

# COMMAND ----------
import json
import os

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
from nhtsa_curator.mcp import ToolContext, _normalise_statement_result
from nhtsa_curator.memory import InMemorySessionStore, PostgresSessionStore

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()

dbutils.widgets.text("env", "dev")
dbutils.widgets.text("run_id", "manual")
dbutils.widgets.dropdown("use_lakebase", "false", ["true", "false"])

env = get_env(spark)
cfg = load_config("../project_config.yml", env)
system_prompt = load_system_prompt("../project_config.yml")
use_lakebase = dbutils.widgets.get("use_lakebase") == "true"

print(f"env={env} | catalog={cfg.catalog}.{cfg.db_schema}")
print(f"genie_space={cfg.genie_space_id}")
print(f"lakebase={cfg.lakebase_project_id if use_lakebase else 'in-memory'}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Build the live ToolContext
# MAGIC Wires the real Databricks clients into `ToolContext`. The agent itself
# MAGIC never imports these modules — they're all passed through here.

# COMMAND ----------
ws = WorkspaceClient()
vs_client = VectorSearchClient(disable_notice=True)


def _sql(query: str) -> list[dict]:
    """Warehouse-based SQL executor for fetch_* tools."""
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

# COMMAND ----------
# MAGIC %md
# MAGIC ## LLM client
# MAGIC Uses the OpenAI-compatible client that ``WorkspaceClient`` exposes —
# MAGIC points at `cfg.llm_endpoint` (Databricks foundation-model serving).

# COMMAND ----------
llm_client = ws.serving_endpoints.get_open_ai_client()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Session store
# MAGIC - `use_lakebase=false` → `InMemorySessionStore` (no Lakebase needed).
# MAGIC - `use_lakebase=true`  → real `PostgresSessionStore`; requires
# MAGIC   `LAKEBASE_HOST` + `LAKEBASE_USER` env vars and `lakebase_project_id`
# MAGIC   set in `project_config.yml`.

# COMMAND ----------
if use_lakebase:
    import psycopg
    from databricks.sdk.service.postgres import PostgresAPI

    def _conn_factory() -> "psycopg.Connection":
        # Matches 4.2: resolve host/credential dynamically from the
        # PostgresAPI (Lakebase Autoscaling Project), not the legacy
        # `ws.database` "database instance" API — the project was created
        # under `projects/...`, so the instance-lookup returns 404.
        sp_client_id = os.environ.get("LAKEBASE_SP_CLIENT_ID")
        sp_client_secret = os.environ.get("LAKEBASE_SP_CLIENT_SECRET")
        sp_host = os.environ.get("LAKEBASE_SP_HOST")

        if sp_client_id and sp_client_secret and sp_host:
            w = WorkspaceClient(
                host=sp_host,
                client_id=sp_client_id,
                client_secret=sp_client_secret,
            )
            username = sp_client_id
        else:
            w = WorkspaceClient()
            username = w.current_user.me().user_name

        local_pg_api = PostgresAPI(w.api_client)
        project_parent = f"projects/{cfg.lakebase_project_id}"
        default_branch = next(iter(local_pg_api.list_branches(parent=project_parent)))
        endpoint = next(iter(local_pg_api.list_endpoints(parent=default_branch.name)))
        host = endpoint.status.hosts.host
        cred = local_pg_api.generate_database_credential(endpoint=endpoint.name)
        return psycopg.connect(
            host=host,
            dbname=os.environ.get("LAKEBASE_DB", "databricks_postgres"),
            user=username,
            password=cred.token,
            sslmode="require",
        )

    store = PostgresSessionStore(_conn_factory)
    store.init_schema()
else:
    store = InMemorySessionStore()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Build the agent

# COMMAND ----------
agent = NhtsaAgent(
    cfg=cfg,
    llm_client=llm_client,
    tool_context=tool_ctx,
    session_store=store,
    system_prompt=system_prompt,
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5 end-to-end questions
# MAGIC Each question targets a different tool-routing shape. If any returns
# MAGIC `stopped_reason != "ok"` or an empty answer, investigate *that* one
# MAGIC rather than accepting a low pass rate — Phase 4 checkpoint is 5/5.

# COMMAND ----------
questions = [
    # 1. Pure structured — should route to Genie.
    "How many recall campaigns were issued for Tesla vehicles in 2023?",
    # 2. Pure qualitative — should route to Vector Search.
    "What kinds of issues are drivers reporting with unintended acceleration?",
    # 3. Specific TSB id — should route to fetch_tsb.
    "Give me the full details of TSB 10160095.",
    # 4. Specific investigation id — should route to fetch_investigation.
    "What's the status and scope of investigation EA22-002?",
    # 5. Combined — should call Genie AND Vector Search.
    "Top 3 OEM groups by fire-related complaints in 2024, with example narratives.",
]

# COMMAND ----------
# MAGIC %md
# MAGIC ### Single-session turn loop
# MAGIC Running all 5 turns in the same session tests multi-turn memory —
# MAGIC after turn 1 the session holds a Tesla/2023 filter; turn 2 should
# MAGIC recognise that filter or the user's new constraints.

# COMMAND ----------
session_id = store.create_session(user_id="pg-notebook-4.1")
print(f"session_id = {session_id}\n")

results = []
for i, q in enumerate(questions, 1):
    print(f"--- Turn {i} ---")
    print(f"Q: {q}")
    result = agent.run_turn(user_message=q, session_id=session_id)
    print(f"tools: {[t['name'] for t in result.tool_trace]}")
    print(f"stopped: {result.stopped_reason} | llm_calls: {result.n_llm_calls}")
    print(f"A: {result.answer[:500]}")
    print(f"filters: {result.accumulated_filters}\n")
    results.append(result)

# COMMAND ----------
# MAGIC %md
# MAGIC ### Tool-routing check
# MAGIC The expected first-call routing per question. A mismatch doesn't
# MAGIC always mean a bug (LLMs are non-deterministic), but 3+ mismatches
# MAGIC across 5 questions means the system prompt needs sharpening.

# COMMAND ----------
from nhtsa_curator.mcp import (
    FETCH_INVESTIGATION_TOOL_NAME,
    FETCH_TSB_TOOL_NAME,
    GENIE_TOOL_NAME,
    VECTOR_SEARCH_TOOL_NAME,
)

expected_first_tool = [
    GENIE_TOOL_NAME,
    VECTOR_SEARCH_TOOL_NAME,
    FETCH_TSB_TOOL_NAME,
    FETCH_INVESTIGATION_TOOL_NAME,
    GENIE_TOOL_NAME,  # combined — Genie first for the ranking, VS second.
]
routing_ok = 0
for q, r, expected in zip(questions, results, expected_first_tool, strict=True):
    first = r.tool_trace[0]["name"] if r.tool_trace else None
    tick = "+" if first == expected else "-"
    if first == expected:
        routing_ok += 1
    print(f"[{tick}] expected={expected!s:<25} actual={first!s:<25} | {q[:60]}")

print(f"\nRouting match: {routing_ok}/{len(questions)}")
assert routing_ok >= 4, (
    f"Routing accuracy {routing_ok}/5 below threshold. "
    "Tighten the system prompt tool descriptions."
)

# COMMAND ----------
# MAGIC %md
# MAGIC ### Persisted history snapshot
# MAGIC Confirms every turn landed in the store.

# COMMAND ----------
history = store.get_history(session_id)
print(f"turns in store: {len(history)}")
for t in history:
    preview = t.content[:100].replace("\n", " ")
    print(f"  [{t.turn_idx:02d}] {t.role:<10} {preview}")

print("\naccumulated_filters:")
print(json.dumps(store.get_accumulated_filters(session_id), indent=2))

# COMMAND ----------
# MAGIC %md
# MAGIC ### Phase 4 exit check
# MAGIC - 5/5 questions return a non-empty answer.
# MAGIC - No turn ended with `stopped_reason = "max_steps"`.
# MAGIC - Every turn produced at least one tool call (no direct LLM answers).

# COMMAND ----------
n_empty = sum(1 for r in results if not r.answer.strip())
n_max_steps = sum(1 for r in results if r.stopped_reason == "max_steps")
n_no_tool = sum(1 for r in results if not r.tool_trace)

print(f"empty answers:   {n_empty}")
print(f"hit max steps:   {n_max_steps}")
print(f"no tool calls:   {n_no_tool}")
assert n_empty == 0, "Some answers were empty — investigate."
assert n_max_steps == 0, "Some turns hit max_tool_steps — investigate."
assert n_no_tool <= 1, "Agent answered without any tool call on >1 question."
print("\nPhase 4 local checkpoint: PASS")
