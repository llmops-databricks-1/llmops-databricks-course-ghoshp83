# Databricks notebook source
# MAGIC %md
# MAGIC # 4.2 — Lakebase session-memory setup
# MAGIC
# MAGIC Creates the two tables the agent needs for multi-turn memory:
# MAGIC `agent_sessions` + `agent_turns`. Idempotent — re-running is safe.
# MAGIC
# MAGIC **When to run this notebook:**
# MAGIC - Once, after Lakebase project is provisioned (one-time per env).
# MAGIC - Re-run only if `DDL_STATEMENTS` in `memory.py` changes (rare).
# MAGIC
# MAGIC **What this notebook does:**
# MAGIC 1. Creates the Lakebase Postgres *project* if it doesn't already
# MAGIC    exist (idempotent — same pattern as the reference project's
# MAGIC    notebook 3.3). Needs **no admin rights** — the notebook user
# MAGIC    owns the project.
# MAGIC 2. Creates the `agent_sessions` + `agent_turns` tables on that
# MAGIC    project via `PostgresSessionStore.init_schema()`.
# MAGIC 3. Smoke-tests a write + read round-trip.
# MAGIC
# MAGIC **What this notebook does NOT do:**
# MAGIC - Seed any rows. The agent writes sessions on demand.
# MAGIC - Set row-level security. Not relevant while the agent is single-tenant;
# MAGIC   revisit if the agent serves multiple end-users.
# MAGIC
# MAGIC **Env vars — all optional:**
# MAGIC - `LAKEBASE_DB` — database name (defaults to `databricks_postgres`).
# MAGIC - SPN auth (`LAKEBASE_SP_CLIENT_ID` / `_SECRET` / `_HOST`) — only
# MAGIC   needed when running as a service principal. In a notebook, the
# MAGIC   interactive user auth is used automatically.

# COMMAND ----------
import os

import psycopg
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.postgres import (
    PostgresAPI,
    Project,
    ProjectDefaultEndpointSettings,
    ProjectSpec,
)
from google.protobuf.duration_pb2 import Duration
from pyspark.sql import SparkSession

from nhtsa_curator.config import get_env, load_config
from nhtsa_curator.memory import (
    DDL_STATEMENTS,
    PostgresSessionStore,
    Turn,
)

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()

dbutils.widgets.text("env", "dev")
dbutils.widgets.text("run_id", "manual")

env = get_env(spark)
cfg = load_config("../project_config.yml", env)

assert cfg.lakebase_project_id, (
    "lakebase_project_id must be set in project_config.yml before running "
    "this notebook."
)

print(f"env={env} | lakebase_project_id={cfg.lakebase_project_id}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Auto-provision the Lakebase project
# MAGIC `get_project` first, fall back to `create_project`. This is the
# MAGIC same idempotent pattern as the reference project's notebook 3.3 —
# MAGIC re-running this cell is safe.
# MAGIC
# MAGIC Autoscale 1–4 CU, suspend after 300s idle (budget guardrail).

# COMMAND ----------
ws = WorkspaceClient()
pg_api = PostgresAPI(ws.api_client)
run_id = dbutils.widgets.get("run_id")

try:
    project = pg_api.get_project(name=f"projects/{cfg.lakebase_project_id}")
    print(f"Lakebase project already exists: {project.name}")
except Exception:
    # Note: older SDKs accepted `budget_policy_id` on ProjectSpec; the
    # current SDK schema dropped it. Budget-policy enforcement now happens
    # at the workspace level, not per-project.
    project = pg_api.create_project(
        project_id=cfg.lakebase_project_id,
        project=Project(
            spec=ProjectSpec(
                display_name=cfg.lakebase_project_id,
                default_endpoint_settings=ProjectDefaultEndpointSettings(
                    autoscaling_limit_min_cu=1,
                    autoscaling_limit_max_cu=4,
                    suspend_timeout_duration=Duration(seconds=300),
                ),
            ),
        ),
    ).wait()
    print(f"Lakebase project created: {project.name}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Connection factory
# MAGIC Host + credential are fetched dynamically from the project's
# MAGIC endpoint on every call, so there is **no LAKEBASE_HOST env var
# MAGIC requirement**. Username = interactive user in a notebook; falls
# MAGIC back to the SPN client_id when the SP env vars are present.
# MAGIC
# MAGIC We hand a factory (not an open connection) to `PostgresSessionStore`
# MAGIC so token rotation is transparent to the agent.

# COMMAND ----------
def _conn_factory() -> psycopg.Connection:
    sp_client_id = os.environ.get("LAKEBASE_SP_CLIENT_ID")
    sp_client_secret = os.environ.get("LAKEBASE_SP_CLIENT_SECRET")
    sp_host = os.environ.get("LAKEBASE_SP_HOST")

    if sp_client_id and sp_client_secret and sp_host:
        w = WorkspaceClient(
            host=sp_host, client_id=sp_client_id, client_secret=sp_client_secret,
        )
        username = sp_client_id
    else:
        w = WorkspaceClient()
        # Raw username — psycopg takes `user=` as a kwarg, not a URI
        # component, so no URL-encoding. If we URL-encoded, Postgres would
        # see the literal "%40" and reject with "password authentication
        # failed for user 'pralay.ghosh%40gmail.com'".
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

# COMMAND ----------
# MAGIC %md
# MAGIC ## Emit + inspect DDL
# MAGIC Printed first so a reviewer can diff expectations against what actually
# MAGIC runs. `init_schema` is idempotent (`CREATE TABLE IF NOT EXISTS`).

# COMMAND ----------
for i, stmt in enumerate(DDL_STATEMENTS, 1):
    print(f"-- statement {i}")
    print(stmt.strip())
    print()

# COMMAND ----------
store = PostgresSessionStore(_conn_factory)
store.init_schema()
print("Schema initialised.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Smoke test — write + read a round-trip
# MAGIC Creates a throwaway session, appends a turn, reads it back, then
# MAGIC deletes. Confirms the schema + connection + JSON handling all work
# MAGIC end to end before the agent ever touches the store.

# COMMAND ----------
sid = store.create_session(user_id=f"smoke-4.2-{run_id}")
store.append_turn(
    Turn(
        session_id=sid,
        turn_idx=0,
        role="user",
        content="Smoke test — please ignore.",
    )
)
store.append_turn(
    Turn(
        session_id=sid,
        turn_idx=1,
        role="assistant",
        content="Acknowledged.",
        tool_calls=[{"name": "noop", "args": {}}],
    )
)
store.set_accumulated_filters(sid, {"make_norm": "Tesla", "model_year": 2023})

hist = store.get_history(sid)
filters = store.get_accumulated_filters(sid)

assert len(hist) == 2, f"expected 2 turns, got {len(hist)}"
assert hist[0].role == "user"
assert hist[1].tool_calls == [{"name": "noop", "args": {}}]
assert filters == {"make_norm": "Tesla", "model_year": 2023}
print(f"Round-trip OK. sid={sid}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Clean up the smoke session
# MAGIC We don't want the smoke-test session polluting ops dashboards.
# MAGIC Deleting the session row cascades to `agent_turns` via the FK.

# COMMAND ----------
def _cleanup(sid: str) -> None:
    conn = _conn_factory()
    try:
        with conn, conn.cursor() as cur:
            cur.execute("DELETE FROM agent_sessions WHERE session_id = %s", (sid,))
    finally:
        conn.close()


_cleanup(sid)
print(f"Deleted smoke session {sid}.")

# COMMAND ----------
# MAGIC %md
# MAGIC ### Phase 4 Lakebase checkpoint
# MAGIC If all cells above completed, the agent is cleared to hit this
# MAGIC Lakebase with `PostgresSessionStore`. Set `use_lakebase=true` in
# MAGIC notebook 4.1 to exercise it.
