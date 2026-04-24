# Databricks notebook source
# MAGIC %md
# MAGIC # 4.3 — Grant `dev_SPN` a Postgres role on the Lakebase project
# MAGIC
# MAGIC One-time setup. Without this, the serving endpoint fails every
# MAGIC Lakebase call with `psycopg.OperationalError: password
# MAGIC authentication failed for user '<sp-uuid>'` — the workspace
# MAGIC issues the SPN a token, but Postgres has no role bound to the
# MAGIC SPN's client_id so it rejects the credential.
# MAGIC
# MAGIC **Why not MLflow's `DatabricksLakebase` resource auto-grant?**
# MAGIC That class targets the older **Database Instance** API. Notebook
# MAGIC 4.2 creates the Lakebase as a **Project** (newer `PostgresAPI`),
# MAGIC so the auto-grant lookup returns `NOT_FOUND: Database instance`
# MAGIC at deploy time. Manual `pg_api.create_role(...)` is the correct
# MAGIC path for Projects.
# MAGIC
# MAGIC **Scope of what this notebook grants** (intentionally narrow):
# MAGIC - Postgres role on the `nhtsa-agent-lakebase-pg` Project's
# MAGIC   `production` branch ONLY. No other Lakebase project affected.
# MAGIC - `USAGE` on `public` schema of the project's `databricks_postgres`
# MAGIC   database.
# MAGIC - `SELECT, INSERT, UPDATE, DELETE` on `agent_sessions` + `agent_turns`
# MAGIC   only. No other tables.
# MAGIC - **Does NOT** grant any UC / workspace / warehouse / endpoint
# MAGIC   permissions. All other agent resources go through the endpoint's
# MAGIC   built-in system SP via MLflow's `resources` list in `log_register_agent`.
# MAGIC
# MAGIC **Re-run safety**: every step is idempotent — safe to re-run at
# MAGIC any time.
# MAGIC
# MAGIC **When to re-run**:
# MAGIC - First-time setup (required before the endpoint can write sessions).
# MAGIC - If `dev_SPN` client_id ever rotates.
# MAGIC - NOT needed on every `register_deploy_agent` run — the role is
# MAGIC   stable across endpoint rebuilds because it's bound to the SPN,
# MAGIC   not to the endpoint.

# COMMAND ----------
import urllib.parse

import psycopg
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import BadRequest
from databricks.sdk.service.postgres import (
    PostgresAPI,
    Role,
    RoleAuthMethod,
    RoleIdentityType,
    RoleRoleSpec,
)

from nhtsa_curator.config import get_env, load_config

# COMMAND ----------
dbutils.widgets.text("env", "dev")
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

assert cfg.lakebase_project_id, (
    "lakebase_project_id must be set in project_config.yml before running this notebook."
)

SCOPE = "dev_SPN"
spn_client_id = dbutils.secrets.get(SCOPE, "client_id")

print(f"env={env}")
print(f"lakebase_project_id={cfg.lakebase_project_id}")
print(f"dev_SPN client_id (first 8): {spn_client_id[:8]}...")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1 — Resolve the Lakebase project + default branch
# MAGIC We scope every grant to this single project's default branch so
# MAGIC nothing touches unrelated Lakebases.

# COMMAND ----------
w = WorkspaceClient()
pg_api = PostgresAPI(w.api_client)

project = pg_api.get_project(name=f"projects/{cfg.lakebase_project_id}")
default_branch = next(iter(pg_api.list_branches(parent=project.name)))
branch_parent = default_branch.name

print(f"Project : {project.name}")
print(f"Branch  : {branch_parent}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2 — Ensure the SPN has a Postgres role on the branch
# MAGIC Idempotent — if the role already exists, we catch and continue.

# COMMAND ----------
# Best-effort pre-check via list_roles; if that misses a role in a non-
# ACTIVE state (seen during re-runs of partial setups), the create_role
# call will fail with BadRequest "already exists" and we treat it as
# success in the except branch below.
existing_roles = {
    r.spec.postgres_role
    for r in pg_api.list_roles(parent=branch_parent)
    if r.spec and r.spec.postgres_role
}

if spn_client_id in existing_roles:
    print(f"✓ Role for {spn_client_id[:8]}... already exists — skipping create_role.")
else:
    print(f"Creating Postgres role for SPN {spn_client_id[:8]}...")
    try:
        pg_api.create_role(
            parent=branch_parent,
            role=Role(
                spec=RoleRoleSpec(
                    identity_type=RoleIdentityType.SERVICE_PRINCIPAL,
                    auth_method=RoleAuthMethod.LAKEBASE_OAUTH_V1,
                    postgres_role=spn_client_id,
                )
            ),
            role_id="nhtsa-agent-spn",
        ).wait()
        print("✓ Role created.")
    except BadRequest as e:
        if "already exists" in str(e).lower():
            print("✓ Role already exists (from a prior partial run) — continuing.")
        else:
            raise

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3 — Grant table-level privileges to the SPN role
# MAGIC Connects as the **interactive user** (you) — your user has owner
# MAGIC role on the project from 4.2, so it can issue GRANTs. The SPN
# MAGIC still cannot read/write until these GRANTs run.
# MAGIC
# MAGIC All GRANTs are idempotent — Postgres silently accepts re-grants
# MAGIC of already-granted privileges.

# COMMAND ----------
endpoint = next(iter(pg_api.list_endpoints(parent=branch_parent)))
host = endpoint.status.hosts.host
pg_credential = pg_api.generate_database_credential(endpoint=endpoint.name)

user = w.current_user.me()
username = urllib.parse.quote_plus(user.user_name)

conn_string = (
    f"postgresql://{username}:{pg_credential.token}@{host}:5432/"
    "databricks_postgres?sslmode=require"
)

# COMMAND ----------
# Tables the agent reads/writes. Kept explicit so the grant blast radius
# is auditable from this notebook alone.
AGENT_TABLES = ("agent_sessions", "agent_turns")

with psycopg.connect(conn_string) as conn:
    with conn.cursor() as cur:
        cur.execute(f'GRANT USAGE ON SCHEMA public TO "{spn_client_id}";')
        for tbl in AGENT_TABLES:
            cur.execute(
                f"GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE {tbl} "
                f'TO "{spn_client_id}";'
            )
    conn.commit()

print("✓ Table grants applied to:")
for tbl in AGENT_TABLES:
    print(f"   - public.{tbl}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 4 — Verify (smoke test as the SPN)
# MAGIC Authenticates as the SPN and runs a trivial `SELECT` to prove the
# MAGIC grant chain works end-to-end before the serving pod tries it.

# COMMAND ----------
spn_ws = WorkspaceClient(
    host=w.config.host,
    client_id=spn_client_id,
    client_secret=dbutils.secrets.get(SCOPE, "client_secret"),
    auth_type="oauth-m2m",
)
spn_pg_api = PostgresAPI(spn_ws.api_client)
spn_credential = spn_pg_api.generate_database_credential(endpoint=endpoint.name)

spn_conn_string = (
    f"postgresql://{spn_client_id}:{spn_credential.token}@{host}:5432/"
    "databricks_postgres?sslmode=require"
)

with psycopg.connect(spn_conn_string) as conn, conn.cursor() as cur:
    cur.execute("SELECT COUNT(*) FROM agent_sessions;")
    (count,) = cur.fetchone()

print(f"✓ SPN auth + SELECT works. agent_sessions row count: {count}")
print(
    "   The serving endpoint will now succeed against Lakebase once "
    "LAKEBASE_SP_* env vars are injected by deploy_agent.py."
)
