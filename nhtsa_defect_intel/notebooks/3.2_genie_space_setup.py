# Databricks notebook source
# MAGIC %md
# MAGIC # 3.2 — Genie space setup + verification
# MAGIC
# MAGIC Genie spaces are **created interactively in the Databricks UI** —
# MAGIC there is no public DDL for "create a space". This notebook owns the
# MAGIC programmatic pieces around the space:
# MAGIC
# MAGIC 1. Apply table + column comments — Genie reads these directly when
# MAGIC    generating SQL, so curating them is the single biggest text-to-SQL
# MAGIC    quality lever.
# MAGIC 2. Verify every Genie-exposed table exists and is queryable.
# MAGIC 3. Print the canonical seed list of trusted queries for paste-into-UI.
# MAGIC 4. Print the manual UI recipe so a teammate can recreate the space
# MAGIC    in another workspace.

# COMMAND ----------
from pyspark.sql import SparkSession

from nhtsa_curator.config import get_env, load_config
from nhtsa_curator.genie import (
    GENIE_TABLES,
    apply_comments,
    default_trusted_queries,
    verify_genie_space,
)

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()

dbutils.widgets.text("env", "dev")
dbutils.widgets.text("run_id", "manual")
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Apply table + column comments (idempotent)

# COMMAND ----------
n_applied = apply_comments(spark, cfg)
print(f"Applied {n_applied} COMMENT statements.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Verify gold tables Genie depends on

# COMMAND ----------
result = verify_genie_space(spark, cfg)
print("space_id        :", result.space_id)
print("tables expected :", len(result.tables_expected))
print("tables present  :", len(result.tables_present))
print("tables missing  :", result.tables_missing or "—")
print("ok              :", result.ok)

if not result.ok:
    print(
        "\nNOT OK. Common causes:\n"
        " - genie_space_id still PLACEHOLDER in project_config.yml\n"
        "   → create the space in the UI (steps below), copy its id,\n"
        "     update project_config.yml, redeploy the bundle.\n"
        " - one or more gold tables missing → run notebook 2.7 first."
    )

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Trusted-query seeds (paste into Genie UI)

# COMMAND ----------
for tq in default_trusted_queries(cfg):
    print(f"--- {tq.name} ---")
    print(f"Q: {tq.question}")
    print(f"SQL:\n{tq.sql}\n")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Manual UI recipe (one-time per env)
# MAGIC
# MAGIC 1. **Genie** in the left nav → **New space**.
# MAGIC 2. Name: `NHTSA Defect Intelligence ({env})`.
# MAGIC 3. Pick the warehouse from `cfg.warehouse_id`
# MAGIC    (`{warehouse_id}` for this env).
# MAGIC 4. Add the following 7 tables (do **not** add silver or bronze):
# MAGIC    - `dim_vehicle`, `dim_component`, `dim_oem_group`, `dim_date`
# MAGIC    - `gold_recalls_fact`, `gold_complaints_fact`,
# MAGIC      `gold_investigations_fact`
# MAGIC 5. **Trusted assets** → paste each of the 5 queries from section 3
# MAGIC    above (name + question + SQL).
# MAGIC 6. Add general instructions (paste verbatim):
# MAGIC    > "Always cite source IDs (campaign_number, odi_number,
# MAGIC    > nhtsa_action_number) in answers. When a question mentions a
# MAGIC    > manufacturer (e.g. 'GM', 'Stellantis'), join via dim_oem_group
# MAGIC    > rather than dim_vehicle.make_norm."
# MAGIC 7. **Save** → copy the space id from the URL → paste into
# MAGIC    `project_config.yml` under the matching env's `genie_space_id`.
# MAGIC 8. Re-run this notebook to confirm `verify_genie_space().ok == True`.

# COMMAND ----------
print(f"warehouse_id for env={env}: {cfg.warehouse_id}")
print(f"tables to add: {', '.join(GENIE_TABLES)}")
