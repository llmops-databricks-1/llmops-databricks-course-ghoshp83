# Databricks notebook source
# MAGIC %md
# MAGIC # 2.6 — investigation PDF parsing (`ai_parse_document`)
# MAGIC
# MAGIC Drains the per-investigation PDF backlog by calling
# MAGIC `ai_parse_document` on docs in `bronze_investigation_documents`
# MAGIC that aren't yet in `silver_investigation_parsed_tracker`.
# MAGIC
# MAGIC The bronze investigation PDF table is populated by a separate
# MAGIC scraper (Phase 2 follow-up) — the table may be empty on first
# MAGIC run and that's expected.

# COMMAND ----------
from pyspark.sql import SparkSession

from nhtsa_curator.config import get_env, load_config
from nhtsa_curator.parsing import parse_investigation_documents

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()

dbutils.widgets.text("env", "dev")
dbutils.widgets.text("run_id", "manual")
dbutils.widgets.text("max_docs_per_run", "100")
env = get_env(spark)
max_docs = int(dbutils.widgets.get("max_docs_per_run"))

cfg = load_config("../project_config.yml", env)

# COMMAND ----------
docs_table = f"{cfg.full_schema_name}.bronze_investigation_documents"
if not spark.catalog.tableExists(docs_table):
    print(
        f"{docs_table} not present — investigation PDF scraper hasn't run yet.\n"
        "Skipping ai_parse_document. (This is expected on first deployment.)"
    )
else:
    result = parse_investigation_documents(spark, cfg, max_docs_per_run=max_docs)
    print(result)

# COMMAND ----------
parsed_table = f"{cfg.full_schema_name}.silver_investigation_parsed"
if spark.catalog.tableExists(parsed_table):
    display(spark.sql(f"SELECT count(*) AS parsed_docs FROM {parsed_table}"))
