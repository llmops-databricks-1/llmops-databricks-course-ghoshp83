# Databricks notebook source
# MAGIC %md
# MAGIC # 1.3 — Investigations (PE / EA / RQ / DP) ingestion
# MAGIC
# MAGIC Pulls the NHTSA `FLAT_INV.zip` bulk dump and MERGE-writes new
# MAGIC rows into `bronze_investigations`. Idempotent on
# MAGIC `nhtsa_action_number`.
# MAGIC
# MAGIC Per-case PDF downloading is handled in Phase 2 (parsing) where it
# MAGIC sits naturally next to `ai_parse_document`.

# COMMAND ----------
from loguru import logger
from pyspark.sql import SparkSession

from nhtsa_curator.bronze import write_investigations_bronze
from nhtsa_curator.config import get_env, load_config, load_sources
from nhtsa_curator.io.http import open_client
from nhtsa_curator.io.investigations import fetch_investigations_bulk

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()

dbutils.widgets.text("env", "dev")
dbutils.widgets.text("run_id", "manual")
env = get_env(spark)
run_id = dbutils.widgets.get("run_id")

cfg = load_config("../project_config.yml", env)
sources = load_sources("../project_config.yml")

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {cfg.full_schema_name}")

# COMMAND ----------
with open_client() as client:
    rows = fetch_investigations_bulk(client, sources.investigations_bulk_csv)
    n = write_investigations_bronze(
        spark=spark,
        cfg=cfg,
        rows=rows,
        ingest_run_id=run_id,
        source_url=sources.investigations_bulk_csv,
    )

logger.info(f"Investigations ingestion complete; rows merged in this run: {n:,}")

# COMMAND ----------
display(
    spark.sql(f"""
        SELECT
            substring(nhtsa_action_number, 1, 2) AS investigation_type,
            count(*) AS n
        FROM {cfg.full_schema_name}.bronze_investigations
        GROUP BY substring(nhtsa_action_number, 1, 2)
        ORDER BY n DESC
    """)
)
