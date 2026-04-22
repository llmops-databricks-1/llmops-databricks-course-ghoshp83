# Databricks notebook source
# MAGIC %md
# MAGIC # 1.1 — Recalls bulk ingestion
# MAGIC
# MAGIC Pulls the NHTSA `FLAT_RCL.zip` bulk dump and MERGE-writes new
# MAGIC rows into `bronze_recalls`. Idempotent on `record_id`.
# MAGIC
# MAGIC See:
# MAGIC - `docs/02_data_sources.md` (recalls section)
# MAGIC - `docs/03_data_model.md` (`bronze_recalls` schema)

# COMMAND ----------
from loguru import logger
from pyspark.sql import SparkSession

from nhtsa_curator.bronze import write_recalls_bronze
from nhtsa_curator.config import get_env, load_config, load_sources
from nhtsa_curator.io.http import open_client
from nhtsa_curator.io.recalls import fetch_recalls_bulk

# COMMAND ----------
# Setup

spark = SparkSession.builder.getOrCreate()

dbutils.widgets.text("env", "dev")
dbutils.widgets.text("run_id", "manual")
env = get_env(spark)
run_id = dbutils.widgets.get("run_id")

cfg = load_config("../project_config.yml", env)
sources = load_sources("../project_config.yml")

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {cfg.full_schema_name}")
logger.info(f"Schema {cfg.full_schema_name} ready (env={env}, run_id={run_id})")

# COMMAND ----------
# Ingest

with open_client() as client:
    rows = fetch_recalls_bulk(client, sources.recalls_bulk_csv)
    n = write_recalls_bronze(
        spark=spark,
        cfg=cfg,
        rows=rows,
        ingest_run_id=run_id,
        source_url=sources.recalls_bulk_csv,
    )

logger.info(f"Recalls ingestion complete; rows merged in this run: {n:,}")

# COMMAND ----------
# Verify

display(
    spark.sql(f"""
        SELECT
            count(*)                              AS row_count,
            count(DISTINCT campno)                AS unique_campaigns,
            min(rcdate)                           AS earliest_record_date,
            max(rcdate)                           AS latest_record_date
        FROM {cfg.full_schema_name}.bronze_recalls
    """)
)
