# Databricks notebook source
# MAGIC %md
# MAGIC # 1.2 — Complaints (VOQ) bulk ingestion
# MAGIC
# MAGIC Pulls the NHTSA `FLAT_CMPL.zip` bulk dump and MERGE-writes new
# MAGIC rows into `bronze_complaints`. Idempotent on `cmplid`.
# MAGIC
# MAGIC PII (VINs, names, phone, dealer info) is intentionally **kept** in
# MAGIC bronze. Silver (Phase 2) is responsible for scrubbing.

# COMMAND ----------
from loguru import logger
from pyspark.sql import SparkSession

from nhtsa_curator.bronze import write_complaints_bronze
from nhtsa_curator.config import get_env, load_config, load_sources
from nhtsa_curator.io.complaints import fetch_complaints_bulk
from nhtsa_curator.io.http import open_client

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
    rows = fetch_complaints_bulk(client, sources.complaints_bulk_csv)
    n = write_complaints_bronze(
        spark=spark,
        cfg=cfg,
        rows=rows,
        ingest_run_id=run_id,
        source_url=sources.complaints_bulk_csv,
    )

logger.info(f"Complaints ingestion complete; rows merged in this run: {n:,}")

# COMMAND ----------
display(
    spark.sql(f"""
        SELECT
            count(*)                          AS row_count,
            count(DISTINCT odino)             AS unique_odi_numbers,
            sum(CASE WHEN crash = 'Y' THEN 1 ELSE 0 END) AS crash_count,
            sum(CASE WHEN fire  = 'Y' THEN 1 ELSE 0 END) AS fire_count
        FROM {cfg.full_schema_name}.bronze_complaints
    """)
)
