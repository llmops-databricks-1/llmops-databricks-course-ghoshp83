# Databricks notebook source
# MAGIC %md
# MAGIC # 1.4 — TSB / Manufacturer Communications ingestion
# MAGIC
# MAGIC NHTSA's May-2024 MfrComms rewrite:
# MAGIC - Replaced the single `FLAT_TSBS.zip` with per-5-year chunks
# MAGIC   (`TSBS_RECEIVED_YYYY-YYYY.zip`). The list lives in
# MAGIC   `project_config.yml → nhtsa_sources.tsbs_bulk_csv` (plural).
# MAGIC - Dropped the per-TSB `pdf_path` URL. The `Summary` field now
# MAGIC   carries up to 4000 chars of the bulletin text inline, so we no
# MAGIC   longer download per-bulletin PDFs. Silver / gold reads from
# MAGIC   `bronze_tsb_index.summary` directly.

# COMMAND ----------
from loguru import logger
from pyspark.sql import SparkSession

from nhtsa_curator.bronze import write_tsbs_bronze
from nhtsa_curator.config import get_env, load_config, load_sources
from nhtsa_curator.io.http import open_client
from nhtsa_curator.io.tsbs import fetch_tsbs_bulk

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
# Single stage — pull every TSB chunk URL and MERGE into bronze_tsb_index.

with open_client() as client:
    rows = fetch_tsbs_bulk(client, sources.tsbs_bulk_csv)
    n_idx = write_tsbs_bronze(
        spark=spark,
        cfg=cfg,
        rows=rows,
        ingest_run_id=run_id,
        source_url=",".join(sources.tsbs_bulk_csv),
    )

logger.info(f"TSB index merged: {n_idx:,} rows")

# COMMAND ----------
display(
    spark.sql(f"""
        SELECT count(*) AS index_rows
        FROM {cfg.full_schema_name}.bronze_tsb_index
    """)
)
