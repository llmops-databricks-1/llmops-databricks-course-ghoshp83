# Databricks notebook source
# MAGIC %md
# MAGIC # 1.5 — SGO AV crash ingestion
# MAGIC
# MAGIC Pulls the latest monthly SGO CSV snapshot and MERGE-writes into
# MAGIC `bronze_sgo_crashes`. SGO snapshots are cumulative; primary key
# MAGIC defaults to `Report ID` (column header from NHTSA).

# COMMAND ----------
from loguru import logger
from pyspark.sql import SparkSession

from nhtsa_curator.bronze import write_sgo_bronze
from nhtsa_curator.config import get_env, load_config, load_sources
from nhtsa_curator.io.http import open_client
from nhtsa_curator.io.sgo import fetch_sgo_csv

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
    rows = fetch_sgo_csv(client, sources.sgo_av_csv)
    n = write_sgo_bronze(
        spark=spark,
        cfg=cfg,
        rows=rows,
        ingest_run_id=run_id,
        source_url=",".join(sources.sgo_av_csv),
    )

logger.info(f"SGO ingestion complete; rows merged in this run: {n:,}")

# COMMAND ----------
display(
    spark.sql(f"""
        SELECT count(*) AS rows
        FROM {cfg.full_schema_name}.bronze_sgo_crashes
    """)
)
