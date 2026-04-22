# Databricks notebook source
# MAGIC %md
# MAGIC # 2.1 — silver_recalls
# MAGIC
# MAGIC Reads `bronze_recalls`, casts NHTSA's stringly-typed fields into
# MAGIC proper SQL types, normalises make → OEM group, and writes
# MAGIC `silver_recalls`.
# MAGIC
# MAGIC Idempotent (CREATE OR REPLACE) — silver is fully reproducible
# MAGIC from bronze, no MERGE needed.

# COMMAND ----------
from pyspark.sql import SparkSession

from nhtsa_curator.config import get_env, load_config
from nhtsa_curator.silver import write_silver_recalls

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()

dbutils.widgets.text("env", "dev")
dbutils.widgets.text("run_id", "manual")
env = get_env(spark)

cfg = load_config("../project_config.yml", env)

# COMMAND ----------
n = write_silver_recalls(
    spark=spark,
    cfg=cfg,
    taxonomy_path="../src/nhtsa_curator/ref/oem_groups.yml",
)

# COMMAND ----------
display(spark.sql(f"""
    SELECT
        count(*)                            AS rows,
        count(DISTINCT campaign_number)     AS campaigns,
        count(DISTINCT make_norm)           AS makes,
        sum(units_affected)                 AS total_units_affected,
        min(record_creation_date)           AS earliest,
        max(record_creation_date)           AS latest
    FROM {cfg.full_schema_name}.silver_recalls
"""))
