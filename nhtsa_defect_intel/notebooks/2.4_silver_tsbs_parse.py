# Databricks notebook source
# MAGIC %md
# MAGIC # 2.4 — silver_tsbs (index typing)
# MAGIC
# MAGIC NHTSA's May-2024 MfrComms rewrite removed the per-TSB PDF URLs and
# MAGIC now carries up to 4000 chars of bulletin text inline in the
# MAGIC `summary` field. There is no longer a second "parse PDFs" stage —
# MAGIC this notebook just types / normalises `bronze_tsb_index` into
# MAGIC `silver_tsbs`, and gold narrative chunks reads `silver_tsbs.summary`
# MAGIC directly (see `gold.py::write_gold_narrative_chunks`).
# MAGIC
# MAGIC Re-running is cheap: `write_silver_tsbs` is CREATE OR REPLACE.

# COMMAND ----------
from pyspark.sql import SparkSession

from nhtsa_curator.config import get_env, load_config
from nhtsa_curator.silver import write_silver_tsbs

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()

dbutils.widgets.text("env", "dev")
dbutils.widgets.text("run_id", "manual")
env = get_env(spark)

cfg = load_config("../project_config.yml", env)

# COMMAND ----------
write_silver_tsbs(spark, cfg, taxonomy_path="../src/nhtsa_curator/ref/oem_groups.yml")

# COMMAND ----------
display(
    spark.sql(f"""
    SELECT count(*) AS index_rows
    FROM {cfg.full_schema_name}.silver_tsbs
""")
)
