# Databricks notebook source
# MAGIC %md
# MAGIC # 2.3 — silver_investigations
# MAGIC
# MAGIC Reads `bronze_investigations`, types it, and adds a computed
# MAGIC `days_open` column (close_date - open_date, or current_date -
# MAGIC open_date for cases still open) plus a derived `status` column.

# COMMAND ----------
from pyspark.sql import SparkSession

from nhtsa_curator.config import get_env, load_config
from nhtsa_curator.silver import write_silver_investigations

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()

dbutils.widgets.text("env", "dev")
dbutils.widgets.text("run_id", "manual")
env = get_env(spark)

cfg = load_config("../project_config.yml", env)

# COMMAND ----------
n = write_silver_investigations(
    spark=spark,
    cfg=cfg,
    taxonomy_path="../src/nhtsa_curator/ref/oem_groups.yml",
)

# COMMAND ----------
display(spark.sql(f"""
    SELECT
        investigation_type,
        status,
        count(*)              AS cases,
        avg(days_open)        AS avg_days_open,
        max(days_open)        AS max_days_open
    FROM {cfg.full_schema_name}.silver_investigations
    GROUP BY investigation_type, status
    ORDER BY investigation_type, status
"""))
