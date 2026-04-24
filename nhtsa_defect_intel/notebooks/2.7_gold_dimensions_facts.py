# Databricks notebook source
# MAGIC %md
# MAGIC # 2.7 — gold dimensions + facts
# MAGIC
# MAGIC Builds the star-schema tables Genie will read:
# MAGIC
# MAGIC | layer | table                       |
# MAGIC |-------|-----------------------------|
# MAGIC | dim   | `dim_vehicle`               |
# MAGIC | dim   | `dim_component`             |
# MAGIC | dim   | `dim_oem_group`             |
# MAGIC | dim   | `dim_date`                  |
# MAGIC | fact  | `gold_recalls_fact`         |
# MAGIC | fact  | `gold_complaints_fact`      |
# MAGIC | fact  | `gold_investigations_fact`  |
# MAGIC
# MAGIC Surrogate keys are deterministic xxhash64 digests so re-runs
# MAGIC produce stable joins. Order matters: dims first, then facts.

# COMMAND ----------
from pyspark.sql import SparkSession

from nhtsa_curator.config import get_env, load_config
from nhtsa_curator.gold import (
    write_dim_component,
    write_dim_date,
    write_dim_oem_group,
    write_dim_vehicle,
    write_gold_complaints_fact,
    write_gold_investigations_fact,
    write_gold_recalls_fact,
)

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()

dbutils.widgets.text("env", "dev")
dbutils.widgets.text("run_id", "manual")
env = get_env(spark)

cfg = load_config("../project_config.yml", env)

# COMMAND ----------
# Dimensions (must come first so facts can join cleanly).
write_dim_vehicle(spark, cfg)
write_dim_component(spark, cfg)
write_dim_oem_group(spark, cfg)
write_dim_date(spark, cfg)

# COMMAND ----------
# Facts.
write_gold_recalls_fact(spark, cfg)
write_gold_complaints_fact(spark, cfg)
write_gold_investigations_fact(spark, cfg)

# COMMAND ----------
display(
    spark.sql(f"""
    SELECT 'dim_vehicle'      AS table, count(*) AS rows FROM {cfg.full_schema_name}.dim_vehicle      UNION ALL
    SELECT 'dim_component',          count(*)         FROM {cfg.full_schema_name}.dim_component      UNION ALL
    SELECT 'dim_oem_group',          count(*)         FROM {cfg.full_schema_name}.dim_oem_group      UNION ALL
    SELECT 'dim_date',               count(*)         FROM {cfg.full_schema_name}.dim_date           UNION ALL
    SELECT 'gold_recalls_fact',      count(*)         FROM {cfg.full_schema_name}.gold_recalls_fact      UNION ALL
    SELECT 'gold_complaints_fact',   count(*)         FROM {cfg.full_schema_name}.gold_complaints_fact   UNION ALL
    SELECT 'gold_investigations_fact', count(*)       FROM {cfg.full_schema_name}.gold_investigations_fact
""")
)
