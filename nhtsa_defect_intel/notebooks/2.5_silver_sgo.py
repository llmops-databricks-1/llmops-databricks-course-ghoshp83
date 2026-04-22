# Databricks notebook source
# MAGIC %md
# MAGIC # 2.5 — silver_sgo_crashes
# MAGIC
# MAGIC Reads `bronze_sgo_crashes` and adds:
# MAGIC - `make_norm` + `oem_group` (via the OEM taxonomy)
# MAGIC - canonical `sae_level` ("L0".."L5") extracted from whatever phrasing
# MAGIC   the SGO file uses (e.g. "Level 4", "L4", "SAE 4")
# MAGIC - typed `incident_date_d`
# MAGIC
# MAGIC SGO column headers vary release-to-release; if a column we expect
# MAGIC isn't present, the silver writer will substitute `NULL` rather
# MAGIC than fail. Update the column names below if NHTSA renames them.

# COMMAND ----------
from pyspark.sql import SparkSession

from nhtsa_curator.config import get_env, load_config
from nhtsa_curator.silver import write_silver_sgo

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()

dbutils.widgets.text("env", "dev")
dbutils.widgets.text("run_id", "manual")
dbutils.widgets.text("key_column", "report_id")
dbutils.widgets.text("make_column", "make")
dbutils.widgets.text("sae_column", "sae_automation_level")
dbutils.widgets.text("incident_date_column", "incident_date")

env = get_env(spark)
cfg = load_config("../project_config.yml", env)

# COMMAND ----------
write_silver_sgo(
    spark=spark,
    cfg=cfg,
    taxonomy_path="../src/nhtsa_curator/ref/oem_groups.yml",
    key_column=dbutils.widgets.get("key_column"),
    make_column=dbutils.widgets.get("make_column"),
    sae_column=dbutils.widgets.get("sae_column"),
    incident_date_column=dbutils.widgets.get("incident_date_column"),
)

# COMMAND ----------
display(spark.sql(f"""
    SELECT sae_level, count(*) AS crashes
    FROM {cfg.full_schema_name}.silver_sgo_crashes
    GROUP BY sae_level
    ORDER BY sae_level
"""))
