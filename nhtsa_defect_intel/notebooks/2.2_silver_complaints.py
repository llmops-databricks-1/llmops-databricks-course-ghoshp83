# Databricks notebook source
# MAGIC %md
# MAGIC # 2.2 — silver_complaints (with PII scrubbing)
# MAGIC
# MAGIC Reads `bronze_complaints`, types the rows, and runs the PII
# MAGIC scrubber over the narrative (`cdescr` → `narrative_clean`).
# MAGIC
# MAGIC PII scrubber removes:
# MAGIC - VINs (17-char alphanumeric)
# MAGIC - US phone numbers
# MAGIC - email addresses
# MAGIC - long bare digit runs
# MAGIC
# MAGIC Replacement tokens are stable strings (`[VIN]`, `[PHONE]`,
# MAGIC `[EMAIL]`, `[NUM]`) so downstream readers can spot scrubbed
# MAGIC fields.

# COMMAND ----------
from pyspark.sql import SparkSession

from nhtsa_curator.config import get_env, load_config
from nhtsa_curator.silver import write_silver_complaints

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()

dbutils.widgets.text("env", "dev")
dbutils.widgets.text("run_id", "manual")
env = get_env(spark)

cfg = load_config("../project_config.yml", env)

# COMMAND ----------
n = write_silver_complaints(
    spark=spark,
    cfg=cfg,
    taxonomy_path="../src/nhtsa_curator/ref/oem_groups.yml",
)

# COMMAND ----------
display(
    spark.sql(f"""
    SELECT
        count(*)                                     AS rows,
        count(DISTINCT make_norm)                    AS makes,
        sum(CAST(crash AS INT))                      AS crash_complaints,
        sum(CAST(fire AS INT))                       AS fire_complaints,
        sum(injured)                                 AS total_injured,
        sum(deaths)                                  AS total_deaths,
        avg(narrative_len_orig)                      AS avg_narrative_len
    FROM {cfg.full_schema_name}.silver_complaints
""")
)

# COMMAND ----------
# MAGIC %md
# MAGIC ### Spot-check: PII scrub
# MAGIC Sample a few narratives and confirm the scrubber replaced the
# MAGIC obvious patterns. If you see a raw VIN or phone make it past,
# MAGIC tighten the regex in `nhtsa_curator/pii.py`.

display(
    spark.sql(f"""
    SELECT odi_number, substring(narrative_clean, 1, 400) AS preview
    FROM {cfg.full_schema_name}.silver_complaints
    WHERE narrative_clean RLIKE '\\\\[(VIN|PHONE|EMAIL|NUM)\\\\]'
    LIMIT 10
""")
)
