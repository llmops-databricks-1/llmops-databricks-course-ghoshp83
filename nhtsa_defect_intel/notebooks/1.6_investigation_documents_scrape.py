# Databricks notebook source
# MAGIC %md
# MAGIC # 1.6 — Investigation PDF scraper
# MAGIC
# MAGIC Populates `bronze_investigation_documents` with one row per
# MAGIC downloaded ODI investigation PDF. Runs after
# MAGIC `1.3_investigations_ingestion` has populated `bronze_investigations`.
# MAGIC
# MAGIC Behaviour:
# MAGIC - Pulls distinct `nhtsa_action_number` values from `bronze_investigations`.
# MAGIC - Skips action numbers already represented in
# MAGIC   `bronze_investigation_documents` (idempotent MERGE on `document_id`).
# MAGIC - Calls `list_documents_for_action` against the NHTSA listing URL
# MAGIC   template (widget `listing_url_template`).
# MAGIC - Downloads each PDF to the UC volume at
# MAGIC   `${volume}/investigation_documents/`.
# MAGIC - Bounded by `max_docs_per_run` so the backlog drains across
# MAGIC   successive runs rather than burning the job timeout on first
# MAGIC   deployment.

# COMMAND ----------
from loguru import logger
from pyspark.sql import SparkSession

from nhtsa_curator.bronze import write_investigation_documents_bronze
from nhtsa_curator.config import get_env, load_config
from nhtsa_curator.io.http import open_client
from nhtsa_curator.io.investigation_documents import scrape_investigation_documents

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()

dbutils.widgets.text("env", "dev")
dbutils.widgets.text("run_id", "manual")
dbutils.widgets.text("max_docs_per_run", "200")
# NHTSA ODI documents listing endpoint. JSON is preferred; the scraper
# falls back to HTML link scraping if the payload isn't JSON. Update
# here (not in code) if NHTSA rotates the path.
dbutils.widgets.text(
    "listing_url_template",
    "https://api.nhtsa.gov/odi/actions/{action_number}/documents",
)

env = get_env(spark)
run_id = dbutils.widgets.get("run_id")
max_docs = int(dbutils.widgets.get("max_docs_per_run"))
listing_url_template = dbutils.widgets.get("listing_url_template")

cfg = load_config("../project_config.yml", env)
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {cfg.full_schema_name}")

# COMMAND ----------
# Action numbers we haven't scraped yet = in bronze_investigations but
# not yet represented in bronze_investigation_documents. Left-anti keeps
# the query cheap even as the docs table grows.
docs_table = f"{cfg.full_schema_name}.bronze_investigation_documents"
if spark.catalog.tableExists(docs_table):
    unscraped_df = spark.sql(f"""
        SELECT DISTINCT i.nhtsa_action_number
        FROM {cfg.full_schema_name}.bronze_investigations i
        LEFT ANTI JOIN {docs_table} d
          ON i.nhtsa_action_number = d.nhtsa_action_number
        WHERE i.nhtsa_action_number IS NOT NULL
    """)
else:
    unscraped_df = spark.sql(f"""
        SELECT DISTINCT nhtsa_action_number
        FROM {cfg.full_schema_name}.bronze_investigations
        WHERE nhtsa_action_number IS NOT NULL
    """)

action_numbers = [r["nhtsa_action_number"] for r in unscraped_df.collect()]
logger.info(f"{len(action_numbers):,} unscraped action numbers")

# COMMAND ----------
volume_dir = f"{cfg.volume_fs_path}/investigation_documents"

with open_client() as client:
    rows = scrape_investigation_documents(
        client=client,
        action_numbers=action_numbers,
        volume_dir=volume_dir,
        listing_url_template=listing_url_template,
        max_docs=max_docs,
    )
    n = write_investigation_documents_bronze(
        spark=spark,
        cfg=cfg,
        rows=rows,
        ingest_run_id=run_id,
        source_url=listing_url_template,
    )

logger.info(f"Investigation docs scrape complete; rows merged: {n:,}")

# COMMAND ----------
display(
    spark.sql(f"""
    SELECT
        substring(nhtsa_action_number, 1, 2) AS investigation_type,
        count(DISTINCT nhtsa_action_number)  AS investigations_with_docs,
        count(*)                             AS documents
    FROM {docs_table}
    GROUP BY substring(nhtsa_action_number, 1, 2)
    ORDER BY documents DESC
""")
)
