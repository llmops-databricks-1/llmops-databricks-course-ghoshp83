# Databricks notebook source
# MAGIC %md
# MAGIC # Data Processing Pipeline
# MAGIC
# MAGIC This notebook processes arXiv papers and syncs the vector search index.
# MAGIC Runs on a schedule to keep the knowledge base up to date.
# MAGIC
# MAGIC Pipeline steps:
# MAGIC 1. Download new PDFs from arXiv
# MAGIC 2. Parse PDFs with AI Parse Documents
# MAGIC 3. Extract and clean chunks
# MAGIC 4. Sync vector search index

# COMMAND ----------

from loguru import logger
from pyspark.sql import SparkSession

from arxiv_curator.config import get_env, load_config
from arxiv_curator.data_processor import DataProcessor
from arxiv_curator.vector_search import VectorSearchManager

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

env = get_env(spark)
cfg = load_config("../../project_config.yml", env=env)

logger.info("Configuration loaded:")
logger.info(f"  Environment: {env}")
logger.info(f"  Catalog: {cfg.catalog}")
logger.info(f"  Schema: {cfg.schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Process New Papers

# COMMAND ----------

processor = DataProcessor(spark=spark, config=cfg)
processor.process_and_save()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Sync Vector Search Index

# COMMAND ----------

vs_manager = VectorSearchManager(config=cfg)
vs_manager.sync_index()

logger.info("✓ Data processing pipeline complete!")
