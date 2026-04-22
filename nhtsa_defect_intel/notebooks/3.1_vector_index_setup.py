# Databricks notebook source
# MAGIC %md
# MAGIC # 3.1 — Vector Search index setup
# MAGIC
# MAGIC Creates (or reuses) the Vector Search **endpoint** and the
# MAGIC **delta-sync index** over `gold_narrative_chunks`.
# MAGIC
# MAGIC - Endpoint: `cfg.vector_search_endpoint`
# MAGIC - Source: `<catalog>.<schema>.gold_narrative_chunks` (CDF enabled in Phase 2)
# MAGIC - Index: `<catalog>.<schema>.gold_narrative_chunks_index`
# MAGIC - Embedding model: `cfg.embedding_endpoint` (managed embeddings)
# MAGIC
# MAGIC Both `ensure_endpoint` and `ensure_index` are idempotent — re-running
# MAGIC the notebook is safe and cheap once the index exists.
# MAGIC
# MAGIC The endpoint may take ~5-10 minutes on cold create; the initial
# MAGIC index sync depends on the row count of `gold_narrative_chunks` and
# MAGIC the embedding endpoint's throughput.

# COMMAND ----------
from databricks.vector_search.client import VectorSearchClient
from pyspark.sql import SparkSession

from nhtsa_curator.config import get_env, load_config
from nhtsa_curator.vector_search import (
    build_default_spec,
    ensure_endpoint,
    ensure_index,
    wait_for_index_online,
)

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()

dbutils.widgets.text("env", "dev")
dbutils.widgets.text("run_id", "manual")
dbutils.widgets.dropdown("wait_for_online", "true", ["true", "false"])

env = get_env(spark)
cfg = load_config("../project_config.yml", env)
wait = dbutils.widgets.get("wait_for_online") == "true"

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Confirm the source table exists and CDF is enabled

# COMMAND ----------
src_table = f"{cfg.full_schema_name}.gold_narrative_chunks"
assert spark.catalog.tableExists(src_table), (
    f"{src_table} not found — run notebook 2.8 first"
)

cdf_props = spark.sql(f"SHOW TBLPROPERTIES {src_table}").collect()
cdf_on = any(
    r["key"] == "delta.enableChangeDataFeed" and r["value"].lower() == "true"
    for r in cdf_props
)
assert cdf_on, "CDF not enabled on source — re-run 2.8 to flip the flag"

print(f"Source row count: {spark.table(src_table).count():,}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Endpoint + index

# COMMAND ----------
client = VectorSearchClient(disable_notice=True)
spec = build_default_spec(cfg)
print(spec)

ensure_endpoint(client, spec.endpoint_name)
ensure_index(client, spec)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Wait for ONLINE (optional — skip when re-running quickly)

# COMMAND ----------
if wait:
    wait_for_index_online(
        client,
        endpoint_name=spec.endpoint_name,
        index_name=spec.index_name,
        timeout_s=3600,
    )
    print("Index is ONLINE.")
else:
    print("Skipped wait_for_online — check status in the Databricks UI.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Smoke query

# COMMAND ----------
from nhtsa_curator.config import VectorSearchConfig
from nhtsa_curator.vector_search import similarity_search
import yaml

with open("../project_config.yml") as fh:
    raw = yaml.safe_load(fh)
vs_cfg = VectorSearchConfig(**raw["vector_search"])

hits = similarity_search(
    client=client,
    endpoint_name=spec.endpoint_name,
    index_name=spec.index_name,
    query_text="sudden unintended acceleration on a Tesla Model 3",
    vs_cfg=vs_cfg,
    columns=[
        "chunk_id", "source_dataset", "source_id",
        "make_norm", "model_year", "component_group", "content",
    ],
)
for h in hits[:5]:
    print(
        f"[{h.get('source_dataset')}] {h.get('source_id')} "
        f"({h.get('make_norm')} {h.get('model_year')}, "
        f"{h.get('component_group')})"
    )
    print(f"  {(h.get('content') or '')[:240]}\n")
