# Databricks notebook source
# MAGIC %md
# MAGIC # 3.3 — Smoke-test retrieval over both surfaces
# MAGIC
# MAGIC Runs 10 hand-picked queries against the Vector Search index and
# MAGIC checks each one returns at least one hit whose metadata matches a
# MAGIC simple expectation (e.g. "Tesla Model 3 acceleration → at least one
# MAGIC chunk with make_norm=Tesla"). Records p50 / p95 latency.
# MAGIC
# MAGIC Genie smoke-test is **manual** — the API for programmatic Genie
# MAGIC queries isn't part of this scope. The 5 trusted-query examples in
# MAGIC notebook 3.2 double as the Genie smoke set: paste each into the
# MAGIC space UI and verify the SQL Genie generates matches the seed.
# MAGIC
# MAGIC Pass criteria (Phase 3 checkpoint):
# MAGIC - VS: ≥ 8 of 10 queries return a sensible top hit.
# MAGIC - Genie: 5 of 5 trusted queries match the seed SQL.

# COMMAND ----------
import statistics
import time

import yaml
from databricks.vector_search.client import VectorSearchClient
from pyspark.sql import SparkSession

from nhtsa_curator.config import VectorSearchConfig, get_env, load_config
from nhtsa_curator.vector_search import build_default_spec, similarity_search

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()

dbutils.widgets.text("env", "dev")
dbutils.widgets.text("run_id", "manual")
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

with open("../project_config.yml") as fh:
    raw = yaml.safe_load(fh)
vs_cfg = VectorSearchConfig(**raw["vector_search"])

client = VectorSearchClient(disable_notice=True)
spec = build_default_spec(cfg)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Hand-picked smoke set
# MAGIC
# MAGIC Each entry: a query and a `must_match` predicate over the top hit.
# MAGIC Predicates are deliberately loose — we're testing that retrieval
# MAGIC isn't broken, not that ranking is perfect.

# COMMAND ----------
SMOKE_QUERIES = [
    {
        "q": "sudden unintended acceleration in a Tesla Model 3",
        "filters": {"make_norm": "Tesla"},
        "must_match": lambda h: h.get("make_norm") == "Tesla",
    },
    {
        "q": "Ford F-150 transmission shifting issue",
        "filters": {"make_norm": "Ford"},
        "must_match": lambda h: h.get("make_norm") == "Ford",
    },
    {
        "q": "airbag deployment failure during low-speed crash",
        "filters": {},
        "must_match": lambda h: (h.get("component_group") or "").lower().startswith(
            "air bag"
        ) or "airbag" in (h.get("content") or "").lower(),
    },
    {
        "q": "engine fire while parked",
        "filters": {},
        "must_match": lambda h: "fire" in (h.get("content") or "").lower(),
    },
    {
        "q": "brake pedal goes to floor",
        "filters": {},
        "must_match": lambda h: (
            "brake" in (h.get("content") or "").lower()
            or (h.get("component_group") or "").lower().startswith("service brakes")
        ),
    },
    {
        "q": "TSB on EGR cooler replacement procedure",
        "filters": {"source_dataset": "tsb"},
        "must_match": lambda h: h.get("source_dataset") == "tsb",
    },
    {
        "q": "complaint about steering wheel locking up at highway speed",
        "filters": {"source_dataset": "complaints"},
        "must_match": lambda h: h.get("source_dataset") == "complaints",
    },
    {
        "q": "Hyundai Sonata engine knock 2015",
        "filters": {"make_norm": "Hyundai", "model_year": 2015},
        "must_match": lambda h: (
            h.get("make_norm") == "Hyundai" and h.get("model_year") == 2015
        ),
    },
    {
        "q": "Takata airbag rupture metal fragments",
        "filters": {},
        "must_match": lambda h: "takata" in (h.get("content") or "").lower()
        or "airbag" in (h.get("content") or "").lower(),
    },
    {
        "q": "battery thermal runaway in EV after charging",
        "filters": {},
        "must_match": lambda h: (
            "battery" in (h.get("content") or "").lower()
            or (h.get("component_group") or "").lower().startswith("electrical")
        ),
    },
]


# COMMAND ----------
def _run_one(q: dict) -> dict:
    t0 = time.perf_counter()
    hits = similarity_search(
        client=client,
        endpoint_name=spec.endpoint_name,
        index_name=spec.index_name,
        query_text=q["q"],
        vs_cfg=vs_cfg,
        filters=q["filters"],
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    top = hits[0] if hits else {}
    passed = bool(hits) and bool(q["must_match"](top))
    return {
        "q": q["q"],
        "n_hits": len(hits),
        "top_source": top.get("source_dataset"),
        "top_make": top.get("make_norm"),
        "top_year": top.get("model_year"),
        "passed": passed,
        "latency_ms": round(latency_ms, 1),
    }


results = [_run_one(q) for q in SMOKE_QUERIES]

# COMMAND ----------
# MAGIC %md
# MAGIC ## Results

# COMMAND ----------
import pandas as pd

df = pd.DataFrame(results)
display(df)

n_pass = int(df["passed"].sum())
print(f"\nPASS: {n_pass}/{len(df)} (target ≥ 8)")
print(f"latency p50: {statistics.median(df['latency_ms']):.1f} ms")
print(f"latency p95: {df['latency_ms'].quantile(0.95):.1f} ms")
print(f"avg n_hits : {df['n_hits'].mean():.1f}")

assert n_pass >= 8, f"Vector Search smoke gate failed: {n_pass}/10"

# COMMAND ----------
# MAGIC %md
# MAGIC ## Genie smoke (manual)
# MAGIC
# MAGIC Open the Genie space, paste each of the 5 trusted-query questions
# MAGIC from notebook 3.2, and confirm Genie generates SQL that:
# MAGIC - Joins on the surrogate key (`vehicle_key`, `oem_group_id`,
# MAGIC   `component_id`) — not on raw text columns.
# MAGIC - Returns the same row count (±5%) as the seed SQL when run.
# MAGIC
# MAGIC Record the 5/5 result in the Phase 3 checkpoint comment.
