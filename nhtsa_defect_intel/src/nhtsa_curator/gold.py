"""Gold-layer builders: star schema for Genie + narrative chunks for VS.

Two distinct shapes live here:

1. **Star schema** (``gold_*_fact`` + ``dim_*``) — narrow, joinable
   tables exposed to the Genie space. Keeping the surface small is the
   single biggest lever on Genie SQL quality, so we expose only:

       - dim_vehicle   (make_norm, model_norm, model_year, vehicle_key)
       - dim_component (component_group, component_leaf, component_id)
       - dim_oem_group (oem_group, oem_group_id)
       - dim_date      (date_key, year, quarter, month, day)
       - gold_recalls_fact
       - gold_complaints_fact
       - gold_investigations_fact

2. **Narrative chunks** (``gold_narrative_chunks``) — the *only* table
   the Vector Search index reads. Sources:
       - silver_complaints.narrative_clean
       - silver_tsb_parsed.full_text (with metadata join from silver_tsbs)
       - silver_investigation_parsed.full_text

   Each chunk carries the metadata columns the agent will use as
   filters (``make_norm``, ``model_year``, ``component_group``,
   ``event_date``, ``source_dataset``).

Surrogate keys are deterministic ``xxhash64`` digests so a full
rebuild produces stable joins for downstream consumers.
"""

from __future__ import annotations

from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from .chunking import chunk_text
from .config import ChunkingConfig, ProjectConfig


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------

def _vehicle_key() -> "F.Column":
    return F.xxhash64(
        F.coalesce(F.col("make_norm"), F.lit("")),
        F.coalesce(F.col("model_norm"), F.lit("")),
        F.coalesce(F.col("model_year").cast("string"), F.lit("")),
    )


def _component_key() -> "F.Column":
    return F.xxhash64(
        F.coalesce(F.col("component_group"), F.lit("")),
        F.coalesce(F.col("component_leaf"), F.lit("")),
    )


def write_dim_vehicle(spark: SparkSession, cfg: ProjectConfig) -> int:
    """Union distinct (make, model, year) tuples across all silver tables."""
    parts = []
    for tbl, has_leaf in [
        (f"{cfg.full_schema_name}.silver_recalls", True),
        (f"{cfg.full_schema_name}.silver_complaints", True),
        (f"{cfg.full_schema_name}.silver_tsbs", True),
    ]:
        if not spark.catalog.tableExists(tbl):
            continue
        parts.append(
            spark.table(tbl).select("make_norm", "model_norm", "model_year")
        )
    if not parts:
        raise RuntimeError("No silver tables found to build dim_vehicle from")

    df = parts[0]
    for p in parts[1:]:
        df = df.unionByName(p)

    df = (
        df.dropDuplicates(["make_norm", "model_norm", "model_year"])
        .withColumn("vehicle_key", _vehicle_key())
    )
    table = f"{cfg.full_schema_name}.dim_vehicle"
    df.write.format("delta").mode("overwrite") \
        .option("overwriteSchema", "true").saveAsTable(table)
    n = spark.table(table).count()
    logger.info(f"dim_vehicle: {n:,} rows")
    return n


def write_dim_component(spark: SparkSession, cfg: ProjectConfig) -> int:
    parts = []
    for tbl in [
        f"{cfg.full_schema_name}.silver_recalls",
        f"{cfg.full_schema_name}.silver_complaints",
        f"{cfg.full_schema_name}.silver_tsbs",
        f"{cfg.full_schema_name}.silver_investigations",
    ]:
        if not spark.catalog.tableExists(tbl):
            continue
        cols = spark.table(tbl).columns
        leaf_col = "component_leaf" if "component_leaf" in cols else "component_raw"
        parts.append(
            spark.table(tbl).select(
                "component_group",
                F.col(leaf_col).alias("component_leaf"),
            )
        )
    if not parts:
        raise RuntimeError("No silver tables found to build dim_component from")

    df = parts[0]
    for p in parts[1:]:
        df = df.unionByName(p)

    df = (
        df.dropDuplicates(["component_group", "component_leaf"])
        .withColumn("component_id", _component_key())
    )
    table = f"{cfg.full_schema_name}.dim_component"
    df.write.format("delta").mode("overwrite") \
        .option("overwriteSchema", "true").saveAsTable(table)
    n = spark.table(table).count()
    logger.info(f"dim_component: {n:,} rows")
    return n


def write_dim_oem_group(spark: SparkSession, cfg: ProjectConfig) -> int:
    """One row per known OEM group + a sentinel for makes outside the taxonomy."""
    parts = []
    for tbl in [
        f"{cfg.full_schema_name}.silver_recalls",
        f"{cfg.full_schema_name}.silver_complaints",
        f"{cfg.full_schema_name}.silver_tsbs",
    ]:
        if not spark.catalog.tableExists(tbl):
            continue
        parts.append(spark.table(tbl).select("oem_group"))
    if not parts:
        raise RuntimeError("No silver tables found to build dim_oem_group from")
    df = parts[0]
    for p in parts[1:]:
        df = df.unionByName(p)
    df = (
        df.dropDuplicates(["oem_group"])
        .withColumn("oem_group_id", F.xxhash64(F.coalesce(F.col("oem_group"), F.lit("UNKNOWN"))))
    )
    table = f"{cfg.full_schema_name}.dim_oem_group"
    df.write.format("delta").mode("overwrite") \
        .option("overwriteSchema", "true").saveAsTable(table)
    n = spark.table(table).count()
    logger.info(f"dim_oem_group: {n:,} rows")
    return n


def write_dim_date(spark: SparkSession, cfg: ProjectConfig) -> int:
    """Calendar dim spanning the union of date columns across silver."""
    bounds = spark.sql(f"""
        SELECT
            LEAST(
                (SELECT min(record_creation_date) FROM {cfg.full_schema_name}.silver_recalls),
                (SELECT min(incident_date)        FROM {cfg.full_schema_name}.silver_complaints),
                (SELECT min(open_date)            FROM {cfg.full_schema_name}.silver_investigations)
            ) AS d_min,
            GREATEST(
                (SELECT max(record_creation_date) FROM {cfg.full_schema_name}.silver_recalls),
                (SELECT max(incident_date)        FROM {cfg.full_schema_name}.silver_complaints),
                (SELECT max(open_date)            FROM {cfg.full_schema_name}.silver_investigations),
                CURRENT_DATE()
            ) AS d_max
    """).collect()[0]

    d_min, d_max = bounds["d_min"], bounds["d_max"]
    if d_min is None or d_max is None:
        # Cold start — fall back to last 30 years.
        d_max = spark.sql("SELECT current_date() AS d").collect()[0]["d"]
        d_min = spark.sql("SELECT add_months(current_date(), -360) AS d") \
            .collect()[0]["d"]

    table = f"{cfg.full_schema_name}.dim_date"
    spark.sql(f"""
        CREATE OR REPLACE TABLE {table} AS
        SELECT
            d                    AS date_key,
            year(d)              AS year,
            quarter(d)           AS quarter,
            month(d)             AS month,
            day(d)               AS day,
            date_format(d,'EEE') AS day_of_week,
            d = current_date()   AS is_today
        FROM (
            SELECT explode(sequence(DATE'{d_min}', DATE'{d_max}', INTERVAL 1 DAY)) AS d
        )
    """)
    n = spark.table(table).count()
    logger.info(f"dim_date: {n:,} rows ({d_min}..{d_max})")
    return n


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------

def _join_keys(df: DataFrame) -> DataFrame:
    return (
        df.withColumn("vehicle_key", _vehicle_key())
        .withColumn("component_id", _component_key())
        .withColumn("oem_group_id", F.xxhash64(F.coalesce(F.col("oem_group"), F.lit("UNKNOWN"))))
    )


def write_gold_recalls_fact(spark: SparkSession, cfg: ProjectConfig) -> int:
    src = spark.table(f"{cfg.full_schema_name}.silver_recalls")
    fact = (
        _join_keys(src)
        .select(
            "record_id",
            "campaign_number",
            "vehicle_key",
            "component_id",
            "oem_group_id",
            "units_affected",
            F.col("owner_notify_date").alias("event_date"),
            "recall_type_code",
            "fmvss",
        )
    )
    table = f"{cfg.full_schema_name}.gold_recalls_fact"
    fact.write.format("delta").mode("overwrite") \
        .option("overwriteSchema", "true").saveAsTable(table)
    n = spark.table(table).count()
    logger.info(f"gold_recalls_fact: {n:,} rows")
    return n


def write_gold_complaints_fact(spark: SparkSession, cfg: ProjectConfig) -> int:
    src = spark.table(f"{cfg.full_schema_name}.silver_complaints")
    fact = (
        _join_keys(src)
        .select(
            "complaint_id",
            "odi_number",
            "vehicle_key",
            "component_id",
            "oem_group_id",
            F.col("incident_date").alias("event_date"),
            "crash",
            "fire",
            "injured",
            "deaths",
            "miles",
            "state",
            "complaint_source",
        )
    )
    table = f"{cfg.full_schema_name}.gold_complaints_fact"
    fact.write.format("delta").mode("overwrite") \
        .option("overwriteSchema", "true") \
        .partitionBy("state").saveAsTable(table)
    n = spark.table(table).count()
    logger.info(f"gold_complaints_fact: {n:,} rows")
    return n


def write_gold_investigations_fact(spark: SparkSession, cfg: ProjectConfig) -> int:
    src = spark.table(f"{cfg.full_schema_name}.silver_investigations")
    src = (
        src.withColumnRenamed("mfr_norm", "make_norm")
        .withColumn("model_norm", F.lit(None).cast("string"))
        .withColumn("model_year", F.lit(None).cast("int"))
        .withColumn("component_leaf", F.lit(None).cast("string"))
    )
    fact = (
        _join_keys(src)
        .select(
            "nhtsa_action_number",
            "investigation_type",
            "status",
            "days_open",
            "vehicle_key",
            "component_id",
            "oem_group_id",
            F.col("open_date").alias("event_date"),
            "close_date",
        )
    )
    table = f"{cfg.full_schema_name}.gold_investigations_fact"
    fact.write.format("delta").mode("overwrite") \
        .option("overwriteSchema", "true").saveAsTable(table)
    n = spark.table(table).count()
    logger.info(f"gold_investigations_fact: {n:,} rows")
    return n


# ---------------------------------------------------------------------------
# Narrative chunks
# ---------------------------------------------------------------------------

def _chunk_udf(chunk_size: int, overlap: int, separator: str):
    def _do(text: str | None):
        return chunk_text(text, chunk_size, overlap, separator)
    return F.udf(_do, T.ArrayType(T.StringType()))


def write_gold_narrative_chunks(
    spark: SparkSession,
    cfg: ProjectConfig,
    chunking: ChunkingConfig,
    embedding_model: str,
    sample_per_source: dict[str, int] | None = None,
) -> int:
    """Build the unified narrative-chunk table from all three text sources.

    Output schema matches what the Vector Search index will read from in
    Phase 3. ``chunk_id`` is a stable sha256 of source_dataset + source_id
    + chunk_idx so the index can do incremental updates.

    ``sample_per_source`` caps the per-source-dataset row count via a
    deterministic ``sampleBy`` (seed=42). Intended for dev/demo runs
    where the full ~8.75M-row corpus would take weeks to embed on the
    shared Foundation Models endpoint — a stratified 500K sample syncs
    in ~24h and is dense enough for a credible retrieval demo. Pass
    ``None`` (the default) for a full production build.
    """
    chunk = _chunk_udf(chunking.chunk_size, chunking.chunk_overlap, chunking.separator)

    # Source A — complaints.
    cmpl = spark.table(f"{cfg.full_schema_name}.silver_complaints").select(
        F.lit("complaints").alias("source_dataset"),
        F.col("odi_number").alias("source_id"),
        F.lit(None).cast("string").alias("parent_doc_id"),
        F.col("narrative_clean").alias("body"),
        "make_norm",
        "model_norm",
        "model_year",
        "component_group",
        "oem_group",
        F.col("incident_date").alias("event_date"),
    )

    # Source B — TSB summaries. NHTSA's MfrComms rewrite removed the
    # per-TSB PDF URLs; the Summary column now carries up to 4000 chars
    # of bulletin text inline, so we chunk the silver_tsbs.summary
    # directly instead of the older parsed-PDF flow. ``silver_tsb_parsed``
    # is still checked so a custom PDF-parse pipeline can supersede this
    # if the env reintroduces one.
    if spark.catalog.tableExists(f"{cfg.full_schema_name}.silver_tsb_parsed"):
        tsb = spark.sql(f"""
            SELECT
                'tsb'                       AS source_dataset,
                p.doc_id                    AS source_id,
                p.doc_id                    AS parent_doc_id,
                p.full_text                 AS body,
                t.make_norm,
                t.model_norm,
                t.model_year,
                t.component_group,
                t.oem_group,
                t.original_date             AS event_date
            FROM {cfg.full_schema_name}.silver_tsb_parsed p
            LEFT JOIN {cfg.full_schema_name}.silver_tsbs t
              ON p.doc_id = t.tsb_id
        """)
    else:
        tsb = spark.table(f"{cfg.full_schema_name}.silver_tsbs").select(
            F.lit("tsb").alias("source_dataset"),
            F.col("tsb_id").alias("source_id"),
            F.col("tsb_id").alias("parent_doc_id"),
            F.col("summary").alias("body"),
            "make_norm",
            "model_norm",
            "model_year",
            "component_group",
            "oem_group",
            F.col("original_date").alias("event_date"),
        )

    # Source C — investigation parsed text joined to silver_investigations.
    if spark.catalog.tableExists(f"{cfg.full_schema_name}.silver_investigation_parsed"):
        inv = spark.sql(f"""
            SELECT
                'investigation'             AS source_dataset,
                p.doc_id                    AS source_id,
                p.doc_id                    AS parent_doc_id,
                p.full_text                 AS body,
                i.mfr_norm                  AS make_norm,
                CAST(NULL AS STRING)        AS model_norm,
                CAST(NULL AS INT)           AS model_year,
                i.component_group           AS component_group,
                i.oem_group                 AS oem_group,
                i.open_date                 AS event_date
            FROM {cfg.full_schema_name}.silver_investigation_parsed p
            LEFT JOIN {cfg.full_schema_name}.silver_investigations i
              ON p.doc_id = i.nhtsa_action_number
        """)
    else:
        inv = None

    parts = [cmpl] + [d for d in (tsb, inv) if d is not None]
    union = parts[0]
    for p in parts[1:]:
        union = union.unionByName(p)

    chunked = (
        union
        .filter(F.col("body").isNotNull() & (F.length("body") > 0))
        .withColumn("chunks", chunk(F.col("body")))
        .select("*", F.posexplode("chunks").alias("chunk_idx", "content"))
        .drop("body", "chunks")
        .withColumn(
            "chunk_id",
            F.sha2(
                F.concat_ws(
                    "::",
                    F.col("source_dataset"),
                    F.coalesce(F.col("source_id"), F.lit("")),
                    F.col("chunk_idx").cast("string"),
                ),
                256,
            ),
        )
        .withColumn("embedding_model", F.lit(embedding_model))
        .withColumn("_chunked_at", F.current_timestamp())
    )

    final = chunked.select(
        "chunk_id",
        "source_dataset",
        "source_id",
        "parent_doc_id",
        "chunk_idx",
        "content",
        "make_norm",
        "model_norm",
        "model_year",
        "component_group",
        "oem_group",
        "event_date",
        "embedding_model",
        "_chunked_at",
    )

    if sample_per_source:
        # Note: we deliberately don't .cache() here — serverless compute
        # blocks PERSIST TABLE, so the chunking UDF re-runs once for the
        # count and once for the final write. Acceptable cost for a
        # one-time dev/demo sample build.
        counts = {
            r["source_dataset"]: r["n"]
            for r in (
                final.groupBy("source_dataset")
                .count()
                .withColumnRenamed("count", "n")
                .collect()
            )
        }
        fractions = {
            src: min(1.0, target / counts[src])
            for src, target in sample_per_source.items()
            if counts.get(src, 0) > 0
        }
        logger.info(
            f"Stratified sample — source counts: {counts}, "
            f"fractions: {fractions}, targets: {sample_per_source}"
        )
        final = final.sampleBy("source_dataset", fractions=fractions, seed=42)

    table = f"{cfg.full_schema_name}.gold_narrative_chunks"
    (
        final
        .write.format("delta").mode("overwrite")
        .option("overwriteSchema", "true")
        .option("delta.enableChangeDataFeed", "true")  # Vector Search needs CDF
        .saveAsTable(table)
    )
    n = spark.table(table).count()
    logger.info(f"gold_narrative_chunks: {n:,} rows")
    return n
