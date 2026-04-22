"""Document parsing via Databricks ``ai_parse_document``.

``ai_parse_document(content)`` is a Databricks built-in SQL function
that takes binary PDF/Office bytes and returns a structured payload
(text, pages, optional table extractions). It runs on a serverless
inference endpoint, so we treat it as a potentially expensive call:

* **Idempotency**: we maintain a parsed-tracker table per dataset
  (``silver_<dataset>_parsed_tracker``) and only invoke ``ai_parse_document``
  on PDFs we haven't seen before. Re-running the silver_parse notebook
  is therefore a no-op for already-parsed docs.
* **Schema stability**: we materialise the parsed output into typed
  columns rather than carrying the raw STRUCT around. The first time
  we see a new ``ai_parse_document`` schema version the writer will
  log a warning and rebuild — see ``silver_<dataset>_parsed`` rebuilds
  triggered by the ``--full-rebuild`` flag in the notebook.
* **Cost guardrails**: callers can pass ``max_docs_per_run`` so the
  silver job stays predictable in runtime + spend. The unparsed
  backlog drains across consecutive runs.

This module deliberately does NOT chunk — chunking is gold's job (the
output here is one row per doc, not per chunk).
"""

from __future__ import annotations

from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from .config import ProjectConfig


_TRACKER_SCHEMA = T.StructType([
    T.StructField("doc_id", T.StringType(), False),
    T.StructField("volume_path", T.StringType(), False),
    T.StructField("parsed_at", T.TimestampType(), False),
    T.StructField("parser_version", T.StringType(), True),
    T.StructField("parse_status", T.StringType(), False),  # ok | error
    T.StructField("error_message", T.StringType(), True),
])


def _ensure_tracker(spark: SparkSession, table: str) -> None:
    if not spark.catalog.tableExists(table):
        spark.createDataFrame([], _TRACKER_SCHEMA) \
            .write.format("delta").saveAsTable(table)
        logger.info(f"Created parse tracker {table}")


def _unparsed_paths(
    spark: SparkSession,
    docs_table: str,
    tracker_table: str,
    doc_id_col: str,
    path_col: str,
    limit: int,
) -> DataFrame:
    """Return ``(doc_id, volume_path)`` for docs that aren't yet parsed."""
    return spark.sql(f"""
        SELECT d.{doc_id_col} AS doc_id, d.{path_col} AS volume_path
        FROM {docs_table} d
        LEFT JOIN {tracker_table} t
          ON d.{doc_id_col} = t.doc_id
        WHERE t.doc_id IS NULL
        LIMIT {int(limit)}
    """)


def parse_documents(
    spark: SparkSession,
    cfg: ProjectConfig,
    *,
    dataset: str,
    docs_table: str,
    doc_id_col: str,
    path_col: str,
    parsed_table: str,
    tracker_table: str,
    max_docs_per_run: int = 200,
    parser_version: str = "ai_parse_document/v1",
) -> dict:
    """Run ``ai_parse_document`` on the next batch of unparsed PDFs.

    Args:
        dataset: short label used in logs (``"tsb"`` | ``"investigation"``).
        docs_table: bronze table with ``doc_id_col`` + ``path_col``
            (e.g. ``bronze_tsb_documents`` with cols ``tsb_id`` +
            ``volume_path``).
        parsed_table: target silver-parsed table; CREATE-if-missing.
        tracker_table: per-dataset tracker for idempotency.
        max_docs_per_run: cap on documents parsed in this invocation.

    Returns:
        Tally dict ``{queued, parsed_ok, parsed_err}``.
    """
    _ensure_tracker(spark, tracker_table)

    todo = _unparsed_paths(
        spark, docs_table, tracker_table, doc_id_col, path_col, max_docs_per_run,
    )
    n_todo = todo.count()
    logger.info(f"[{dataset}] {n_todo} docs queued for ai_parse_document")
    if n_todo == 0:
        return {"queued": 0, "parsed_ok": 0, "parsed_err": 0}

    # Read PDF bytes from the UC volume via binaryFile, join to the queue
    # so we keep the doc_id alongside the parsed payload.
    todo.createOrReplaceTempView(f"_{dataset}_todo")

    parsed = spark.sql(f"""
        WITH pdfs AS (
            SELECT
                t.doc_id,
                t.volume_path,
                ai_parse_document(read_files(t.volume_path, format => 'binaryFile').content) AS parsed
            FROM _{dataset}_todo t
        )
        SELECT
            doc_id,
            volume_path,
            parsed.text          AS full_text,
            parsed.pages         AS pages,
            parsed.metadata      AS doc_metadata,
            current_timestamp()  AS parsed_at,
            '{parser_version}'   AS parser_version
        FROM pdfs
    """)

    # Persist parse outputs (append) and tracker rows in lockstep.
    if not spark.catalog.tableExists(parsed_table):
        # Materialise schema from the first batch so we don't have to
        # hand-maintain the ai_parse_document return shape.
        parsed.limit(0).write.format("delta") \
            .option("delta.enableChangeDataFeed", "true") \
            .saveAsTable(parsed_table)
        logger.info(f"Created {parsed_table}")

    # We can't easily distinguish per-row failure inside ai_parse_document
    # from here — Databricks raises on the whole call. We assume success
    # on the rows that materialise. Failures bubble up to the notebook
    # and the tracker isn't updated, so the next run will retry.
    parsed.write.mode("append").saveAsTable(parsed_table)

    tracker_rows = parsed.select(
        F.col("doc_id"),
        F.col("volume_path"),
        F.col("parsed_at"),
        F.col("parser_version"),
        F.lit("ok").alias("parse_status"),
        F.lit(None).cast("string").alias("error_message"),
    )
    tracker_rows.write.mode("append").saveAsTable(tracker_table)

    n_ok = tracker_rows.count()
    logger.info(f"[{dataset}] parsed_ok={n_ok}")
    return {"queued": n_todo, "parsed_ok": n_ok, "parsed_err": n_todo - n_ok}


# ---------------------------------------------------------------------------
# Convenience wrappers per dataset — keeps notebook code uncluttered.
# ---------------------------------------------------------------------------

def parse_tsb_documents(
    spark: SparkSession,
    cfg: ProjectConfig,
    *,
    max_docs_per_run: int = 200,
) -> dict:
    return parse_documents(
        spark,
        cfg,
        dataset="tsb",
        docs_table=f"{cfg.full_schema_name}.bronze_tsb_documents",
        doc_id_col="tsb_id",
        path_col="volume_path",
        parsed_table=f"{cfg.full_schema_name}.silver_tsb_parsed",
        tracker_table=f"{cfg.full_schema_name}.silver_tsb_parsed_tracker",
        max_docs_per_run=max_docs_per_run,
    )


def parse_investigation_documents(
    spark: SparkSession,
    cfg: ProjectConfig,
    *,
    max_docs_per_run: int = 200,
) -> dict:
    return parse_documents(
        spark,
        cfg,
        dataset="investigation",
        docs_table=f"{cfg.full_schema_name}.bronze_investigation_documents",
        doc_id_col="document_id",
        path_col="volume_path",
        parsed_table=f"{cfg.full_schema_name}.silver_investigation_parsed",
        tracker_table=f"{cfg.full_schema_name}.silver_investigation_parsed_tracker",
        max_docs_per_run=max_docs_per_run,
    )
