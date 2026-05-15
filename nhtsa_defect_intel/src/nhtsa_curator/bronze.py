"""Bronze writers for the NHTSA datasets.

One ``write_<dataset>_bronze`` function per dataset. Each:

1. Consumes an iterator of dicts from the corresponding ``io.<dataset>``
   reader.
2. Adds the standard provenance columns (``_ingested_at``,
   ``_ingest_run_id``, ``_source_url``, ``_raw``).
3. MERGEs into the bronze delta table on the dataset's natural key, so
   re-running a job is idempotent.

Bronze tables live at ``${catalog}.${schema}.bronze_<dataset>``. See
``docs/03_data_model.md`` for the full schema.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from itertools import islice
from typing import TYPE_CHECKING

from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import types as T

from .config import ProjectConfig

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


# ---------------------------------------------------------------------------
# Common provenance columns
# ---------------------------------------------------------------------------

_PROVENANCE_FIELDS = [
    T.StructField("_ingest_run_id", T.StringType(), False),
    T.StructField("_ingested_at", T.TimestampType(), False),
    T.StructField("_source_url", T.StringType(), True),
    T.StructField("_raw", T.StringType(), True),
]


def _bronze_table(cfg: ProjectConfig, dataset: str) -> str:
    return f"{cfg.full_schema_name}.bronze_{dataset}"


def _create_table_if_missing(
    spark: SparkSession,
    table_name: str,
    schema: T.StructType,
) -> None:
    """Create an empty delta table with the given schema if it doesn't exist."""
    if not spark.catalog.tableExists(table_name):
        empty = spark.createDataFrame([], schema)
        (
            empty.write.format("delta")
            .option("delta.enableChangeDataFeed", "true")
            .saveAsTable(table_name)
        )
        logger.info(f"Created bronze table {table_name}")


def _evolve_target_schema(
    spark: SparkSession,
    table_name: str,
    df: DataFrame,
) -> None:
    """ALTER TABLE ADD COLUMNS for any source column missing on target.

    NHTSA periodically appends fields to its flat-file schemas (e.g.
    May-2025 added ``do_not_drive`` / ``park_outside`` to recalls). A
    fresh tuple in code then diverges from a bronze table created by
    an earlier run, which makes ``MERGE ... INSERT`` fail on the new
    columns. Widening the target in place keeps the pipeline
    idempotent without requiring a manual table drop.
    """
    target_cols = {f.name for f in spark.table(table_name).schema.fields}
    source_schema = {f.name: f.dataType.simpleString() for f in df.schema.fields}
    missing = [c for c in df.columns if c not in target_cols]
    if not missing:
        return
    add_clause = ", ".join(f"{c} {source_schema[c]}" for c in missing)
    spark.sql(f"ALTER TABLE {table_name} ADD COLUMNS ({add_clause})")
    logger.info(f"Evolved {table_name}: added {missing}")


def _merge_on_key(
    spark: SparkSession,
    table_name: str,
    df: DataFrame,
    key_cols: list[str],
) -> int:
    """MERGE ``df`` into ``table_name`` on ``key_cols``, insert-if-not-matched.

    Returns the row count of ``df`` (note: this is the input size, not
    the merged delta).
    """
    _evolve_target_schema(spark, table_name, df)
    df.createOrReplaceTempView("_bronze_source")
    on_clause = " AND ".join(f"target.{c} = source.{c}" for c in key_cols)
    cols = df.columns
    insert_cols = ", ".join(cols)
    insert_vals = ", ".join(f"source.{c}" for c in cols)
    spark.sql(f"""
        MERGE INTO {table_name} target
        USING _bronze_source source
        ON {on_clause}
        WHEN NOT MATCHED THEN INSERT ({insert_cols}) VALUES ({insert_vals})
    """)
    n = df.count()
    logger.info(f"Merged {n:,} rows into {table_name}")
    return n


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


def _schema_for(columns: list[str]) -> T.StructType:
    """All NHTSA flat-file fields land as STRING in bronze; typing is silver."""
    fields = [T.StructField(c, T.StringType(), True) for c in columns]
    return T.StructType(fields + _PROVENANCE_FIELDS)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_recalls_bronze(
    spark: SparkSession,
    cfg: ProjectConfig,
    rows: Iterable[dict],
    ingest_run_id: str,
    source_url: str,
) -> int:
    """Write recall rows to ``bronze_recalls``.

    Idempotency key: ``record_id`` (NHTSA's unique row id; campaign ids
    fan out to many records, one per make/model/year affected).
    """
    from .io._flat_files import RECALLS_COLUMNS

    table = _bronze_table(cfg, "recalls")
    schema = _schema_for(list(RECALLS_COLUMNS))
    _create_table_if_missing(spark, table, schema)
    return _write_iterable(
        spark,
        rows,
        list(RECALLS_COLUMNS),
        schema,
        table,
        key_cols=["record_id"],
        ingest_run_id=ingest_run_id,
        source_url=source_url,
    )


def write_complaints_bronze(
    spark: SparkSession,
    cfg: ProjectConfig,
    rows: Iterable[dict],
    ingest_run_id: str,
    source_url: str,
) -> int:
    """Write complaint rows to ``bronze_complaints``. Idempotent on cmplid."""
    from .io._flat_files import COMPLAINTS_COLUMNS

    table = _bronze_table(cfg, "complaints")
    schema = _schema_for(list(COMPLAINTS_COLUMNS))
    _create_table_if_missing(spark, table, schema)
    return _write_iterable(
        spark,
        rows,
        list(COMPLAINTS_COLUMNS),
        schema,
        table,
        key_cols=["cmplid"],
        ingest_run_id=ingest_run_id,
        source_url=source_url,
    )


def write_investigations_bronze(
    spark: SparkSession,
    cfg: ProjectConfig,
    rows: Iterable[dict],
    ingest_run_id: str,
    source_url: str,
) -> int:
    """Write investigation rows to ``bronze_investigations``."""
    from .io._flat_files import INVESTIGATIONS_COLUMNS

    table = _bronze_table(cfg, "investigations")
    schema = _schema_for(list(INVESTIGATIONS_COLUMNS))
    _create_table_if_missing(spark, table, schema)
    return _write_iterable(
        spark,
        rows,
        list(INVESTIGATIONS_COLUMNS),
        schema,
        table,
        key_cols=["nhtsa_action_number"],
        ingest_run_id=ingest_run_id,
        source_url=source_url,
    )


def write_tsbs_bronze(
    spark: SparkSession,
    cfg: ProjectConfig,
    rows: Iterable[dict],
    ingest_run_id: str,
    source_url: str,
) -> int:
    """Write TSB / MfrComms index rows to ``bronze_tsb_index``.

    NHTSA's MfrComms schema fans one communication out across every
    (product, component) pair, so ``tsb_id`` + ``nhtsa_item_number``
    alone are NOT unique. We merge on the composite natural key the
    NHTSA data dictionary describes so re-running a month's ingest
    stays idempotent instead of inserting n×(product×component) dupes.
    """
    from .io._flat_files import TSBS_COLUMNS

    table = _bronze_table(cfg, "tsb_index")
    schema = _schema_for(list(TSBS_COLUMNS))
    _create_table_if_missing(spark, table, schema)
    return _write_iterable(
        spark,
        rows,
        list(TSBS_COLUMNS),
        schema,
        table,
        key_cols=[
            "nhtsa_item_number",
            "tsb_id",
            "maketxt",
            "modeltxt",
            "yeartxt",
            "component_desc",
        ],
        ingest_run_id=ingest_run_id,
        source_url=source_url,
    )


_INVESTIGATION_DOCS_COLUMNS: tuple[str, ...] = (
    "document_id",
    "nhtsa_action_number",
    "document_url",
    "document_title",
    "volume_path",
    "file_name",
)


def write_investigation_documents_bronze(
    spark: SparkSession,
    cfg: ProjectConfig,
    rows: Iterable[dict],
    ingest_run_id: str,
    source_url: str,
) -> int:
    """Write scraped investigation PDF rows to ``bronze_investigation_documents``.

    Idempotency key: ``document_id`` (stable hash of action number +
    PDF URL). Re-running the scraper on the same action numbers will
    re-download PDFs but the MERGE keeps the bronze table deduped, so
    the downstream ``ai_parse_document`` tracker never sees duplicates.
    """
    table = _bronze_table(cfg, "investigation_documents")
    schema = _schema_for(list(_INVESTIGATION_DOCS_COLUMNS))
    _create_table_if_missing(spark, table, schema)
    return _write_iterable(
        spark,
        rows,
        list(_INVESTIGATION_DOCS_COLUMNS),
        schema,
        table,
        key_cols=["document_id"],
        ingest_run_id=ingest_run_id,
        source_url=source_url,
    )


def write_sgo_bronze(
    spark: SparkSession,
    cfg: ProjectConfig,
    rows: Iterable[dict],
    ingest_run_id: str,
    source_url: str,
    key_column: str = "Report ID",
) -> int:
    """Write SGO rows to ``bronze_sgo_crashes``.

    SGO CSVs evolve over time so we infer the schema from the first
    batch rather than hard-coding a column list. ``key_column`` is the
    natural id (default "Report ID" — pass the actual header you find).
    """
    table = _bronze_table(cfg, "sgo_crashes")
    materialised = list(rows)
    if not materialised:
        logger.info("SGO snapshot empty, nothing to write.")
        return 0

    # ADS and ADAS snapshots share most but not all columns; take the
    # union (ordered by first-seen) so no fields are silently dropped.
    seen: dict[str, None] = {}
    for r in materialised:
        for k in r:
            if k not in seen:
                seen[k] = None
    columns = list(seen.keys())
    schema = _schema_for(columns)
    _create_table_if_missing(spark, table, schema)
    return _write_iterable(
        spark,
        materialised,
        columns,
        schema,
        table,
        key_cols=[_safe_col(key_column)],
        ingest_run_id=ingest_run_id,
        source_url=source_url,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_col(name: str) -> str:
    """Normalise SGO header (spaces/punct) to a Delta-safe column name."""
    from .io._flat_files import safe_col

    return safe_col(name)


_DEFAULT_BATCH_SIZE = 200_000


def _write_iterable(
    spark: SparkSession,
    rows: Iterable[dict],
    columns: list[str],
    schema: T.StructType,
    table: str,
    key_cols: list[str],
    ingest_run_id: str,
    source_url: str,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> int:
    """Stream an iterable of dicts to delta with provenance + MERGE.

    Batched to keep driver heap bounded: NHTSA's FLAT_CMPL dump is ~2.5M
    rows and materialising all of them (plus a ``_raw`` JSON string per
    row) into Python memory before ``createDataFrame`` crashed the
    driver with DRIVER_UNEXPECTED_FAILURE on a small cluster. Each
    batch is built, MERGEd, and dropped before the next one starts.
    """
    # _ingested_at is NOT NULL in the schema, so stamp it on the
    # python side — letting withColumn(current_timestamp()) fill it
    # later fails because Spark validates nullability before the
    # post-hoc column replacement runs.
    ingested_at = datetime.now(tz=UTC)
    it = iter(rows)
    total = 0
    batch_num = 0

    while True:
        chunk = list(islice(it, batch_size))
        if not chunk:
            break
        batch_num += 1
        records = []
        for r in chunk:
            rec = {c: r.get(c) for c in columns}
            rec["_ingest_run_id"] = ingest_run_id
            rec["_ingested_at"] = ingested_at
            rec["_source_url"] = source_url
            rec["_raw"] = json.dumps(r, default=str)
            records.append(rec)

        df = spark.createDataFrame(records, schema=schema)
        n = _merge_on_key(spark, table, df, key_cols)
        total += n
        logger.info(
            f"{table} batch {batch_num}: merged {n:,} rows (running total: {total:,})"
        )

    if total == 0:
        logger.info(f"No rows to write to {table}")

    return total
