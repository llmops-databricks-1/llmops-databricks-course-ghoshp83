"""Silver transformations for the NHTSA datasets.

Silver tables are typed, deduplicated, PII-scrubbed views over bronze.
They are the canonical source for everything downstream: gold facts,
gold narrative chunks, and the agent's tool surface.

Each ``write_silver_<dataset>`` function:

1. Reads from the corresponding bronze table.
2. Casts NHTSA's stringly-typed fields into proper SQL types
   (dates from ``YYYYMMDD``, ints from numeric strings, booleans
   from Y/N).
3. Adds normalised columns (``make_norm``, ``oem_group``,
   ``component_group``).
4. Scrubs PII out of free-text narratives (complaints only).
5. CREATE OR REPLACE TABLE — silver is fully reproducible from bronze,
   so we don't need MERGE here. The cost of rewriting is small at the
   sizes we're dealing with (~10M complaints, ~100k recalls), and it
   keeps the logic dramatically simpler.

Bronze is preserved verbatim; silver can be torn down and rebuilt at
any time.
"""

from __future__ import annotations

from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from .config import ProjectConfig
from .taxonomy import make_to_oem_group

# ---------------------------------------------------------------------------
# Shared UDFs (registered lazily so import doesn't require a SparkSession).
# ---------------------------------------------------------------------------

_OEM_RESULT = T.StructType(
    [
        T.StructField("make_norm", T.StringType(), True),
        T.StructField("oem_group", T.StringType(), True),
    ]
)


def _register_udfs(spark: SparkSession, taxonomy_path: str) -> None:
    """Register the OEM resolver UDF once per SparkSession.

    ``scrub_pii``, ``component_group``, and ``component_leaf`` used to be
    Python UDFs too. They are now native Spark column expressions (see
    ``_scrub_pii_col`` / ``_component_group_col`` below) — row-by-row
    Python UDFs dominated runtime on the 2.5M-row complaints write.
    Only ``resolve_oem`` remains a UDF because it does a taxonomy lookup
    that isn't worth inlining as a broadcast join at this scale.
    """

    def _resolve_oem(make: str | None) -> tuple[str | None, str | None]:
        canonical, group = make_to_oem_group(make, taxonomy_path)
        return (canonical, group)

    spark.udf.register("resolve_oem", _resolve_oem, _OEM_RESULT)


# Mirror of pii.py regexes; Spark uses Java regex but none of the
# patterns below use Python-specific constructs. Kept as a tuple so
# ``_scrub_pii_col`` can fold them in order (VIN → phone → email →
# long-digit), matching the pure-Python implementation.
_PII_REGEXES: tuple[tuple[str, str], ...] = (
    (r"\b(?=[A-HJ-NPR-Z0-9]*\d)[A-HJ-NPR-Z0-9]{17}\b", "[VIN]"),
    (
        r"(?<!\d)(?:\+?1[-.\s]?)?\(?\b\d{3}\b\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)",
        "[PHONE]",
    ),
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[EMAIL]"),
    (r"(?<!\d)\d{8,}(?!\d)", "[NUM]"),
)


def _scrub_pii_col(col_name: str) -> F.Column:
    """Native regex chain equivalent of ``pii.scrub_text``.

    Stays in the JVM so 2.5M-row complaint narratives don't cross the
    Py4J boundary per row.
    """
    out = F.col(col_name)
    for pattern, token in _PII_REGEXES:
        out = F.regexp_replace(out, pattern, token)
    return out


def _component_group_col(col_name: str) -> F.Column:
    """Top-level NHTSA component group; matches ``taxonomy.component_group``.

    Native equivalent of the Python helper: split on ``:``, take the
    first segment, trim, title-case only if input was all-upper.
    """
    head = F.trim(F.split(F.col(col_name), ":", 2).getItem(0))
    return F.when(head == F.upper(head), F.initcap(head)).otherwise(head)


def _component_leaf_col(col_name: str) -> F.Column:
    """Leaf NHTSA component name; matches ``taxonomy.component_leaf``."""
    parts = F.split(F.col(col_name), ":")
    leaf = F.trim(F.element_at(parts, -1))
    return F.when(leaf == F.upper(leaf), F.initcap(leaf)).otherwise(leaf)


def _yyyymmdd(col: str) -> F.Column:
    """Parse NHTSA's ``YYYYMMDD`` strings into a DATE.

    Uses ``try_to_date`` because NHTSA flat files routinely contain empty
    strings and sentinel zeros in date columns (e.g. blank ``faildate``,
    ``00000000`` ``endman``). Under Databricks' default ANSI mode, a
    plain ``to_date`` throws ``CANNOT_PARSE_TIMESTAMP`` on those rows
    and aborts the entire job; ``try_to_date`` coerces the bad value to
    NULL, which is what downstream expects.
    """
    return F.expr(f"try_to_date({col}, 'yyyyMMdd')")


def _try_int(col: str) -> F.Column:
    """Lenient int cast; NHTSA uses blank strings for missing numerics."""
    return F.expr(f"try_cast({col} as int)")


def _try_long(col: str) -> F.Column:
    """Lenient long cast; NHTSA uses blank strings for missing numerics."""
    return F.expr(f"try_cast({col} as long)")


def _yn_to_bool(col: str) -> F.Column:
    """Map NHTSA's Y/N (with N as default for blanks) to a bool."""
    return F.when(F.upper(F.col(col)) == "Y", F.lit(True)).otherwise(F.lit(False))


# ---------------------------------------------------------------------------
# Per-dataset writers
# ---------------------------------------------------------------------------


def write_silver_recalls(
    spark: SparkSession,
    cfg: ProjectConfig,
    taxonomy_path: str = "src/nhtsa_curator/ref/oem_groups.yml",
) -> int:
    """Build ``silver_recalls`` from ``bronze_recalls``."""
    _register_udfs(spark, taxonomy_path)

    bronze = f"{cfg.full_schema_name}.bronze_recalls"
    silver = f"{cfg.full_schema_name}.silver_recalls"

    df = spark.table(bronze)
    oem = F.expr("resolve_oem(maketxt)")

    out = df.withColumn("_oem", oem).select(
        F.col("record_id").cast("string").alias("record_id"),
        F.col("campno").alias("campaign_number"),
        F.col("mfgcampno").alias("mfr_campaign_number"),
        F.col("_oem.make_norm").alias("make_norm"),
        F.col("_oem.oem_group").alias("oem_group"),
        F.col("modeltxt").alias("model_norm"),
        _try_int("yeartxt").alias("model_year"),
        F.col("compname").alias("component_raw"),
        _component_group_col("compname").alias("component_group"),
        _component_leaf_col("compname").alias("component_leaf"),
        F.col("rcltypecd").alias("recall_type_code"),
        _try_long("potaff").alias("units_affected"),
        _yyyymmdd("odate").alias("owner_notify_date"),
        _yyyymmdd("rcdate").alias("record_creation_date"),
        _yyyymmdd("bgman").alias("manufacture_begin_date"),
        _yyyymmdd("endman").alias("manufacture_end_date"),
        F.col("fmvss").alias("fmvss"),
        F.col("desc_defect").alias("defect_description"),
        F.col("conequence_defect").alias("consequence_description"),
        F.col("corrective_action").alias("corrective_action"),
        F.col("notes").alias("notes"),
        F.col("_ingested_at").alias("_bronze_ingested_at"),
        F.current_timestamp().alias("_silver_at"),
    )
    out.write.format("delta").mode("overwrite").option(
        "overwriteSchema", "true"
    ).saveAsTable(silver)
    n = spark.table(silver).count()
    logger.info(f"silver_recalls: {n:,} rows")
    return n


def write_silver_complaints(
    spark: SparkSession,
    cfg: ProjectConfig,
    taxonomy_path: str = "src/nhtsa_curator/ref/oem_groups.yml",
) -> int:
    """Build ``silver_complaints`` with PII-scrubbed narratives."""
    _register_udfs(spark, taxonomy_path)

    bronze = f"{cfg.full_schema_name}.bronze_complaints"
    silver = f"{cfg.full_schema_name}.silver_complaints"

    df = spark.table(bronze)
    oem = F.expr("resolve_oem(maketxt)")

    out = df.withColumn("_oem", oem).select(
        F.col("cmplid").alias("complaint_id"),
        F.col("odino").alias("odi_number"),
        F.col("_oem.make_norm").alias("make_norm"),
        F.col("_oem.oem_group").alias("oem_group"),
        F.col("modeltxt").alias("model_norm"),
        _try_int("yeartxt").alias("model_year"),
        F.col("compdesc").alias("component_raw"),
        _component_group_col("compdesc").alias("component_group"),
        _component_leaf_col("compdesc").alias("component_leaf"),
        _yyyymmdd("faildate").alias("incident_date"),
        _yyyymmdd("ldate").alias("loaded_date"),
        _yyyymmdd("datea").alias("amend_date"),
        _yn_to_bool("crash").alias("crash"),
        _yn_to_bool("fire").alias("fire"),
        _try_int("injured").alias("injured"),
        _try_int("deaths").alias("deaths"),
        _try_long("miles").alias("miles"),
        F.col("city").alias("city"),
        F.col("state").alias("state"),
        _scrub_pii_col("cdescr").alias("narrative_clean"),
        F.length(F.col("cdescr")).alias("narrative_len_orig"),
        F.col("cmpl_type").alias("complaint_source"),
        F.col("_ingested_at").alias("_bronze_ingested_at"),
        F.current_timestamp().alias("_silver_at"),
    )
    out.write.format("delta").mode("overwrite").option(
        "overwriteSchema", "true"
    ).partitionBy("model_year").saveAsTable(silver)
    n = spark.table(silver).count()
    logger.info(f"silver_complaints: {n:,} rows (PII-scrubbed)")
    return n


def write_silver_investigations(
    spark: SparkSession,
    cfg: ProjectConfig,
    taxonomy_path: str = "src/nhtsa_curator/ref/oem_groups.yml",
) -> int:
    """Build ``silver_investigations`` with computed ``days_open``."""
    _register_udfs(spark, taxonomy_path)

    bronze = f"{cfg.full_schema_name}.bronze_investigations"
    silver = f"{cfg.full_schema_name}.silver_investigations"

    df = spark.table(bronze)
    oem = F.expr("resolve_oem(mfr_name)")

    # NHTSA encodes the investigation type as the two-character prefix on
    # ``nhtsa_action_number`` (e.g. "PE09-023" → PE, "EA10-001" → EA,
    # "RQ11-003" → RQ, "AQ09-001" → AQ, "DP05-002" → DP). The separate
    # ``investigation_type`` / ``action_letter_date`` fields the prior
    # schema exposed no longer exist in FLAT_INV.
    out = (
        df.withColumn("_oem", oem)
        .withColumn("open_d", _yyyymmdd("action_open_date"))
        .withColumn("close_d", _yyyymmdd("action_close_date"))
        .select(
            F.col("nhtsa_action_number").alias("nhtsa_action_number"),
            F.substring(F.col("nhtsa_action_number"), 1, 2).alias("investigation_type"),
            F.col("subject").alias("subject"),
            F.col("summary").alias("summary"),
            F.col("campno").alias("linked_campaign_number"),
            F.col("_oem.make_norm").alias("mfr_norm"),
            F.col("_oem.oem_group").alias("oem_group"),
            F.col("maketxt").alias("make_raw"),
            F.col("modeltxt").alias("model_norm"),
            _try_int("yeartxt").alias("model_year"),
            F.col("component_name").alias("component_raw"),
            _component_group_col("component_name").alias("component_group"),
            F.col("open_d").alias("open_date"),
            F.col("close_d").alias("close_date"),
            F.when(
                F.col("close_d").isNotNull() & F.col("open_d").isNotNull(),
                F.datediff(F.col("close_d"), F.col("open_d")),
            )
            .otherwise(
                F.when(
                    F.col("open_d").isNotNull(),
                    F.datediff(F.current_date(), F.col("open_d")),
                ).otherwise(F.lit(None).cast("int"))
            )
            .alias("days_open"),
            F.when(F.col("close_d").isNotNull(), F.lit("closed"))
            .otherwise(F.lit("open"))
            .alias("status"),
            F.col("_ingested_at").alias("_bronze_ingested_at"),
            F.current_timestamp().alias("_silver_at"),
        )
    )
    out.write.format("delta").mode("overwrite").option(
        "overwriteSchema", "true"
    ).saveAsTable(silver)
    n = spark.table(silver).count()
    logger.info(f"silver_investigations: {n:,} rows")
    return n


def write_silver_tsbs(
    spark: SparkSession,
    cfg: ProjectConfig,
    taxonomy_path: str = "src/nhtsa_curator/ref/oem_groups.yml",
) -> int:
    """Build ``silver_tsbs`` (index-only; parsed text is a separate table)."""
    _register_udfs(spark, taxonomy_path)

    bronze = f"{cfg.full_schema_name}.bronze_tsb_index"
    silver = f"{cfg.full_schema_name}.silver_tsbs"

    df = spark.table(bronze)
    oem = F.expr("resolve_oem(maketxt)")

    # NHTSA's MfrComms schema replaced ``pdf_path`` with a 4000-char
    # inline ``summary``. We surface the new metadata (communication_type,
    # mfr_component_system/subsystem) since silver is the last point
    # where the raw bronze names are still available.
    out = df.withColumn("_oem", oem).select(
        F.col("tsb_id").alias("tsb_id"),
        F.col("nhtsa_item_number").alias("nhtsa_item_number"),
        F.col("replacement_bulletin_no").alias("replacement_bulletin_no"),
        F.col("_oem.make_norm").alias("make_norm"),
        F.col("_oem.oem_group").alias("oem_group"),
        F.col("modeltxt").alias("model_norm"),
        _try_int("yeartxt").alias("model_year"),
        F.col("component_desc").alias("component_raw"),
        _component_group_col("component_desc").alias("component_group"),
        F.col("mfr_component_system").alias("mfr_component_system"),
        F.col("mfr_component_subsystem").alias("mfr_component_subsystem"),
        F.col("communication_type").alias("communication_type"),
        F.col("summary").alias("summary"),
        _yyyymmdd("orig_date").alias("original_date"),
        _yyyymmdd("changed_date").alias("changed_date"),
        F.year(_yyyymmdd("orig_date")).alias("bulletin_year"),
        F.col("_ingested_at").alias("_bronze_ingested_at"),
        F.current_timestamp().alias("_silver_at"),
    )
    out.write.format("delta").mode("overwrite").option(
        "overwriteSchema", "true"
    ).partitionBy("bulletin_year").saveAsTable(silver)
    n = spark.table(silver).count()
    logger.info(f"silver_tsbs: {n:,} rows")
    return n


def write_silver_sgo(
    spark: SparkSession,
    cfg: ProjectConfig,
    taxonomy_path: str = "src/nhtsa_curator/ref/oem_groups.yml",
    key_column: str = "report_id",
    make_column: str = "make",
    sae_column: str = "sae_automation_level",
    incident_date_column: str = "incident_date",
) -> int:
    """Build ``silver_sgo_crashes`` with normalised SAE level + reporting entity.

    Schema is data-driven (we let bronze infer columns) so the column
    names passed in must match what bronze actually stored. Defaults
    reflect the SGO column headers normalised to snake_case by the
    bronze writer.
    """
    _register_udfs(spark, taxonomy_path)

    bronze = f"{cfg.full_schema_name}.bronze_sgo_crashes"
    silver = f"{cfg.full_schema_name}.silver_sgo_crashes"

    df = spark.table(bronze)
    bronze_cols = set(df.columns)

    # Resolve OEM only if make column is actually present.
    if make_column in bronze_cols:
        df = df.withColumn("_oem", F.expr(f"resolve_oem({make_column})"))
        df = df.withColumn("make_norm", F.col("_oem.make_norm"))
        df = df.withColumn("oem_group", F.col("_oem.oem_group"))
    else:
        df = df.withColumn("make_norm", F.lit(None).cast("string"))
        df = df.withColumn("oem_group", F.lit(None).cast("string"))

    # Canonicalise SAE level to "L0".."L5" — SGO files use various
    # phrasings ("Level 4", "L4", "SAE 4"). We extract the first digit.
    if sae_column in bronze_cols:
        df = df.withColumn(
            "sae_level",
            F.concat(F.lit("L"), F.regexp_extract(F.col(sae_column), r"(\d)", 1)),
        )
    else:
        df = df.withColumn("sae_level", F.lit(None).cast("string"))

    # Incident date → DATE. SGO mixes formats: most rows are ISO
    # ``YYYY-MM-DD``, but NHTSA also publishes month-precision strings
    # like ``"MAR-2026"`` when the reporter only provided a month, and
    # the ADAS snapshot occasionally uses ``MM/DD/YYYY``. Plain
    # ``to_date`` under ANSI mode throws ``CAST_INVALID_INPUT`` on the
    # non-ISO rows and aborts the job, so we ``try_to_date`` each known
    # format and coalesce; unknown shapes land as NULL.
    if incident_date_column in bronze_cols:
        c = incident_date_column
        df = df.withColumn(
            "incident_date_d",
            F.coalesce(
                F.expr(f"try_to_date(`{c}`, 'yyyy-MM-dd')"),
                F.expr(f"try_to_date(`{c}`, 'MM/dd/yyyy')"),
                F.expr(f"try_to_date(`{c}`, 'MMM-yyyy')"),
                F.expr(f"try_to_date(`{c}`)"),
            ),
        )
    else:
        df = df.withColumn("incident_date_d", F.lit(None).cast("date"))

    df = df.withColumn("_silver_at", F.current_timestamp())
    if key_column not in bronze_cols:
        raise ValueError(
            f"SGO bronze missing expected key column '{key_column}'; "
            f"saw {sorted(bronze_cols)}"
        )

    df.write.format("delta").mode("overwrite").option(
        "overwriteSchema", "true"
    ).saveAsTable(silver)
    n = spark.table(silver).count()
    logger.info(f"silver_sgo_crashes: {n:,} rows")
    return n
