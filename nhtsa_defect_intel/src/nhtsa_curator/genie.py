"""Genie space companion helpers.

Genie spaces are configured **interactively in the Databricks UI** —
there is no public DDL for "create a space". What Phase 3 owns
programmatically is everything *around* the space:

    1. The list of tables that should be exposed (the star-schema
       contract — narrow on purpose, see `docs/phase3_implementation.md`).
    2. ``COMMENT ON`` statements that drive Genie's column
       descriptions (Genie reads these directly when generating SQL,
       so curating them is the single biggest text-to-SQL lever).
    3. Verification queries we run against the configured space to
       confirm a space-id was wired correctly and that the space sees
       the tables we expect.
    4. A list of "trusted assets" (saved queries / example questions)
       to seed the space with — recorded in YAML so deployment is
       reproducible.

The space-id itself is stored in ``project_config.yml`` under
``genie_space_id`` (one per env). It's set manually after the first
space is created in the UI, then committed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from .config import ProjectConfig

# Tables exposed to Genie. Order matters only for human readability;
# Genie discovers the join graph from FK columns.
GENIE_TABLES: tuple[str, ...] = (
    "dim_vehicle",
    "dim_component",
    "dim_oem_group",
    "dim_date",
    "gold_recalls_fact",
    "gold_complaints_fact",
    "gold_investigations_fact",
)

# Per-table / per-column natural-language descriptions. Genie surfaces
# these to the LLM at SQL-generation time. Be specific about *units*
# (years are 4-digit ints; counts are integers; flags are 0/1) and
# *grain* (one row per…) — Genie hallucinates less when the grain is
# explicit.
TABLE_COMMENTS: dict[str, str] = {
    "dim_vehicle": (
        "One row per (make_norm, model_norm, model_year) combination "
        "observed across recalls, complaints, and TSBs. Join via "
        "vehicle_key (xxhash64 surrogate)."
    ),
    "dim_component": (
        "One row per (component_group, component_leaf) seen in NHTSA "
        "data. Component group is the top-level category (e.g. "
        "'Engine And Engine Cooling'); leaf is the specific subsystem "
        "(e.g. 'Cylinder Block'). Join via component_id."
    ),
    "dim_oem_group": (
        "Corporate parent groupings of vehicle makes (e.g. Chevrolet "
        "→ General Motors). Use this when an analyst asks about a "
        "manufacturer rather than a single brand. Join via oem_group_id."
    ),
    "dim_date": (
        "Calendar dimension keyed on date_key (DATE). One row per day "
        "from the earliest event in the silver layer to today."
    ),
    "gold_recalls_fact": (
        "One row per (campaign_number, vehicle, component) — i.e. the "
        "same recall campaign expands to multiple rows when it covers "
        "multiple makes/models/years. event_date = owner_notify_date. "
        "units_affected is a per-row count."
    ),
    "gold_complaints_fact": (
        "One row per consumer complaint (odi_number is unique). "
        "crash/fire/injured/deaths are 0/1 flags except deaths and "
        "injured which are integer counts. event_date = incident_date."
    ),
    "gold_investigations_fact": (
        "One row per NHTSA investigation (nhtsa_action_number is "
        "unique). status is 'open' or 'closed'; days_open is "
        "computed (close_date - open_date) clamped to today for open "
        "cases. event_date = open_date."
    ),
}

# Column-level comments. Listed only where the column name isn't
# self-explanatory; Genie's prompt budget is finite, so be selective.
COLUMN_COMMENTS: dict[tuple[str, str], str] = {
    ("dim_vehicle", "make_norm"): "Manufacturer brand (canonical, title-cased).",
    ("dim_vehicle", "model_norm"): "Model name (canonical, title-cased).",
    ("dim_vehicle", "model_year"): "4-digit model year as integer.",
    ("dim_component", "component_group"): "Top-level NHTSA component category.",
    (
        "dim_component",
        "component_leaf",
    ): "Most-specific subsystem in the NHTSA hierarchy.",
    (
        "dim_oem_group",
        "oem_group",
    ): "Corporate parent (e.g. 'General Motors', 'Stellantis', 'Tesla').",
    (
        "gold_recalls_fact",
        "recall_type_code",
    ): "NHTSA recall type — 'V'=vehicle, 'E'=equipment, 'T'=tire, 'C'=child seat.",
    (
        "gold_recalls_fact",
        "fmvss",
    ): "Federal Motor Vehicle Safety Standard number, if the recall references one.",
    (
        "gold_recalls_fact",
        "units_affected",
    ): "Number of vehicles/units in this campaign row.",
    ("gold_complaints_fact", "crash"): "1 if the consumer reported a crash, else 0.",
    ("gold_complaints_fact", "fire"): "1 if the consumer reported a fire, else 0.",
    ("gold_complaints_fact", "injured"): "Integer count of injured persons reported.",
    ("gold_complaints_fact", "deaths"): "Integer count of fatalities reported.",
    ("gold_complaints_fact", "miles"): "Vehicle mileage at incident, if reported.",
    (
        "gold_complaints_fact",
        "complaint_source",
    ): "Channel: 'OWNR' (owner-submitted) | 'MFR' (mfr-forwarded).",
    (
        "gold_investigations_fact",
        "investigation_type",
    ): (
        "PE (Preliminary Evaluation), EA (Engineering Analysis), "
        "DP (Defect Petition), RQ (Recall Query), TT (Trend), AQ (Audit Query)."
    ),
    ("gold_investigations_fact", "status"): "'open' or 'closed'.",
    (
        "gold_investigations_fact",
        "days_open",
    ): "Calendar days the investigation has been open (or was open, if closed).",
}


# ---------------------------------------------------------------------------
# Trusted asset seeds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrustedQuery:
    """A 'good' canonical question + SQL pair to seed the Genie space.

    Genie uses trusted assets as few-shot examples when generating SQL
    for new questions. Curating ~10-20 of these covers the common
    question shapes and dramatically improves SQL quality.
    """

    name: str
    question: str
    sql: str
    description: str = ""


def default_trusted_queries(cfg: ProjectConfig) -> list[TrustedQuery]:
    """Return the seed trusted-query set for the NHTSA Genie space."""
    fq = cfg.full_schema_name
    return [
        TrustedQuery(
            name="recalls_by_make_year",
            question="How many recall campaigns were issued for Tesla vehicles in 2023?",
            sql=f"""
                SELECT COUNT(DISTINCT f.campaign_number) AS n_campaigns
                FROM {fq}.gold_recalls_fact f
                JOIN {fq}.dim_vehicle v ON v.vehicle_key = f.vehicle_key
                WHERE v.make_norm = 'Tesla'
                  AND YEAR(f.event_date) = 2023
            """.strip(),
            description="Recall campaign count per make + year — common volume question.",
        ),
        TrustedQuery(
            name="complaints_with_fire_by_oem_group",
            question=(
                "Which OEM groups had the most fire-related complaints "
                "in the last 3 years?"
            ),
            sql=f"""
                SELECT g.oem_group, COUNT(*) AS n_fire_complaints
                FROM {fq}.gold_complaints_fact f
                JOIN {fq}.dim_oem_group g ON g.oem_group_id = f.oem_group_id
                WHERE f.fire = 1
                  AND f.event_date >= add_months(current_date(), -36)
                GROUP BY g.oem_group
                ORDER BY n_fire_complaints DESC
                LIMIT 10
            """.strip(),
        ),
        TrustedQuery(
            name="open_investigations_by_component",
            question="What component groups have the most open investigations right now?",
            sql=f"""
                SELECT c.component_group, COUNT(*) AS n_open
                FROM {fq}.gold_investigations_fact i
                JOIN {fq}.dim_component c ON c.component_id = i.component_id
                WHERE i.status = 'open'
                GROUP BY c.component_group
                ORDER BY n_open DESC
            """.strip(),
        ),
        TrustedQuery(
            name="injuries_per_make_per_year",
            question=(
                "Total reported injuries from complaints, by make, by year "
                "— last 5 years."
            ),
            sql=f"""
                SELECT v.make_norm, YEAR(f.event_date) AS yr,
                       SUM(f.injured) AS injuries
                FROM {fq}.gold_complaints_fact f
                JOIN {fq}.dim_vehicle v ON v.vehicle_key = f.vehicle_key
                WHERE f.event_date >= add_months(current_date(), -60)
                GROUP BY v.make_norm, YEAR(f.event_date)
                ORDER BY yr DESC, injuries DESC
            """.strip(),
        ),
        TrustedQuery(
            name="recall_campaigns_units_by_component",
            question="Top 10 component groups by units recalled (all-time).",
            sql=f"""
                SELECT c.component_group, SUM(f.units_affected) AS units
                FROM {fq}.gold_recalls_fact f
                JOIN {fq}.dim_component c ON c.component_id = f.component_id
                GROUP BY c.component_group
                ORDER BY units DESC
                LIMIT 10
            """.strip(),
        ),
    ]


# ---------------------------------------------------------------------------
# DDL emitters
# ---------------------------------------------------------------------------


def emit_table_comment_ddl(cfg: ProjectConfig) -> list[str]:
    """Emit ``COMMENT ON TABLE`` statements for every Genie-exposed table.

    Idempotent — Spark SQL's ``COMMENT ON`` overwrites in place.
    """
    out = []
    for table in GENIE_TABLES:
        comment = TABLE_COMMENTS.get(table)
        if not comment:
            continue
        safe = comment.replace("'", "''")
        out.append(f"COMMENT ON TABLE {cfg.full_schema_name}.{table} IS '{safe}'")
    return out


def emit_column_comment_ddl(cfg: ProjectConfig) -> list[str]:
    """Emit ``ALTER TABLE ... CHANGE COLUMN ... COMMENT`` statements.

    Skips columns whose host table doesn't yet exist (cold-start safety).
    """
    out = []
    for (table, column), comment in COLUMN_COMMENTS.items():
        safe = comment.replace("'", "''")
        out.append(
            f"ALTER TABLE {cfg.full_schema_name}.{table} "
            f"ALTER COLUMN {column} COMMENT '{safe}'"
        )
    return out


def apply_comments(spark: Any, cfg: ProjectConfig) -> int:
    """Run all comment DDL against the workspace.

    Returns the number of statements that succeeded. Logs and skips
    failures (typically: table not yet built, or column renamed) rather
    than aborting — the comments are advisory metadata, not load-bearing.
    """
    n_ok = 0
    for stmt in emit_table_comment_ddl(cfg) + emit_column_comment_ddl(cfg):
        try:
            spark.sql(stmt)
            n_ok += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("Comment DDL skipped: {} ({})", stmt[:80], exc)
    logger.info("Applied {} Genie comment statements", n_ok)
    return n_ok


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


@dataclass
class GenieSpaceVerification:
    """Summary returned by :func:`verify_genie_space`."""

    space_id: str
    tables_expected: list[str] = field(default_factory=list)
    tables_present: list[str] = field(default_factory=list)
    tables_missing: list[str] = field(default_factory=list)
    ok: bool = False


def verify_genie_space(spark: Any, cfg: ProjectConfig) -> GenieSpaceVerification:
    """Confirm the gold tables Genie depends on actually exist.

    This does **not** introspect the Genie space itself (no public
    SDK at the moment). It validates the *contract*: every table the
    space is configured against must exist and be queryable.
    """
    space_id = cfg.genie_space_id or "<not-set>"
    present, missing = [], []
    for table in GENIE_TABLES:
        fq = f"{cfg.full_schema_name}.{table}"
        try:
            spark.sql(f"SELECT 1 FROM {fq} LIMIT 1").collect()
            present.append(table)
        except Exception:  # noqa: BLE001
            missing.append(table)

    return GenieSpaceVerification(
        space_id=space_id,
        tables_expected=list(GENIE_TABLES),
        tables_present=present,
        tables_missing=missing,
        ok=(
            not missing
            and space_id != "<not-set>"
            and not space_id.startswith("PLACEHOLDER")
        ),
    )
