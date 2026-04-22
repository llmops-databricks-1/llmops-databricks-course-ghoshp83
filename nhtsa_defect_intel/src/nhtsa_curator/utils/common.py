"""Common helpers shared by notebooks and deployment scripts.

Kept deliberately thin: each function is either reading a Databricks-
environment-only value (with a safe default when running locally) or
setting MLflow tracking URIs so notebooks can be developed against a
CLI profile without special-casing the notebook code.
"""

from __future__ import annotations

import os
from typing import Any


def get_widget(name: str, default: str | None = None) -> str | None:
    """Return a Databricks notebook widget value with a fallback default.

    When running outside a Databricks notebook (CI, pytest, CLI smoke),
    ``dbutils`` is not importable — we return ``default`` without a
    warning so the caller can treat the path uniformly.
    """
    try:
        from databricks.sdk.runtime import dbutils  # noqa: PLC0415 — lazy; avoids CI imports
    except Exception:  # noqa: BLE001
        return default
    try:
        return dbutils.widgets.get(name)
    except Exception:  # noqa: BLE001 — widget absent → fallback
        return default


def set_mlflow_tracking_uri() -> None:
    """Point MLflow at the right tracking/registry URI for the caller.

    Inside a Databricks runtime the defaults work out of the box; from a
    local CLI / notebook session we read a ``PROFILE`` env var (the same
    one the reference project uses) and build
    ``databricks://<profile>``. Calling this in a Databricks notebook is
    a no-op — the early return keeps it safe to place at the top of every
    deployment script.
    """
    if "DATABRICKS_RUNTIME_VERSION" in os.environ:
        return

    import mlflow  # noqa: PLC0415

    profile = os.environ.get("PROFILE")
    if not profile:
        return

    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")


def get_delta_table_version(spark: Any, full_table_name: str) -> str:
    """Return the latest version of a Delta table.

    Used by the register/deploy scripts to tag the MLflow run with the
    gold snapshot that the agent was validated against — crucial for
    reproducing a prod eval result later.
    """
    from delta.tables import DeltaTable  # noqa: PLC0415 — Spark-only

    delta_table = DeltaTable.forName(spark, full_table_name)
    return str(delta_table.history().select("version").first()[0])


__all__ = [
    "get_widget",
    "set_mlflow_tracking_uri",
    "get_delta_table_version",
]
