"""Vector Search endpoint + index helpers for ``gold_narrative_chunks``.

The index is a **delta-sync index with managed embeddings**: we hand
Databricks a Delta table name and an embedding-endpoint name, and the
service keeps the index in lockstep with the source table via Change
Data Feed. This is why ``write_gold_narrative_chunks`` sets
``delta.enableChangeDataFeed = true`` on the source.

Why managed embeddings (and not self-managed):
    - No extra Spark job to embed + upsert. The service reads CDF and
      calls the embedding endpoint for us.
    - The embedding model is a config knob (``cfg.embedding_endpoint``),
      so switching models is a re-create, not a re-embed pipeline.
    - The source table already carries the metadata filters the agent
      needs (``make_norm``, ``model_year``, ``component_group``,
      ``event_date``, ``source_dataset``, ``oem_group``) — the index
      inherits those as filterable columns.

This module stays thin — it wraps the ``VectorSearchClient`` with
idempotent helpers so the notebook and the refresh job can both call
``ensure_endpoint`` + ``ensure_index`` without caring about prior state.

Typical lifecycle:

    client = VectorSearchClient(disable_notice=True)
    ensure_endpoint(client, cfg.vector_search_endpoint)
    ensure_index(client, cfg, source_table, index_name)
    wait_for_index_online(client, cfg.vector_search_endpoint, index_name)

    # Later (cron): the service auto-syncs from CDF. ``trigger_sync`` is
    # only needed if the on-demand refresh job wants a forced run.
    trigger_sync(client, cfg.vector_search_endpoint, index_name)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from loguru import logger

from .config import ProjectConfig, VectorSearchConfig

# Columns from ``gold_narrative_chunks`` we want the index to carry as
# filterable / returned metadata. The body column ``content`` is the
# embedded text and must be listed separately via ``embedding_source_column``.
INDEX_COLUMNS: list[str] = [
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
]

# Primary key of the index — stable sha256 computed in gold.py.
PRIMARY_KEY: str = "chunk_id"
EMBEDDING_SOURCE_COLUMN: str = "content"


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


def ensure_endpoint(
    client: Any, endpoint_name: str, endpoint_type: str = "STANDARD"
) -> None:
    """Create the Vector Search endpoint if it doesn't already exist.

    Endpoints are the compute substrate for indexes; a single endpoint
    hosts many indexes and is what the agent connects to at query time.
    They take several minutes to provision — so this function blocks
    until the endpoint reports ``ONLINE`` state.

    Args:
        client: a ``databricks.vector_search.client.VectorSearchClient``.
        endpoint_name: the endpoint name (e.g. from
            ``cfg.vector_search_endpoint``).
        endpoint_type: ``STANDARD`` for most workloads;
            ``STORAGE_OPTIMIZED`` for very large indexes.
    """
    try:
        ep = client.get_endpoint(endpoint_name)
        logger.info(
            "Endpoint {} already exists (state={})",
            endpoint_name,
            ep.get("endpoint_status", {}).get("state"),
        )
        return
    except Exception:  # noqa: BLE001 — SDK raises a generic error on 404
        pass

    logger.info("Creating endpoint {} ({})", endpoint_name, endpoint_type)
    client.create_endpoint(name=endpoint_name, endpoint_type=endpoint_type)
    _wait_for_endpoint_online(client, endpoint_name)


def _wait_for_endpoint_online(
    client: Any,
    endpoint_name: str,
    timeout_s: int = 1800,
    poll_s: int = 30,
) -> None:
    """Block until the endpoint reports ``ONLINE``. Raises on timeout."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        ep = client.get_endpoint(endpoint_name)
        state = ep.get("endpoint_status", {}).get("state")
        logger.info("Endpoint {} state={}", endpoint_name, state)
        if state == "ONLINE":
            return
        time.sleep(poll_s)
    raise TimeoutError(f"Endpoint {endpoint_name} did not reach ONLINE in {timeout_s}s")


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IndexSpec:
    """Immutable description of the index we want to ensure exists.

    ``source_table``, ``index_name``, ``primary_key``, and
    ``embedding_source_column`` must all agree with the columns
    produced by ``write_gold_narrative_chunks`` — drifting any of these
    means a full re-create.
    """

    endpoint_name: str
    source_table: str
    index_name: str
    primary_key: str
    embedding_source_column: str
    embedding_model_endpoint: str
    pipeline_type: str = "TRIGGERED"  # or "CONTINUOUS"


def ensure_index(client: Any, spec: IndexSpec) -> Any:
    """Create the delta-sync index if it doesn't exist.

    Idempotent: if an index already exists at ``index_name`` we return
    it as-is. Changing the embedding model or primary key requires a
    manual ``delete_index`` first — we don't silently re-create.
    """
    try:
        idx = client.get_index(
            endpoint_name=spec.endpoint_name, index_name=spec.index_name
        )
        logger.info("Index {} already exists — reusing", spec.index_name)
        return idx
    except Exception:  # noqa: BLE001
        pass

    logger.info(
        "Creating delta-sync index {} from {} (model={})",
        spec.index_name,
        spec.source_table,
        spec.embedding_model_endpoint,
    )
    return client.create_delta_sync_index(
        endpoint_name=spec.endpoint_name,
        source_table_name=spec.source_table,
        index_name=spec.index_name,
        primary_key=spec.primary_key,
        embedding_source_column=spec.embedding_source_column,
        embedding_model_endpoint_name=spec.embedding_model_endpoint,
        pipeline_type=spec.pipeline_type,
    )


def wait_for_index_online(
    client: Any,
    endpoint_name: str,
    index_name: str,
    timeout_s: int = 3600,
    poll_s: int = 30,
) -> None:
    """Block until the index is ``ONLINE`` *and* has synced at least once.

    Initial sync scans every row of the source table through the
    embedding endpoint, so on a fresh build this can take many minutes.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        idx = client.get_index(endpoint_name=endpoint_name, index_name=index_name)
        desc = idx.describe() if hasattr(idx, "describe") else idx
        status = desc.get("status", {}) if isinstance(desc, dict) else {}
        state = status.get("detailed_state") or status.get("ready")
        ready = status.get("ready") is True
        logger.info("Index {} state={} ready={}", index_name, state, ready)
        if ready:
            return
        time.sleep(poll_s)
    raise TimeoutError(f"Index {index_name} did not come online in {timeout_s}s")


def trigger_sync(client: Any, endpoint_name: str, index_name: str) -> None:
    """Force an incremental sync from the source Delta table.

    For ``TRIGGERED`` pipeline_type this is the refresh knob cron calls;
    for ``CONTINUOUS`` it's usually a no-op (the service keeps syncing).
    """
    idx = client.get_index(endpoint_name=endpoint_name, index_name=index_name)
    idx.sync()
    logger.info("Triggered sync for {}", index_name)


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


def similarity_search(
    client: Any,
    endpoint_name: str,
    index_name: str,
    query_text: str,
    vs_cfg: VectorSearchConfig,
    filters: dict | None = None,
    columns: list[str] | None = None,
) -> list[dict]:
    """Run a similarity search and return a list of dict hits.

    Filters follow Vector Search syntax:
        {"source_dataset": "complaints", "make_norm": "Tesla"}
        {"model_year >=": 2020}

    Returns one dict per hit, keyed by column name plus a ``score`` key.
    """
    idx = client.get_index(endpoint_name=endpoint_name, index_name=index_name)
    resp = idx.similarity_search(
        query_text=query_text,
        columns=columns or INDEX_COLUMNS,
        num_results=vs_cfg.num_results,
        filters=filters or {},
    )
    # The SDK returns a nested shape; flatten for caller ergonomics.
    # Note: ``manifest`` is at the top level of the response (sibling of
    # ``result``), not nested inside it. We check both positions so this
    # stays resilient if the SDK shape shifts.
    result = resp.get("result", {}) if isinstance(resp, dict) else {}
    data = result.get("data_array", []) or []
    manifest = (
        (resp.get("manifest") if isinstance(resp, dict) else None)
        or (result.get("manifest") if isinstance(result, dict) else None)
        or {}
    )
    col_names = [c["name"] for c in manifest.get("columns", [])]
    hits: list[dict] = []
    for row in data:
        hit = dict(zip(col_names, row, strict=False))
        hits.append(hit)
    return hits


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def build_default_spec(cfg: ProjectConfig) -> IndexSpec:
    """Produce the canonical spec for the NHTSA narrative-chunk index."""
    source_table = f"{cfg.full_schema_name}.gold_narrative_chunks"
    index_name = f"{cfg.full_schema_name}.gold_narrative_chunks_index"
    return IndexSpec(
        endpoint_name=cfg.vector_search_endpoint,
        source_table=source_table,
        index_name=index_name,
        primary_key=PRIMARY_KEY,
        embedding_source_column=EMBEDDING_SOURCE_COLUMN,
        embedding_model_endpoint=cfg.embedding_endpoint,
        pipeline_type="TRIGGERED",
    )
