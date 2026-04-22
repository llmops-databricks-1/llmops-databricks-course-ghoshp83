"""NHTSA complaints (VOQ) reader.

Same pattern as recalls: bulk flat-file dump is primary, REST API is a
fallback for targeted lookups.

NB: Complaint narratives can contain consumer PII (names, phone, VIN).
PII scrubbing happens in the silver layer (Phase 2), not here. Bronze
intentionally preserves the raw payload.
"""

from __future__ import annotations

from collections.abc import Iterator

from loguru import logger

from .._typing import Json
from ._flat_files import COMPLAINTS_COLUMNS, parse_flat_file, read_zip_member
from .http import NhtsaHttpClient


def fetch_complaints_bulk(
    client: NhtsaHttpClient,
    bulk_zip_url: str,
) -> Iterator[dict]:
    """Stream the complaints bulk flat file as dicts."""
    logger.info(f"Downloading complaints bulk dump from {bulk_zip_url}")
    payload = client.get_bytes(bulk_zip_url)
    text = read_zip_member(payload, "FLAT_CMPL")
    n = 0
    for row in parse_flat_file(text, COMPLAINTS_COLUMNS):
        n += 1
        yield row
    logger.info(f"Complaints bulk yielded {n:,} rows")


def fetch_complaints_by_vehicle(
    client: NhtsaHttpClient,
    api_base: str,
    make: str,
    model: str,
    model_year: int,
) -> list[Json]:
    """Hit the complaints REST API for a specific make/model/year."""
    params = {"make": make, "model": model, "modelYear": str(model_year)}
    payload = client.get_json(api_base, params=params)
    return payload.get("results", []) if isinstance(payload, dict) else []
