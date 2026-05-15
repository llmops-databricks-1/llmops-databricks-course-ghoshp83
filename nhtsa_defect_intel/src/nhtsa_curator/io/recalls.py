"""NHTSA recalls reader.

The bulk flat-file dump is the primary path (it's complete and updated
nightly). The recalls REST API is provided as a fallback for targeted
incremental pulls (e.g. one make/model/year).

Yields dicts already keyed by stable column names; downstream Spark
writes happen in :mod:`nhtsa_curator.bronze`.
"""

from __future__ import annotations

from collections.abc import Iterator

from loguru import logger

from .._typing import Json
from ._flat_files import RECALLS_COLUMNS, parse_flat_file, read_zip_member
from .http import NhtsaHttpClient


def fetch_recalls_bulk(
    client: NhtsaHttpClient,
    bulk_zip_url: str,
) -> Iterator[dict]:
    """Stream the recalls bulk flat file as dicts.

    The remote payload is a ZIP containing ``FLAT_RCL.txt``. We download
    in-memory because the file is typically < 100 MB.
    """
    logger.info(f"Downloading recalls bulk dump from {bulk_zip_url}")
    payload = client.get_bytes(bulk_zip_url)
    text = read_zip_member(payload, "FLAT_RCL")
    n = 0
    for row in parse_flat_file(text, RECALLS_COLUMNS):
        n += 1
        yield row
    logger.info(f"Recalls bulk yielded {n:,} rows")


def fetch_recalls_by_vehicle(
    client: NhtsaHttpClient,
    api_base: str,
    make: str,
    model: str,
    model_year: int,
) -> list[Json]:
    """Hit the recalls REST API for a specific make/model/year.

    Returns the raw ``results`` array. Useful for spot checks and for
    backfilling a single entity, but not the primary ingestion path.
    """
    params = {"make": make, "model": model, "modelYear": str(model_year)}
    payload = client.get_json(api_base, params=params)
    return payload.get("results", []) if isinstance(payload, dict) else []
