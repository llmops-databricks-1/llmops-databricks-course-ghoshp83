"""NHTSA defect investigations reader.

Two artefacts:

1. ``FLAT_INV.txt`` (bulk dump) → case-level metadata for every PE / EA
   / RQ / DP action ever opened.
2. Per-case PDF documents — defect petitions, MFR responses, closing
   reports — published on static.nhtsa.gov under predictable paths.

This module exposes the case-metadata reader (`fetch_investigations_bulk`)
and a generic PDF downloader. The actual list of per-case PDF URLs is
built in Phase 2 by the parser, since investigation case URLs need a
small amount of HTML scraping that we keep separate from bulk ingest.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from loguru import logger

from ._flat_files import INVESTIGATIONS_COLUMNS, parse_flat_file, read_zip_member
from .http import NhtsaHttpClient


def fetch_investigations_bulk(
    client: NhtsaHttpClient,
    bulk_zip_url: str,
) -> Iterator[dict]:
    """Stream the investigations flat-file dump as dicts."""
    logger.info(f"Downloading investigations bulk dump from {bulk_zip_url}")
    payload = client.get_bytes(bulk_zip_url)
    text = read_zip_member(payload, "FLAT_INV")
    n = 0
    for row in parse_flat_file(text, INVESTIGATIONS_COLUMNS):
        n += 1
        yield row
    logger.info(f"Investigations bulk yielded {n:,} rows")


def download_investigation_pdf(
    client: NhtsaHttpClient,
    pdf_url: str,
    dest_dir: str | Path,
    filename: str,
) -> Path:
    """Download a single investigation document PDF.

    ``dest_dir`` should be a UC volume path; ``filename`` is typically a
    sanitized form of the document title or the URL's terminal segment.
    """
    dest = Path(dest_dir) / filename
    return client.stream_to_file(pdf_url, dest)
